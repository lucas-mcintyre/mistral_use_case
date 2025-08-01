"""Launch a *classifier* fine-tune on the Mistral platform and evaluate it.

Run example
-----------
```bash
export MISTRAL_API_KEY="sk-…"
python3 src/baselines/fine_tune_mistral.py \                                         
  --train      data_processed/train.jsonl \       
  --val        data_processed/val.jsonl \          
  --test       data_processed/test.jsonl \                                                                                
  --output_dir models/fine_tune \
  --base_model ministral-3b-latest
```
"""

from __future__ import annotations
import argparse, json, os, pathlib, time, hashlib, itertools, joblib
from typing import List, Dict

from mistralai import Mistral
from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb  

from dotenv import load_dotenv
load_dotenv()

CACHE_FILE = "ft_cls_cache.joblib"
BATCH = 32

# ───────────────────────────────────────────────────── helpers ──

def sha1(txt: str) -> str:
    return hashlib.sha1(txt.encode()).hexdigest()

class KVCache:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.mem: Dict[str, str] = joblib.load(path) if path.exists() else {}
    def get(self, k):
        return self.mem.get(k)
    def set(self, k, v):
        self.mem[k] = v; joblib.dump(self.mem, self.path)

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# write data -----------------------------------------------------

def write_cls_jsonl(src: str, dst: pathlib.Path, cap: int | None):
    """Return n_rows and sorted_unique_labels."""
    labels, n = set(), 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if cap and n >= cap:
                break
            rec = json.loads(line)
            fout.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": rec["text"]},
                            {"role": "assistant", "content": rec["label"]}
                        ],
                        "labels": {"leaf_path": rec["label"]}
                    },
                    ensure_ascii=False
                ) + "\n"
            )
            labels.add(rec["label"]); n += 1
    return n, sorted(labels)

# chunk util -----------------------------------------------------

def chunked(it, size):
    it = iter(it)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk: break
        yield chunk

# upload ---------------------------------------------------------

def upload_file(client: Mistral, path: pathlib.Path):
    resp = client.files.upload(
        file={
            "file_name": path.name, 
            "content": open(path, "rb")})
    print("⬆️  Uploaded", path.name, "→", resp.id)
    return resp.id

# create job -----------------------------------------------------

def create_and_start_job(client: Mistral,
                         train_id: str,
                         val_id: str,
                         *,
                         base_model: str,
                         steps: int,
                         lr: float,
                         labels: List[str]) -> str:

    job = client.fine_tuning.jobs.create(
        model=base_model,
        job_type="classifier",
        training_files=[{"file_id": train_id, "weight": 1}],
        validation_files=[val_id],
        classifier_targets=[{"name": "leaf_path", "labels": labels}],
        hyperparameters={"training_steps": steps, "learning_rate": lr},
        auto_start=True,          # ← start automatically after validation
        integrations=[
            {
                "project": "finetuning",
                "api_key": os.environ["WANDB_API_KEY"],
            }
        ]
    )

    print("🚀 Job", job.id, "created (auto-start enabled)")
    return job.id

# poll -----------------------------------------------------------

def wait(client: Mistral, job_id: str, sleep=30):
    while True:
        j = client.fine_tuning.jobs.get(job_id=job_id)
        if j.status in {"SUCCESS", "CANCELLED", "FAILED_VALIDATION", "CANCELLATION_REQUESTED", "FAILED", "CANCELLED"}: return j
        print("⏳", j.status, "— sleep", sleep, "s"); time.sleep(sleep)

# classify -------------------------------------------------------

def classify_batch(client: Mistral, model: str, texts: List[str]):
    """Use classifier API for the fine-tuned model."""
    try:
        preds = client.classifiers.classify(model=model, inputs=texts).results
        return [max(p[1], key=p[1].get) for p in preds]
    except Exception as e:
        print(f"Error in batch classification: {e}")
        # Fallback: return unknown for all predictions
        return ["unknown"] * len(texts)

# eval -----------------------------------------------------------

def eval_model(client: Mistral, model: str, test_jsonl: str, limit: int | None, cache: KVCache):
    y_true, y_pred = [], []
    for batch in chunked(load_jsonl(test_jsonl), BATCH):
        texts = [r["text"] for r in batch]
        gold  = [r["label"] for r in batch]
        uncached, idxs = [], []
        batch_pred = [None]*len(texts)
        for i, t in enumerate(texts):
            hit = cache.get(sha1(t))
            if hit is None:
                uncached.append(t); idxs.append(i)
            else:
                batch_pred[i] = hit
                # batch_pred[i] is a ClassificationTargetResult object, not subscriptable
                # Use .scores to get the dict of label:score
                batch_pred[i] = max(batch_pred[i].scores, key=batch_pred[i].scores.get)
        
        if uncached:
            fresh = classify_batch(client, model, uncached)
            # fresh is a list of dicts, each dict contains the scores of all labels for a single prediction
            for i, p in zip(idxs, fresh):
                batch_pred[i] = p
                cache.set(sha1(texts[i]), p)
                batch_pred[i] = max(batch_pred[i].scores, key=batch_pred[i].scores.get)
        
        y_true.extend(gold); y_pred.extend(batch_pred)
        if limit and len(y_true) >= limit:
            y_true, y_pred = y_true[:limit], y_pred[:limit]; break
    
    # Debug: show some predictions vs true labels
    print(f"\n🔍 Sample predictions (first 5):")
    for i in range(min(5, len(y_true))):
        print(f"  True: '{y_true[i]}' | Pred: '{y_pred[i]}'")
    
    # Normalize predictions to match true labels
    unique_true_labels = set(y_true)
    normalized_preds = []
    for pred in y_pred:
        # Try to find the closest match in true labels
        best_match = None
        best_score = 0
        for true_label in unique_true_labels:
            # Simple string similarity (you could use more sophisticated methods)
            if pred.lower() == true_label.lower():
                best_match = true_label
                break
            elif pred.lower() in true_label.lower() or true_label.lower() in pred.lower():
                score = len(set(pred.lower()) & set(true_label.lower())) / len(set(pred.lower()) | set(true_label.lower()))
                if score > best_score:
                    best_score = score
                    best_match = true_label
        
        if best_match is None:
            # If no good match, use the first true label as fallback
            best_match = list(unique_true_labels)[0]
            print(f"Could not match prediction '{pred}' to any true label, using fallback")
        
        normalized_preds.append(best_match)
    
    return y_true, normalized_preds

# main -----------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output_dir", default="models/fine_tune")
    p.add_argument("--base_model", default="mistral-small-latest")
    p.add_argument("--training_steps", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train", type=int)
    p.add_argument("--max_val", type=int)
    p.add_argument("--limit_test", type=int)
    args = p.parse_args()

    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    n_tr, labels = write_cls_jsonl(args.train, out/"train.jsonl", args.max_train)
    n_val, _     = write_cls_jsonl(args.val,   out/"val.jsonl",   args.max_val)
    print(f"Prepared {n_tr} train / {n_val} val rows • labels = {len(labels)}")

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    tr_id  = upload_file(client, out/"train.jsonl")
    val_id = upload_file(client, out/"val.jsonl")
    job_id = create_and_start_job(client, tr_id, val_id,
                                  base_model=args.base_model,
                                  steps=args.training_steps,
                                  lr=args.learning_rate,
                                  labels=labels)
    job = wait(client, job_id)
    if job.status != "SUCCESS":
        raise SystemExit("❌ job ended with status " + job.status)
    model_name = job.fine_tuned_model
    print("✅ Fine-tuned model:", model_name)
    # 4. Evaluate
    cache = KVCache(out / CACHE_FILE)
    y_true, y_pred = eval_model(client, model_name, args.test, args.limit_test, cache)
    metrics = {
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "model": model_name,
        "n_train": n_tr,
        "n_val": n_val,
    }
    json.dump(metrics, open(out / "metrics.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
    #client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    #canceled_jobs = client.fine_tuning.jobs.cancel(job_id = 'db819816-be67-4ad5-9d07-48b1ddea9f3a')
    #jobs = client.fine_tuning.jobs.list()
    #print(jobs)
    #print(canceled_jobs)
    #retrieve_job = client.fine_tuning.jobs.get(job_id = 'cbbef2d3-9439-47fb-80f2-f3514c79a73b')
    #print(retrieve_job)
    #started_job = client.fine_tuning.jobs.start(job_id='1a18379c-9522-46e7-97ad-deaab8280712')
    #print(started_job)
    #job = client.fine_tuning.jobs.get(job_id='1a18379c-9522-46e7-97ad-deaab8280712')
    #print(job)
    