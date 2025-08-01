"""Tiny evaluation helpers for classification baselines **plus** a
remote‑evaluation helper for fine‑tuned Mistral models.

You can execute the script with the following command:
python -m eval_lib.core mistral \
    --model ft:classifier:ministral-3b-latest:eca5aeb1:20250731:0ba03b81 \
    --test data/test.jsonl \
    --cache src/ft_cls_cache.joblib \
    --limit 1000 \
    --report \
    > src/evaluation_report.txt

"""


from __future__ import annotations
import argparse, json, pathlib, sys, os, hashlib, itertools, joblib, time
from typing import List, Dict, Any, Tuple, OrderedDict as _OrderedDict
from collections import OrderedDict

from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             classification_report, confusion_matrix)

try:
    from mistralai import Mistral  # optional – only needed for the *mistral* mode
except ImportError:  # keep import‑time deps minimal
    Mistral = None  # type: ignore

# ‑‑ general helpers ────────────────────────────────────────────────────────────
_SUPPORTED_TXT = {".txt", ""}
_CACHE_FILE    = "src/ft_cls_cache.joblib"
_BATCH_DEFAULT = 32

sha1 = lambda t: hashlib.sha1(t.encode()).hexdigest()

class KVCache:
    """Simple on‑disk key‑value cache using *joblib* (SHA‑1 → prediction)."""
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.mem: Dict[str, Any] = joblib.load(path) if path.exists() else {}
    def get(self, k):
        return self.mem.get(k)
    def set(self, k, v):
        self.mem[k] = v; joblib.dump(self.mem, self.path)

# ‑‑ classic low‑level I/O ──────────────────────────────────────────────────────

def _read_labels(path: str | pathlib.Path) -> List[str]:
    """Load labels or predictions from .txt / .json / .jsonl (see README)."""
    p = pathlib.Path(path)
    if p.suffix in _SUPPORTED_TXT:
        return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    elif p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data[k] for k in sorted(data, key=lambda x: int(x) if str(x).isdigit() else x)]
        raise ValueError(f"Expected list or dict in JSON file {path}, got {type(data).__name__}")
    elif p.suffix == ".jsonl":
        labels = []
        for line in p.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            if isinstance(obj, dict) and "label" in obj:
                labels.append(obj["label"])
            else:
                labels.append(obj)
        return labels
    else:
        raise ValueError(f"Unsupported extension: {p.suffix}. Use .txt, .json or .jsonl")

# ‑‑ metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: List[str], y_pred: List[str]) -> OrderedDict:
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)=} vs {len(y_pred)=}")
    m = OrderedDict()
    m["accuracy"]        = accuracy_score(y_true, y_pred)
    m["f1_micro"]        = f1_score(y_true, y_pred, average="micro", zero_division=0)
    m["f1_macro"]        = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    m["recall_micro"]    = recall_score(y_true, y_pred, average="micro", zero_division=0)
    return m

def print_metrics(metrics: OrderedDict):
    longest = max(len(k) for k in metrics)
    for k, v in metrics.items():
        print(f"{k:<{longest}} : {v:.4f}")

# ‑‑ classic evaluate helpers (unchanged) ───────────────────────────────────────

def evaluate_files(ref_path: str, pred_path: str, *, report: bool = False):
    y_true = _read_labels(ref_path)
    y_pred = _read_labels(pred_path)
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)
    if report:
        print("\nClassification report (macro‑averaged):")
        print(classification_report(y_true, y_pred, digits=4))
    return metrics


def evaluate_mappings(id2path_path: str,
                      preds_path: str,
                      leaf2id_path: str | None = None,
                      *,
                      report: bool = False):
    id2path: Dict[int, str] = json.load(open(id2path_path, encoding="utf-8"))
    id2path = {int(k): v for k, v in id2path.items()}

    preds_raw = json.load(open(preds_path, encoding="utf-8")) if preds_path.endswith(".json") else None
    if preds_raw is None:
        preds_list = _read_labels(preds_path)
        preds_raw = {i: label for i, label in enumerate(preds_list)}
    else:
        preds_raw = {int(k): v for k, v in preds_raw.items()}

    missing = [k for k in preds_raw if k not in id2path]
    if missing:
        print(f"[WARN] {len(missing)} ids present in predictions but absent from id2path; they will be ignored.")

    if all(isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()) for v in preds_raw.values()):
        if leaf2id_path is None:
            raise ValueError("Predictions seem numeric – provide --leaf2id to map back to leaf names.")
        leaf2id = json.load(open(leaf2id_path, encoding="utf-8"))
        id2leaf = {v: k for k, v in leaf2id.items()}
        preds_raw = {k: id2leaf[int(v)] for k, v in preds_raw.items()}

    common_ids = sorted(k for k in preds_raw if k in id2path)
    y_true = [id2path[k] for k in common_ids]
    y_pred = [preds_raw[k] for k in common_ids]

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)
    if report:
        print("\nClassification report (macro‑averaged):")
        print(classification_report(y_true, y_pred, digits=4))
    return metrics

# ‑‑ NEW: evaluate a remote Mistral model ───────────────────────────────────────

def _chunked(it, size):
    it = iter(it)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def _classify_batch(client: "Mistral", model: str, texts: List[str]):
    """Robust batch classification using the *classifier* endpoint with fallback."""
    try:
        preds = client.classifiers.classify(model=model, inputs=texts).results
        # Each item is Tuple(str,label_scores) depending on SDK version; handle mapping
        out = []
        for p in preds:
            scores = p["leaf_path"].scores
            out.append(max(scores, key=scores.get))
        return out
    except Exception as e:
        print(f"[WARN] Batch classify failed → {e}. Falling back to single calls…")
        out = []
        for t in texts:
            try:
                resp = client.chat(model=model, messages=[{"role": "user", "content": t}])
                out.append(resp.choices[0].message.content.strip())
            except Exception as e2:
                print(f"  · Single classify on '{t[:30]}…' failed: {e2}")
                out.append("unknown")
        return out


def evaluate_mistral(test_jsonl: str,
                     model: str,
                     *,
                     api_key: str | None = None,
                     batch: int = _BATCH_DEFAULT,
                     cache_path: str | pathlib.Path | None = None,
                     limit: int | None = None,
                     report: bool = False):
    """Run the hosted *model* over **test_jsonl** and print metrics + sample preds.

    The test file must contain lines like `{ "text": …, "label": … }`.
    Predictions are cached in *ft_cls_cache.joblib* to avoid duplicate charges.
    """
    if Mistral is None:
        raise ImportError("mistralai not installed – `pip install mistralai` first.")
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("Set MISTRAL_API_KEY env var or pass api_key explicitly.")
    client = Mistral(api_key=api_key)

    cache = KVCache(pathlib.Path(cache_path or _CACHE_FILE))
    print(f"🔧 Using cache: {cache_path or _CACHE_FILE}")
    print(f"🤖 Model: {model}")
    print(f"📁 Test file: {test_jsonl}")
    print(f"📦 Batch size: {batch}")
    if limit:
        print(f"⏹️  Limit: {limit} examples")

    # Count total lines first
    total_lines = sum(1 for _ in open(test_jsonl, "r", encoding="utf-8"))
    print(f"📊 Total examples in file: {total_lines}")

    y_true, y_pred = [], []
    batch_count = 0
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for batch_records in _chunked(f, batch):
            batch_count += 1
            print(f"\n🔄 Processing batch {batch_count} / {total_lines//batch} ({len(batch_records)} examples)")
            
            records = [json.loads(l) for l in batch_records]
            texts = [r["text"] for r in records]
            gold  = [r["label"] for r in records]

            need_api, idxs = [], []
            preds_batch: List[str | None] = [None]*len(texts)
            for i, t in enumerate(texts):
                hit = cache.get(sha1(t))
                if hit is None:
                    need_api.append(t); idxs.append(i)
                else:
                    preds_batch[i] = hit

            cached_count = len(texts) - len(need_api)
            api_count = len(need_api)
            print(f"   📋 Cached: {cached_count}, API calls needed: {api_count}")

            if need_api:
                print(f"   🌐 Making {api_count} API calls...")
                fresh = _classify_batch(client, model, need_api)
                for i, p in zip(idxs, fresh):
                    preds_batch[i] = p; cache.set(sha1(texts[i]), p)
                print(f"   ✅ API calls completed")

            y_true.extend(gold); y_pred.extend(preds_batch)
            print(f"   📈 Progress: {len(y_true)}/{total_lines} examples processed")
            
            if limit and len(y_true) >= limit:
                y_true, y_pred = y_true[:limit], y_pred[:limit]
                print(f"   ⏹️  Reached limit of {limit} examples")
                break

    print(f"\n🎯 Evaluation completed! Processed {len(y_true)} examples in {batch_count} batches")

    # quick sanity sample
    print("\n🔍 Sample predictions (first 5):")
    for i in range(min(5, len(y_true))):
        print(f"  {i+1}. True='{y_true[i]}' | Pred='{y_pred[i]}'")

    # Create confusion matrix for top 15 most frequent predictions
    print("\n📊 Creating confusion matrix for top 15 most frequent predictions...")
    
    # Count frequency of each true label
    from collections import Counter
    true_counts = Counter(y_true)
    top_15_labels = [label for label, count in true_counts.most_common(15)]
    
    print(f"📈 Top 15 most frequent labels:")
    for i, (label, count) in enumerate(true_counts.most_common(15)):
        print(f"  {i+1:2d}. {label[:50]:<50} ({count} examples)")
    
    # Filter data to only include top 15 labels
    top_15_mask = [true in top_15_labels for true in y_true]
    y_true_top15 = [true for i, true in enumerate(y_true) if top_15_mask[i]]
    y_pred_top15 = [pred for i, pred in enumerate(y_pred) if top_15_mask[i]]
    
    print(f"📊 Creating confusion matrix for {len(y_true_top15)} examples (top 15 labels)")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_top15, y_pred_top15, labels=top_15_labels)
    
    # Print confusion matrix
    print("\n🔢 Confusion Matrix (Top 15 Labels):")
    print("Rows = True Labels, Columns = Predicted Labels")
    print("=" * 80)
    
    # Header row with predicted labels (truncated)
    header = "True\\Pred"
    for i, label in enumerate(top_15_labels):
        header += f" | {i+1:2d}"
    print(header)
    print("-" * 80)
    
    # Matrix rows
    for i, (true_label, row) in enumerate(zip(top_15_labels, cm)):
        row_str = f"{i+1:2d}     "
        for j, count in enumerate(row):
            if count > 0:
                row_str += f" | {count:2d}"
            else:
                row_str += f" |  ."
        print(row_str)
    
    # Summary statistics for confusion matrix
    print("\n📈 Confusion Matrix Summary:")
    total_correct = sum(cm[i][i] for i in range(len(top_15_labels)))
    total_predictions = sum(sum(row) for row in cm)
    accuracy_top15 = total_correct / total_predictions if total_predictions > 0 else 0
    print(f"  ✅ Correct predictions: {total_correct}/{total_predictions} = {accuracy_top15:.3f}")
    
    # Most confused pairs
    print("\n🤔 Most Confused Label Pairs:")
    confusion_pairs = []
    for i, true_label in enumerate(top_15_labels):
        for j, pred_label in enumerate(top_15_labels):
            if i != j and cm[i][j] > 0:  # Not diagonal and has confusion
                confusion_pairs.append((true_label, pred_label, cm[i][j]))
    
    # Sort by confusion count
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, (true_label, pred_label, count) in enumerate(confusion_pairs[:10]):  # Top 10 confusions
        print(f"  {i+1:2d}. '{true_label[:30]:<30}' → '{pred_label[:30]:<30}' ({count} times)")

    metrics = compute_metrics(y_true, y_pred)
    print("\n=== Evaluation on", test_jsonl, "===")
    print_metrics(metrics)
    if report:
        print("\nFull classification report (macro‑averaged):")
        print(classification_report(y_true, y_pred, digits=4))
    return metrics

# ‑‑ CLI entry‑point ────────────────────────────────────────────────────────────

def _cli():
    ap = argparse.ArgumentParser(description="Evaluate predictions vs ground truth.")
    sub = ap.add_subparsers(dest="mode", required=True)

    # classic (labels.txt / preds.txt)
    p_files = sub.add_parser("files", help="labels.txt / preds.txt mode")
    p_files.add_argument("--ref", required=True)
    p_files.add_argument("--pred", required=True)
    p_files.add_argument("--report", action="store_true")

    # mappings (id2path + preds)
    p_map = sub.add_parser("mappings", help="id2path.json + preds.json mode")
    p_map.add_argument("--id2path", required=True)
    p_map.add_argument("--preds", required=True)
    p_map.add_argument("--leaf2id", required=False)
    p_map.add_argument("--report", action="store_true")

    # NEW: remote Mistral model
    p_mis = sub.add_parser("mistral", help="Run fine‑tuned Mistral model on a JSONL test set")
    p_mis.add_argument("--model", required=True, help="Fine‑tuned model id, e.g. ft:mistral-small:abcd")
    p_mis.add_argument("--test", required=True, help="JSONL with {text,label}")
    p_mis.add_argument("--batch", type=int, default=_BATCH_DEFAULT)
    p_mis.add_argument("--cache", default=_CACHE_FILE)
    p_mis.add_argument("--limit", type=int)
    p_mis.add_argument("--report", action="store_true")

    args = ap.parse_args()

    if args.mode == "files":
        evaluate_files(args.ref, args.pred, report=args.report)
    elif args.mode == "mappings":
        evaluate_mappings(args.id2path, args.preds, args.leaf2id, report=args.report)
    elif args.mode == "mistral":
        evaluate_mistral(args.test, args.model,
                         batch=args.batch,
                         cache_path=args.cache,
                         limit=args.limit,
                         report=args.report)

if __name__ == "__main__":
    _cli()
