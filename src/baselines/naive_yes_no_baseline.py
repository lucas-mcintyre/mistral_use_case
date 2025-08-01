
"""Zero-shot leaf prediction with the Mistral chat API.

Naive strategy
========
1. Use a TF-IDF retrieval over the 2 257 leaf names to pick the top-K (default 30)
   most relevant candidate categories for each product description– cheap and local.
2. Ask the Mistral model whether the product belongs to *each* candidate leaf
   with a deterministic yes/no prompt.
3. Select the leaf as soon as a positive response ("Oui") is given (fall back to retrieval rank
   if the API never says yes).

Example usage
-------------
    export MISTRAL_API_KEY="sk-..."
    python src/baselines/naive_yes_no_baseline.py \
        --test  data_processed/test.jsonl \
        --mapping data_processed/mappings/leaf2id.json \
        --model mistral-small-latest \
        --output_dir models/naive_yes_no
"""

import argparse, json, os, pathlib, hashlib, joblib
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from mistralai import Mistral
from mistralai.models import UserMessage
from tqdm import tqdm

CACHE_FILE = "api_cache.joblib"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sha1(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()

class PromptCache:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        if self.path.exists():
            print(f"[PromptCache] Loading cache from {self.path}")
            self.mem = joblib.load(self.path)
        else:
            print(f"[PromptCache] No cache found at {self.path}, starting fresh.")
            self.mem = {}

    def get(self, key):
        print(f"[PromptCache] Getting key {key[:8]}...")  
        return self.mem.get(key)

    def set(self, key, value):
        print(f"[PromptCache] Setting key {key[:8]}...")  
        self.mem[key] = value
        joblib.dump(self.mem, self.path)
        print(f"[PromptCache] Cached response for key {key[:8]}...")

# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF retrieval over leaf names
# ──────────────────────────────────────────────────────────────────────────────

def build_leaf_retriever(leaf_names: List[str]):
    print(f"[TFIDF] Building TF-IDF vectorizer for {len(leaf_names)} leaf names...")
    vec = TfidfVectorizer()
    X = vec.fit_transform(leaf_names)
    print("[TFIDF] Vectorizer and matrix built.")
    print(f"[TFIDF] Shape of matrix: {X.shape}")  
    return vec, X

def topk_candidates(vec, X, query: str, leaf_names: List[str], k: int):
    print(f"[TFIDF] Transforming query: {query[:60]}...")  
    q = vec.transform([query])
    sims = (q @ X.T).toarray().ravel()
    print(f"[TFIDF] Similarities computed, shape: {sims.shape}")  
    top_idx = sims.argsort()[-k:][::-1]
    print(f"[TFIDF] Top-{k} candidates for query: {query[:60]}... -> {[leaf_names[i] for i in top_idx[:3]]} ...")
    print(f"[TFIDF] Top indices: {top_idx[:5]}")  
    return [leaf_names[i] for i in top_idx]

# ──────────────────────────────────────────────────────────────────────────────
# Mistral yes/no scoring
# ──────────────────────────────────────────────────────────────────────────────

def ask_yes_no(client: Mistral, model: str, product_text: str, leaf: str, cache: PromptCache):
    print(f"[ASK] Preparing prompt for leaf '{leaf[:40]}...'")  
    prompt = (f"Le produit suivant appartient-il à la catégorie décrite ci-dessous ?\n\n"
              f"### Produit\n{product_text}\n\n"
              f"### Catégorie\n{leaf}\n\n"
              f"Répondez strictement par \"Oui\" ou \"Non\".")
    key = sha1(prompt)
    cached = cache.get(key)
    if cached is not None:
        print(f"[ASK] Cache hit for leaf '{leaf[:40]}...'")
        print(f"[ASK] Returning cached value: {cached}")  
        return cached

    print(f"[ASK] Querying Mistral for leaf '{leaf[:40]}...'")
    msg = [UserMessage(content=prompt)]
    print(f"[ASK] Sending prompt to Mistral API...")  
    resp = client.chat.complete(
        model=model,
        messages=msg
    )
    answer = resp.choices[0].message.content.strip().lower()
    print(f"[ASK] Mistral response: {answer}")
    pred_yes = answer.startswith("oui")
    print(f"[ASK] Prediction is {'YES' if pred_yes else 'NO'}")  
    cache.set(key, pred_yes)
    return pred_yes

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    print(f"[LOAD] Loading test data from {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 100 == 0 and i > 0:
                print(f"[LOAD] Loaded {i} lines so far...")  
            yield json.loads(line)
    print(f"[LOAD] Finished loading all lines from {path}")  

def eval_yes_no(test_path, leaf2id_path, output_dir, model_name,
                   k_candidates=30, limit=None):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Output dir: {output_dir}")
    cache = PromptCache(output_dir / CACHE_FILE)

    print(f"[SETUP] Initializing Mistral client with model '{model_name}'")
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    # load leaves & retrieval engine
    print(f"[LOAD] Loading leaf2id mapping from {leaf2id_path}")
    leaf2id = json.load(open(leaf2id_path, encoding="utf-8"))
    print(f"[LOAD] Loaded {len(leaf2id)} leaves from mapping.")  
    leaf_names = list(leaf2id.keys())
    vec, X = build_leaf_retriever(leaf_names)

    # Count total examples first
    total_examples = sum(1 for _ in load_jsonl(test_path))
    print(f"[SETUP] Total examples to process: {total_examples}")
    if limit:
        print(f"[SETUP] Will process up to {limit} examples (limit applied)")
        total_examples = min(total_examples, limit)

    y_true, y_pred = [], []
    print(f"[EVAL] Starting naive yes/no evaluation...")
    print(f"[EVAL] Will check up to {k_candidates} candidates per example")
    
    for i, rec in enumerate(load_jsonl(test_path)):
        if limit and i >= limit:
            print(f"[EVAL] Reached limit of {limit} examples.")
            break
            
        text = rec["text"]
        true_leaf = rec["label"]
        print(f"\n[EVAL] Example {i+1}/{total_examples}:")
        print(f"  Text: {text[:80]}...")
        print(f"  True label: {true_leaf}")
        
        # Get candidates
        print(f"  [RETRIEVAL] Getting top-{k_candidates} candidates...")
        cands = topk_candidates(vec, X, text, leaf_names, k=k_candidates)
        print(f"  [RETRIEVAL] Top 5 candidates: {cands[:5]}")
        
        # Check each candidate
        chosen = None
        api_calls_made = 0
        cache_hits = 0
        
        for j, leaf in enumerate(cands):
            print(f"    [CAND {j+1}/{len(cands)}] Checking: {leaf[:60]}...")
            
            # Check if this candidate is the true label
            if leaf == true_leaf:
                print(f"    [CAND {j+1}] ⭐ This is the TRUE label!")
            
            if ask_yes_no(client, model_name, text, leaf, cache):
                print(f"    [CAND {j+1}] ✅ Mistral said 'Oui'! Selecting this leaf.")
                chosen = leaf
                break
            else:
                print(f"    [CAND {j+1}] ❌ Mistral said 'Non', continuing...")
        
        if chosen is None:
            chosen = cands[0]  # fallback to retrieval best
            print(f"    [FALLBACK] No 'Oui' found, using top retrieval candidate: {chosen}")
        
        y_true.append(true_leaf)
        y_pred.append(chosen)
        
        # Show result for this example
        correct = (chosen == true_leaf)
        status = "✅ CORRECT" if correct else "❌ WRONG"
        print(f"  [RESULT] Prediction: {chosen}")
        print(f"  [RESULT] {status} (true: {true_leaf})")
        
        # Show running stats
        correct_so_far = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
        accuracy_so_far = correct_so_far / len(y_pred)
        print(f"  [STATS] Running accuracy: {correct_so_far}/{len(y_pred)} = {accuracy_so_far:.3f}")

    # Final metrics
    print(f"\n[METRICS] Computing final metrics for {len(y_pred)} examples...")
    print(f"[METRICS] Number of predictions: {len(y_pred)}")  
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"[METRICS] F1 micro: {f1_micro:.4f}, F1 macro: {f1_macro:.4f}")  
    
    # Show some sample predictions
    print(f"\n[SAMPLES] Sample predictions (first 10):")
    for i in range(min(10, len(y_pred))):
        correct = "✅" if y_pred[i] == y_true[i] else "❌"
        print(f"  {i+1}. {correct} Pred: '{y_pred[i]}' | True: '{y_true[i]}'")
    
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    metrics = {"f1_micro": f1_micro, "f1_macro": f1_macro}
    print(f"[METRICS] Saving metrics to {output_dir / 'metrics.json'}")  
    json.dump(metrics, open(output_dir / "metrics.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[METRICS] Saving report to {output_dir / 'report.json'}")  
    json.dump(report, open(output_dir / "report.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("[METRICS] Final Results:")
    print(json.dumps(metrics, indent=2))
    print(f"[DONE] Metrics and report written to {output_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--output_dir", default="models/naive_yes_no")
    ap.add_argument("--model", default="mistral-small-latest")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on #examples to score (cost control)")
    args = ap.parse_args()

    print("[MAIN] Starting naive yes/no baseline script with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("[MAIN] Calling eval_yes_no...")  
    eval_yes_no(args.test, args.mapping, args.output_dir, args.model,
                   limit=args.limit)