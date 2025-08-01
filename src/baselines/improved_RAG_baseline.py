"""Improved retrieval + single-prompt baseline.

What’s new vs. naive_yes_no
---------------------------
* **Dual retrieval**: TF-IDF (k_tfidf) ∪ FAISS on `mistral-embed` vectors (k_embed).
* **One-shot decision prompt**: the model sees _all_ candidates at once and must
  reply with the index.
* **Retrieval recall report** so you can grid-search ``k_tfidf`` & ``k_embed``
  until the true label is covered ≥ 95 %.

Example (default k_tfidf=5, k_embed=15)
---------------------------------------
```bash
export MISTRAL_API_KEY="sk-..."
python src/baselines/improved_RAG_baseline.py \
  --test      data_processed/test.jsonl \
  --mapping   data_processed/mappings/leaf2id.json \
  --model     mistral-small-latest \
  --output_dir models/improved_RAG
```
"""

import argparse, json, os, pathlib, hashlib, joblib, faiss, numpy as np
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from mistralai import Mistral
from mistralai.models import UserMessage
from tqdm import tqdm

CACHE_LLM   = "llm_cache.joblib"
CACHE_EMBED = "embed_cache.joblib"
FAISS_INDEX = "faiss_index.bin"

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def sha1(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()

class KVCache:
    """Simple disk-backed dict for both embeddings and LLM calls."""
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.mem = joblib.load(path) if path.exists() else {}
    def get(self, k):
        return self.mem.get(k)
    def set(self, k, v):
        self.mem[k] = v
        joblib.dump(self.mem, self.path)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Build retrieval engines
# ──────────────────────────────────────────────────────────────────────────────

def build_tfidf(leaf_names: List[str]):
    vec = TfidfVectorizer()
    X = vec.fit_transform(leaf_names)
    return vec, X

def build_faiss(leaf_names: List[str], client: Mistral, embed_cache: KVCache, faiss_path: pathlib.Path):
    print(f"[FAISS] Checking for existing index at {faiss_path}")
    if faiss_path.exists():
        print(f"[FAISS] Loading existing index from {faiss_path}")
        index = faiss.read_index(str(faiss_path))
        # We still need to build the vectors matrix for search
        vectors = []
        for leaf in leaf_names:
            hid = sha1(leaf)
            emb = embed_cache.get(hid)
            if emb is None:
                print(f"[FAISS] Missing embedding for leaf: {leaf[:50]}...")
                resp = client.embeddings.create(model="mistral-embed", inputs=[leaf])
                emb = resp.data[0].embedding
                embed_cache.set(hid, emb)
            vectors.append(emb)
        mat = np.asarray(vectors).astype("float32")
        faiss.normalize_L2(mat)
        print(f"[FAISS] Loaded existing index with {index.ntotal} vectors")
        return index, mat
    else:
        print(f"[FAISS] Building new index...")
        dims = 1024  # mistral-embed returns 1024-dim vectors
        index = faiss.IndexFlatIP(dims)
        vectors = []
        for i, leaf in enumerate(leaf_names):
            if i % 100 == 0:
                print(f"[FAISS] Processing leaf {i+1}/{len(leaf_names)}: {leaf[:50]}...")
            hid = sha1(leaf)
            emb = embed_cache.get(hid)
            if emb is None:
                resp = client.embeddings.create(model="mistral-embed", inputs=[leaf])
                emb = resp.data[0].embedding
                embed_cache.set(hid, emb)
            vectors.append(emb)
        mat = np.asarray(vectors).astype("float32")
        # normalise to unit length (cosine sim)
        faiss.normalize_L2(mat)
        index.add(mat)
        print(f"[FAISS] Built index with {index.ntotal} vectors")
        print(f"[FAISS] Saving index to {faiss_path}")
        faiss.write_index(index, str(faiss_path))
        return index, mat

# ──────────────────────────────────────────────────────────────────────────────
# 2. Candidate retrieval
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_candidates(product_text: str, *,
                        vec, X_tfidf, tfidf_k: int,
                        client, faiss_index, leaf_names: List[str], embed_k: int):
    # TF-IDF part
    print(f"      [TFIDF] Computing similarities for top-{tfidf_k}...")
    q = vec.transform([product_text])
    sims = (q @ X_tfidf.T).toarray().ravel()
    tf_idx = sims.argsort()[-tfidf_k:][::-1]
    cand_ids = set(tf_idx)
    print(f"      [TFIDF] Top-{tfidf_k} indices: {tf_idx}")

    # Embedding part
    print(f"      [EMBED] Computing embedding and FAISS search for top-{embed_k}...")
    emb = client.embeddings.create(model="mistral-embed", inputs=[product_text]).data[0].embedding
    emb = np.asarray(emb, dtype="float32")[None, :]
    faiss.normalize_L2(emb)
    _, faiss_idx = faiss_index.search(emb, embed_k)
    cand_ids.update(faiss_idx[0])
    print(f"      [EMBED] FAISS top-{embed_k} indices: {faiss_idx[0]}")
    
    final_candidates = [leaf_names[i] for i in cand_ids]
    print(f"      [MERGE] Combined {len(cand_ids)} unique candidates from TF-IDF + FAISS")
    return final_candidates

# ──────────────────────────────────────────────────────────────────────────────
# 3. Single prompt classification
# ──────────────────────────────────────────────────────────────────────────────

def llm_choose(client: Mistral, model: str, product_text: str, cands: List[str], llm_cache: KVCache):
    numbered = [f"{i+1}. {c}" for i, c in enumerate(cands)]
    prompt = ("Tu es un classificateur produit, expert dans la catégorisation et les descriptions des produits. "
              "Je vais te donner une description d'un produit, et tu devras choisir la catégorie la plus appropriée parmi les choix donnés par la suite."
              f"Description du produit: \"{product_text[:400]}\"\n"
              "Choisis la catégorie la plus appropriée, mais réponds UNIQUEMENT par le numéro de la catégorie, sans aucun autre commentaire."
              + "\n Les choix possibles sont : " + "\n ".join(numbered))
    hid = sha1(prompt)
    cached = llm_cache.get(hid)
    if cached is not None:
        print(f"    [CACHE] ✅ Using cached answer: {cached}")
        return cached
    
    print(f"    [API] 🌐 Querying Mistral for choice from {len(cands)} candidates...")
    resp = client.chat.complete(model=model, messages=[UserMessage(content=prompt)], temperature=0, max_tokens=4)
    answer = resp.choices[0].message.content.strip()
    print(f"    [API] 📝 Response: '{answer}'")
    llm_cache.set(hid, answer)
    return answer

# Parse the model's answer safely

def parse_choice(answer: str, n_opts: int):
    for tok in answer.split():
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < n_opts:
                print(f"[PARSE] Parsed choice: {idx}")
                return idx
    print(f"[PARSE] No valid choice found in answer: {answer}")
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 4. Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def run(test_path, mapping_path, output_dir, model_name,
        k_tfidf, k_embed, limit=None):
    out = pathlib.Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Output directory: {out}")
    llm_cache   = KVCache(out / CACHE_LLM)
    embed_cache = KVCache(out / CACHE_EMBED)
    print(f"[SETUP] Initialized caches: LLM={out / CACHE_LLM}, Embed={out / CACHE_EMBED}")

    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    print(f"[SETUP] Initialized Mistral client with model: {model_name}")

    # Load leaves and retrieval engines
    print(f"[LOAD] Loading leaf mapping from {mapping_path}")
    leaf2id = json.load(open(mapping_path, encoding="utf-8"))
    leaf_names = list(leaf2id.keys())
    print(f"[LOAD] Loaded {len(leaf_names):,} leaf categories")
    
    print(f"[BUILD] Building retrieval indices...")
    print(f"[BUILD] Building TF-IDF index...")
    vec, X_tfidf = build_tfidf(leaf_names)
    print(f"[BUILD] Building FAISS index...")
    faiss_index, _ = build_faiss(leaf_names, client, embed_cache, out / FAISS_INDEX)
    print(f"[BUILD] ✅ Built TF-IDF and FAISS indices")

    # Count total examples first
    total_examples = sum(1 for _ in load_jsonl(test_path))
    print(f"[SETUP] Total examples to process: {total_examples}")
    if limit:
        print(f"[SETUP] Will process up to {limit} examples (limit applied)")
        total_examples = min(total_examples, limit)

    # Metrics accumulators
    y_true, y_pred = [], []
    retrieval_hits = 0
    api_calls = 0
    cache_hits = 0

    print(f"[EVAL] Starting evaluation with k_tfidf={k_tfidf}, k_embed={k_embed}")
    print(f"[EVAL] Will retrieve up to {k_tfidf + k_embed} candidates per example")
    
    for i, rec in enumerate(load_jsonl(test_path)):
        if limit and i >= limit:
            print(f"[EVAL] Reached limit of {limit} examples")
            break
            
        text, true_leaf = rec["text"], rec["label"]
        print(f"\n[EVAL] Example {i+1}/{total_examples}:")
        print(f"  Text: {text[:80]}...")
        print(f"  True label: {true_leaf}")
        
        # Retrieve candidates
        print(f"  [RETRIEVAL] Getting candidates...")
        cands = retrieve_candidates(text, vec=vec, X_tfidf=X_tfidf, tfidf_k=k_tfidf,
                                    client=client, faiss_index=faiss_index,
                                    leaf_names=leaf_names, embed_k=k_embed)
        print(f"  [RETRIEVAL] Found {len(cands)} unique candidates")
        print(f"  [RETRIEVAL] Top 5 candidates: {cands[:5]}")
        
        # Check if true label is in candidates
        if true_leaf in cands:
            retrieval_hits += 1
            print(f"  [RETRIEVAL] ✅ True label found in candidates!")
        else:
            print(f"  [RETRIEVAL] ❌ True label NOT found in candidates")
        
        # LLM decision
        print(f"  [LLM] Asking model to choose from {len(cands)} candidates...")
        answer = llm_choose(client, model_name, text, cands, llm_cache)
        idx = parse_choice(answer, len(cands))
        
        if idx is not None:
            chosen = cands[idx]
            print(f"  [LLM] Model chose candidate {idx+1}: {chosen}")
        else:
            chosen = cands[0]  # fallback to first candidate
            print(f"  [LLM]  Could not parse model answer, using first candidate: {chosen}")
        
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
        retrieval_recall_so_far = retrieval_hits / len(y_true)
        print(f"  [STATS] Running accuracy: {correct_so_far}/{len(y_pred)} = {accuracy_so_far:.3f}")
        print(f"  [STATS] Running retrieval recall: {retrieval_hits}/{len(y_true)} = {retrieval_recall_so_far:.3f}")

    # Final metrics
    print(f"\n[METRICS] Computing final metrics for {len(y_pred)} examples...")
    recall = retrieval_hits / len(y_true)
    print(f"[METRICS] Retrieval recall@{k_tfidf+k_embed}: {recall:.3%}")
    
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"[METRICS] F1 micro: {f1_micro:.4f}, F1 macro: {f1_macro:.4f}")
    
    # Show some sample predictions
    print(f"\n[SAMPLES] Sample predictions (first 10):")
    for i in range(min(10, len(y_pred))):
        correct = "✅" if y_pred[i] == y_true[i] else "❌"
        print(f"  {i+1}. {correct} Pred: '{y_pred[i]}' | True: '{y_true[i]}'")
    
    metrics = {"retrieval_recall": recall, "f1_micro": f1_micro, "f1_macro": f1_macro}
    print(f"[METRICS] Saving metrics to {out / 'metrics.json'}")
    json.dump(metrics, open(out / "metrics.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[METRICS] Saving params to {out / 'params.json'}")
    json.dump({"k_tfidf": k_tfidf, "k_embed": k_embed}, open(out / "params.json", "w"), indent=2)
    print(f"[METRICS] Final Results:")
    print(json.dumps(metrics, indent=2))
    print(f"[DONE] Results saved to {out}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--output_dir", default="models/improved_RAG")
    ap.add_argument("--model", default="mistral-small-latest")
    ap.add_argument("--k_tfidf", type=int, default=10)
    ap.add_argument("--k_embed", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run(args.test, args.mapping, args.output_dir, args.model,
        k_tfidf=args.k_tfidf, k_embed=args.k_embed, limit=args.limit)