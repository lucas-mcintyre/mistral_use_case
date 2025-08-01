#!/usr/bin/env python
"""
predict_one.py – classify a single snippet with your fine-tuned model
Usage:
    export MISTRAL_API_KEY="sk-…"      # once
    python predict_one.py "My new text to classify"
"""

import os, sys, json, hashlib, pathlib, joblib
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()


# ── paths & helpers ─────────────────────────────────────────────
ROOT        = pathlib.Path("models/fine_tune")   # adjust if needed
CACHE_PATH  = ROOT / "ft_cls_cache.joblib"       # same file the training script uses
METRICS_JSON= ROOT / "metrics.json"              # stores the model id
sha1 = lambda t: hashlib.sha1(t.encode()).hexdigest()

# ── load the fine-tuned model name ──────────────────────────────
model_id = json.load(open(METRICS_JSON))["model"]   # e.g. "ft:mistral-small-latest:abcd1234"
model = 'ft:classifier:ministral-3b-latest:eca5aeb1:20250731:0ba03b81'

# ── lazy-initialise cache ──────────────────────────────────────
cache = joblib.load(CACHE_PATH) if CACHE_PATH.exists() else {}

def predict(text: str) -> str:
    h = sha1(text)
    if h in cache:                       # instant, no API call
        return cache[h]

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    resp = client.classifiers.classify(model=model, inputs=[text]).results[0]
    
    label = max(resp["leaf_path"].scores, key=resp["leaf_path"].scores.get)
    cache[h] = label                     # persist for next time
    joblib.dump(cache, CACHE_PATH)
    return label

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Need exactly one text argument")
    print(predict(sys.argv[1]))
