#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deterministic preprocessing from raw CSV → train/val/test JSONL

Example:
    python datascripts/prepare_data.py \
        --input      data_raw/fr_amazon_catalog.csv \
        --output_dir data_processed \
        --val_frac   0.10 \
        --test_frac  0.10 \
        --seed       42
"""

import argparse, json, pathlib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from text_cleaning import clean_text

# ────────────────────────────────────────────────────────────────
def main(cfg):
    rng = np.random.RandomState(cfg.seed)
    out = pathlib.Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Reading raw …")
    df = pd.read_csv(cfg.input, encoding="utf-16", sep="\t")

    # ------------------------------------------------------------------
    # 1 . Basic cleaning
    # ------------------------------------------------------------------
    print("Cleaning text …")
    df = df.dropna(subset=["Product Category"])            # must have a label
    df["text"] = (df["Product Name"].fillna("") + " " +
                  df["Product Category"].fillna("")).apply(clean_text)

    df["path"] = df["Product Category"].str.strip()
    df["leaf"] = df["path"].str.split(" > ").str[-1]

    # ------------------------------------------------------------------
    # 2 . Category-level split  (80 % seen  / 20 % unseen)
    # ------------------------------------------------------------------
    all_paths = df["path"].unique()
    rng.shuffle(all_paths)
    n_seen = int(len(all_paths) * 0.80)
    seen_paths   = set(all_paths[:n_seen])
    unseen_paths = set(all_paths[n_seen:])

    df_seen   = df[df["path"].isin(seen_paths)].copy()
    df_unseen = df[df["path"].isin(unseen_paths)].copy()   # will live only in test

    # ------------------------------------------------------------------
    # 3 . Row-level split inside seen categories: 80 train /10 val /10 test
    #     -> no stratification (avoids 1-sample classes issue)
    # ------------------------------------------------------------------
    train_seen, tmp_seen = train_test_split(
        df_seen, test_size=cfg.val_frac + cfg.test_frac, random_state=cfg.seed)

    rel_test = cfg.test_frac / (cfg.val_frac + cfg.test_frac)
    val_seen,  test_seen = train_test_split(
        tmp_seen, test_size=rel_test, random_state=cfg.seed)

    train = train_seen
    val   = val_seen
    test  = pd.concat([test_seen, df_unseen], ignore_index=True)

    # ------------------------------------------------------------------
    # 4 . Build label-to-id mapping *only from the training leaves*
    # ------------------------------------------------------------------
    leaf2id = {p: i for i, p in enumerate(sorted(train["path"].unique()))}
    id2path = {v: k for k, v in leaf2id.items()}

    (out / "mappings").mkdir(exist_ok=True)
    json.dump(leaf2id, open(out / "mappings" / "leaf2id.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(id2path, open(out / "mappings" / "id2path.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 5 . Dump splits to JSONL
    # ------------------------------------------------------------------
    def dump(split_df: pd.DataFrame, name: str):
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                obj = {"text": row["text"], "label": row["path"]}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Wrote {name}: {len(split_df):,} rows → {path}")

    dump(train, "train")
    dump(val,   "val")
    dump(test,  "test")

    # ------------------------------------------------------------------
    # 6 . Console report
    # ------------------------------------------------------------------
    print("\nSplit summary")
    print(f"Seen categories (train/val/test):  {len(seen_paths):,}")
    print(f"Unseen categories (test only):     {len(unseen_paths):,}")
    print(f"Train rows: {len(train):,}  |  Val rows: {len(val):,}  |  Test rows: {len(test):,}")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_frac",  type=float, default=0.10)   # fraction of *rows* inside seen cats
    p.add_argument("--test_frac", type=float, default=0.10)   #   »       »         »
    p.add_argument("--seed",      type=int,   default=42)
    args = p.parse_args()
    main(args)
