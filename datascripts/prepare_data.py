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
    # 2 . Simple row-level split: 80 train / 10 val / 10 test
    # ------------------------------------------------------------------
    # No more separation between seen/unseen categories
    # All categories can appear in train/val/test
    train, tmp = train_test_split(
        df, test_size=cfg.val_frac + cfg.test_frac, random_state=cfg.seed)

    rel_test = cfg.test_frac / (cfg.val_frac + cfg.test_frac)
    val, test = train_test_split(
        tmp, test_size=rel_test, random_state=cfg.seed)

    # ------------------------------------------------------------------
    # 3 . Filter out parent paths when child paths exist
    # ------------------------------------------------------------------
    
    def filter_parent_paths(paths):
        """Remove parent paths when child paths exist.
        
        For example, if we have:
        - "A > B > C"
        - "A > B > C > D"
        
        We keep only "A > B > C > D" and remove "A > B > C"
        """
        paths_list = sorted(paths)
        filtered_paths = []
        
        for path in paths_list:
            is_parent = False
            for other_path in paths_list:
                if other_path != path and other_path.startswith(path + " > "):
                    is_parent = True
                    break
            if not is_parent:
                filtered_paths.append(path)
        
        return filtered_paths
    
    # Get all unique paths from training data
    all_train_paths = train["path"].unique()
    
    # Filter out parent paths
    filtered_paths = filter_parent_paths(all_train_paths)
    
    print(f"Original training paths: {len(all_train_paths):,}")
    print(f"After filtering parent paths: {len(filtered_paths):,}")
    
    # Filter out categories with fewer than 8 products in training data
    print(f"\nFiltering categories by minimum product count...")
    category_counts = train["path"].value_counts()
    categories_with_enough_products = category_counts[category_counts >= 8].index.tolist()
    
    # Keep only categories that are both leaf nodes AND have enough products
    final_categories = [cat for cat in filtered_paths if cat in categories_with_enough_products]
    
    print(f"Categories with ≥8 products: {len(categories_with_enough_products):,}")
    print(f"Categories that are leaf nodes AND have ≥8 products: {len(final_categories):,}")
    
    # Update filtered_paths to use the final filtered list
    filtered_paths = final_categories
    
    # ------------------------------------------------------------------
    # 4 . Build label-to-id mapping *only from the filtered training paths*
    # ------------------------------------------------------------------
    leaf2id = {p: i for i, p in enumerate(sorted(filtered_paths))}
    id2path = {v: k for k, v in leaf2id.items()}

    (out / "mappings").mkdir(exist_ok=True)
    json.dump(leaf2id, open(out / "mappings" / "leaf2id.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(id2path, open(out / "mappings" / "id2path.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 5 . Filter data to only include rows with filtered paths
    # ------------------------------------------------------------------
    # Keep only rows where the path is in our filtered set
    train_filtered = train[train["path"].isin(filtered_paths)].copy()
    val_filtered = val[val["path"].isin(filtered_paths)].copy()
    test_filtered = test[test["path"].isin(filtered_paths)].copy()
    
    print(f"After filtering data:")
    print(f"  Train rows: {len(train_filtered):,} (was {len(train):,})")
    print(f"  Val rows: {len(val_filtered):,} (was {len(val):,})")
    print(f"  Test rows: {len(test_filtered):,} (was {len(test):,})")
    
    # ------------------------------------------------------------------
    # 6 . Dump splits to JSONL
    # ------------------------------------------------------------------
    def dump(split_df: pd.DataFrame, name: str):
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                obj = {"text": row["text"], "label": row["path"]}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Wrote {name}: {len(split_df):,} rows → {path}")

    dump(train_filtered, "train")
    dump(val_filtered,   "val")
    dump(test_filtered,  "test")

    # ------------------------------------------------------------------
    # 7 . Console report
    # ------------------------------------------------------------------
    print("\nSplit summary")
    print(f"Total unique paths: {len(df['path'].unique()):,}")
    print(f"Number of products: {len(df):,}")
    print(f"Number of products in train: {len(train_filtered):,}")
    print(f"Number of products in val: {len(val_filtered):,}")
    print(f"Number of products in test: {len(test_filtered):,}")
    print(f"Train rows: {len(train_filtered):,}  |  Val rows: {len(val_filtered):,}  |  Test rows: {len(test_filtered):,}")
    print(f"Final unique paths (leaf nodes only): {len(filtered_paths):,}")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_frac",  type=float, default=0.10)   # fraction of total rows
    p.add_argument("--test_frac", type=float, default=0.10)   # fraction of total rows
    p.add_argument("--seed",      type=int,   default=42)
    args = p.parse_args()
    main(args)
