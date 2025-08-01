#!/usr/bin/env python3
"""
Create child-leaf level category labels from full category paths to ease
classification for the Mistral fine-tuning pipeline.

Given input JSONL dataset files (train / validation / test) where each
line has at least:

{
  "text": "...",
  "category_path": "Parent > Sub-parent > ... > Leaf"
}

the script will:

1. Derive a *compact* version of the label that is **unique** across the
   dataset but as short as possible, following the rules:

   * Start with the last component (leaf).
   * If the same leaf appears in more than one path, prefix it with the
     penultimate component — e.g. "Couvre-chefs > Accessoires".
   * If a collision still exists, continue prefixing components from the
     right until the label is unique (worst-case becomes the full path).

2. Optionally perform text cleaning via `text_cleaning.clean_text`
   (falls back to an identity function if the module is not available).

3. Create *new* JSONL files beside the originals
   (e.g. `train_childleaf.jsonl`) that do **not** overwrite the inputs.

4. Write `label2id_childleaf.json` and `id2label_childleaf.json`
   mappings next to the dataset.

You can then fine-tune again with the untouched `fine_tune_mistral.py`,
pointing it to the new files, e.g.:

```bash
python fine_tune_mistral.py \
    --train_file data/train_childleaf.jsonl \
    --validation_file data/valid_childleaf.jsonl \
    --output_dir runs/mistral_childleaf
```
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

try:
    from text_cleaning import clean_text  # user-provided util
except ImportError:  # graceful fallback
    def clean_text(x: str) -> str:
        return x


def split_path(path: str) -> List[str]:
    """Split a category path on '>' and trim whitespace."""
    return [part.strip() for part in path.split(">")]


def derive_unique_labels(paths: List[List[str]]) -> List[str]:
    """
    Derive the shortest unique label for every path.
    Algorithm:
        1. Start with the leaf component.
        2. While duplicates remain, prefix the next component from the right.
    """
    n = len(paths)
    depth_used = [1] * n  # how many components from the right are used
    labels = [" > ".join(p[-1:]) for p in paths]
    
    print(f"Starting with {n} paths")
    print(f"Initial labels (first 10): {labels[:10]}")

    # helper to build label with k right-most components
    def build_label(p: List[str], k: int) -> str:
        return " > ".join(p[-k:])

    iteration = 0
    # iterate until no duplicates or max depth reached
    while True:
        iteration += 1
        print(f"\nIteration {iteration}")
        
        buckets: Dict[str, List[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            buckets[lbl].append(idx)

        dup_groups = [idxs for idxs in buckets.values() if len(idxs) > 1]
        print(f"Found {len(dup_groups)} duplicate groups")
        
        if not dup_groups:
            print("✅ All labels are now unique!")
            break  # all unique

        # Show some examples of duplicates
        for i, group in enumerate(dup_groups[:3]):  # Show first 3 groups
            dup_label = labels[group[0]]
            print(f"   Duplicate group {i+1}: '{dup_label}' appears {len(group)} times at indices {group[:5]}{'...' if len(group) > 5 else ''}")

        # Track if we made any progress this iteration
        progress_made = False
        
        for group in dup_groups:
            for idx in group:
                # Increase depth if possible
                if depth_used[idx] < len(paths[idx]):
                    old_label = labels[idx]
                    depth_used[idx] += 1
                    labels[idx] = build_label(paths[idx], depth_used[idx])
                    print(f"   🔧 Extended path {idx}: '{old_label}' → '{labels[idx]}' (depth {depth_used[idx]})")
                    progress_made = True
                # else: already at full path -> leave as is
                else:
                    print(f"   Path {idx} already at max depth ({len(paths[idx])})")
        
        # If no progress was made, we have unresolvable duplicates
        if not progress_made:
            print("No progress made - unresolvable duplicates detected!")
            print(" Unresolvable duplicate groups:")
            
            # Print all unresolvable duplicates for investigation
            for i, group in enumerate(dup_groups):
                dup_label = labels[group[0]]
                print(f"   Group {i+1}: '{dup_label}' appears {len(group)} times")
                print(f"      Indices: {group}")
                print(f"      Full paths:")
                for idx in group:
                    full_path = " > ".join(paths[idx])
                    print(f"        {idx}: {full_path}")
                print()
            
            print("Continuing with unresolvable duplicates - you may want to investigate these in the database")
            break
    
    print(f"\n Final result: {len(set(labels))} unique labels out of {n} paths")
    print(f" Final labels (first 10): {labels[:10]}")
    
    # Show some statistics
    unique_labels = set(labels)
    label_lengths = [len(label) for label in labels]
    print(f"Label length stats: min={min(label_lengths)}, max={max(label_lengths)}, avg={sum(label_lengths)/len(label_lengths):.1f}")
    print(f"Depth usage stats: min={min(depth_used)}, max={max(depth_used)}, avg={sum(depth_used)/len(depth_used):.1f}")
    
    return labels


def load_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_split(in_path: Path, out_path: Path,
                  label_lookup: dict, collect_paths: bool) -> List[List[str]]:
    """
    Process a dataset split file.
    Returns the list of category path components if collect_paths is True.
    """
    records = load_jsonl(in_path)
    path_components: List[List[str]] = []

    for rec in records:
        path_str = rec["label"]
        comps = split_path(path_str)
        if collect_paths:
            path_components.append(comps)

        rec["text"] = clean_text(rec["text"])
        # label will be filled later once we know unique mapping
        rec["_components"] = comps
    # write temp file
    write_jsonl(out_path.with_suffix(".tmp"), records)
    return path_components


def finalize_split(tmp_path: Path, final_path: Path,
                   label_func) -> None:
    """
    Replace _components temp field with final label string.
    """
    records = load_jsonl(tmp_path)
    new_records = []
    for rec in records:
        comps = rec.pop("_components")
        label_str = label_func(tuple(comps))  # deterministic
        # Keep only text and label fields
        new_rec = {
            "text": rec["text"],
            "label": label_str  # Use the leaf name directly, not an ID
        }
        new_records.append(new_rec)
    write_jsonl(final_path, new_records)
    tmp_path.unlink()  # cleanup


def main():
    parser = argparse.ArgumentParser(
        description="Generate child-leaf dataset for Mistral fine-tuning.")
    parser.add_argument(
        "--data_dir", type=Path, required=True,
        help="Directory containing train.jsonl / valid.jsonl (and optionally test.jsonl)")
    parser.add_argument(
        "--train_file", type=str, default="train.jsonl",
        help="Name of the training split file")
    parser.add_argument(
        "--valid_file", type=str, default="valid.jsonl",
        help="Name of the validation split file")
    parser.add_argument(
        "--test_file", type=str, default=None,
        help="Optional test split file name")
    args = parser.parse_args()

    splits = {"train": args.train_file, "valid": args.valid_file}
    if args.test_file:
        splits["test"] = args.test_file

    # First pass: load all splits to gather category paths
    all_paths: List[List[str]] = []
    temp_files = {}
    for split_name, fname in splits.items():
        in_path = args.data_dir / fname
        out_tmp = args.data_dir / f"{fname}.tmp"
        comps = process_split(in_path, out_tmp, None, collect_paths=True)
        all_paths.extend(comps)
        temp_files[split_name] = out_tmp

    # Derive unique labels
    unique_labels = derive_unique_labels(all_paths)

    # Helper to map components tuple -> label string
    comp2label = {}
    for comps, lbl in zip(all_paths, unique_labels):
        comp2label[tuple(comps)] = lbl

    def label_func(comps: tuple) -> str:
        return comp2label[comps]

    # Second pass: create final jsonl per split
    for split_name, tmp_path in temp_files.items():
        original_name = splits[split_name]
        out_path = args.data_dir / original_name.replace(
            ".jsonl", "_childleaf.jsonl")
        finalize_split(tmp_path, out_path, label_func)
        print(f"Wrote {out_path}")

    # Save unique labels list for reference
    unique_labels_set = sorted(set(unique_labels))
    map_dir = args.data_dir
    with (map_dir / "unique_labels_childleaf.json").open("w", encoding="utf-8") as f:
        json.dump(unique_labels_set, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(unique_labels_set)} unique labels to unique_labels_childleaf.json")


if __name__ == "__main__":
    main()
