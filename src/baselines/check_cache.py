#!/usr/bin/env python3
"""
Check if existing FAISS index and caches are available.

This script helps you verify what cache files exist and their sizes.

Usage:
    python src/baselines/check_cache.py --cache_dir models/improved_RAG
"""

import argparse
import pathlib
import os


def check_cache(cache_dir: str):
    """Check what cache files exist in the given directory."""
    cache_path = pathlib.Path(cache_dir)
    
    if not cache_path.exists():
        print(f"❌ Cache directory does not exist: {cache_dir}")
        return False
    
    print(f"📁 Checking cache directory: {cache_dir}")
    print("=" * 50)
    
    cache_files = {
        "faiss_index.bin": "FAISS vector index",
        "embed_cache.joblib": "Embedding cache",
        "llm_cache.joblib": "LLM response cache"
    }
    
    all_exist = True
    
    for filename, description in cache_files.items():
        file_path = cache_path / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {filename:<20} {description:<20} {size_mb:.1f} MB")
        else:
            print(f"❌ {filename:<20} {description:<20} MISSING")
            all_exist = False
    
    print("=" * 50)
    
    if all_exist:
        print("🎉 All cache files are available!")
        print("💡 You can use --shared_cache to reuse these caches in grid search.")
        return True
    else:
        print("⚠️  Some cache files are missing.")
        print("💡 Run the improved_RAG baseline first to create the caches.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check if cache files exist")
    parser.add_argument("--cache_dir", default="models/improved_RAG", 
                       help="Path to cache directory")
    
    args = parser.parse_args()
    
    check_cache(args.cache_dir)


if __name__ == "__main__":
    main() 