#!/usr/bin/env python3
"""
Quick parameter testing script for improved RAG.

This script tests a few key combinations to quickly understand the parameter space
before running a full grid search.

Usage:
    python src/baselines/quick_param_test.py \
        --test data_processed/test.jsonl \
        --mapping data_processed/mappings/leaf2id.json \
        --limit 50
"""

import argparse
import json
import pathlib
import os
from improved_RAG_baseline import run as run_improved_rag


def quick_test(test_path: str, mapping_path: str, output_dir: str, model_name: str, limit: int = None, shared_cache_dir: str = None):
    """
    Quick test of key parameter combinations.
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use shared cache directory if provided
    if shared_cache_dir:
        cache_dir = pathlib.Path(shared_cache_dir)
        print(f"🔗 Using shared cache directory: {cache_dir}")
        if not cache_dir.exists():
            print(f"⚠️  Shared cache directory {cache_dir} does not exist. Will create new caches.")
    else:
        cache_dir = output_path
        print(f"📁 Using local cache directory: {cache_dir}")
    
    # Key combinations to test
    combinations = [
        # Conservative (few candidates)
        {"k_tfidf": 5, "k_embed": 5, "name": "conservative"},
        # Balanced
        {"k_tfidf": 10, "k_embed": 10, "name": "balanced"},
        # TF-IDF heavy
        {"k_tfidf": 15, "k_embed": 5, "name": "tfidf_heavy"},
        # Embedding heavy
        {"k_tfidf": 5, "k_embed": 15, "name": "embed_heavy"},
        # Aggressive (many candidates)
        {"k_tfidf": 15, "k_embed": 15, "name": "aggressive"},
        # Extended range tests
        {"k_tfidf": 20, "k_embed": 20, "name": "extended_balanced"},
        {"k_tfidf": 25, "k_embed": 15, "name": "extended_tfidf"},
        {"k_tfidf": 15, "k_embed": 25, "name": "extended_embed"},
        {"k_tfidf": 30, "k_embed": 30, "name": "very_aggressive"},
        {"k_tfidf": 40, "k_embed": 20, "name": "tfidf_dominant"},
    ]
    
    results = []
    
    print(f"🚀 Quick parameter test with {len(combinations)} combinations")
    print(f"📁 Test file: {test_path}")
    print(f"📁 Mapping file: {mapping_path}")
    print(f"⏹️  Limit: {limit if limit else 'all'}")
    print("=" * 60)
    
    for i, combo in enumerate(combinations):
        print(f"\n🔄 [{i+1}/{len(combinations)}] Testing {combo['name']}: k_tfidf={combo['k_tfidf']}, k_embed={combo['k_embed']}")
        
        combo_output = output_path / combo["name"]
        
        # Create symlinks or copy cache files to share them
        if shared_cache_dir and pathlib.Path(shared_cache_dir).exists():
            # Ensure the output directory exists first
            combo_output.mkdir(parents=True, exist_ok=True)
            
            # Create symlinks to shared cache files
            for cache_file in ["llm_cache.joblib", "embed_cache.joblib", "faiss_index.bin"]:
                shared_cache_path = pathlib.Path(shared_cache_dir) / cache_file
                local_cache_path = combo_output / cache_file
                if shared_cache_path.exists():
                    if not local_cache_path.exists():
                        try:
                            # Try to create a relative symlink
                            relative_path = os.path.relpath(shared_cache_path, combo_output)
                            local_cache_path.symlink_to(relative_path)
                            print(f"      🔗 Linked {cache_file} to shared cache")
                        except (OSError, FileExistsError):
                            # If symlink fails, copy the file
                            import shutil
                            try:
                                shutil.copy2(shared_cache_path, local_cache_path)
                                print(f"      📋 Copied {cache_file} from shared cache")
                            except Exception as e:
                                print(f"      ⚠️  Could not copy {cache_file}: {e}")
                                # Continue without this cache file
                    else:
                        print(f"      ✅ {cache_file} already exists")
                else:
                    print(f"      ⚠️  {cache_file} not found in shared cache")
        
        try:
            run_improved_rag(
                test_path=test_path,
                mapping_path=mapping_path,
                output_dir=str(combo_output),
                model_name=model_name,
                k_tfidf=combo["k_tfidf"],
                k_embed=combo["k_embed"],
                limit=limit,
                shared_cache_dir=shared_cache_dir
            )
            
            # Load results
            metrics_path = combo_output / "metrics.json"
            if metrics_path.exists():
                metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
                
                result = {
                    "name": combo["name"],
                    "k_tfidf": combo["k_tfidf"],
                    "k_embed": combo["k_embed"],
                    "total_candidates": combo["k_tfidf"] + combo["k_embed"],
                    "retrieval_recall": metrics["retrieval_recall"],
                    "f1_micro": metrics["f1_micro"],
                    "f1_macro": metrics["f1_macro"]
                }
                
                results.append(result)
                
                print(f"✅ {combo['name']}: recall={metrics['retrieval_recall']:.3f}, "
                      f"f1_micro={metrics['f1_micro']:.3f}, "
                      f"f1_macro={metrics['f1_macro']:.3f}")
            else:
                print(f"❌ No metrics found for {combo['name']}")
                
        except Exception as e:
            print(f"❌ Error with {combo['name']}: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 60)
    
    if results:
        # Sort by F1 micro
        results.sort(key=lambda x: x["f1_micro"], reverse=True)
        
        print(f"{'Name':<12} {'k_tfidf':<8} {'k_embed':<8} {'Total':<6} {'Recall':<8} {'F1 Micro':<10} {'F1 Macro':<10}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['name']:<12} {result['k_tfidf']:<8} {result['k_embed']:<8} "
                  f"{result['total_candidates']:<6} {result['retrieval_recall']:<8.3f} "
                  f"{result['f1_micro']:<10.3f} {result['f1_macro']:<10.3f}")
        
        # Best combination
        best = results[0]
        print(f"\n🏆 Best combination: {best['name']}")
        print(f"   k_tfidf={best['k_tfidf']}, k_embed={best['k_embed']}")
        print(f"   F1 Micro: {best['f1_micro']:.4f}")
        print(f"   F1 Macro: {best['f1_macro']:.4f}")
        print(f"   Retrieval Recall: {best['retrieval_recall']:.4f}")
        
        # Save results
        results_path = output_path / "quick_test_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {results_path}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if best["retrieval_recall"] < 0.95:
            print(f"   ⚠️  Retrieval recall is below 95%. Consider increasing k_tfidf or k_embed.")
        else:
            print(f"   ✅ Retrieval recall is good (≥95%).")
        
        if best["total_candidates"] > 20:
            print(f"   💰 High candidate count ({best['total_candidates']}). Consider if this is cost-effective.")
        else:
            print(f"   ✅ Reasonable candidate count ({best['total_candidates']}).")
        
        print(f"   🎯 For full grid search, focus around k_tfidf={best['k_tfidf']}±10, k_embed={best['k_embed']}±10")
        
        # Additional recommendations based on total candidates
        if best["total_candidates"] > 40:
            print(f"   ⚠️  Very high candidate count ({best['total_candidates']}). Consider cost vs performance trade-off.")
        elif best["total_candidates"] > 30:
            print(f"   💰 High candidate count ({best['total_candidates']}). Test if performance justifies cost.")
        else:
            print(f"   ✅ Reasonable candidate count ({best['total_candidates']}).")
    
    else:
        print("❌ No successful results to display")


def main():
    parser = argparse.ArgumentParser(description="Quick parameter testing for improved RAG")
    parser.add_argument("--test", required=True, help="Path to test JSONL file")
    parser.add_argument("--mapping", required=True, help="Path to leaf2id.json mapping")
    parser.add_argument("--output_dir", default="models/grid_search/quick_test", help="Output directory")
    parser.add_argument("--model", default="mistral-small-latest", help="Mistral model to use")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of examples to process")
    parser.add_argument("--shared_cache", type=str, 
                       help="Path to shared cache directory (e.g., models/improved_RAG)")
    
    args = parser.parse_args()
    
    quick_test(
        test_path=args.test,
        mapping_path=args.mapping,
        output_dir=args.output_dir,
        model_name=args.model,
        limit=args.limit,
        shared_cache_dir=args.shared_cache
    )


if __name__ == "__main__":
    main() 