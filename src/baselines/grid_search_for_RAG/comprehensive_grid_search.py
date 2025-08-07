#!/usr/bin/env python3
"""
Comprehensive grid search for optimal k_tfidf and k_embed parameters across multiple models.

This script tests different combinations of k_tfidf and k_embed across multiple Mistral models
and provides cost analysis and visualization.

Usage:
    python src/baselines/comprehensive_grid_search.py \
        --test data_processed/test.jsonl \
        --mapping data_processed/mappings/leaf2id.json \
        --shared_cache models/improved_RAG \
        --limit 100
"""

import argparse
import json
import pathlib
import time
import os
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from improved_RAG_baseline import run as run_improved_rag

# Model pricing (per 1M tokens) - approximate costs
MODEL_PRICING = {
    "mistral-small-latest": {
        "input": 0.10,      # $0.10 per 1M input tokens
        "output": 0.30      # $0.30 per 1M output tokens
    },
    "mistral-medium-latest": {
        "input": 0.4,      # $0.4 per 1M input tokens
        "output": 2.0      # $2.0 per 1M output tokens
    },
    "mistral-large-latest": {
        "input": 2.0,      # $2.0 per 1M input tokens
        "output": 6.0      # $6.0 per 1M output tokens
    }
}

# Embedding pricing (per 1M tokens)
EMBEDDING_PRICING = {
    "mistral-embed": 0.10   # $0.10 per 1M tokens
}


def estimate_cost(model_name: str, total_candidates: int, num_examples: int, avg_input_length: int = 200, avg_output_length: int = 5):
    """
    Estimate the cost for running the improved RAG model.
    
    Args:
        model_name: Name of the Mistral model
        total_candidates: Total number of candidates (k_tfidf + k_embed)
        num_examples: Number of examples to process
        avg_input_length: Average input length in tokens
        avg_output_length: Average output length in tokens
    
    Returns:
        Dictionary with cost breakdown
    """
    if model_name not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model_name}")
    
    pricing = MODEL_PRICING[model_name]
    
    # Calculate tokens
    # Input: product description + candidate list
    input_tokens_per_example = avg_input_length + (total_candidates * 20)  # ~20 tokens per candidate
    total_input_tokens = input_tokens_per_example * num_examples
    
    # Output: model response (usually just a number)
    total_output_tokens = avg_output_length * num_examples
    
    # Embedding tokens (one embedding per example)
    embedding_tokens = avg_input_length * num_examples
    
    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    embedding_cost = (embedding_tokens / 1_000_000) * EMBEDDING_PRICING["mistral-embed"]
    
    total_cost = input_cost + output_cost + embedding_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "embedding_cost": embedding_cost,
        "total_cost": total_cost,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "embedding_tokens": embedding_tokens,
        "cost_per_example": total_cost / num_examples
    }


def comprehensive_grid_search(test_path: str,
                             mapping_path: str,
                             output_dir: str,
                             models: List[str],
                             k_tfidf_range: List[int],
                             k_embed_range: List[int],
                             limit: int = None,
                             shared_cache_dir: str = None) -> pd.DataFrame:
    """
    Perform comprehensive grid search across multiple models and parameters.
    """
    base_output = pathlib.Path(output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Use shared cache directory if provided
    if shared_cache_dir:
        cache_dir = pathlib.Path(shared_cache_dir)
        print(f"🔗 Using shared cache directory: {cache_dir}")
        if not cache_dir.exists():
            print(f"⚠️  Shared cache directory {cache_dir} does not exist. Will create new caches.")
    else:
        cache_dir = base_output
        print(f"📁 Using local cache directory: {cache_dir}")
    
    results = []
    total_combinations = len(models) * len(k_tfidf_range) * len(k_embed_range)
    current_combination = 0
    
    print(f"🔍 Starting comprehensive grid search with {total_combinations} combinations")
    print(f"🤖 Models: {models}")
    print(f"📊 k_tfidf range: {k_tfidf_range}")
    print(f"📊 k_embed range: {k_embed_range}")
    print(f"⏹️  Example limit: {limit if limit else 'all'}")
    print("=" * 80)
    
    for model_name in models:
        for k_tfidf in k_tfidf_range:
            for k_embed in k_embed_range:
                current_combination += 1
                total_candidates = k_tfidf + k_embed
                
                print(f"\n🔄 [{current_combination}/{total_combinations}] Testing {model_name}: k_tfidf={k_tfidf}, k_embed={k_embed} (total candidates: {total_candidates})")
                
                # Create unique output directory for this combination
                combo_output = base_output / f"{model_name}_k_tfidf_{k_tfidf}_k_embed_{k_embed}"
                
                # Create symlinks or copy cache files to share them
                if shared_cache_dir and pathlib.Path(shared_cache_dir).exists():
                    combo_output.mkdir(parents=True, exist_ok=True)
                    
                    for cache_file in ["llm_cache.joblib", "embed_cache.joblib", "faiss_index.bin"]:
                        shared_cache_path = pathlib.Path(shared_cache_dir) / cache_file
                        local_cache_path = combo_output / cache_file
                        if shared_cache_path.exists():
                            if not local_cache_path.exists():
                                try:
                                    relative_path = os.path.relpath(shared_cache_path, combo_output)
                                    local_cache_path.symlink_to(relative_path)
                                    print(f"      🔗 Linked {cache_file} to shared cache")
                                except (OSError, FileExistsError):
                                    import shutil
                                    try:
                                        shutil.copy2(shared_cache_path, local_cache_path)
                                        print(f"      📋 Copied {cache_file} from shared cache")
                                    except Exception as e:
                                        print(f"      ⚠️  Could not copy {cache_file}: {e}")
                            else:
                                print(f"      ✅ {cache_file} already exists")
                        else:
                            print(f"      ⚠️  {cache_file} not found in shared cache")
                
                try:
                    start_time = time.time()
                    
                    # Run the improved RAG baseline
                    run_improved_rag(
                        test_path=test_path,
                        mapping_path=mapping_path,
                        output_dir=str(combo_output),
                        model_name=model_name,
                        k_tfidf=k_tfidf,
                        k_embed=k_embed,
                        limit=limit,
                        shared_cache_dir=shared_cache_dir
                    )
                    
                    # Load results
                    metrics_path = combo_output / "metrics.json"
                    if metrics_path.exists():
                        metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
                        
                        # Estimate cost
                        cost_estimate = estimate_cost(model_name, total_candidates, limit or 1000)
                        
                        result = {
                            "model": model_name,
                            "k_tfidf": k_tfidf,
                            "k_embed": k_embed,
                            "total_candidates": total_candidates,
                            "retrieval_recall": metrics["retrieval_recall"],
                            "f1_micro": metrics["f1_micro"],
                            "f1_macro": metrics["f1_macro"],
                            "execution_time": time.time() - start_time,
                            "total_cost": cost_estimate["total_cost"],
                            "cost_per_example": cost_estimate["cost_per_example"],
                            "input_cost": cost_estimate["input_cost"],
                            "output_cost": cost_estimate["output_cost"],
                            "embedding_cost": cost_estimate["embedding_cost"]
                        }
                        
                        results.append(result)
                        
                        print(f"✅ Results: recall={metrics['retrieval_recall']:.3f}, "
                              f"f1_micro={metrics['f1_micro']:.3f}, "
                              f"f1_macro={metrics['f1_macro']:.3f}, "
                              f"cost=${cost_estimate['total_cost']:.4f}")
                    else:
                        print(f"❌ No metrics.json found at {metrics_path}")
                        
                except Exception as e:
                    print(f"❌ Error with {model_name}, k_tfidf={k_tfidf}, k_embed={k_embed}: {e}")
                    results.append({
                        "model": model_name,
                        "k_tfidf": k_tfidf,
                        "k_embed": k_embed,
                        "total_candidates": total_candidates,
                        "retrieval_recall": None,
                        "f1_micro": None,
                        "f1_macro": None,
                        "execution_time": None,
                        "total_cost": None,
                        "cost_per_example": None,
                        "input_cost": None,
                        "output_cost": None,
                        "embedding_cost": None,
                        "error": str(e)
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    results_path = base_output / "comprehensive_grid_search_results.json"
    df.to_json(results_path, orient="records", indent=2)
    print(f"\n💾 Results saved to {results_path}")
    
    return df


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Create comprehensive visualizations of the grid search results.
    """
    output_path = pathlib.Path(output_dir)
    
    # Filter out failed runs
    valid_df = df.dropna(subset=["f1_micro", "f1_macro"])
    
    if valid_df.empty:
        print("❌ No valid results to visualize!")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Grid Search Results: Performance vs Cost Analysis', fontsize=16, fontweight='bold')
    
    # Helper function to add jitter to overlapping points
    def add_jitter(data, jitter_amount=0.02):
        """Add small random jitter to prevent overlapping points"""
        jitter = np.random.normal(0, jitter_amount, len(data))
        return data + jitter
    
    # 1. F1 Micro by model and total candidates
    ax1 = axes[0, 0]
    for model in valid_df['model'].unique():
        model_data = valid_df[valid_df['model'] == model]
        # Add jitter to x-axis to separate overlapping points
        x_jittered = add_jitter(model_data['total_candidates'], jitter_amount=0.8)
        ax1.scatter(x_jittered, model_data['f1_micro'], 
                   label=model, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Total Candidates (k_tfidf + k_embed)')
    ax1.set_ylabel('F1 Micro')
    ax1.set_title('F1 Micro vs Total Candidates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. F1 Macro by model and total candidates
    ax2 = axes[0, 1]
    for model in valid_df['model'].unique():
        model_data = valid_df[valid_df['model'] == model]
        # Add jitter to x-axis to separate overlapping points
        x_jittered = add_jitter(model_data['total_candidates'], jitter_amount=0.8)
        ax2.scatter(x_jittered, model_data['f1_macro'], 
                   label=model, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Total Candidates (k_tfidf + k_embed)')
    ax2.set_ylabel('F1 Macro')
    ax2.set_title('F1 Macro vs Total Candidates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cost per example by model and total candidates
    ax3 = axes[0, 2]
    for model in valid_df['model'].unique():
        model_data = valid_df[valid_df['model'] == model]
        # Add jitter to x-axis to separate overlapping points
        x_jittered = add_jitter(model_data['total_candidates'], jitter_amount=0.8)
        ax3.scatter(x_jittered, model_data['cost_per_example'], 
                   label=model, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Total Candidates (k_tfidf + k_embed)')
    ax3.set_ylabel('Cost per Example ($)')
    ax3.set_title('Cost per Example vs Total Candidates')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. F1 Micro vs Cost per example
    ax4 = axes[1, 0]
    for model in valid_df['model'].unique():
        model_data = valid_df[valid_df['model'] == model]
        # Add jitter to both axes to separate overlapping points
        x_jittered = add_jitter(model_data['cost_per_example'], jitter_amount=0.0001)
        y_jittered = add_jitter(model_data['f1_micro'], jitter_amount=0.005)
        ax4.scatter(x_jittered, y_jittered, 
                   label=model, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Cost per Example ($)')
    ax4.set_ylabel('F1 Micro')
    ax4.set_title('F1 Micro vs Cost per Example')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. F1 Macro vs Cost per example
    ax5 = axes[1, 1]
    for model in valid_df['model'].unique():
        model_data = valid_df[valid_df['model'] == model]
        # Add jitter to both axes to separate overlapping points
        x_jittered = add_jitter(model_data['cost_per_example'], jitter_amount=0.0001)
        y_jittered = add_jitter(model_data['f1_macro'], jitter_amount=0.005)
        ax5.scatter(x_jittered, y_jittered, 
                   label=model, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax5.set_xlabel('Cost per Example ($)')
    ax5.set_ylabel('F1 Macro')
    ax5.set_title('F1 Macro vs Cost per Example')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Heatmap of F1 Micro by k_tfidf and k_embed (for the best model)
    ax6 = axes[1, 2]
    best_model = valid_df.loc[valid_df['f1_micro'].idxmax(), 'model']
    best_model_data = valid_df[valid_df['model'] == best_model]
    
    # Create pivot table for heatmap
    pivot_data = best_model_data.pivot_table(
        values='f1_micro', 
        index='k_embed', 
        columns='k_tfidf', 
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax6)
    ax6.set_title(f'F1 Micro Heatmap - {best_model}')
    ax6.set_xlabel('k_tfidf')
    ax6.set_ylabel('k_embed')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_path / "comprehensive_grid_search_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to {plot_path}")
    
    # Create additional detailed plots
    create_detailed_plots(valid_df, output_path)


def create_detailed_plots(df: pd.DataFrame, output_path: pathlib.Path):
    """
    Create additional detailed plots for deeper analysis.
    """
    # Helper function to add jitter to overlapping points
    def add_jitter(data, jitter_amount=0.02):
        """Add small random jitter to prevent overlapping points"""
        jitter = np.random.normal(0, jitter_amount, len(data))
        return data + jitter
    
    # 1. Performance comparison across models
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1 Micro comparison
    ax1 = axes[0]
    model_performance = df.groupby('model')['f1_micro'].agg(['mean', 'std', 'max']).reset_index()
    bars = ax1.bar(model_performance['model'], model_performance['mean'], 
                   yerr=model_performance['std'], capsize=5, alpha=0.7)
    ax1.set_ylabel('F1 Micro (Mean ± Std)')
    ax1.set_title('Average F1 Micro by Model')
    ax1.grid(True, alpha=0.3)
    
    # Add max values as text
    for i, (_, row) in enumerate(model_performance.iterrows()):
        ax1.text(i, row['mean'] + row['std'] + 0.01, f"Max: {row['max']:.3f}", 
                ha='center', va='bottom', fontsize=9)
    
    # Cost comparison
    ax2 = axes[1]
    model_cost = df.groupby('model')['cost_per_example'].agg(['mean', 'std']).reset_index()
    bars = ax2.bar(model_cost['model'], model_cost['mean'], 
                   yerr=model_cost['std'], capsize=5, alpha=0.7)
    ax2.set_ylabel('Cost per Example ($)')
    ax2.set_title('Average Cost per Example by Model')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    detailed_plot_path = output_path / "model_comparison.png"
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Model comparison saved to {detailed_plot_path}")
    
    # 2. Cost-effectiveness analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Calculate efficiency (F1 Micro / Cost)
    df['efficiency'] = df['f1_micro'] / df['cost_per_example']
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        # Add jitter to x-axis to separate overlapping points
        x_jittered = add_jitter(model_data['total_candidates'], jitter_amount=0.8)
        ax.scatter(x_jittered, model_data['efficiency'], 
                  label=model, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Total Candidates (k_tfidf + k_embed)')
    ax.set_ylabel('Efficiency (F1 Micro / Cost)')
    ax.set_title('Cost-Effectiveness Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    efficiency_plot_path = output_path / "cost_effectiveness.png"
    plt.savefig(efficiency_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Cost-effectiveness analysis saved to {efficiency_plot_path}")


def analyze_comprehensive_results(df: pd.DataFrame) -> Dict:
    """
    Analyze comprehensive grid search results and provide recommendations.
    """
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE GRID SEARCH ANALYSIS")
    print("=" * 80)
    
    # Filter out failed runs
    valid_df = df.dropna(subset=["f1_micro", "f1_macro"])
    
    if valid_df.empty:
        print("❌ No valid results found!")
        return {}
    
    print(f"✅ Valid combinations: {len(valid_df)}/{len(df)}")
    
    analysis = {}
    
    # Best overall performer
    best_overall_idx = valid_df["f1_micro"].idxmax()
    best_overall = valid_df.loc[best_overall_idx]
    
    print(f"\n🏆 BEST OVERALL PERFORMER:")
    print(f"   Model: {best_overall['model']}")
    print(f"   k_tfidf={best_overall['k_tfidf']}, k_embed={best_overall['k_embed']}")
    print(f"   F1 Micro: {best_overall['f1_micro']:.4f}")
    print(f"   F1 Macro: {best_overall['f1_macro']:.4f}")
    print(f"   Cost per Example: ${best_overall['cost_per_example']:.4f}")
    print(f"   Total Candidates: {best_overall['total_candidates']}")
    
    analysis["best_overall"] = best_overall.to_dict()
    
    # Best per model
    print(f"\n🥇 BEST PERFORMER BY MODEL:")
    for model in valid_df['model'].unique():
        model_data = valid_df[valid_df['model'] == model]
        best_model_idx = model_data["f1_micro"].idxmax()
        best_model = model_data.loc[best_model_idx]
        
        print(f"   {model}:")
        print(f"     k_tfidf={best_model['k_tfidf']}, k_embed={best_model['k_embed']}")
        print(f"     F1 Micro: {best_model['f1_micro']:.4f}")
        print(f"     Cost per Example: ${best_model['cost_per_example']:.4f}")
        
        analysis[f"best_{model}"] = best_model.to_dict()
    
    # Most cost-effective
    valid_df['efficiency'] = valid_df['f1_micro'] / valid_df['cost_per_example']
    most_efficient_idx = valid_df["efficiency"].idxmax()
    most_efficient = valid_df.loc[most_efficient_idx]
    
    print(f"\n⚡ MOST COST-EFFECTIVE:")
    print(f"   Model: {most_efficient['model']}")
    print(f"   k_tfidf={most_efficient['k_tfidf']}, k_embed={most_efficient['k_embed']}")
    print(f"   F1 Micro: {most_efficient['f1_micro']:.4f}")
    print(f"   Cost per Example: ${most_efficient['cost_per_example']:.4f}")
    print(f"   Efficiency: {most_efficient['efficiency']:.2f}")
    
    analysis["most_efficient"] = most_efficient.to_dict()
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Average F1 Micro: {valid_df['f1_micro'].mean():.4f}")
    print(f"   Average F1 Macro: {valid_df['f1_macro'].mean():.4f}")
    print(f"   Average Cost per Example: ${valid_df['cost_per_example'].mean():.4f}")
    print(f"   Average Total Candidates: {valid_df['total_candidates'].mean():.1f}")
    
    # Model comparison
    print(f"\n🤖 MODEL COMPARISON:")
    for model in valid_df['model'].unique():
        model_data = valid_df[valid_df['model'] == model]
        print(f"   {model}:")
        print(f"     Avg F1 Micro: {model_data['f1_micro'].mean():.4f}")
        print(f"     Avg Cost per Example: ${model_data['cost_per_example'].mean():.4f}")
        print(f"     Best F1 Micro: {model_data['f1_micro'].max():.4f}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Comprehensive grid search for optimal parameters across multiple models")
    parser.add_argument("--test", required=True, help="Path to test JSONL file")
    parser.add_argument("--mapping", required=True, help="Path to leaf2id.json mapping")
    parser.add_argument("--output_dir", default="models/comprehensive_grid_search", help="Output directory")
    parser.add_argument("--models", nargs="+", default=["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
                       help="Mistral models to test")
    parser.add_argument("--k_tfidf_range", nargs="+", type=int, default=[15, 20, 25, 30], 
                       help="Range of k_tfidf values to test")
    parser.add_argument("--k_embed_range", nargs="+", type=int, default=[15, 20, 25, 30], 
                       help="Range of k_embed values to test")
    parser.add_argument("--limit", type=int, help="Limit number of examples to process")
    parser.add_argument("--shared_cache", type=str, 
                       help="Path to shared cache directory (e.g., models/improved_RAG)")
    
    args = parser.parse_args()
    
    # Perform comprehensive grid search
    df = comprehensive_grid_search(
        test_path=args.test,
        mapping_path=args.mapping,
        output_dir=args.output_dir,
        models=args.models,
        k_tfidf_range=args.k_tfidf_range,
        k_embed_range=args.k_embed_range,
        limit=args.limit,
        shared_cache_dir=args.shared_cache
    )
    
    # Create visualizations
    create_visualizations(df, args.output_dir)
    
    # Analyze results
    analysis = analyze_comprehensive_results(df)
    
    # Save analysis
    analysis_path = pathlib.Path(args.output_dir) / "comprehensive_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Analysis saved to {analysis_path}")
    print(f"🎯 Comprehensive grid search completed! Check the results and visualizations above.")


if __name__ == "__main__":
    main() 