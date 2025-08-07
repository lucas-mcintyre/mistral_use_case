# Parameter Search for Improved RAG Model

This guide explains how to find the optimal `k_tfidf` and `k_embed` parameters for your improved RAG model while efficiently reusing existing caches.

## Overview

The improved RAG model uses two retrieval methods:
- **TF-IDF retrieval**: `k_tfidf` candidates based on term frequency
- **Embedding retrieval**: `k_embed` candidates based on semantic similarity

Finding the right balance between these parameters is crucial for performance and cost efficiency.

## Quick Start

### 1. Check Existing Caches

First, verify that you have existing cache files from a previous run:

```bash
python src/baselines/check_cache.py --cache_dir models/improved_RAG_filtered_dataset
```

You should see:
```
✅ faiss_index.bin      FAISS vector index   6.3 MB
✅ embed_cache.joblib   Embedding cache      14.3 MB
✅ llm_cache.joblib     LLM response cache   0.0 MB
```

### 2. Quick Parameter Test

Run a quick test with extended combinations to understand the parameter space:

```bash
python src/baselines/grid_search_for_RAG/quick_param_test_extended.py \
    --test data_processed_filtered/test.jsonl \
    --mapping data_processed_filtered/mappings/leaf2id.json \
    --shared_cache models/improved_RAG_filtered_dataset \
    --limit 50
```

This tests:
- **Conservative**: k_tfidf=5, k_embed=5 (10 total candidates)
- **Balanced**: k_tfidf=10, k_embed=10 (20 total candidates)
- **TF-IDF heavy**: k_tfidf=15, k_embed=5 (20 total candidates)
- **Embedding heavy**: k_tfidf=5, k_embed=15 (20 total candidates)
- **Aggressive**: k_tfidf=15, k_embed=15 (30 total candidates)
- **Extended ranges**: Higher values up to k_tfidf=50, k_embed=50

### 3. Comprehensive Grid Search

Run a comprehensive grid search across multiple models with cost analysis:

```bash
python src/baselines/grid_search_for_RAG/comprehensive_grid_search.py \
    --test data_processed_filtered/test.jsonl \
    --mapping data_processed_filtered/mappings/leaf2id.json \
    --shared_cache models/improved_RAG_filtered_dataset \
    --k_tfidf_range 15 20 25 30 \
    --k_embed_range 15 20 25 30 \
    --models mistral-small-latest mistral-medium-latest mistral-large-latest \
    --limit 100
```

## Using Shared Caches

### Why Use Shared Caches?

1. **Speed**: Avoid rebuilding FAISS index and re-computing embeddings
2. **Cost**: Avoid re-making expensive embedding API calls
3. **Consistency**: Ensure all parameter combinations use the same retrieval base

### How It Works

The scripts automatically:
1. **Link** existing cache files to each parameter combination directory
2. **Reuse** the FAISS index and embedding cache
3. **Share** LLM response cache (avoid duplicate API calls)

### Cache Files

- `faiss_index.bin`: Pre-built FAISS index for fast similarity search
- `embed_cache.joblib`: Cached embeddings for all leaf categories
- `llm_cache.joblib`: Cached LLM responses to avoid duplicate API calls

## Parameter Selection Strategy

### 1. Retrieval Recall Target

Aim for **≥95% retrieval recall** - this means the true label is found in the candidate set 95% of the time.

```bash
# Check recall for different combinations
python src/baselines/grid_search_for_RAG/comprehensive_grid_search.py \
    --test data_processed_filtered/test.jsonl \
    --mapping data_processed_filtered/mappings/leaf2id.json \
    --shared_cache models/improved_RAG_filtered_dataset \
    --k_tfidf_range 5 10 15 20 \
    --k_embed_range 5 10 15 20 \
    --limit 100
```

### 2. Performance vs Cost Trade-off

- **Higher k_tfidf/k_embed** = Better recall but more API calls
- **Lower k_tfidf/k_embed** = Fewer API calls but potentially lower recall

### 3. TF-IDF vs Embedding Balance

- **TF-IDF heavy**: Good for exact keyword matches
- **Embedding heavy**: Good for semantic similarity
- **Balanced**: Often the best approach

## Example Workflows

### Workflow 1: Conservative Approach

```bash
# Start with small ranges
python src/baselines/grid_search_for_RAG/comprehensive_grid_search.py \
    --test data_processed_filtered/test.jsonl \
    --mapping data_processed_filtered/mappings/leaf2id.json \
    --shared_cache models/improved_RAG_filtered_dataset \
    --k_tfidf_range 5 10 15 \
    --k_embed_range 5 10 15 \
    --models mistral-small-latest \
    --limit 50
```

### Workflow 2: Aggressive Approach

```bash
# Test larger ranges for maximum performance
python src/baselines/grid_search_for_RAG/comprehensive_grid_search.py \
    --test data_processed_filtered/test.jsonl \
    --mapping data_processed_filtered/mappings/leaf2id.json \
    --shared_cache models/improved_RAG_filtered_dataset \
    --k_tfidf_range 10 15 20 25 \
    --k_embed_range 10 15 20 25 \
    --models mistral-small-latest mistral-medium-latest mistral-large-latest \
    --limit 100
```

### Workflow 3: Focused Search

```bash
# Based on quick test results, focus on promising region
python src/baselines/grid_search_for_RAG/comprehensive_grid_search.py \
    --test data_processed_filtered/test.jsonl \
    --mapping data_processed_filtered/mappings/leaf2id.json \
    --shared_cache models/improved_RAG_filtered_dataset \
    --k_tfidf_range 8 9 10 11 12 \
    --k_embed_range 8 9 10 11 12 \
    --models mistral-small-latest \
    --limit 100
```

## Using with eval_lib

You can also use the unified evaluation framework with shared caches:

```bash
python -m eval_lib.core improved_rag \
    --test data_processed_filtered/test.jsonl \
    --mapping data_processed_filtered/mappings/leaf2id.json \
    --shared_cache models/improved_RAG_filtered_dataset \
    --k_tfidf 10 \
    --k_embed 10 \
    --limit 100
```

## Interpreting Results

### Key Metrics

1. **Retrieval Recall**: Should be ≥95% for good performance
2. **F1 Micro**: Overall accuracy across all classes
3. **F1 Macro**: Balanced accuracy across classes
4. **Total Candidates**: Cost consideration
5. **API Cost**: Estimated cost per parameter combination

### Example Output

```
🔍 CACHE VERIFICATION:
✅ faiss_index.bin      FAISS vector index   6.3 MB
✅ embed_cache.joblib   Embedding cache      14.3 MB
✅ llm_cache.joblib     LLM response cache   0.0 MB

🚀 QUICK PARAMETER TEST:
Testing 8 combinations...
🥇 Best F1 Micro: k_tfidf=10, k_embed=10
   F1 Micro: 0.7845, F1 Macro: 0.7234
   Retrieval Recall: 0.9600
   Total Candidates: 20

📊 COMPREHENSIVE GRID SEARCH:
Testing 16 combinations across 3 models...
🏆 BEST COMBINATIONS:
🥇 mistral-large-latest: k_tfidf=25, k_embed=25
   F1 Micro: 0.8234, Cost: $12.45
🥈 mistral-medium-latest: k_tfidf=20, k_embed=20  
   F1 Micro: 0.8123, Cost: $8.23
🥉 mistral-small-latest: k_tfidf=15, k_embed=15
   F1 Micro: 0.7945, Cost: $4.12

💡 RECOMMENDATIONS:
- For best performance: mistral-large-latest with k_tfidf=25, k_embed=25
- For cost efficiency: mistral-small-latest with k_tfidf=15, k_embed=15
- For balanced approach: mistral-medium-latest with k_tfidf=20, k_embed=20
```

## Troubleshooting

### No Cache Files Found

If you don't have existing cache files:

```bash
# Run the improved RAG baseline first to create caches
python src/baselines/improved_RAG_baseline.py \
    --test data_processed_filtered/test.jsonl \
    --mapping data_processed_filtered/mappings/leaf2id.json \
    --output_dir models/improved_RAG_filtered_dataset \
    --limit 10
```

### Cache Linking Issues

If symlinks fail, the scripts will automatically copy the files instead.

### Memory Issues

For large datasets, consider:
- Reducing the limit parameter
- Using smaller parameter ranges
- Running on a machine with more RAM

## Best Practices

1. **Start with quick test** to understand the parameter space
2. **Use shared caches** to save time and money
3. **Focus on recall ≥95%** for good performance
4. **Balance performance with cost** (total candidates)
5. **Test on a subset first** before full evaluation
6. **Document your findings** for future reference
7. **Use the filtered dataset** (data_processed_filtered) for consistent results

## Cost Estimation

### API Pricing (as of 2024)
- **Mistral Small**: $0.10 per 1M input tokens, $0.30 per 1M output tokens
- **Mistral Medium**: $0.40 per 1M input tokens, $2.0 per 1M output tokens  
- **Mistral Large**: $2.0 per 1M input tokens, $6.0 per 1M output tokens
- **Embeddings**: $0.10 per 1M tokens

### Cost Optimization Tips
1. **Start with Mistral Small** for initial exploration
2. **Use shared caches** to avoid redundant API calls
3. **Test on smaller subsets** before full evaluation
4. **Focus on promising parameter ranges** based on quick tests

## File Organization

```
src/baselines/
├── check_cache.py                           # Cache verification utility
├── improved_RAG_baseline.py                 # Main RAG baseline
├── README_parameter_search.md               # This guide
└── grid_search_for_RAG/                     # Parameter search tools
    ├── quick_param_test_extended.py         # Extended quick parameter exploration
    └── comprehensive_grid_search.py         # Full grid search with cost analysis
```

## Dataset Information

The parameter search is designed for the filtered dataset:
- **Dataset**: `data_processed_filtered/`
- **Categories**: 369 leaf categories (≥8 products each)
- **Train/Val/Test**: 15,941 / 1,976 / 1,950 products
- **Cache directory**: `models/improved_RAG_filtered_dataset/` 