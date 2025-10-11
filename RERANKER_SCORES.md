# Understanding Reranker Scores

## BGE Reranker (BAAI/bge-reranker-v2-m3)

The BGE reranker produces **negative scores** by default. This is normal behavior for this model.

### Score Characteristics:

- **Range**: Typically -10.0 to +10.0
- **Higher is better**: -0.96 is MORE relevant than -6.99
- **Most relevant results**: Usually between -3.0 and +5.0
- **Less relevant results**: Usually below -5.0

### Score Interpretation:

| Score Range | Relevance Level | Description |
|-------------|-----------------|-------------|
| +3.0 to +10.0 | Excellent | Highly relevant, strong semantic match |
| 0.0 to +3.0 | Very Good | Strong relevance to query |
| -3.0 to 0.0 | Good | Relevant, useful results |
| -5.0 to -3.0 | Moderate | Some relevance, may be useful |
| Below -5.0 | Low | Weak relevance, consider filtering |

### Using Score Thresholds:

When using `score_threshold` with BGE reranker, use **negative values**:

```python
# Good examples for BGE reranker:
response = client.retrieve(
    query="What is semantic chunking?",
    collection_name="my_collection",
    top_k=20,
    score_threshold=-5.0,  # ✅ Filter out weakly relevant results
)

# Or for stricter filtering:
response = client.retrieve(
    query="What is semantic chunking?",
    collection_name="my_collection",
    top_k=20,
    score_threshold=-3.0,  # ✅ Keep only moderately to highly relevant
)

# ❌ WRONG - This will filter out ALL results:
response = client.retrieve(
    query="What is semantic chunking?",
    collection_name="my_collection",
    top_k=20,
    score_threshold=0.01,  # ❌ Too high for BGE negative scores
)
```

### Example Output:

```
🎯 Reranking:
   Reranking 33 chunks using bge...
   ✓ Reranked to 20 chunks (398.11ms)
   ✓ Score range: -6.9961 to -0.9624  # Higher (less negative) is better!

   After score threshold (-5.0): 15 chunks  # Kept results with score >= -5.0
```

### Recommended Thresholds by Use Case:

| Use Case | Recommended Threshold | Rationale |
|----------|----------------------|-----------|
| General retrieval | No threshold or -6.0 | Keep most results |
| Document generation | -5.0 | Moderate quality filter |
| Question answering | -3.0 | High quality only |
| Fact verification | -2.0 | Very high confidence |

## Cohere Reranker (Alternative)

If you prefer **normalized 0-1 scores**, you can use Cohere reranker instead:

```python
# In .env:
COHERE_API_KEY=your_cohere_key_here

# Cohere produces 0-1 scores:
# - 1.0 = Perfect match
# - 0.8-1.0 = Highly relevant
# - 0.5-0.8 = Moderately relevant
# - Below 0.5 = Less relevant

response = client.retrieve(
    query="What is semantic chunking?",
    collection_name="my_collection",
    top_k=20,
    score_threshold=0.5,  # ✅ Works well with Cohere's 0-1 range
)
```

## Key Takeaways:

1. **BGE reranker** uses negative scores (higher = better)
2. **Use negative thresholds** like -5.0, -3.0, or -2.0 with BGE
3. **No threshold** (None) returns all results sorted by relevance
4. **Different models** use different score ranges - adjust accordingly
