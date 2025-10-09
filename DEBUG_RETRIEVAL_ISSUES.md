# Debugging Retrieval Issues

## 🔍 Issues You're Experiencing

Based on your API response:
```json
{
  "chunks": [],                      // ❌ No chunks returned!
  "chunks_after_reranking": 0,       // ❌ All chunks filtered out!
  "keyword_search_chunks": 0,        // ❌ BM25 not working
  "total_chunks_retrieved": 16,      // ✅ Got 16 chunks from vector search
  "chunks_after_dedup": 16           // ✅ Deduplication worked
}
```

## 🐛 Root Causes

### Issue 1: No Chunks Returned (`chunks_after_reranking: 0`)

**Possible causes:**

1. **High score_threshold** - You might have set `score_threshold` too high
   - Vector similarity scores are typically 0.0-1.0
   - Setting threshold > 0.5 might filter everything out

2. **top_k=0** - Check if `top_k` was accidentally set to 0

3. **MongoDB fetch failing silently** - Content not being retrieved from MongoDB

### Issue 2: BM25 Not Working (`keyword_search_chunks: 0`)

**Root cause:** Content is stored in MongoDB, not in Qdrant payload.

BM25 keyword search requires **content in Qdrant** to build a searchable index. When content is stored in MongoDB:
- Qdrant only has: `vectors + metadata + mongodb_id`
- Qdrant does NOT have: `content` field
- BM25 cannot build index → skips keyword search

## ✅ Solutions

### Solution 1: Fix No Chunks Issue

**Test without score_threshold:**

```bash
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "semantic chunking",
    "collection_name": "insta_rag_test_collection",
    "top_k": 10,
    "enable_hyde": true,
    "enable_keyword_search": true,
    "score_threshold": null,
    "return_full_chunks": true,
    "deduplicate": true
  }'
```

**Or test with low threshold:**

```bash
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "semantic chunking",
    "collection_name": "insta_rag_test_collection",
    "top_k": 10,
    "score_threshold": 0.1
  }'
```

### Solution 2: Enable BM25 (Fix Content Storage)

**Option A: Store content in Qdrant (Recommended for BM25)**

Modify your document upload to **NOT** use MongoDB content storage:

```python
from insta_rag import RAGClient, RAGConfig

# Create config WITHOUT MongoDB
config = RAGConfig.from_env()
config.mongodb = None  # Disable MongoDB content storage

client = RAGClient(config)

# Upload documents - content will be stored in Qdrant
response = client.add_documents(
    documents=[...],
    collection_name="your_collection"
)

# Now BM25 will work!
response = client.retrieve(
    query="your query",
    collection_name="your_collection",
    enable_keyword_search=True  # BM25 will now work
)
```

**Option B: Enhance BM25 to fetch from MongoDB (Future Enhancement)**

This would require modifying `src/insta_rag/retrieval/keyword_search.py` to fetch content from MongoDB during corpus building.

**Option C: Use Vector Search Only (Current Workaround)**

```python
response = client.retrieve(
    query="your query",
    collection_name="your_collection",
    enable_hyde=True,           # Keep HyDE
    enable_keyword_search=False # Disable BM25
)
```

## 🧪 Testing Steps

### Step 1: Run Debug Test Script

```bash
# Test the API endpoint
python test_api_retrieve.py
```

This will show detailed diagnostics.

### Step 2: Check API Server Logs

When you call the `/api/v1/retrieve` endpoint, the server now prints detailed logs:

```
✓ MongoDB connected: Test_Insta_RAG
Warning: HyDE generation failed: Error code: 404 - ...
   BM25 corpus built: 0 documents indexed
   Warning: BM25 index not available, skipping keyword search
   ✓ Fetched content for 16 chunks from MongoDB
   Step 6: Selecting top-10 chunks from 16 ranked chunks
   After top-k selection: 10 chunks
   After score threshold (0.9): 0 chunks (filtered out: 10)  ← HERE'S THE PROBLEM!
   ✓ Final chunks to return: 0
```

Look for:
- MongoDB fetch count
- BM25 corpus size
- Score threshold filtering

### Step 3: Test with Simple Python Script

```python
from insta_rag import RAGClient, RAGConfig

config = RAGConfig.from_env()
client = RAGClient(config)

# Test with NO threshold
response = client.retrieve(
    query="semantic chunking",
    collection_name="insta_rag_test_collection",
    top_k=10,
    score_threshold=None,  # No filtering
    enable_hyde=False,     # Disable for faster test
    enable_keyword_search=False
)

print(f"Success: {response.success}")
print(f"Chunks returned: {len(response.chunks)}")
print(f"Stats: {response.retrieval_stats.to_dict()}")

if response.chunks:
    chunk = response.chunks[0]
    print(f"\nFirst chunk score: {chunk.relevance_score}")
    print(f"Content preview: {chunk.content[:200]}")
else:
    print("\n❌ No chunks returned!")
```

## 📊 Understanding Typical Scores

Vector similarity scores (COSINE distance):

| Score Range | Meaning |
|------------|---------|
| 0.9 - 1.0  | Extremely similar (exact or near-exact match) |
| 0.7 - 0.9  | Very similar (good match) |
| 0.5 - 0.7  | Moderately similar (may be relevant) |
| 0.3 - 0.5  | Somewhat similar (loosely related) |
| 0.0 - 0.3  | Not very similar (likely not relevant) |

**Typical scores you'll see:** 0.15 - 0.4 for normal searches

**Setting `score_threshold=0.7` will likely filter out ALL results!**

## ✅ Recommended Settings

### For Best Results:

```python
response = client.retrieve(
    query="your question",
    collection_name="your_collection",
    top_k=20,                      # Good default
    enable_hyde=True,              # Better retrieval quality
    enable_keyword_search=False,   # Disable (won't work with MongoDB storage)
    score_threshold=None,          # Let top-k handle selection
    return_full_chunks=True,       # Get full content
    deduplicate=True               # Remove duplicates
)
```

### For Faster Queries:

```python
response = client.retrieve(
    query="your question",
    collection_name="your_collection",
    top_k=10,                      # Fewer results
    enable_hyde=False,             # Skip HyDE (saves ~1.3s)
    enable_keyword_search=False,   # Skip BM25
    score_threshold=0.001,           # Basic quality filter
    return_full_chunks=False,      # Truncated content
    deduplicate=True
)
```

## 🔧 Quick Fixes

### Fix 1: Remove score_threshold

```bash
# DON'T DO THIS:
curl ... -d '{"score_threshold": 0.9}'  # ❌ Too high!

# DO THIS:
curl ... -d '{"score_threshold": null}'  # ✅ No filtering
```

### Fix 2: Use Reasonable Threshold

```bash
curl ... -d '{"score_threshold": 0.15}'  # ✅ Reasonable
```

### Fix 3: Increase top_k

```bash
curl ... -d '{"top_k": 20}'  # ✅ More results
```

## 📖 API Documentation

Visit your Swagger UI for complete API documentation:

```
http://localhost:8000/docs
```

Look for the `/api/v1/retrieve` endpoint with:
- Parameter descriptions
- Default values
- Example requests
- Example responses

## 🆘 Still Having Issues?

Run this comprehensive diagnostic:

```python
from insta_rag import RAGClient, RAGConfig

config = RAGConfig.from_env()
client = RAGClient(config)

print("=" * 80)
print("DIAGNOSTIC TEST")
print("=" * 80)

# Test 1: Simple search (no Phase 2 features)
print("\n1. Testing basic search...")
response1 = client.search(
    query="test",
    collection_name="insta_rag_test_collection",
    top_k=5
)
print(f"   Chunks: {len(response1.chunks)}")
print(f"   Success: {response1.success}")

# Test 2: Retrieve with no filters
print("\n2. Testing retrieve (no threshold)...")
response2 = client.retrieve(
    query="test",
    collection_name="insta_rag_test_collection",
    top_k=5,
    score_threshold=None,
    enable_hyde=False,
    enable_keyword_search=False
)
print(f"   Chunks: {len(response2.chunks)}")
print(f"   Success: {response2.success}")

# Test 3: Check collection
print("\n3. Checking collection...")
try:
    info = client.get_collection_info("insta_rag_test_collection")
    print(f"   Vectors: {info['vectors_count']}")
    print(f"   Status: {info['status']}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 80)
```

This will help identify the exact issue!
