"""Test the /api/v1/retrieve endpoint to debug issues."""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000/api/v1/retrieve"

# Test 1: Basic retrieval (no threshold)
print("=" * 80)
print("TEST 1: Basic Retrieval (Phase 2)")
print("=" * 80)

request_data = {
    "query": "semantic chunking",
    "collection_name": "insta_rag_test_collection",
    "top_k": 10,
    "enable_hyde": True,
    "enable_keyword_search": True,
    "score_threshold": None,  # No threshold
    "return_full_chunks": True,
    "deduplicate": True
}

print(f"\nRequest:")
print(json.dumps(request_data, indent=2))

response = requests.post(API_URL, json=request_data)
result = response.json()

print(f"\nResponse:")
print(json.dumps(result, indent=2))

# Analyze the response
print(f"\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print(f"\nSuccess: {result['success']}")
print(f"Chunks returned: {len(result.get('chunks', []))}")

stats = result.get('stats', {})
print(f"\nStats:")
print(f"  Total chunks retrieved: {stats.get('total_chunks_retrieved', 0)}")
print(f"  Vector chunks: {stats.get('vector_search_chunks', 0)}")
print(f"  Keyword chunks: {stats.get('keyword_search_chunks', 0)}")
print(f"  After dedup: {stats.get('chunks_after_dedup', 0)}")
print(f"  Final chunks: {stats.get('chunks_after_reranking', 0)}")

print(f"\nGenerated Queries:")
for key, val in result.get('queries_generated', {}).items():
    print(f"  {key}: {val[:80]}..." if len(str(val)) > 80 else f"  {key}: {val}")

# Check if chunks have content
if result.get('chunks'):
    print(f"\nFirst Chunk:")
    chunk = result['chunks'][0]
    print(f"  Score: {chunk.get('relevance_score', 0)}")
    print(f"  Content length: {len(chunk.get('content', ''))}")
    print(f"  Content preview: {chunk.get('content', '')[:200]}...")
else:
    print(f"\n⚠️ WARNING: No chunks returned!")
    print(f"   This means:")
    print(f"   1. Score threshold too high, OR")
    print(f"   2. MongoDB content fetch failed, OR")
    print(f"   3. No matching documents found")

if stats.get('keyword_search_chunks', 0) == 0:
    print(f"\n⚠️ WARNING: BM25 not working!")
    print(f"   Reason: Content is stored in MongoDB, not in Qdrant payload")
    print(f"   BM25 needs content in Qdrant to build search index")

print("\n" + "=" * 80)
