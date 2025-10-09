"""Test the /api/v1/retrieve endpoint with correct settings."""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000/api/v1/retrieve"

print("=" * 80)
print("Testing /api/v1/retrieve Endpoint (Phase 2 - HyDE + BM25)")
print("=" * 80)

# Correct request with no score threshold
request_data = {
    "query": "semantic chunking",
    "collection_name": "insta_rag_test_collection",
    "top_k": 20,
    "enable_hyde": True,
    "enable_keyword_search": False,  # Disabled (won't work with MongoDB storage)
    "score_threshold": None,  # ✅ No threshold - return all results!
    "return_full_chunks": True,
    "deduplicate": True
}

print(f"\n📤 Request:")
print(json.dumps(request_data, indent=2))

print(f"\n⏳ Sending request to {API_URL}...")

try:
    response = requests.post(API_URL, json=request_data, timeout=30)
    response.raise_for_status()
    result = response.json()

    print(f"\n✅ Response received!")
    print("=" * 80)

    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"  Success: {result.get('success')}")
    print(f"  Chunks returned: {len(result.get('chunks', []))}")
    print(f"  Query: {result.get('query')}")

    # Queries generated (HyDE)
    queries = result.get('queries_generated', {})
    if queries:
        print(f"\n🔍 QUERIES GENERATED (HyDE):")
        for key, val in queries.items():
            val_str = str(val)
            print(f"  {key}: {val_str[:80]}..." if len(val_str) > 80 else f"  {key}: {val_str}")

    # Stats
    stats = result.get('stats', {})
    print(f"\n📈 PERFORMANCE STATS:")
    print(f"  Query generation: {stats.get('query_generation_time_ms', 0):.2f}ms")
    print(f"  Vector search: {stats.get('vector_search_time_ms', 0):.2f}ms")
    print(f"  Keyword search: {stats.get('keyword_search_time_ms', 0):.2f}ms")
    print(f"  Total time: {stats.get('total_time_ms', 0):.2f}ms")

    print(f"\n📦 RETRIEVAL STATS:")
    print(f"  Total chunks retrieved: {stats.get('total_chunks_retrieved', 0)}")
    print(f"  Vector chunks: {stats.get('vector_search_chunks', 0)}")
    print(f"  Keyword chunks: {stats.get('keyword_search_chunks', 0)}")
    print(f"  After deduplication: {stats.get('chunks_after_dedup', 0)}")
    print(f"  Final chunks: {stats.get('chunks_after_reranking', 0)}")

    # Show chunks
    chunks = result.get('chunks', [])
    if chunks:
        print(f"\n📄 CHUNKS (showing first 3 of {len(chunks)}):")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n  Chunk {i}:")
            print(f"    Rank: {chunk.get('rank', 0)}")
            print(f"    Relevance Score: {chunk.get('relevance_score', 0):.4f}")
            print(f"    Vector Score: {chunk.get('vector_score', 0):.4f}")
            print(f"    Content Length: {len(chunk.get('content', ''))} chars")
            print(f"    Content Preview: {chunk.get('content', '')[:150]}...")
            print(f"    Metadata: {list(chunk.get('metadata', {}).keys())}")

        # Score distribution
        print(f"\n📊 SCORE DISTRIBUTION:")
        scores = [c.get('relevance_score', 0) for c in chunks]
        print(f"  Highest: {max(scores):.4f}")
        print(f"  Lowest: {min(scores):.4f}")
        print(f"  Average: {sum(scores)/len(scores):.4f}")

        print(f"\n💡 RECOMMENDATION:")
        avg_score = sum(scores)/len(scores)
        if avg_score < 0.3:
            print(f"  Scores are in normal range (0.15-0.40)")
            print(f"  Use score_threshold=0.15 or null for best results")
        else:
            print(f"  Scores are good!")
            print(f"  Use score_threshold=0.2-0.3 for quality filtering")
    else:
        print(f"\n⚠️  NO CHUNKS RETURNED!")
        print(f"  Check if:")
        print(f"  1. Collection exists and has data")
        print(f"  2. score_threshold is not too high")
        print(f"  3. MongoDB content is accessible")

    # Sources
    sources = result.get('sources', [])
    if sources:
        print(f"\n📚 SOURCES:")
        for source in sources:
            print(f"  {source.get('source')}: {source.get('chunks_count')} chunks (avg: {source.get('avg_relevance', 0):.4f})")

    # Errors
    errors = result.get('errors', [])
    if errors:
        print(f"\n⚠️  ERRORS:")
        for error in errors:
            print(f"  - {error}")

    print(f"\n" + "=" * 80)
    print("✅ TEST COMPLETE!")
    print("=" * 80)

except requests.exceptions.ConnectionError:
    print(f"\n❌ ERROR: Cannot connect to {API_URL}")
    print(f"   Make sure the API server is running:")
    print(f"   cd testing_api")
    print(f"   uvicorn main:app --reload")

except requests.exceptions.Timeout:
    print(f"\n❌ ERROR: Request timeout (>30s)")
    print(f"   The API might be processing a large query")

except requests.exceptions.HTTPError as e:
    print(f"\n❌ HTTP ERROR: {e}")
    print(f"   Response: {e.response.text}")

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()

print()
