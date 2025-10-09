"""Test Phase 2 Complete: HyDE + BM25 Keyword Search with Content Verification."""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000/api/v1/retrieve"

print("=" * 80)
print("Testing Phase 2 Complete: HyDE + BM25 Keyword Search")
print("=" * 80)

# Correct request with BOTH features enabled
request_data = {
    "query": "semantic chunking",
    "collection_name": "insta_rag_test_collection",
    "top_k": 20,
    "enable_hyde": True,           # ✅ HyDE enabled
    "enable_keyword_search": True,  # ✅ BM25 enabled
    "score_threshold": None,        # ✅ No threshold - return all results!
    "return_full_chunks": True,
    "deduplicate": True
}

print(f"\n📤 Request Configuration:")
print(f"  Query: '{request_data['query']}'")
print(f"  Collection: {request_data['collection_name']}")
print(f"  Top-K: {request_data['top_k']}")
print(f"  HyDE Enabled: {request_data['enable_hyde']}")
print(f"  Keyword Search (BM25) Enabled: {request_data['enable_keyword_search']}")
print(f"  Score Threshold: {request_data['score_threshold']}")
print(f"  Return Full Chunks: {request_data['return_full_chunks']}")

print(f"\n⏳ Sending request to {API_URL}...")

try:
    response = requests.post(API_URL, json=request_data, timeout=60)
    response.raise_for_status()
    result = response.json()

    print(f"\n✅ Response received!")
    print("=" * 80)

    # Summary
    print(f"\n📊 RESPONSE SUMMARY:")
    print(f"  Success: {result.get('success')}")
    print(f"  Chunks returned: {len(result.get('chunks', []))} chunks")
    print(f"  Original query: {result.get('query')}")

    # HyDE Query Generation
    queries = result.get('queries_generated', {})
    if queries:
        print(f"\n🔍 HYDE QUERY GENERATION:")
        print(f"  ✓ HyDE is working!")
        for key, val in queries.items():
            val_str = str(val)
            if len(val_str) > 100:
                print(f"  {key}: {val_str[:100]}...")
            else:
                print(f"  {key}: {val_str}")
    else:
        print(f"\n⚠️  HYDE QUERY GENERATION:")
        print(f"  No queries_generated field - HyDE may have failed")

    # Performance Stats
    stats = result.get('stats', {})
    print(f"\n⏱️  PERFORMANCE STATS:")
    print(f"  Query generation time: {stats.get('query_generation_time_ms', 0):.2f}ms")
    print(f"  Vector search time: {stats.get('vector_search_time_ms', 0):.2f}ms")
    print(f"  Keyword search time: {stats.get('keyword_search_time_ms', 0):.2f}ms")
    print(f"  Total time: {stats.get('total_time_ms', 0):.2f}ms")

    # Retrieval Stats
    print(f"\n📦 RETRIEVAL STATS:")
    print(f"  Total chunks retrieved: {stats.get('total_chunks_retrieved', 0)}")
    print(f"  Vector search chunks: {stats.get('vector_search_chunks', 0)}")
    print(f"  Keyword search chunks: {stats.get('keyword_search_chunks', 0)}")
    print(f"  After deduplication: {stats.get('chunks_after_dedup', 0)}")
    print(f"  Final chunks: {stats.get('chunks_after_reranking', 0)}")

    # BM25 Status Check
    keyword_chunks = stats.get('keyword_search_chunks', 0)
    if keyword_chunks > 0:
        print(f"\n✅ BM25 KEYWORD SEARCH: WORKING!")
        print(f"  Found {keyword_chunks} chunks via keyword search")
    else:
        print(f"\n⚠️  BM25 KEYWORD SEARCH: NOT WORKING")
        print(f"  Possible reasons:")
        print(f"  1. BM25 corpus failed to build (check server logs)")
        print(f"  2. No keyword matches for query '{request_data['query']}'")
        print(f"  3. MongoDB content not accessible")

    # Content Verification
    chunks = result.get('chunks', [])
    if chunks:
        print(f"\n📄 CONTENT VERIFICATION:")
        print(f"  Total chunks: {len(chunks)}")

        # Check if chunks have content
        chunks_with_content = 0
        total_content_length = 0
        for chunk in chunks:
            content = chunk.get('content', '')
            if content and len(content.strip()) > 0:
                chunks_with_content += 1
                total_content_length += len(content)

        print(f"  Chunks with content: {chunks_with_content}/{len(chunks)}")
        print(f"  Total content length: {total_content_length} chars")
        print(f"  Average content length: {total_content_length/chunks_with_content if chunks_with_content > 0 else 0:.0f} chars")

        if chunks_with_content == len(chunks):
            print(f"  ✅ ALL chunks have content!")
        elif chunks_with_content > 0:
            print(f"  ⚠️  {len(chunks) - chunks_with_content} chunks are missing content")
        else:
            print(f"  ❌ NO chunks have content!")

        # Show sample chunks
        print(f"\n📝 SAMPLE CHUNKS (first 3):")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n  Chunk {i}:")
            print(f"    Rank: {chunk.get('rank', 0)}")
            print(f"    Relevance Score: {chunk.get('relevance_score', 0):.4f}")
            print(f"    Vector Score: {chunk.get('vector_score', 0):.4f}")

            content = chunk.get('content', '')
            if content:
                print(f"    Content Length: {len(content)} chars")
                print(f"    Content Preview: {content[:200]}...")
            else:
                print(f"    Content: ❌ EMPTY!")

            metadata = chunk.get('metadata', {})
            print(f"    Metadata Keys: {list(metadata.keys())}")
            if 'source' in metadata:
                print(f"    Source: {metadata['source']}")

        # Score distribution
        print(f"\n📊 SCORE DISTRIBUTION:")
        scores = [c.get('relevance_score', 0) for c in chunks]
        print(f"  Highest: {max(scores):.4f}")
        print(f"  Lowest: {min(scores):.4f}")
        print(f"  Average: {sum(scores)/len(scores):.4f}")
    else:
        print(f"\n❌ NO CHUNKS RETURNED!")
        print(f"  Possible reasons:")
        print(f"  1. Collection is empty")
        print(f"  2. Score threshold too high (current: {request_data['score_threshold']})")
        print(f"  3. No matches found for query")

    # Sources
    sources = result.get('sources', [])
    if sources:
        print(f"\n📚 SOURCES:")
        for source in sources:
            print(f"  {source.get('source')}: {source.get('chunks_count')} chunks (avg score: {source.get('avg_relevance', 0):.4f})")

    # Errors
    errors = result.get('errors', [])
    if errors:
        print(f"\n⚠️  ERRORS:")
        for error in errors:
            print(f"  - {error}")

    # Final Assessment
    print(f"\n" + "=" * 80)
    print("PHASE 2 FEATURE ASSESSMENT:")
    print("=" * 80)

    hyde_working = bool(queries)
    bm25_working = keyword_chunks > 0
    content_working = chunks_with_content > 0 if chunks else False

    print(f"  HyDE Query Generation: {'✅ WORKING' if hyde_working else '❌ NOT WORKING'}")
    print(f"  BM25 Keyword Search: {'✅ WORKING' if bm25_working else '❌ NOT WORKING'}")
    print(f"  Content Return: {'✅ WORKING' if content_working else '❌ NOT WORKING'}")

    if hyde_working and bm25_working and content_working:
        print(f"\n🎉 ALL PHASE 2 FEATURES WORKING!")
    else:
        print(f"\n⚠️  Some features need attention. Check server logs for details.")

    print("=" * 80)

except requests.exceptions.ConnectionError:
    print(f"\n❌ ERROR: Cannot connect to {API_URL}")
    print(f"   Make sure the API server is running:")
    print(f"   cd testing_api")
    print(f"   uvicorn main:app --reload")

except requests.exceptions.Timeout:
    print(f"\n❌ ERROR: Request timeout (>60s)")
    print(f"   The API might be processing a large query or building BM25 corpus")

except requests.exceptions.HTTPError as e:
    print(f"\n❌ HTTP ERROR: {e}")
    print(f"   Response: {e.response.text}")

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()

print()
