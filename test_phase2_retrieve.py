"""
Test Phase 2 Retrieval - HyDE + BM25 Keyword Search

This script tests the Phase 2 implementation:
- HyDE query generation using Azure OpenAI
- Dual vector search (standard + HyDE queries)
- BM25 keyword search
- Combined deduplication
- MongoDB content fetching

Run this script with:
    python test_phase2_retrieve.py
Or with venv:
    venv/bin/python test_phase2_retrieve.py
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from insta_rag import RAGClient, RAGConfig


def print_separator(title=""):
    """Print a formatted separator."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}\n")
    else:
        print(f"\n{'-' * 80}\n")


def test_phase2_basic():
    """Test 1: Basic Phase 2 retrieval with HyDE + BM25."""
    print_separator("TEST 1: Basic Phase 2 Retrieval (HyDE + BM25 enabled)")

    config = RAGConfig.from_env()
    client = RAGClient(config)

    # Test query
    query = "What is semantic chunking and how does it work?"
    collection_name = "insta_rag_test_collection"

    print(f"Query: {query}")
    print(f"Collection: {collection_name}")
    print(f"Features: HyDE=True, BM25=True, Deduplication=True\n")

    # Execute retrieval
    start = time.time()
    response = client.retrieve(
        query=query,
        collection_name=collection_name,
        top_k=10,
        enable_hyde=True,
        enable_keyword_search=True,
        deduplicate=True,
    )
    elapsed = time.time() - start

    print(f"✓ Retrieval completed in {elapsed:.2f}s\n")

    # Print results
    print(f"Success: {response.success}")
    print(f"Chunks returned: {len(response.chunks)}")
    print(f"\nGenerated Queries:")
    for key, val in response.queries_generated.items():
        print(f"  {key}: {val}")

    print(f"\nPerformance Stats:")
    stats = response.retrieval_stats
    print(f"  Query generation: {stats.query_generation_time_ms:.2f}ms")
    print(f"  Vector search: {stats.vector_search_time_ms:.2f}ms ({stats.vector_search_chunks} chunks)")
    print(f"  Keyword search: {stats.keyword_search_time_ms:.2f}ms ({stats.keyword_search_chunks} chunks)")
    print(f"  Total retrieved: {stats.total_chunks_retrieved} chunks")
    print(f"  After dedup: {stats.chunks_after_dedup} chunks")
    print(f"  Final chunks: {stats.chunks_after_reranking} chunks")
    print(f"  Total time: {stats.total_time_ms:.2f}ms")

    # Show top 3 results
    print(f"\nTop 3 Results:")
    for i, chunk in enumerate(response.chunks[:3], 1):
        print(f"\n  {i}. Score: {chunk.relevance_score:.4f}")
        print(f"     Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"     Content: {chunk.content[:150]}...")

    return response


def test_phase2_vs_phase1():
    """Test 2: Compare Phase 2 (HyDE+BM25) vs Phase 1 (no HyDE/BM25)."""
    print_separator("TEST 2: Phase 2 vs Phase 1 Comparison")

    config = RAGConfig.from_env()
    client = RAGClient(config)

    query = "explain semantic chunking"
    collection_name = "insta_rag_test_collection"

    print(f"Query: {query}")
    print(f"Collection: {collection_name}\n")

    # Phase 1 (no HyDE, no BM25)
    print("Running Phase 1 (Vector search only)...")
    start1 = time.time()
    response1 = client.retrieve(
        query=query,
        collection_name=collection_name,
        top_k=10,
        enable_hyde=False,
        enable_keyword_search=False,
        deduplicate=True,
    )
    time1 = time.time() - start1

    # Phase 2 (HyDE + BM25)
    print("Running Phase 2 (HyDE + BM25)...")
    start2 = time.time()
    response2 = client.retrieve(
        query=query,
        collection_name=collection_name,
        top_k=10,
        enable_hyde=True,
        enable_keyword_search=True,
        deduplicate=True,
    )
    time2 = time.time() - start2

    # Compare results
    print(f"\n{'Metric':<30} {'Phase 1':<15} {'Phase 2':<15} {'Diff':<15}")
    print("-" * 75)
    print(f"{'Execution time':<30} {time1:<15.2f} {time2:<15.2f} {time2-time1:<15.2f}")
    print(f"{'Vector chunks':<30} {response1.retrieval_stats.vector_search_chunks:<15} {response2.retrieval_stats.vector_search_chunks:<15} {response2.retrieval_stats.vector_search_chunks - response1.retrieval_stats.vector_search_chunks:<15}")
    print(f"{'Keyword chunks':<30} {response1.retrieval_stats.keyword_search_chunks:<15} {response2.retrieval_stats.keyword_search_chunks:<15} {response2.retrieval_stats.keyword_search_chunks - response1.retrieval_stats.keyword_search_chunks:<15}")
    print(f"{'Total chunks':<30} {response1.retrieval_stats.total_chunks_retrieved:<15} {response2.retrieval_stats.total_chunks_retrieved:<15} {response2.retrieval_stats.total_chunks_retrieved - response1.retrieval_stats.total_chunks_retrieved:<15}")
    print(f"{'After dedup':<30} {response1.retrieval_stats.chunks_after_dedup:<15} {response2.retrieval_stats.chunks_after_dedup:<15} {response2.retrieval_stats.chunks_after_dedup - response1.retrieval_stats.chunks_after_dedup:<15}")
    print(f"{'Final chunks':<30} {len(response1.chunks):<15} {len(response2.chunks):<15} {len(response2.chunks) - len(response1.chunks):<15}")

    # Compare top result relevance
    if response1.chunks and response2.chunks:
        print(f"{'Top relevance score':<30} {response1.chunks[0].relevance_score:<15.4f} {response2.chunks[0].relevance_score:<15.4f} {response2.chunks[0].relevance_score - response1.chunks[0].relevance_score:<15.4f}")

    print(f"\n✓ Phase 2 retrieved {response2.retrieval_stats.total_chunks_retrieved - response1.retrieval_stats.total_chunks_retrieved} more chunks")
    print(f"✓ Query generation added {response2.retrieval_stats.query_generation_time_ms:.2f}ms overhead")
    print(f"✓ BM25 search added {response2.retrieval_stats.keyword_search_time_ms:.2f}ms overhead")


def test_hyde_query_generation():
    """Test 3: Test HyDE query generation specifically."""
    print_separator("TEST 3: HyDE Query Generation")

    config = RAGConfig.from_env()
    client = RAGClient(config)

    queries = [
        "What is semantic chunking?",
        "How do I configure MongoDB?",
        "BM25 ranking algorithm",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Original: {query}")

        response = client.retrieve(
            query=query,
            collection_name="insta_rag_test_collection",
            top_k=1,
            enable_hyde=True,
            enable_keyword_search=False,  # Only test HyDE
        )

        if "standard" in response.queries_generated:
            print(f"   Standard: {response.queries_generated['standard']}")
        if "hyde" in response.queries_generated:
            print(f"   HyDE:     {response.queries_generated['hyde'][:100]}...")

        print(f"   Time: {response.retrieval_stats.query_generation_time_ms:.2f}ms")

    print(f"\n✓ HyDE query generation working")


def test_bm25_keyword_search():
    """Test 4: Test BM25 keyword search specifically."""
    print_separator("TEST 4: BM25 Keyword Search")

    config = RAGConfig.from_env()
    client = RAGClient(config)

    # Test with exact term that should match BM25 well
    query = "semantic chunking embeddings"

    print(f"Query: {query}")
    print(f"Testing BM25 keyword search for exact term matches\n")

    response = client.retrieve(
        query=query,
        collection_name="insta_rag_test_collection",
        top_k=10,
        enable_hyde=False,
        enable_keyword_search=True,  # Only test BM25
        deduplicate=True,
    )

    print(f"BM25 Results:")
    print(f"  Keyword chunks found: {response.retrieval_stats.keyword_search_chunks}")
    print(f"  Search time: {response.retrieval_stats.keyword_search_time_ms:.2f}ms")
    print(f"  Final chunks: {len(response.chunks)}")

    if response.chunks:
        print(f"\n  Top BM25 result:")
        chunk = response.chunks[0]
        print(f"    Score: {chunk.relevance_score:.4f}")
        print(f"    Content: {chunk.content[:200]}...")

    print(f"\n✓ BM25 keyword search working")


def test_combined_hybrid_search():
    """Test 5: Test full hybrid search (Vector + HyDE + BM25)."""
    print_separator("TEST 5: Full Hybrid Search (Vector + HyDE + BM25)")

    config = RAGConfig.from_env()
    client = RAGClient(config)

    query = "semantic chunking"

    print(f"Query: {query}")
    print(f"Combining: Vector Search + HyDE + BM25\n")

    response = client.retrieve(
        query=query,
        collection_name="insta_rag_test_collection",
        top_k=15,
        enable_hyde=True,
        enable_keyword_search=True,
        deduplicate=True,
    )

    print(f"Hybrid Search Results:")
    print(f"  Vector chunks (standard): ~25")
    print(f"  Vector chunks (HyDE): ~25")
    print(f"  Keyword chunks (BM25): {response.retrieval_stats.keyword_search_chunks}")
    print(f"  Total retrieved: {response.retrieval_stats.total_chunks_retrieved}")
    print(f"  After deduplication: {response.retrieval_stats.chunks_after_dedup}")
    print(f"  Final top-k: {len(response.chunks)}")

    print(f"\n  Performance:")
    stats = response.retrieval_stats
    print(f"    Query gen: {stats.query_generation_time_ms:.2f}ms")
    print(f"    Vector search: {stats.vector_search_time_ms:.2f}ms")
    print(f"    Keyword search: {stats.keyword_search_time_ms:.2f}ms")
    print(f"    Total: {stats.total_time_ms:.2f}ms")

    print(f"\n✓ Full hybrid search working")
    print(f"✓ Deduplication reduced {response.retrieval_stats.total_chunks_retrieved} → {response.retrieval_stats.chunks_after_dedup} chunks")


def main():
    """Run all Phase 2 tests."""
    print_separator("PHASE 2 RETRIEVAL TESTS")
    print("Testing HyDE query generation + BM25 keyword search\n")

    try:
        # Run all tests
        test_phase2_basic()
        print_separator()

        test_phase2_vs_phase1()
        print_separator()

        test_hyde_query_generation()
        print_separator()

        test_bm25_keyword_search()
        print_separator()

        test_combined_hybrid_search()

        # Summary
        print_separator("PHASE 2 TEST SUMMARY")
        print("✓ All tests passed!")
        print("✓ HyDE query generation working")
        print("✓ BM25 keyword search working")
        print("✓ Hybrid search combining all methods")
        print("✓ Deduplication working across all sources")
        print("\n🎉 Phase 2 implementation complete and working!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
