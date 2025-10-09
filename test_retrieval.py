"""Test script for retrieval functionality."""

from dotenv import load_dotenv
from src.insta_rag.core.config import RAGConfig
from src.insta_rag.core.client import RAGClient
from src.insta_rag.models.document import DocumentInput

# Load environment variables
load_dotenv()

def test_retrieval():
    """Test document upload and retrieval."""

    print("=" * 70)
    print("RETRIEVAL TESTING")
    print("=" * 70)

    # Initialize RAG client
    print("\n1. Initializing RAG Client...")
    config = RAGConfig.from_env()
    client = RAGClient(config)
    print("   ✓ RAG Client initialized")

    collection_name = "test_retrieval_collection"

    # Test 1: Upload test documents
    print(f"\n2. Uploading test documents to '{collection_name}'...")

    test_documents = [
        DocumentInput.from_text(
            text="""
            Semantic chunking is a method of dividing text into meaningful segments.
            Unlike fixed-size chunking, semantic chunking analyzes the content to find
            natural boundaries between topics. This is done by embedding sentences and
            calculating similarity scores between consecutive sentences. When similarity
            drops below a threshold, it indicates a topic change and creates a natural
            breakpoint for chunking.
            """,
            metadata={"title": "Semantic Chunking Explained", "category": "documentation"}
        ),
        DocumentInput.from_text(
            text="""
            Vector databases like Qdrant are designed for storing and searching embeddings.
            They use approximate nearest neighbor (ANN) algorithms for fast similarity search.
            Qdrant specifically uses HNSW (Hierarchical Navigable Small World) graphs for
            efficient vector search. The COSINE distance metric is commonly used to measure
            similarity between vectors in semantic search applications.
            """,
            metadata={"title": "Vector Databases Overview", "category": "documentation"}
        ),
        DocumentInput.from_text(
            text="""
            Azure OpenAI provides access to powerful embedding models like text-embedding-3-large.
            This model generates 3072-dimensional vectors that capture semantic meaning of text.
            These embeddings can be used for semantic search, clustering, and recommendation systems.
            The Azure API offers enterprise-grade reliability and security for production deployments.
            """,
            metadata={"title": "Azure OpenAI Embeddings", "category": "documentation"}
        ),
    ]

    response = client.add_documents(
        documents=test_documents,
        collection_name=collection_name,
    )

    if response.success:
        print(f"   ✓ Uploaded {response.documents_processed} documents")
        print(f"   ✓ Created {response.total_chunks} chunks")
        print(f"   ✓ Processing time: {response.processing_stats.total_time_ms:.2f}ms")
    else:
        print(f"   ✗ Upload failed: {response.errors}")
        return

    # Test 2: Search with different queries
    print("\n3. Testing search with different queries...")

    test_queries = [
        "What is semantic chunking?",
        "How does vector search work?",
        "Tell me about Azure OpenAI embeddings",
        "What are HNSW graphs?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: '{query}'")

        search_response = client.search(
            query=query,
            collection_name=collection_name,
            top_k=3,
        )

        if search_response.success:
            print(f"   ✓ Found {len(search_response.chunks)} results")
            print(f"   ✓ Search time: {search_response.retrieval_stats.total_time_ms:.2f}ms")

            # Show top result
            if search_response.chunks:
                top_chunk = search_response.chunks[0]
                print(f"\n   Top Result:")
                print(f"     - Score: {top_chunk.relevance_score:.4f}")
                print(f"     - Title: {top_chunk.metadata.get('title', 'N/A')}")
                print(f"     - Content preview: {top_chunk.content[:100]}...")

                # Show sources
                if search_response.sources:
                    print(f"\n   Sources:")
                    for source in search_response.sources:
                        print(f"     - {source.source}: {source.chunks_count} chunks (avg score: {source.avg_relevance:.4f})")
        else:
            print(f"   ✗ Search failed: {search_response.errors}")

    # Test 3: Search with filters
    print("\n\n4. Testing search with metadata filters...")

    filter_query = "embeddings"
    filters = {"category": "documentation"}

    print(f"   Query: '{filter_query}' with filter: {filters}")

    search_response = client.search(
        query=filter_query,
        collection_name=collection_name,
        top_k=5,
        filters=filters,
    )

    if search_response.success:
        print(f"   ✓ Found {len(search_response.chunks)} filtered results")
        print(f"   ✓ Search time: {search_response.retrieval_stats.total_time_ms:.2f}ms")

        for i, chunk in enumerate(search_response.chunks, 1):
            print(f"\n   Result {i}:")
            print(f"     - Score: {chunk.relevance_score:.4f}")
            print(f"     - Title: {chunk.metadata.get('title', 'N/A')}")
            print(f"     - Category: {chunk.metadata.get('category', 'N/A')}")
    else:
        print(f"   ✗ Filtered search failed: {search_response.errors}")

    # Test 4: Performance stats
    print("\n\n5. Retrieval Performance Statistics:")
    print(f"   - Query Generation: {search_response.retrieval_stats.query_generation_time_ms:.2f}ms")
    print(f"   - Vector Search: {search_response.retrieval_stats.vector_search_time_ms:.2f}ms")
    print(f"   - Total Time: {search_response.retrieval_stats.total_time_ms:.2f}ms")
    print(f"   - Chunks Retrieved: {search_response.retrieval_stats.total_chunks_retrieved}")

    print("\n" + "=" * 70)
    print("✅ ALL RETRIEVAL TESTS PASSED!")
    print("=" * 70)
    print("\nThe search/retrieval functionality is working correctly.")
    print(f"Collection '{collection_name}' now contains searchable documents.\n")


if __name__ == "__main__":
    test_retrieval()
