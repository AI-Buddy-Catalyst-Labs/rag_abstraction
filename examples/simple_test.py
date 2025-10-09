"""Simple test script for insta_rag library."""

from dotenv import load_dotenv

from insta_rag import DocumentInput, RAGClient, RAGConfig

# Load environment variables
load_dotenv()


def test_basic_functionality():
    """Test basic RAG functionality with text input."""

    print("Testing insta_rag library...")
    print("-" * 60)

    # Initialize client
    print("1. Initializing RAG client...")
    config = RAGConfig.from_env()
    client = RAGClient(config)
    print("   ✓ Client initialized\n")

    # Create test document
    print("2. Creating test document...")
    test_text = """
    Artificial Intelligence (AI) is revolutionizing how we work and live.
    Machine learning models can now understand natural language, generate images,
    and even write code.

    The RAG (Retrieval-Augmented Generation) pattern combines the power of
    large language models with external knowledge bases. This allows AI systems
    to provide more accurate and up-to-date information.

    Key components of a RAG system include:
    - Document chunking and processing
    - Vector embeddings for semantic search
    - Vector databases for efficient storage and retrieval
    - Reranking for improved result quality

    By implementing RAG, organizations can build AI applications that leverage
    their proprietary data while maintaining accuracy and reducing hallucinations.
    """

    doc = DocumentInput.from_text(
        text=test_text,
        metadata={
            "source": "test_script",
            "topic": "AI and RAG",
        },
    )
    print("   ✓ Document created\n")

    # Process document
    print("3. Processing document...")
    response = client.add_documents(
        documents=[doc],
        collection_name="test_collection",
        metadata={"test_run": True},
    )

    if response.success:
        print("   ✓ Document processed successfully")
        print(f"   - Chunks created: {response.total_chunks}")
        print(f"   - Total tokens: {response.processing_stats.total_tokens}")
        print(f"   - Processing time: {response.processing_stats.total_time_ms:.2f}ms\n")

        # Show chunk details
        print("4. Chunk details:")
        for i, chunk in enumerate(response.chunks, 1):
            print(f"\n   Chunk {i}:")
            print(f"   - ID: {chunk.chunk_id}")
            print(f"   - Tokens: {chunk.metadata.token_count}")
            print(f"   - Method: {chunk.metadata.chunking_method}")
            print(f"   - Content preview: {chunk.content[:80]}...")
    else:
        print("   ✗ Processing failed")
        print("   Errors:", response.errors)

    print("\n" + "-" * 60)
    print("Test completed!")


if __name__ == "__main__":
    test_basic_functionality()
