import os
import pytest
from dotenv import load_dotenv
from insta_rag import DocumentInput, RAGClient, RAGConfig

# Load environment variables from .env file
load_dotenv()

# Check for required environment variables
skip_test = not all(
    os.getenv(key) for key in ["QDRANT_URL", "QDRANT_API_KEY", "OPENAI_API_KEY"]
)


@pytest.mark.skipif(
    skip_test,
    reason="Skipping integration test: Required environment variables are not set.",
)
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
    collection_name = "integration_test_collection"
    print(f"3. Processing document into collection: {collection_name}...")
    response = client.add_documents(
        documents=[doc],
        collection_name=collection_name,
        metadata={"test_run": True},
    )

    assert response.success, f"Processing failed with errors: {response.errors}"
    print("   ✓ Document processed successfully")
    print(f"   - Chunks created: {response.total_chunks}")
    print(f"   - Total tokens: {response.processing_stats.total_tokens}")
    print(f"   - Processing time: {response.processing_stats.total_time_ms:.2f}ms\n")

    # Retrieve document
    print("4. Retrieving document...")
    retrieve_response = client.retrieve(
        query="What is RAG?",
        collection_name=collection_name,
    )

    assert retrieve_response.success, (
        f"Retrieval failed with errors: {retrieve_response.errors}"
    )
    assert len(retrieve_response.chunks) > 0, "No chunks were retrieved."
    print("   ✓ Document retrieved successfully")
    print(f"   - Retrieved {len(retrieve_response.chunks)} chunks.")

    # Clean up
    print("\n5. Cleaning up...")
    client.vectordb.client.delete_collection(collection_name=collection_name)
    print(f"   ✓ Collection '{collection_name}' deleted.")

    print("\n" + "-" * 60)
    print("Test completed!")
