"""Test MongoDB integration."""

import sys
from pathlib import Path

# Add parent directory to path to import insta_rag
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from insta_rag import DocumentInput, RAGClient, RAGConfig

load_dotenv(Path(__file__).parent.parent / ".env")


def test_mongodb_integration():
    """Test MongoDB integration with insta_rag."""

    print("=" * 60)
    print("Testing MongoDB Integration")
    print("=" * 60)
    print()

    # Initialize client
    print("1. Initializing RAG client with MongoDB...")
    config = RAGConfig.from_env()

    if not config.mongodb:
        print("   ✗ MongoDB not configured in .env")
        print("   Please set MONGO_CONNECTION_STRING in .env file")
        return

    client = RAGClient(config)
    print(f"   ✓ Client initialized")
    print(f"   ✓ MongoDB enabled: {config.mongodb.enabled}")
    print(f"   ✓ Database: {config.mongodb.database_name}")
    print()

    # Create test document
    print("2. Creating test document...")
    test_text = """
    MongoDB Integration Test Document

    This document tests the MongoDB integration with insta_rag.
    The content will be stored in MongoDB, and only the reference
    will be stored in Qdrant's metadata.

    Benefits of this approach:
    - Reduced Qdrant storage size
    - Centralized content management
    - Easy content updates without re-embedding
    - Better separation of concerns
    """

    doc = DocumentInput.from_text(
        text=test_text,
        metadata={
            "source": "mongodb_test",
            "test_type": "integration",
        },
    )
    print("   ✓ Document created")
    print()

    # Process document
    print("3. Processing document (content will be stored in MongoDB)...")
    response = client.add_documents(
        documents=[doc],
        collection_name="mongodb_test_collection",
        metadata={"test_run": True},
    )

    if response.success:
        print("   ✓ Document processed successfully")
        print(f"   - Chunks created: {response.total_chunks}")
        print(f"   - Processing time: {response.processing_stats.total_time_ms:.2f}ms")
        print()

        # Check MongoDB storage
        print("4. Verifying MongoDB storage...")
        if client.mongodb:
            for chunk in response.chunks:
                # Try to retrieve from MongoDB
                mongo_doc = client.mongodb.get_chunk_content(chunk.chunk_id)
                if mongo_doc:
                    print(f"   ✓ Chunk {chunk.chunk_id[:8]}... found in MongoDB")
                    print(f"     - MongoDB ID: {mongo_doc['_id']}")
                    print(f"     - Content length: {len(mongo_doc['content'])} chars")
                else:
                    print(f"   ✗ Chunk {chunk.chunk_id[:8]}... NOT found in MongoDB")

            print()

            # Get collection stats
            print("5. MongoDB Collection Statistics...")
            stats = client.mongodb.get_collection_stats("mongodb_test_collection")
            print(f"   - Total chunks: {stats['total_chunks']}")
            print(f"   - Total documents: {stats['total_documents']}")
            print(f"   - Total content size: {stats['total_content_size_bytes']} bytes")
        else:
            print("   ✗ MongoDB client not available")

    else:
        print("   ✗ Processing failed")
        print("   Errors:", response.errors)

    print()
    print("=" * 60)
    print("MongoDB Integration Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_mongodb_integration()
