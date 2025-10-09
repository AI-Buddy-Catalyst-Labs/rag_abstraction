"""Test script to verify Qdrant connection and credentials."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_qdrant_connection():
    """Test Qdrant connection and basic operations."""
    print("=" * 60)
    print("QDRANT CONNECTION TEST")
    print("=" * 60)

    # Get credentials from environment
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    print(f"\n1. Environment Variables:")
    print(f"   QDRANT_URL: {qdrant_url}")
    print(f"   QDRANT_API_KEY: {'*' * 20 if qdrant_api_key else 'NOT SET'}")

    if not qdrant_url or not qdrant_api_key:
        print("\n❌ ERROR: Qdrant credentials not found in .env file")
        return False

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        print("\n2. Initializing Qdrant Client...")
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
            prefer_grpc=False,
        )
        print("   ✓ Client initialized successfully")

        print("\n3. Testing connection...")
        collections = client.get_collections()
        print(f"   ✓ Connection successful!")
        print(f"   Found {len(collections.collections)} existing collections:")
        for col in collections.collections:
            print(f"     - {col.name}")

        print("\n4. Testing collection creation...")
        test_collection_name = "test_connection_collection"

        # Delete if exists
        if any(c.name == test_collection_name for c in collections.collections):
            print(f"   Deleting existing test collection '{test_collection_name}'...")
            client.delete_collection(test_collection_name)

        # Create test collection
        print(f"   Creating test collection '{test_collection_name}'...")
        client.create_collection(
            collection_name=test_collection_name,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        print("   ✓ Collection created successfully")

        print("\n5. Getting collection info...")
        info = client.get_collection(collection_name=test_collection_name)
        print(f"   ✓ Collection info retrieved:")
        print(f"     - Name: {test_collection_name}")
        print(f"     - Vector size: 3072")
        print(f"     - Distance metric: COSINE")
        print(f"     - Points count: {info.points_count}")
        print(f"     - Status: {info.status}")

        print("\n6. Testing vector upsert...")
        from qdrant_client.models import PointStruct
        import uuid

        # Create a test point
        test_point = PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.1] * 3072,  # Dummy 3072-dimensional vector
            payload={
                "content": "This is a test chunk",
                "chunk_id": "test_chunk_1",
                "document_id": "test_doc_1"
            }
        )

        client.upsert(
            collection_name=test_collection_name,
            points=[test_point]
        )
        print("   ✓ Vector upserted successfully")

        print("\n7. Testing vector search...")
        search_results = client.query_points(
            collection_name=test_collection_name,
            query=[0.1] * 3072,
            limit=1,
            with_payload=True,
        )
        print(f"   ✓ Search completed successfully")
        print(f"   Found {len(search_results.points)} result(s)")
        if search_results.points:
            print(f"     - Score: {search_results.points[0].score}")
            print(f"     - Content: {search_results.points[0].payload.get('content', '')}")

        print("\n8. Cleaning up...")
        client.delete_collection(test_collection_name)
        print(f"   ✓ Test collection '{test_collection_name}' deleted")

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour Qdrant credentials are working correctly.")
        print("The insta_rag library is ready to use Qdrant.\n")

        return True

    except ImportError as e:
        print(f"\n❌ ERROR: Missing required package")
        print(f"   {str(e)}")
        print("\n   Please install: pip install qdrant-client")
        return False

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPossible issues:")
        print("  1. Check if the QDRANT_URL is correct")
        print("  2. Check if the QDRANT_API_KEY is valid")
        print("  3. Check if your Qdrant cluster is running")
        print("  4. Check your network connection")
        return False


if __name__ == "__main__":
    success = test_qdrant_connection()
    exit(0 if success else 1)
