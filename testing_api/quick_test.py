#!/usr/bin/env python
"""Quick diagnostic test for insta_rag setup."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def test_imports():
    """Test if all imports work."""
    print("=" * 60)
    print("Test 1: Checking Imports")
    print("=" * 60)
    try:
        from insta_rag import DocumentInput, RAGClient, RAGConfig

        print("✓ insta_rag imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("Test 2: Configuration Loading")
    print("=" * 60)
    try:
        from insta_rag import RAGConfig

        config = RAGConfig.from_env()
        print("✓ Configuration loaded")
        print(f"  - Qdrant URL: {config.vectordb.url[:50]}...")
        print(f"  - Embedding provider: {config.embedding.provider}")
        print(f"  - MongoDB enabled: {config.mongodb is not None}")
        print(f"  - gRPC disabled: {not config.vectordb.prefer_grpc}")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False


def test_qdrant_connection():
    """Test Qdrant connection."""
    print("\n" + "=" * 60)
    print("Test 3: Qdrant Connection")
    print("=" * 60)
    try:
        from qdrant_client import QdrantClient

        import os

        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=False,  # Use HTTP
            timeout=60,
        )

        collections = client.get_collections()
        print("✓ Qdrant connection successful")
        print(f"  - Collections count: {len(collections.collections)}")
        print(f"  - Using HTTP REST API (gRPC disabled)")
        return True
    except Exception as e:
        print(f"✗ Qdrant connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if QDRANT_URL is accessible")
        print("  2. Verify QDRANT_API_KEY is correct")
        print("  3. Ensure network/firewall allows HTTPS")
        return False


def test_mongodb_connection():
    """Test MongoDB connection."""
    print("\n" + "=" * 60)
    print("Test 4: MongoDB Connection")
    print("=" * 60)
    try:
        import os

        from pymongo import MongoClient

        conn_str = os.getenv("MONGO_CONNECTION_STRING")
        if not conn_str:
            print("⚠ MongoDB not configured (optional)")
            return True

        client = MongoClient(conn_str, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        dbs = client.list_database_names()

        print("✓ MongoDB connection successful")
        print(f"  - Databases: {len(dbs)}")
        return True
    except Exception as e:
        print(f"✗ MongoDB connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify MongoDB is running")
        print("  2. Check connection string format")
        print("  3. Note: Your connection string uses port 5432 (PostgreSQL port)")
        print("     MongoDB typically uses port 27017")
        return False


def test_client_initialization():
    """Test RAG client initialization."""
    print("\n" + "=" * 60)
    print("Test 5: RAG Client Initialization")
    print("=" * 60)
    try:
        from insta_rag import RAGClient, RAGConfig

        config = RAGConfig.from_env()
        client = RAGClient(config)

        print("✓ RAG Client initialized successfully")
        print(f"  - Embedder: {type(client.embedder).__name__}")
        print(f"  - Vector DB: {type(client.vectordb).__name__}")
        print(f"  - MongoDB: {type(client.mongodb).__name__ if client.mongodb else 'Not configured'}")
        return True
    except Exception as e:
        print(f"✗ Client initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_simple_operation():
    """Test a simple document operation."""
    print("\n" + "=" * 60)
    print("Test 6: Simple Document Processing")
    print("=" * 60)
    try:
        from insta_rag import DocumentInput, RAGClient, RAGConfig

        config = RAGConfig.from_env()
        client = RAGClient(config)

        # Create simple test document
        doc = DocumentInput.from_text(
            text="This is a quick test document to verify the system works.",
            metadata={"test": "quick_diagnostic"},
        )

        # Process it
        response = client.add_documents(
            documents=[doc], collection_name="quick_diagnostic_test"
        )

        if response.success:
            print("✓ Document processing successful")
            print(f"  - Chunks created: {response.total_chunks}")
            print(f"  - Processing time: {response.processing_stats.total_time_ms:.2f}ms")
            return True
        else:
            print(f"✗ Document processing failed: {response.errors}")
            return False

    except Exception as e:
        print(f"✗ Operation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "insta_rag Quick Diagnostic Test" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Qdrant Connection", test_qdrant_connection()))
    results.append(("MongoDB Connection", test_mongodb_connection()))
    results.append(("Client Initialization", test_client_initialization()))
    results.append(("Simple Operation", test_simple_operation()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} {name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Start the testing API: cd testing_api && ./run.sh")
        print("  2. Open Swagger UI: http://localhost:8000/docs")
        print("  3. Run integration tests: ./test_requests.sh")
    else:
        print(
            "\n⚠ Some tests failed. Please review the errors above and check TROUBLESHOOTING.md"
        )
        print("\nCommon fixes:")
        print("  - Ensure all API keys in .env are correct")
        print("  - Check network connectivity to Qdrant and MongoDB")
        print("  - Verify gRPC is disabled (prefer_grpc=False)")


if __name__ == "__main__":
    main()
