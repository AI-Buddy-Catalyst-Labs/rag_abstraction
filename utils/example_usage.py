"""Example usage of insta_rag library with Qdrant."""

from dotenv import load_dotenv
from src.insta_rag.core.config import RAGConfig
from src.insta_rag.core.client import RAGClient

# Load environment variables from .env
load_dotenv()


def main():
    """Demonstrate basic usage of insta_rag."""

    print("Initializing RAG Client...")

    # Create configuration from environment variables
    config = RAGConfig.from_env()

    # Initialize RAG client
    client = RAGClient(config)

    print("✓ RAG Client initialized successfully!")
    print("\nConfiguration:")
    print(f"  - Vector DB: {config.vectordb.provider}")
    print(f"  - Vector DB URL: {config.vectordb.url}")
    print(f"  - Embedding Provider: {config.embedding.provider}")
    print(f"  - Embedding Model: {config.embedding.model}")
    print(f"  - Embedding Dimensions: {config.embedding.dimensions}")
    print(f"  - Chunking Method: {config.chunking.method}")
    print(f"  - Reranking Enabled: {config.reranking.enabled}")

    # Example: Create a collection
    collection_name = "my_documents"

    print("\n\nExample operations:")
    print("=" * 60)

    # Check if collection exists
    print(f"\n1. Checking if collection '{collection_name}' exists...")
    exists = client.vectordb.collection_exists(collection_name)
    print(f"   Collection exists: {exists}")

    if not exists:
        print(f"\n2. Creating collection '{collection_name}'...")
        client.vectordb.create_collection(
            collection_name=collection_name,
            vector_size=config.embedding.dimensions,
            distance_metric="cosine",
        )
        print("   ✓ Collection created successfully")

    # Get collection info
    print("\n3. Getting collection info...")
    info = client.vectordb.get_collection_info(collection_name)
    print(f"   Collection: {info['name']}")
    print(f"   Vectors count: {info['vectors_count']}")
    print(f"   Status: {info['status']}")

    print("\n" + "=" * 60)
    print("✅ All operations completed successfully!")
    print("\nYour insta_rag setup is working correctly with Qdrant.")
    print("\nNext steps:")
    print("  1. Add PDF documents using client.add_document()")
    print("  2. Search documents using client.search()")
    print("  3. Explore the examples/ directory for more use cases")


if __name__ == "__main__":
    main()
