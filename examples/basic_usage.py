"""Basic usage example for insta_rag library."""

import os
from pathlib import Path

from dotenv import load_dotenv

from insta_rag import DocumentInput, RAGClient, RAGConfig

# Load environment variables
load_dotenv()


def main():
    """Demonstrate basic RAG operations."""

    # Step 1: Create configuration from environment variables
    print("=" * 60)
    print("Step 1: Initializing RAG Client")
    print("=" * 60)

    config = RAGConfig.from_env()
    client = RAGClient(config)

    print("✓ RAG Client initialized successfully")
    print(f"  - Embedding provider: {config.embedding.provider}")
    print(f"  - Embedding model: {config.embedding.model}")
    print(f"  - Vector DB: {config.vectordb.provider}")
    print()

    # Step 2: Prepare documents
    print("=" * 60)
    print("Step 2: Preparing Documents")
    print("=" * 60)

    # Example 1: From PDF file
    documents = []

    # Check if sample PDF exists
    sample_pdf = Path("examples/sample_document.pdf")
    if sample_pdf.exists():
        doc1 = DocumentInput.from_file(
            file_path=sample_pdf,
            metadata={
                "user_id": "user_123",
                "document_type": "business_document",
                "is_standalone": True,
            },
        )
        documents.append(doc1)
        print(f"✓ Added PDF document: {sample_pdf}")

    # Example 2: From text
    sample_text = """
    This is a sample document for testing the insta_rag library.

    The library provides a modular, plug-and-play RAG system that abstracts
    all RAG complexity into three primary operations: Input, Update, and Retrieve.

    Key features include:
    - Semantic chunking for better context preservation
    - Support for multiple embedding providers (OpenAI, Azure OpenAI)
    - Vector storage with Qdrant
    - Hybrid retrieval with HyDE and BM25
    - Reranking with Cohere

    The system is designed to be extensible, allowing you to add new chunking
    methods, embedding providers, or vector databases without breaking existing code.
    """

    doc2 = DocumentInput.from_text(
        text=sample_text,
        metadata={
            "user_id": "user_123",
            "document_type": "knowledge_base",
            "source_name": "sample_text",
        },
    )
    documents.append(doc2)
    print("✓ Added text document")
    print()

    # Step 3: Add documents to collection
    print("=" * 60)
    print("Step 3: Adding Documents to Knowledge Base")
    print("=" * 60)

    collection_name = "test_collection"

    response = client.add_documents(
        documents=documents,
        collection_name=collection_name,
        metadata={"project": "insta_rag_demo"},
    )

    print(f"\n✓ Documents processed successfully!")
    print(f"  - Documents processed: {response.documents_processed}")
    print(f"  - Total chunks created: {response.total_chunks}")
    print(f"  - Total tokens: {response.processing_stats.total_tokens}")
    print(f"  - Chunking time: {response.processing_stats.chunking_time_ms:.2f}ms")
    print(f"  - Embedding time: {response.processing_stats.embedding_time_ms:.2f}ms")
    print(f"  - Upload time: {response.processing_stats.upload_time_ms:.2f}ms")
    print(f"  - Total time: {response.processing_stats.total_time_ms:.2f}ms")

    if response.errors:
        print(f"\n⚠ Errors encountered:")
        for error in response.errors:
            print(f"  - {error}")

    # Display chunk information
    print(f"\nChunk Details:")
    for i, chunk in enumerate(response.chunks[:3]):  # Show first 3 chunks
        print(f"\nChunk {i + 1}:")
        print(f"  - ID: {chunk.chunk_id}")
        print(f"  - Token count: {chunk.metadata.token_count}")
        print(f"  - Method: {chunk.metadata.chunking_method}")
        print(f"  - Preview: {chunk.content[:100]}...")

    if len(response.chunks) > 3:
        print(f"\n  ... and {len(response.chunks) - 3} more chunks")

    # Step 4: Get collection info
    print("\n" + "=" * 60)
    print("Step 4: Collection Information")
    print("=" * 60)

    info = client.get_collection_info(collection_name)
    print(f"Collection: {info['name']}")
    print(f"  - Total vectors: {info['vectors_count']}")
    print(f"  - Status: {info['status']}")

    # List all collections
    print("\nAll collections:")
    collections = client.list_collections()
    for coll in collections:
        print(f"  - {coll}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
