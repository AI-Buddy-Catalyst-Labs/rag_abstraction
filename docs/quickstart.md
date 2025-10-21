# Quickstart Guide

This guide provides a hands-on walkthrough of the core features of the `insta_rag` library. You'll learn how to install the library, configure your environment, add documents, and perform advanced retrieval queries.

## 1. Installation

First, ensure you have Python 3.9+ and install the library. Using `uv` is recommended.

```bash
# Install the package in editable mode
uv pip install -e .
```

For more detailed installation instructions, see the [Installation Guide](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/blob/main/docs/installation.md).

## 2. Environment Setup

Create a `.env` file in your project root and add the necessary API keys and URLs. The library will automatically load these variables.

```env
# Required: Qdrant Vector Database
QDRANT_URL="https://your-qdrant-instance.cloud.qdrant.io"
QDRANT_API_KEY="your_qdrant_api_key"

# Required: OpenAI or Azure OpenAI for embeddings
AZURE_OPENAI_ENDPOINT="https://your-instance.openai.azure.com/"
AZURE_OPENAI_API_KEY="your_azure_key"
AZURE_EMBEDDING_DEPLOYMENT="text-embedding-3-large"

# Required for HyDE Query Generation
AZURE_LLM_DEPLOYMENT="gpt-4"

# Optional: Cohere for reranking
COHERE_API_KEY="your_cohere_api_key"
```

## 3. Initialize the RAG Client

The `RAGClient` is the main entry point to the library. It's configured via a `RAGConfig` object, which can be easily loaded from your environment variables.

```python
from insta_rag import RAGClient, RAGConfig

# Load configuration from .env file
config = RAGConfig.from_env()

# Initialize the client
client = RAGClient(config)

print("✓ RAG Client initialized successfully")
```

## 4. Add Documents to a Collection

You can add documents from files (PDF, TXT) or raw text. The library handles text extraction, semantic chunking, embedding, and storage in a single command.

```python
from insta_rag import DocumentInput

# Prepare documents from different sources
documents = (
    [
        # From a PDF file
        DocumentInput.from_file(
            "path/to/your/document.pdf",
            metadata={"user_id": "user_123", "document_type": "report"},
        ),
        # From a raw text string
        DocumentInput.from_text(
            "This is the content of a short document about insta_rag.",
            metadata={"source": "manual", "author": "Gemini"},
        ),
    ],
)

# Process and store the documents in a collection
response = client.add_documents(
    documents=documents,
    collection_name="my_knowledge_base",
    metadata={"project": "quickstart_demo"},  # Global metadata for this batch
)

# Review the results
if response.success:
    print(f"✓ Processed {response.documents_processed} documents")
    print(f"✓ Created {response.total_chunks} chunks")
    print(f"✓ Total time: {response.processing_stats.total_time_ms:.2f}ms")
else:
    print(f"✗ Errors: {response.errors}")
```

For a detailed explanation of the [Document Management Guide](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/blob/main/docs/guides/document-management.md).

## 5. Perform a Retrieval Query

The `retrieve()` method performs a hybrid search query, combining semantic search, keyword search, and query expansion to find the most relevant results.

By default, **HyDE query generation** and **BM25 keyword search** are enabled for the highest quality results.

```python
# Perform a retrieval query
response = client.retrieve(
    query="What is semantic chunking?", collection_name="my_knowledge_base", top_k=5
)

# Print the results
if response.success:
    print(f"✓ Retrieved {len(response.chunks)} chunks")
    print(f"\nGenerated Queries: {response.queries_generated}")

    for i, chunk in enumerate(response.chunks):
        print(f"\n--- Result {i + 1} (Score: {chunk.relevance_score:.4f}) ---")
        print(f"Source: {chunk.metadata.source}")
        print(chunk.content)
```

### Understanding the Retrieval Response

- `response.chunks`: A list of the top `k` most relevant document chunks.
- `response.queries_generated`: Shows the original query, the optimized standard query, and the hypothetical answer (HyDE) used for searching.
- `response.retrieval_stats`: Provides a detailed performance breakdown, including timings for each stage and the number of chunks found by each method.

## 6. Controlling Retrieval Features

You can easily enable or disable advanced retrieval features to balance speed and quality.

```python
# Fast mode: Vector search only (like a traditional RAG)
fast_response = client.retrieve(
    query="your question",
    collection_name="my_knowledge_base",
    enable_hyde=False,
    enable_keyword_search=False,
    enable_reranking=False,  # Assuming reranking is a future feature
)

# High-quality mode: Vector search + HyDE (no keyword search)
quality_response = client.retrieve(
    query="your question",
    collection_name="my_knowledge_base",
    enable_hyde=True,
    enable_keyword_search=False,
)
```

To learn more about the advanced retrieval pipeline, see the [Retrieval Guide](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/blob/main/docs/guides/retrieval.md).

## 7. Document Management

The library also supports updating and deleting documents.

```python
# Example: Delete all documents for a specific user
delete_response = client.update_documents(
    collection_name="my_knowledge_base",
    update_strategy="delete",
    filters={"user_id": "user_123"},
)

print(f"Deleted {delete_response.chunks_deleted} chunks.")
```

For more, see the [Document Management Guide](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/blob/main/docs/guides/document-management.md).
