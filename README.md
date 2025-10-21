# insta_rag

`insta_rag` is a modular, plug-and-play Python library for building advanced Retrieval-Augmented Generation (RAG) pipelines. It abstracts the complexity of document processing, embedding, and hybrid retrieval into a simple, configuration-driven client.

## Core Features

- **Semantic Chunking**: Splits documents at natural topic boundaries to preserve context.
- **Hybrid Retrieval**: Combines semantic vector search with BM25 keyword search for the best of both worlds.
- **Query Transformation (HyDE)**: Uses an LLM to generate hypothetical answers, improving retrieval relevance.
- **Reranking**: Integrates with state-of-the-art rerankers like Cohere to intelligently re-order results.
- **Pluggable Architecture**: Easily extend the library by adding new chunkers, embedders, or vector databases.
- **Hybrid Storage**: Optional integration with MongoDB for cost-effective content storage, keeping Qdrant lean for vector search.

## Quick Start

### 1. Installation

```bash
# Recommended: using uv
uv pip install insta-rag

# Or with pip
pip install insta-rag
```

### 2. Basic Usage

```python
from insta_rag import RAGClient, RAGConfig, DocumentInput

# Load configuration from environment variables (.env file)
config = RAGConfig.from_env()
client = RAGClient(config)

# 1. Add documents to a collection
documents = [DocumentInput.from_text("Your first document content.")]
client.add_documents(documents, collection_name="my_docs")

# 2. Retrieve relevant information
response = client.retrieve(
    query="What is this document about?", collection_name="my_docs"
)

# Print the most relevant chunk
if response.chunks:
    print(response.chunks[0].content)
```

## Documentation

For detailed guides on installation, configuration, and advanced features, please see the **[Full Documentation](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki)**.

Key sections include:

- **[Installation Guide](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki/installation)**
- **[Quickstart Guide](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki/quickstart)**
- **Guides**
  - [Document Management](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki/guides/document-management)
  - [Advanced Retrieval](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki/guides/retrieval)
  - [Storage Backends](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki/guides/storage-backends)

## License

This project is licensed under the [MIT License](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/blob/main/LICENSE).
