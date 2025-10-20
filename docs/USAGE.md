# insta_rag Usage Guide

This guide shows you how to use the insta_rag library for document processing and retrieval.

## Installation

```bash
# Install the library (development mode)
pip install -e .

# Install required dependencies
pip install openai qdrant-client pdfplumber PyPDF2 tiktoken numpy python-dotenv
```

## Quick Start

### 1. Set Up Environment Variables

Create a `.env` file with your API keys:

```env
# Qdrant Vector Database
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# Option 1: Azure OpenAI (recommended for enterprise)
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Option 2: Standard OpenAI
OPENAI_API_KEY=your_openai_api_key

# Cohere (for reranking - optional)
COHERE_API_KEY=your_cohere_api_key
```

### 2. Basic Usage

```python
from insta_rag import RAGClient, RAGConfig, DocumentInput

# Initialize client from environment variables
config = RAGConfig.from_env()
client = RAGClient(config)

# Add documents from different sources
documents = [
    # From PDF file
    DocumentInput.from_file(
        "path/to/document.pdf",
        metadata={"user_id": "user_123", "document_type": "business_doc"},
    ),
    # From text
    DocumentInput.from_text(
        "Your text content here...", metadata={"source": "web_scrape"}
    ),
]

# Process and store documents
response = client.add_documents(
    documents=documents,
    collection_name="my_knowledge_base",
    metadata={"project": "my_project"},
)

print(f"Processed {response.total_chunks} chunks")
print(f"Total time: {response.processing_stats.total_time_ms}ms")
```

## Detailed Usage

### Configuration

#### Using Environment Variables (Recommended)

```python
config = RAGConfig.from_env()
```

#### Custom Configuration

```python
from insta_rag.core.config import (
    RAGConfig,
    VectorDBConfig,
    EmbeddingConfig,
    ChunkingConfig,
    PDFConfig,
)

config = RAGConfig(
    vectordb=VectorDBConfig(
        url="https://your-qdrant-instance.cloud.qdrant.io", api_key="your_api_key"
    ),
    embedding=EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-large",
        api_key="your_openai_key",
        dimensions=3072,
    ),
    chunking=ChunkingConfig(
        method="semantic", max_chunk_size=1000, overlap_percentage=0.2
    ),
    pdf=PDFConfig(parser="pdfplumber", validate_text=True),
)
```

### Document Input

#### From PDF Files

```python
doc = DocumentInput.from_file(
    file_path="document.pdf",
    metadata={
        "user_id": "user_123",
        "document_type": "contract",
        "department": "legal",
    },
)
```

#### From Text Files

```python
doc = DocumentInput.from_file(
    file_path="document.txt", metadata={"source": "knowledge_base"}
)
```

#### From Raw Text

```python
doc = DocumentInput.from_text(
    text="Your document content...",
    metadata={"source": "web_scrape", "url": "https://example.com"},
)
```

### Processing Documents

```python
response = client.add_documents(
    documents=[doc1, doc2, doc3],
    collection_name="my_collection",
    metadata={"batch_id": "batch_001"},
    batch_size=100,  # Embedding batch size
    validate_chunks=True,  # Enable quality validation
)

# Check results
if response.success:
    print(f"✓ Processed {response.documents_processed} documents")
    print(f"✓ Created {response.total_chunks} chunks")

    # Access chunks
    for chunk in response.chunks:
        print(f"Chunk {chunk.chunk_id}: {chunk.metadata.token_count} tokens")
else:
    print("Errors:", response.errors)
```

### Collection Management

```python
# List all collections
collections = client.list_collections()
print("Available collections:", collections)

# Get collection info
info = client.get_collection_info("my_collection")
print(f"Vectors: {info['vectors_count']}")
print(f"Status: {info['status']}")
```

## Advanced Configuration

### Chunking Strategies

```python
from insta_rag.core.config import ChunkingConfig

# Semantic chunking (default - best quality)
chunking = ChunkingConfig(
    method="semantic",
    max_chunk_size=1000,
    overlap_percentage=0.2,
    semantic_threshold_percentile=95,
)

# For faster processing with less accuracy
chunking = ChunkingConfig(
    method="recursive", max_chunk_size=800, overlap_percentage=0.15
)
```

### PDF Processing

```python
from insta_rag.core.config import PDFConfig

pdf_config = PDFConfig(
    parser="pdfplumber",  # or "pypdf2"
    extract_images=False,
    extract_tables=False,
    validate_text=True,
)
```

### Embedding Providers

#### Azure OpenAI

```python
from insta_rag.core.config import EmbeddingConfig

embedding = EmbeddingConfig(
    provider="azure_openai",
    model="text-embedding-3-large",
    api_key="your_key",
    api_base="https://your-instance.openai.azure.com/",
    api_version="2024-02-01",
    deployment_name="text-embedding-3-large",
    dimensions=3072,
)
```

#### Standard OpenAI

```python
embedding = EmbeddingConfig(
    provider="openai",
    model="text-embedding-3-large",
    api_key="your_key",
    dimensions=3072,
)
```

## Metadata Management

Metadata is crucial for filtering and organization:

### Document-Level Metadata

```python
doc = DocumentInput.from_file(
    "document.pdf",
    metadata={
        # User identification
        "user_id": "user_123",
        # Document categorization
        "document_type": "business_document",
        "department": "sales",
        # Lifecycle management
        "is_standalone": True,
        # Template association
        "template_id": "template_456",
        # Custom fields
        "status": "active",
        "tags": ["contract", "2024"],
        "priority": "high",
    },
)
```

### Global Metadata

Applied to all chunks in a batch:

```python
response = client.add_documents(
    documents=documents,
    collection_name="my_collection",
    metadata={
        "project": "project_name",
        "batch_id": "batch_001",
        "uploaded_by": "user_123",
        "timestamp": "2024-01-15",
    },
)
```

## Error Handling

```python
from insta_rag.exceptions import (
    PDFEncryptedError,
    PDFCorruptedError,
    PDFEmptyError,
    ChunkingError,
    EmbeddingError,
    VectorDBError,
)

try:
    response = client.add_documents(documents, "collection")
except PDFEncryptedError:
    print("PDF is password-protected")
except PDFCorruptedError:
    print("PDF file is corrupted")
except PDFEmptyError:
    print("No text could be extracted from PDF")
except EmbeddingError as e:
    print(f"Embedding generation failed: {e}")
except VectorDBError as e:
    print(f"Vector database error: {e}")
```

## Performance Optimization

### Batch Processing

```python
# Process large batches efficiently
response = client.add_documents(
    documents=large_document_list,
    collection_name="large_collection",
    batch_size=100,  # Adjust based on API limits
)
```

### Monitoring

```python
response = client.add_documents(documents, "collection")

stats = response.processing_stats
print(f"Chunking: {stats.chunking_time_ms}ms")
print(f"Embedding: {stats.embedding_time_ms}ms")
print(f"Upload: {stats.upload_time_ms}ms")
print(f"Total: {stats.total_time_ms}ms")
print(f"Tokens processed: {stats.total_tokens}")
```

## Complete Example

See `examples/basic_usage.py` for a complete working example.

```bash
# Run the example
python examples/basic_usage.py
```

## Next Steps

- Implement `update_documents()` for document updates and deletions
- Implement `retrieve()` for hybrid search with reranking
- Add support for more document types
- Implement custom chunking strategies

## Troubleshooting

### Common Issues

1. **"Collection not found"**: Create the collection first or let `add_documents()` auto-create it
1. **"Embedding API error"**: Check API keys and rate limits
1. **"PDF extraction failed"**: Try a different parser or check PDF quality
1. **"Token count exceeded"**: Reduce `max_chunk_size` in configuration

### Debug Mode

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

## Support

For issues and questions:

- GitHub Issues: https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/issues
- Documentation: See README.md
