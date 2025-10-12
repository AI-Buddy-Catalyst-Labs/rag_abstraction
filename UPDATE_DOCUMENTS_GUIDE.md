# Knowledge Base Update Operations - Complete Guide

## Overview

The `update_documents()` method provides comprehensive document management capabilities for your RAG knowledge base. This implementation adds flexible CRUD operations to the insta_rag library.

## Features Implemented

### 1. **Four Update Strategies**

#### REPLACE Strategy
- Delete existing documents and add new ones
- Use Case: User uploads a new version of an existing document
- Supports filtering by metadata or document IDs

```python
response = client.update_documents(
    collection_name="my_collection",
    update_strategy="replace",
    filters={"user_id": "123", "document_type": "report"},
    new_documents=[new_doc1, new_doc2],
)
```

#### APPEND Strategy
- Add new documents without deleting any existing ones
- Use Case: Adding new documents to the knowledge base
- Simple addition operation

```python
response = client.update_documents(
    collection_name="my_collection",
    update_strategy="append",
    new_documents=[new_doc1, new_doc2],
    metadata_updates={"category": "technical_docs"}
)
```

#### DELETE Strategy
- Remove documents matching specified criteria
- Use Case: Remove outdated or irrelevant documents
- Supports both filter-based and ID-based deletion

```python
# Delete by filters
response = client.update_documents(
    collection_name="my_collection",
    update_strategy="delete",
    filters={"status": "archived"}
)

# Delete by document IDs
response = client.update_documents(
    collection_name="my_collection",
    update_strategy="delete",
    document_ids=["doc-123", "doc-456"]
)
```

#### UPSERT Strategy
- Update if document exists, insert if it doesn't
- Use Case: Synchronizing external data sources
- Automatically detects existence by document_id

```python
response = client.update_documents(
    collection_name="my_collection",
    update_strategy="upsert",
    new_documents=[doc_with_id_1, doc_with_id_2],
)
```

### 2. **Metadata-Only Updates**

Update metadata fields without reprocessing content:

```python
response = client.update_documents(
    collection_name="my_collection",
    update_strategy="delete",  # Placeholder
    filters={"document_type": "report"},
    metadata_updates={"status": "reviewed", "updated_at": "2025-01-01"},
    reprocess_chunks=False,  # Key parameter!
)
```

## Method Signature

```python
def update_documents(
    self,
    collection_name: str,
    update_strategy: str,  # "replace", "append", "delete", "upsert"
    filters: Optional[Dict[str, Any]] = None,
    document_ids: Optional[List[str]] = None,
    new_documents: Optional[List[DocumentInput]] = None,
    metadata_updates: Optional[Dict[str, Any]] = None,
    reprocess_chunks: bool = True,
) -> UpdateDocumentsResponse
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `collection_name` | `str` | Target Qdrant collection |
| `update_strategy` | `str` | Operation type: "replace", "append", "delete", "upsert" |
| `filters` | `Dict[str, Any]` | Metadata-based selection criteria |
| `document_ids` | `List[str]` | Specific document IDs to target |
| `new_documents` | `List[DocumentInput]` | Replacement or additional documents |
| `metadata_updates` | `Dict[str, Any]` | Metadata fields to update |
| `reprocess_chunks` | `bool` | If True, regenerate chunks and embeddings |

### Response Structure

```python
@dataclass
class UpdateDocumentsResponse:
    success: bool
    strategy_used: str
    documents_affected: int
    chunks_deleted: int
    chunks_added: int
    chunks_updated: int
    updated_document_ids: List[str]
    errors: List[str]
```

## Implementation Details

### New Helper Methods in VectorDB Layer

1. **`get_document_ids()`** - Get unique document IDs matching filters
2. **`count_chunks()`** - Count chunks matching criteria
3. **`get_chunk_ids_by_documents()`** - Get all chunk IDs for specific documents
4. **`update_metadata()`** - Update metadata without reprocessing content

### MongoDB Integration

- Automatically handles content deletion from MongoDB when enabled
- Maintains consistency between Qdrant and MongoDB
- New methods: `delete_chunks_by_ids()`, `delete_chunks_by_document_ids()`

### Error Handling

- `ValidationError`: Invalid parameters or strategy
- `CollectionNotFoundError`: Target collection doesn't exist
- `NoDocumentsFoundError`: No documents match filters/IDs
- `VectorDBError`: Qdrant operation failures

## Testing

### 1. **Run the Test Script**

```bash
# Make sure you're in the project directory
cd /home/macorov/Documents/GitHub/insta_rag

# Activate virtual environment
source venv/bin/activate

# Run the comprehensive test script
python test_update_documents.py
```

This will test all 4 strategies plus metadata updates.

### 2. **Use the Testing API**

Start the testing API server:

```bash
cd testing_api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Available endpoints:

- **Generic Update**: `POST /api/v1/test/documents/update`
- **Delete**: `POST /api/v1/test/documents/update/delete`
- **Append**: `POST /api/v1/test/documents/update/append`
- **Replace**: `POST /api/v1/test/documents/update/replace`
- **Upsert**: `POST /api/v1/test/documents/update/upsert`
- **Metadata**: `POST /api/v1/test/documents/update/metadata`

### 3. **Example API Request**

```bash
curl -X POST "http://localhost:8000/api/v1/test/documents/update" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "test_collection",
    "update_strategy": "replace",
    "filters": {"user_id": "123"},
    "new_documents_text": ["Updated content for the document."],
    "metadata_updates": {"status": "updated"}
  }'
```

## Usage Examples

### Example 1: Replace Documents by Filter

```python
from insta_rag import RAGClient, RAGConfig, DocumentInput

# Initialize client
config = RAGConfig.from_env()
client = RAGClient(config)

# Create replacement documents
new_docs = [
    DocumentInput.from_text(
        text="Updated version of the document with new information.",
        metadata={"version": 2, "updated_by": "admin"}
    )
]

# Replace all documents matching the filter
response = client.update_documents(
    collection_name="knowledge_base",
    update_strategy="replace",
    filters={"template_id": "report_template_v1"},
    new_documents=new_docs,
)

print(f"Replaced {response.documents_affected} documents")
print(f"Deleted {response.chunks_deleted} chunks")
print(f"Added {response.chunks_added} chunks")
```

### Example 2: Bulk Delete by Metadata

```python
# Delete all archived documents
response = client.update_documents(
    collection_name="knowledge_base",
    update_strategy="delete",
    filters={"status": "archived", "age_days": ">90"},
)

print(f"Deleted {response.chunks_deleted} chunks")
print(f"Affected {response.documents_affected} documents")
```

### Example 3: Upsert with Document IDs

```python
# Prepare documents with explicit IDs for upsert
docs = [
    DocumentInput.from_text(
        text="Content for document 1",
        metadata={"document_id": "user-123-profile", "type": "profile"}
    ),
    DocumentInput.from_text(
        text="Content for document 2",
        metadata={"document_id": "user-123-settings", "type": "settings"}
    ),
]

# Upsert - will update if exists, insert if not
response = client.update_documents(
    collection_name="user_data",
    update_strategy="upsert",
    new_documents=docs,
)

print(f"Updated: {response.chunks_updated} chunks")
print(f"Inserted: {response.chunks_added} chunks")
```

### Example 4: Metadata-Only Update

```python
# Update metadata without reprocessing content
response = client.update_documents(
    collection_name="knowledge_base",
    update_strategy="delete",  # Placeholder
    filters={"document_type": "manual"},
    metadata_updates={
        "status": "reviewed",
        "reviewed_by": "john.doe",
        "review_date": "2025-01-15"
    },
    reprocess_chunks=False,  # Don't regenerate embeddings
)

print(f"Updated metadata for {response.chunks_updated} chunks")
```

## Performance Characteristics

### Operation Complexity

| Strategy | Time Complexity | Notes |
|----------|----------------|-------|
| APPEND | O(n) | n = new documents |
| DELETE | O(m) | m = chunks to delete |
| REPLACE | O(m + n) | m = delete, n = add |
| UPSERT | O(k + m + n) | k = existence checks |

### Best Practices

1. **Use Filters Wisely**: Add metadata fields to enable efficient filtering
2. **Batch Operations**: Group related updates into single calls
3. **Metadata-Only Updates**: Use when only metadata changes (faster)
4. **Document IDs**: Provide explicit IDs for upsert operations
5. **Monitor Performance**: Check `chunks_deleted/added/updated` in response

## Error Handling Example

```python
from insta_rag.exceptions import (
    CollectionNotFoundError,
    NoDocumentsFoundError,
    ValidationError
)

try:
    response = client.update_documents(
        collection_name="my_collection",
        update_strategy="replace",
        filters={"user_id": "123"},
        new_documents=new_docs,
    )

    if response.success:
        print(f"✓ Update successful!")
    else:
        print(f"✗ Update failed: {response.errors}")

except CollectionNotFoundError as e:
    print(f"Collection doesn't exist: {e}")

except NoDocumentsFoundError as e:
    print(f"No matching documents: {e}")

except ValidationError as e:
    print(f"Invalid parameters: {e}")
```

## Files Modified/Created

### Core Library Files

1. **`src/insta_rag/vectordb/base.py`** - Added abstract helper methods
2. **`src/insta_rag/vectordb/qdrant.py`** - Implemented Qdrant helper methods
3. **`src/insta_rag/mongodb_client.py`** - Added batch deletion methods
4. **`src/insta_rag/core/client.py`** - Implemented main `update_documents()` method
5. **`src/insta_rag/models/response.py`** - Already had `UpdateDocumentsResponse` model

### Testing Files

1. **`testing_api/main.py`** - Added 6 test endpoints for update operations
2. **`test_update_documents.py`** - Comprehensive test script for all strategies
3. **`UPDATE_DOCUMENTS_GUIDE.md`** - This documentation file

## Summary

The `update_documents()` method provides a complete CRUD interface for managing documents in your RAG knowledge base. Key features:

- ✅ **4 Update Strategies**: replace, append, delete, upsert
- ✅ **Flexible Filtering**: By metadata or document IDs
- ✅ **Metadata-Only Updates**: No reprocessing when only metadata changes
- ✅ **MongoDB Integration**: Automatic content synchronization
- ✅ **Comprehensive Error Handling**: Clear error messages for all failure cases
- ✅ **Testing Infrastructure**: Test script + API endpoints
- ✅ **Production-Ready**: Full error handling, logging, and validation

## Next Steps

1. Run the test script: `python test_update_documents.py`
2. Start the testing API: `cd testing_api && uvicorn main:app --reload`
3. Try the example code snippets above
4. Integrate into your application

Enjoy your new document management capabilities! 🚀
