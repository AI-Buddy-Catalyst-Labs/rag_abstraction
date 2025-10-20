# MongoDB Integration Guide

## Overview

The insta_rag library now supports MongoDB integration for efficient content storage. Instead of storing full text content in Qdrant, the content is stored in MongoDB and only the reference (MongoDB ID) is stored in Qdrant's metadata.

## Architecture

### Without MongoDB (Default)

```
Document → Chunking → Embedding → Qdrant (vectors + full content)
```

### With MongoDB (New)

```
Document → Chunking → Embedding → MongoDB (full content)
                                → Qdrant (vectors + MongoDB reference)
```

## Benefits

1. **Reduced Qdrant Storage**: Only vectors and metadata in Qdrant
1. **Centralized Content**: All content in MongoDB for easy management
1. **Easy Updates**: Update content without re-embedding
1. **Better Separation**: Vectors and content stored separately
1. **Cost Effective**: Cheaper storage for large text content

## Setup

### 1. Environment Configuration

Add to your `.env` file:

```env
# MongoDB Connection (already in your .env)
MONGO_CONNECTION_STRING=mongodb://root:password/?directConnection=true

# Database Name (optional, defaults to Test_Insta_RAG)
MONGO_DATABASE_NAME=Test_Insta_RAG
```

### 2. Install MongoDB Driver

```bash
pip install pymongo>=4.6.0
```

Or update from requirements:

```bash
pip install -r requirements-rag.txt
```

## Usage

### Automatic Configuration

MongoDB integration is automatically enabled when connection string is present:

```python
from insta_rag import RAGClient, RAGConfig

# Load config from .env (automatically includes MongoDB if configured)
config = RAGConfig.from_env()
client = RAGClient(config)

# Check if MongoDB is enabled
if client.mongodb:
    print("MongoDB integration active")
```

### Adding Documents

The API remains the same - MongoDB integration is transparent:

```python
from insta_rag import DocumentInput

doc = DocumentInput.from_text("Your document content here")

response = client.add_documents(documents=[doc], collection_name="my_collection")

# Content is now in MongoDB, reference in Qdrant
print(f"Processed {response.total_chunks} chunks")
```

### Retrieving Content

When retrieving, content is automatically fetched from MongoDB:

```python
# Get chunk content from MongoDB
if client.mongodb:
    chunk_content = client.mongodb.get_chunk_content(chunk_id)
    print(chunk_content["content"])
```

## MongoDB Collections

### document_contents

Stores actual chunk content:

```json
{
  "_id": "ObjectId",
  "chunk_id": "doc_123_chunk_0",
  "content": "Full text content of the chunk...",
  "document_id": "doc_123",
  "collection_name": "my_collection",
  "metadata": {
    "token_count": 150,
    "chunk_index": 0,
    ...
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### document_metadata

Stores aggregated document information:

```json
{
  "_id": "ObjectId",
  "document_id": "doc_123",
  "collection_name": "my_collection",
  "total_chunks": 5,
  "metadata": {...},
  "created_at": "2024-01-15T10:30:00Z"
}
```

## MongoDB Client API

### Store Chunk Content

```python
mongo_id = client.mongodb.store_chunk_content(
    chunk_id="chunk_123",
    content="Full text content",
    document_id="doc_123",
    collection_name="my_collection",
    metadata={"key": "value"},
)
```

### Retrieve Content

```python
# By chunk_id
chunk_doc = client.mongodb.get_chunk_content("chunk_123")

# By MongoDB _id
chunk_doc = client.mongodb.get_chunk_content_by_mongo_id(mongo_id)

# All chunks for a document
chunks = client.mongodb.get_chunks_by_document("doc_123")
```

### Delete Content

```python
# Delete single chunk
client.mongodb.delete_chunk("chunk_123")

# Delete all chunks for a document
count = client.mongodb.delete_chunks_by_document("doc_123")

# Delete all chunks in a collection
count = client.mongodb.delete_chunks_by_collection("my_collection")
```

### Get Statistics

```python
stats = client.mongodb.get_collection_stats("my_collection")

print(f"Total chunks: {stats['total_chunks']}")
print(f"Total documents: {stats['total_documents']}")
print(f"Content size: {stats['total_content_size_bytes']} bytes")
```

## Qdrant Metadata Structure

When MongoDB is enabled, Qdrant stores metadata with references:

```json
{
  "mongodb_id": "65a1b2c3d4e5f6g7h8i9j0k1",
  "content_storage": "mongodb",
  "document_id": "doc_123",
  "chunk_index": 0,
  "token_count": 150,
  ...
}
```

Note: The `content` field in Qdrant is empty when MongoDB is enabled.

## Testing MongoDB Integration

### Run Test Script

```bash
cd testing_api
python test_mongodb.py
```

### Expected Output

```
Testing MongoDB Integration
================================================

1. Initializing RAG client with MongoDB...
   ✓ Client initialized
   ✓ MongoDB enabled: True
   ✓ Database: Test_Insta_RAG

2. Creating test document...
   ✓ Document created

3. Processing document...
   Storing content in MongoDB...
   ✓ Stored 3 chunks in MongoDB
   ✓ Document processed successfully

4. Verifying MongoDB storage...
   ✓ Chunk found in MongoDB
     - MongoDB ID: 65a1b2c3d4e5f6g7h8i9j0k1
     - Content length: 250 chars

5. MongoDB Collection Statistics...
   - Total chunks: 3
   - Total documents: 1
   - Total content size: 750 bytes
```

## Migration

### From Qdrant-only to MongoDB

If you have existing data in Qdrant:

1. Existing collections continue to work (content in Qdrant)
1. New documents will use MongoDB storage
1. To migrate existing data, re-process documents with MongoDB enabled

### Disabling MongoDB

To disable MongoDB and revert to Qdrant-only storage:

```python
# Remove or comment out in .env
# MONGO_CONNECTION_STRING=...

# Or explicitly disable in code
config.mongodb = None
```

## Monitoring

### Check MongoDB Connection

```python
if client.mongodb:
    try:
        client.mongodb.client.admin.command("ping")
        print("MongoDB connected")
    except Exception as e:
        print(f"MongoDB error: {e}")
```

### View Collections

```bash
# Using MongoDB shell
mongosh "$MONGO_CONNECTION_STRING"

> use Test_Insta_RAG
> db.document_contents.countDocuments()
> db.document_metadata.countDocuments()
```

## Performance Considerations

1. **Batch Operations**: Use `store_chunks_batch()` for multiple chunks
1. **Indexing**: Indexes are automatically created on `chunk_id`, `document_id`, and `collection_name`
1. **Connection Pooling**: MongoDB client uses connection pooling automatically
1. **Network Latency**: Consider co-locating MongoDB and Qdrant for best performance

## Troubleshooting

### "MongoDB not installed"

```bash
pip install pymongo>=4.6.0
```

### "Failed to connect to MongoDB"

- Check connection string is correct
- Verify MongoDB server is running
- Check network connectivity
- Verify authentication credentials

### "Collection not found"

MongoDB collections are created automatically on first insert.

### Content not found in MongoDB

- Verify MongoDB was enabled during document processing
- Check chunk_id is correct
- Verify database name matches configuration

## API Testing with MongoDB

### Swagger UI

1. Start API: `cd testing_api && ./run.sh`
1. Open: http://localhost:8000/docs
1. Test endpoint: `POST /api/v1/test/documents/add`
1. Check response includes MongoDB storage confirmation

### cURL Test

```bash
curl -X POST http://localhost:8000/api/v1/test/documents/add \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test document for MongoDB storage",
    "collection_name": "test_collection"
  }'
```

## Best Practices

1. **Always use MongoDB for production**: Better scalability and management
1. **Regular backups**: Backup MongoDB collections regularly
1. **Monitor storage**: Use `get_collection_stats()` to track growth
1. **Clean up**: Delete old collections when no longer needed
1. **Indexing**: Add custom indexes for your query patterns

## Future Enhancements

- [ ] Content versioning
- [ ] Full-text search in MongoDB
- [ ] Content compression
- [ ] Multi-tenancy support
- [ ] Backup and restore utilities
