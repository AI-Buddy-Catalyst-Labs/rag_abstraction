# New Testing API Architecture

## Overview
The testing_api now uses library components directly and stores chunk content in MongoDB separately from vectors in Qdrant.

## Architecture Changes

### Library (insta_rag)
- **Database Agnostic**: No MongoDB dependency
- **Stores everything in Qdrant by default**: Content + Vectors + Metadata
- **User-provided metadata**: Library doesn't enforce metadata structure
- **Filter-based deletion**: Delete by `document_id` using Qdrant filters

### Testing API
- **Uses library components individually**:
  - `SemanticChunker` for chunking
  - `OpenAIEmbedder` for embeddings
  - `QdrantVectorDB` for vector storage
- **MongoDB for content storage**: Chunks' text content stored separately
- **Flexible metadata**: User defines what to store in Qdrant

## Upload Flow

1. **Receive Document** → testing_api endpoint
2. **Chunk Document** → Use `SemanticChunker`
3. **For each chunk**:
   - Store content in MongoDB → Get `mongodb_id`
   - Generate embedding using `OpenAIEmbedder`
   - Prepare metadata: `{"mongodb_id": "...", "document_id": "...", ...user metadata...}`
4. **Store in Qdrant**: Vectors + Metadata (NO CONTENT)
   - Use `QdrantVectorDB.upsert()`
   - content parameter: empty string `""`

## Retrieval Flow

1. **Query** → testing_api endpoint
2. **Search Qdrant** → Use `QdrantVectorDB.search()`
3. **Get results with metadata** → Contains `mongodb_id`
4. **Fetch content from MongoDB** → Use `mongodb_id`
5. **Combine and return** → Results with full content

## Delete Flow

1. **Delete request** → testing_api endpoint with `document_ids`
2. **Delete from Qdrant** → Use `QdrantVectorDB.delete_by_document_ids()`
3. **Delete from MongoDB** → Use `MongoDBStorage.delete_chunks_by_document_ids()`

## Benefits

- **Library is database-agnostic**: Users can integrate any storage
- **Efficient vector operations**: Qdrant handles vectors and metadata
- **Flexible content storage**: MongoDB for content, any DB for other data
- **User-controlled metadata**: No enforced structure
- **Better separation of concerns**: Vectors vs. Content storage

## Migration Notes

- Old `RAGClient.add_documents()` now stores content in Qdrant
- Old MongoDB integration removed from library
- Testing API implements the MongoDB pattern as an example
- Users can follow this pattern with their preferred database
