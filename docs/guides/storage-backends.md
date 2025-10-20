# Guide: Storage Backends

`insta_rag` supports two primary storage configurations for your document chunks: **Qdrant-Only** and **Hybrid (Qdrant + MongoDB)**. This guide explains the difference and how to configure them.

## Storage Architectures

### 1. Qdrant-Only Mode (Default)

In this mode, all data associated with a chunk is stored directly in the Qdrant vector database.

*   **Architecture**: `Document → Chunking → Embedding → Qdrant (stores vectors, metadata, and full text content)`
*   **Qdrant Payload**: Contains the vector embedding, all metadata, and the full `content` of the chunk.
*   **Pros**: Simple to set up, requires only one database.
*   **Cons**: Can be less cost-effective for very large text content, as vector databases are optimized and priced for vector search, not bulk text storage.

### 2. Hybrid Mode: Qdrant + MongoDB (Recommended for Production)

In this mode, storage is split: Qdrant stores the vectors for fast searching, and MongoDB stores the actual text content.

*   **Architecture**: `Document → Chunking → Embedding → MongoDB (stores full content) & Qdrant (stores vectors + a reference to MongoDB)`
*   **Qdrant Payload**: Contains the vector embedding and metadata, but the `content` field is empty. Instead, it stores a `mongodb_id` pointing to the document in MongoDB.
*   **MongoDB Document**: Contains the `chunk_id`, the full `content`, and a copy of the metadata.
*   **Pros**:
    *   **Cost-Effective**: Leverages MongoDB for cheaper, efficient bulk text storage.
    *   **Separation of Concerns**: Qdrant handles what it does best (vector search), and MongoDB handles content storage.
    *   **Flexibility**: Allows you to manage and update content in MongoDB without needing to re-index vectors in Qdrant.
*   **Cons**: Requires managing a second database (MongoDB).

## Configuration

The storage mode is **automatically determined** based on your environment configuration.

### Enabling Hybrid Mode (Qdrant + MongoDB)

To enable hybrid mode, simply add your MongoDB connection string to your `.env` file. If the connection string is present, `insta_rag` will automatically use the hybrid storage architecture.

```env
# .env file

# Qdrant Configuration
QDRANT_URL="..."
QDRANT_API_KEY="..."

# MongoDB Configuration (enables Hybrid Mode)
MONGO_CONNECTION_STRING="mongodb://user:password@host:port/"
MONGO_DATABASE_NAME="your_db_name" # Optional, defaults to Test_Insta_RAG
```

### Using Qdrant-Only Mode

To use the Qdrant-only mode, simply omit or comment out the `MONGO_CONNECTION_STRING` from your `.env` file.

```env
# .env file

# Qdrant Configuration
QDRANT_URL="..."
QDRANT_API_KEY="..."

# MONGO_CONNECTION_STRING is not present, so Qdrant-only mode is used.
```

## How It Works During Retrieval

The library handles the difference in storage transparently during retrieval:

1.  A search query is sent to Qdrant.
2.  Qdrant returns a list of matching vectors and their payloads.
3.  The `RAGClient` inspects the payload of each result.
4.  If a `mongodb_id` is present, it automatically fetches the full content from MongoDB.
5.  If there is no `mongodb_id`, it uses the `content` directly from the Qdrant payload.

The final `RetrievalResponse` will contain the full text content regardless of which storage backend was used.

## Best Practices

*   For **local development and testing**, the **Qdrant-only** mode is often simpler and sufficient.
*   For **production environments**, especially with a large volume of documents, the **Hybrid (Qdrant + MongoDB)** mode is highly recommended for its cost-effectiveness and scalability.
