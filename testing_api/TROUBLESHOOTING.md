# Troubleshooting Guide - Testing API

## Common Issues and Solutions

### 1. Qdrant Connection Errors

#### Error: "failed to connect to all addresses... gRPC"

**Problem:** Qdrant client trying to use gRPC but the server doesn't support it or there's a firewall issue.

**Solution:**

The library now uses HTTP REST API by default (gRPC disabled). This is already configured.

If you still see issues, verify in `.env`:
```env
QDRANT_PREFER_GRPC=false
```

**Manual fix if needed:**

```python
from insta_rag import RAGConfig
from insta_rag.core.config import VectorDBConfig

config = RAGConfig.from_env()
config.vectordb.prefer_grpc = False  # Force HTTP
config.vectordb.timeout = 60  # Increase timeout

client = RAGClient(config)
```

#### Error: "Connection timeout"

**Solutions:**

1. **Verify Qdrant URL is accessible:**
```bash
curl -I https://qdrant-okc4ss8owk0ggwg4ccwsoks0.aibuddy-coolify-inventory.aukikaurnab.com/
```

2. **Check API key:**
```bash
curl https://qdrant-okc4ss8owk0ggwg4ccwsoks0.aibuddy-coolify-inventory.aukikaurnab.com/collections \
  -H "api-key: your_api_key"
```

3. **Increase timeout in code:**
```python
config.vectordb.timeout = 120  # 2 minutes
```

### 2. MongoDB Connection Errors

#### Error: "Failed to connect to MongoDB"

**Solutions:**

1. **Verify connection string:**
```bash
# Test MongoDB connection
mongosh "mongodb://root:password/?directConnection=true"
```

2. **Check if MongoDB is using correct port:**
Your connection string shows port `5432` which is typically PostgreSQL. MongoDB usually uses `27017`.

Verify with:
```bash
# Try standard MongoDB port
mongosh "mongodb://root:password/?directConnection=true"
```

3. **Disable MongoDB temporarily:**
```env
# In .env, comment out:
# MONGO_CONNECTION_STRING=...
```

### 3. Testing API Issues

#### Error: "RAG client not initialized"

**Cause:** Configuration missing or invalid.

**Solution:**

1. **Check startup logs:**
```
✓ RAG Client initialized successfully
✓ MongoDB connected: Test_Insta_RAG
```

If you see errors, fix the configuration.

2. **Verify environment variables:**
```bash
cd /home/macorov/Documents/GitHub/insta_rag
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('QDRANT_URL:', os.getenv('QDRANT_URL'))
print('AZURE_OPENAI_API_KEY:', bool(os.getenv('AZURE_OPENAI_API_KEY')))
print('MONGO_CONNECTION_STRING:', bool(os.getenv('MONGO_CONNECTION_STRING')))
"
```

#### Error: "Import errors"

**Solution:**
```bash
# Reinstall insta_rag
cd /home/macorov/Documents/GitHub/insta_rag
pip install -e .

# Install testing_api requirements
cd testing_api
pip install -r requirements.txt
```

### 4. Collection Issues

#### Using Single Test Collection

All tests now use a single collection: `insta_rag_test_collection`

**Benefits:**
- Easier to manage and clean up
- Consistent testing environment
- No collection name conflicts

**To view/delete test collection:**

```python
from insta_rag import RAGClient, RAGConfig

config = RAGConfig.from_env()
client = RAGClient(config)

# List all collections
collections = client.list_collections()
print(collections)

# Get test collection info
info = client.get_collection_info("insta_rag_test_collection")
print(f"Vectors: {info['vectors_count']}")

# Delete test collection if needed (via Qdrant client directly)
client.vectordb.client.delete_collection("insta_rag_test_collection")
```

### 5. API Endpoint Errors

#### All endpoints use the same collection

The testing API now uses `insta_rag_test_collection` for all tests by default.

**Default collection name:** `insta_rag_test_collection`

**Override if needed:**
```bash
# In request body
curl -X POST http://localhost:8000/api/v1/test/documents/add \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test",
    "collection_name": "my_custom_collection"
  }'
```

### 6. Performance Issues

#### Slow embedding generation

**Solutions:**

1. **Use smaller batch sizes:**
```python
config.embedding.batch_size = 50  # Default is 100
```

2. **Check Azure OpenAI quota:**
```bash
# Verify API key works
curl https://your-instance.openai.azure.com/openai/deployments?api-version=2024-02-01 \
  -H "api-key: your_key"
```

#### Slow vector search

**Solutions:**

1. **Already using HTTP (faster than gRPC for your setup)**

2. **Reduce search limits:**
```python
config.retrieval.vector_search_limit = 10  # Default is 25
```

### 7. Quick Diagnostic Commands

#### Test Qdrant Connection
```bash
cd /home/macorov/Documents/GitHub/insta_rag
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(
    url='https://qdrant-okc4ss8owk0ggwg4ccwsoks0.aibuddy-coolify-inventory.aukikaurnab.com/',
    api_key='edfBd7pP251ev2uiRcjcBGt7QXJe1P70',
    prefer_grpc=False,
    timeout=60
)
print('Collections:', client.get_collections())
"
```

#### Test MongoDB Connection
```bash
python -c "
from pymongo import MongoClient
client = MongoClient('mongodb://root:password/?directConnection=true')
print('Databases:', client.list_database_names())
"
```

#### Test Azure OpenAI
```bash
python -c "
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key='your_key',
    api_version='2024-02-01',
    azure_endpoint='your_endpoint'
)
print('API works!')
"
```

### 8. Clean Test Data

#### Remove all test data

```python
from insta_rag import RAGClient, RAGConfig

config = RAGConfig.from_env()
client = RAGClient(config)

# Delete Qdrant test collection
try:
    client.vectordb.client.delete_collection("insta_rag_test_collection")
    print("Qdrant collection deleted")
except:
    print("Collection doesn't exist or already deleted")

# Delete MongoDB test data
if client.mongodb:
    count = client.mongodb.delete_chunks_by_collection("insta_rag_test_collection")
    print(f"Deleted {count} chunks from MongoDB")
```

## Getting Help

### Check Logs

When running the API, watch for these messages:

```
✓ RAG Client initialized successfully
✓ MongoDB connected: Test_Insta_RAG
```

If you see errors instead, they will indicate what's wrong.

### Enable Debug Mode

```bash
# Run with verbose logging
export PYTHONUNBUFFERED=1
python testing_api/main.py
```

### Test Individual Components

```bash
# Test just configuration
python -c "from insta_rag import RAGConfig; config = RAGConfig.from_env(); print('OK')"

# Test just client initialization
python -c "from insta_rag import RAGClient, RAGConfig; client = RAGClient(RAGConfig.from_env()); print('OK')"
```

## Summary of Fixes Applied

1. ✅ **Disabled gRPC by default** - Uses HTTP REST API (more compatible)
2. ✅ **Increased timeout to 60 seconds** - Handles slower networks
3. ✅ **Single test collection** - All tests use `insta_rag_test_collection`
4. ✅ **Better error messages** - Clearer indication of what failed
5. ✅ **Automatic fallbacks** - HTTP when gRPC fails

These fixes should resolve the connection issues you were experiencing!
