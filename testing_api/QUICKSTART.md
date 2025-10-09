# Quick Start Guide - Testing API

## Step-by-Step Setup and Testing

### 1. Install Dependencies

```bash
# Navigate to the testing_api directory
cd /home/macorov/Documents/GitHub/insta_rag/testing_api

# Install FastAPI dependencies
pip install fastapi uvicorn python-multipart pydantic python-dotenv

# Or install from requirements file
pip install -r requirements.txt
```

### 2. Ensure insta_rag is Installed

```bash
# Go back to project root
cd /home/macorov/Documents/GitHub/insta_rag

# Install insta_rag library
pip install -e .

# Or use uv
uv pip install -e .
```

### 3. Verify Environment Variables

Make sure your `.env` file in the project root has:

```bash
# Check if .env exists and has required variables
cat /home/macorov/Documents/GitHub/insta_rag/.env | grep -E "(QDRANT|AZURE_OPENAI|OPENAI)"
```

Should show:
```
QDRANT_URL=...
QDRANT_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_EMBEDDING_DEPLOYMENT=...
```

### 4. Run the Testing API

#### Option A: Direct Python

```bash
cd /home/macorov/Documents/GitHub/insta_rag/testing_api
python main.py
```

#### Option B: Using Uvicorn (Recommended)

```bash
cd /home/macorov/Documents/GitHub/insta_rag/testing_api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Option C: With Custom Port

```bash
uvicorn main:app --reload --port 8080
```

### 5. Verify API is Running

```bash
# Test health endpoint
curl http://localhost:8000/

# Or open in browser
# http://localhost:8000/
```

Expected response:
```json
{
  "status": "healthy",
  "message": "insta_rag Testing API is running",
  "components": {
    "rag_client": "initialized",
    "config": "loaded",
    "embeddings": "configured",
    "vector_db": "configured"
  }
}
```

### 6. Access Interactive Documentation

Open your browser and go to:

**Swagger UI (Interactive):**
```
http://localhost:8000/docs
```

**ReDoc (Alternative):**
```
http://localhost:8000/redoc
```

**OpenAPI JSON:**
```
http://localhost:8000/openapi.json
```

### 7. Run Automated Tests

```bash
cd /home/macorov/Documents/GitHub/insta_rag/testing_api

# Make script executable (if not already)
chmod +x test_requests.sh

# Run all tests
./test_requests.sh
```

## Quick Test Commands

### Test 1: Health Check
```bash
curl http://localhost:8000/
```

### Test 2: Configuration
```bash
curl http://localhost:8000/api/v1/test/config
```

### Test 3: Chunking
```bash
curl -X POST http://localhost:8000/api/v1/test/chunking \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test document. It has multiple sentences. We will test chunking.",
    "max_chunk_size": 1000,
    "overlap_percentage": 0.2
  }'
```

### Test 4: Embeddings
```bash
curl -X POST http://localhost:8000/api/v1/test/embedding \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Test sentence"]}'
```

### Test 5: Vector DB Collections
```bash
curl http://localhost:8000/api/v1/test/vectordb/collections
```

### Test 6: Document Processing
```bash
curl -X POST http://localhost:8000/api/v1/test/documents/add \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test document for complete pipeline.",
    "collection_name": "test_collection",
    "metadata": {"source": "quickstart"}
  }'
```

### Test 7: PDF Upload (if you have a PDF)
```bash
curl -X POST http://localhost:8000/api/v1/test/pdf/upload \
  -F "file=@/path/to/your/document.pdf"
```

### Test 8: Integration Test (All Components)
```bash
curl http://localhost:8000/api/v1/test/integration
```

## Using Swagger UI (Browser)

1. **Open Swagger UI**: http://localhost:8000/docs

2. **Try an endpoint:**
   - Click on any endpoint (e.g., `POST /api/v1/test/chunking`)
   - Click "Try it out" button
   - Fill in the request body
   - Click "Execute"
   - View the response

3. **Test file uploads:**
   - Go to `POST /api/v1/test/pdf/upload`
   - Click "Try it out"
   - Choose file to upload
   - Click "Execute"

## Import into Postman

1. **Import OpenAPI spec:**
   - Open Postman
   - Click "Import"
   - Choose "File"
   - Select `openapi.yaml` from testing_api directory
   - All endpoints will be imported automatically

2. **Set base URL:**
   - In Postman, set variable `{{baseUrl}}` to `http://localhost:8000`

3. **Test endpoints:**
   - Select any request
   - Click "Send"

## Using with Insomnia

1. **Import OpenAPI spec:**
   - Open Insomnia
   - Click "Create" → "Import From"
   - Select "File"
   - Choose `openapi.yaml`

2. **Test endpoints:**
   - Select any request
   - Click "Send"

## Troubleshooting

### API won't start

**Error: "No module named 'fastapi'"**
```bash
pip install fastapi uvicorn python-multipart
```

**Error: "No module named 'insta_rag'"**
```bash
cd /home/macorov/Documents/GitHub/insta_rag
pip install -e .
```

**Error: "Address already in use"**
```bash
# Use different port
uvicorn main:app --reload --port 8001
```

### Tests failing

**Error: "RAG client not initialized"**
- Check `.env` file has all required variables
- Verify environment variables are loaded:
  ```bash
  python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('QDRANT_URL:', os.getenv('QDRANT_URL'))"
  ```

**Error: "Connection refused" to Qdrant**
- Verify Qdrant URL is correct and accessible
- Test connection:
  ```bash
  curl $QDRANT_URL
  ```

**Error: "Invalid API key"**
- Check Azure OpenAI or OpenAI API key is correct
- Verify key has proper permissions

### Check Logs

The API prints logs to console. Look for:
```
✓ RAG Client initialized successfully
```

If you see:
```
✗ Failed to initialize RAG Client: ...
```
Then check the error message and fix configuration.

## Production Deployment

For production use (not recommended for this testing API):

```bash
# With multiple workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or with systemd service
sudo systemctl start insta-rag-testing-api
```

## Next Steps

After successful testing:

1. ✅ All endpoints working → Library is ready to use
2. ❌ Some tests failing → Review error messages and fix configuration
3. 📝 Document any issues in GitHub Issues

## Stop the API

Press `Ctrl+C` in the terminal where the API is running.

## Summary of Commands

```bash
# Install
cd /home/macorov/Documents/GitHub/insta_rag/testing_api
pip install -r requirements.txt
cd .. && pip install -e .

# Run
cd testing_api
uvicorn main:app --reload --port 8000

# Test
curl http://localhost:8000/
./test_requests.sh

# Access UI
# Open browser: http://localhost:8000/docs
```

That's it! Your testing API should now be running and ready to validate all insta_rag components.
