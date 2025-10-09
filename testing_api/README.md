# insta_rag Testing API

FastAPI application for testing all insta_rag library modules and components.

## Purpose

This API provides endpoints to test and validate:
- Configuration loading and validation
- Chunking functionality (semantic chunking, utilities)
- Embedding generation (OpenAI/Azure OpenAI)
- Vector database operations (Qdrant)
- PDF processing
- Complete document processing pipeline
- Integration testing

## Installation

### 1. Install Dependencies

```bash
# From the testing_api directory
pip install -r requirements.txt

# Also ensure insta_rag is installed
cd ..
pip install -e .
```

### 2. Set Up Environment Variables

Ensure your `.env` file in the project root has all required variables:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

## Running the API

### Development Mode

```bash
# From the testing_api directory
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## API Documentation

Once running, access interactive documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Health Check

**GET** `/`
- Check API health and component status

```bash
curl http://localhost:8000/
```

### Configuration Testing

**GET** `/api/v1/test/config`
- Test configuration loading and validation

```bash
curl http://localhost:8000/api/v1/test/config
```

### Chunking Testing

**POST** `/api/v1/test/chunking`
- Test semantic chunking functionality

```bash
curl -X POST http://localhost:8000/api/v1/test/chunking \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your test text here. Multiple sentences for chunking.",
    "max_chunk_size": 1000,
    "overlap_percentage": 0.2
  }'
```

**GET** `/api/v1/test/chunking/utils`
- Test chunking utility functions

```bash
curl http://localhost:8000/api/v1/test/chunking/utils
```

### Embedding Testing

**POST** `/api/v1/test/embedding`
- Test embedding generation

```bash
curl -X POST http://localhost:8000/api/v1/test/embedding \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Test sentence 1", "Test sentence 2"]
  }'
```

### Vector Database Testing

**GET** `/api/v1/test/vectordb/collections`
- List all collections

```bash
curl http://localhost:8000/api/v1/test/vectordb/collections
```

**GET** `/api/v1/test/vectordb/collection/{collection_name}`
- Get collection information

```bash
curl http://localhost:8000/api/v1/test/vectordb/collection/test_collection
```

### PDF Processing Testing

**POST** `/api/v1/test/pdf/upload`
- Test PDF text extraction

```bash
curl -X POST http://localhost:8000/api/v1/test/pdf/upload \
  -F "file=@/path/to/document.pdf"
```

### Document Processing Testing

**POST** `/api/v1/test/documents/add`
- Test complete document processing with text

```bash
curl -X POST http://localhost:8000/api/v1/test/documents/add \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document text here",
    "collection_name": "test_collection",
    "metadata": {"source": "api_test"}
  }'
```

**POST** `/api/v1/test/documents/add-file`
- Test complete document processing with file upload

```bash
curl -X POST "http://localhost:8000/api/v1/test/documents/add-file?collection_name=test_collection" \
  -F "file=@/path/to/document.pdf"
```

### Integration Testing

**GET** `/api/v1/test/integration`
- Run comprehensive integration test of all components

```bash
curl http://localhost:8000/api/v1/test/integration
```

## Testing Workflow

### 1. Quick Health Check

```bash
curl http://localhost:8000/
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

### 2. Validate Configuration

```bash
curl http://localhost:8000/api/v1/test/config
```

### 3. Test Individual Components

```bash
# Test chunking
curl -X POST http://localhost:8000/api/v1/test/chunking \
  -H "Content-Type: application/json" \
  -d '{"text": "Test text for chunking. Multiple sentences here."}'

# Test embeddings
curl -X POST http://localhost:8000/api/v1/test/embedding \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Test sentence"]}'

# Test vector DB
curl http://localhost:8000/api/v1/test/vectordb/collections
```

### 4. Test Complete Pipeline

```bash
curl -X POST http://localhost:8000/api/v1/test/documents/add \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Complete pipeline test document.",
    "collection_name": "test_collection"
  }'
```

### 5. Run Integration Test

```bash
curl http://localhost:8000/api/v1/test/integration
```

## Using Swagger UI

1. Open browser: `http://localhost:8000/docs`
2. Explore all endpoints with interactive documentation
3. Try out endpoints directly from the browser
4. View request/response schemas

## Response Format

All endpoints return JSON with a consistent structure:

### Success Response
```json
{
  "success": true,
  "data": {...},
  "errors": []
}
```

### Error Response
```json
{
  "success": false,
  "errors": ["Error message 1", "Error message 2"]
}
```

## Troubleshooting

### API won't start

1. Check Python version: `python --version` (need 3.9+)
2. Verify dependencies: `pip install -r requirements.txt`
3. Check environment variables in `.env`

### Tests failing

1. Verify Qdrant is accessible: `curl $QDRANT_URL`
2. Check API keys are correct
3. Review logs for specific error messages

### Import errors

```bash
# Ensure insta_rag is installed
cd /home/macorov/Documents/GitHub/insta_rag
pip install -e .
```

## Development

### Adding New Test Endpoints

1. Add new endpoint function in `main.py`
2. Define Pydantic models for request/response
3. Add error handling
4. Update this README

### Running Tests

```bash
# Manual testing with curl
./test_api.sh  # If you create a test script

# Or use pytest (future enhancement)
pytest tests/
```

## Production Deployment

For production deployment, use a production ASGI server:

```bash
# With Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

# Or configure with systemd/supervisor
```

## Security Notes

- This is a **testing API** - not for production use
- No authentication implemented
- File uploads stored temporarily and deleted
- Use in development/testing environments only

## Support

For issues with the testing API:
- Check FastAPI logs
- Verify insta_rag library is working: `python examples/simple_test.py`
- Review endpoint documentation at `/docs`
