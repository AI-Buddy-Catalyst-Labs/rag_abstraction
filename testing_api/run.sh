#!/bin/bash

# Run script for insta_rag Testing API
# Usage: ./run.sh [port]

PORT=${1:-8000}

echo "================================================"
echo "Starting insta_rag Testing API"
echo "================================================"
echo ""

# Check if in correct directory
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found. Please run from testing_api directory."
    exit 1
fi

# Check if .env exists in parent directory
if [ ! -f "../.env" ]; then
    echo "Warning: .env file not found in parent directory"
    echo "Please create .env with required variables:"
    echo "  - QDRANT_URL"
    echo "  - QDRANT_API_KEY"
    echo "  - AZURE_OPENAI_ENDPOINT"
    echo "  - AZURE_OPENAI_API_KEY"
    echo ""
fi

# Check if fastapi is installed
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: FastAPI not installed"
    echo "Run: pip install -r requirements.txt"
    exit 1
fi

# Check if insta_rag is installed
python -c "import insta_rag" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: insta_rag not installed"
    echo "Run from project root: pip install -e ."
    exit 1
fi

echo "✓ Dependencies verified"
echo "✓ Starting API on port $PORT"
echo ""
echo "Access points:"
echo "  - Health Check: http://localhost:$PORT/"
echo "  - Swagger UI:   http://localhost:$PORT/docs"
echo "  - ReDoc:        http://localhost:$PORT/redoc"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the API
uvicorn main:app --reload --host 0.0.0.0 --port $PORT
