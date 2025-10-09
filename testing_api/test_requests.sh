#!/bin/bash

# Test script for insta_rag Testing API
# Usage: ./test_requests.sh

BASE_URL="http://localhost:8000"

echo "================================================"
echo "insta_rag Testing API - Test Suite"
echo "================================================"
echo ""

# Test 1: Health Check
echo "Test 1: Health Check"
echo "-------------------"
curl -s "$BASE_URL/" | jq
echo ""
echo ""

# Test 2: Configuration
echo "Test 2: Configuration Validation"
echo "--------------------------------"
curl -s "$BASE_URL/api/v1/test/config" | jq
echo ""
echo ""

# Test 3: Chunking Utils
echo "Test 3: Chunking Utilities"
echo "--------------------------"
curl -s "$BASE_URL/api/v1/test/chunking/utils" | jq
echo ""
echo ""

# Test 4: Chunking
echo "Test 4: Semantic Chunking"
echo "-------------------------"
curl -s -X POST "$BASE_URL/api/v1/test/chunking" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test document. It has multiple sentences. We will test semantic chunking. The system should split this intelligently.",
    "max_chunk_size": 1000,
    "overlap_percentage": 0.2
  }' | jq
echo ""
echo ""

# Test 5: Embedding
echo "Test 5: Embedding Generation"
echo "----------------------------"
curl -s -X POST "$BASE_URL/api/v1/test/embedding" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Test sentence one", "Test sentence two"]
  }' | jq
echo ""
echo ""

# Test 6: Vector DB Collections
echo "Test 6: Vector DB Collections"
echo "-----------------------------"
curl -s "$BASE_URL/api/v1/test/vectordb/collections" | jq
echo ""
echo ""

# Test 7: Document Processing
echo "Test 7: Document Processing"
echo "---------------------------"
curl -s -X POST "$BASE_URL/api/v1/test/documents/add" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test document for the complete processing pipeline. It will be chunked, embedded, and stored in the vector database.",
    "collection_name": "test_collection",
    "metadata": {"source": "test_script", "test": true}
  }' | jq
echo ""
echo ""

# Test 8: Integration Test
echo "Test 8: Integration Test (All Components)"
echo "-----------------------------------------"
curl -s "$BASE_URL/api/v1/test/integration" | jq
echo ""
echo ""

echo "================================================"
echo "Test Suite Complete"
echo "================================================"
