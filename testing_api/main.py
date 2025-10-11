"""FastAPI application for testing insta_rag library modules."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import insta_rag modules
from insta_rag import DocumentInput, RAGClient, RAGConfig
from insta_rag.chunking.semantic import SemanticChunker
from insta_rag.chunking.utils import count_tokens_accurate, validate_chunk_quality
from insta_rag.embedding.openai import OpenAIEmbedder
from insta_rag.exceptions import (
    ChunkingError,
    ConfigurationError,
    EmbeddingError,
    PDFEmptyError,
    PDFEncryptedError,
    VectorDBError,
)
from insta_rag.models.chunk import Chunk, ChunkMetadata
from insta_rag.pdf_processing import extract_text_from_pdf, validate_pdf
from insta_rag.vectordb.qdrant import QdrantVectorDB

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="insta_rag Testing API",
    description="API for testing all insta_rag library modules and components",
    version="0.0.1",
)

# Global RAG client (initialized on startup)
rag_client: Optional[RAGClient] = None

# Single collection name for all tests
TEST_COLLECTION_NAME = "insta_rag_test_collection"


# Pydantic Models for API
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    message: str
    components: Dict[str, str]


class ConfigResponse(BaseModel):
    """Configuration validation response."""

    valid: bool
    config: Dict[str, Any]
    errors: List[str] = []


class TextChunkRequest(BaseModel):
    """Request for chunking text."""

    text: str = Field(..., description="Text to chunk")
    max_chunk_size: int = Field(1000, description="Maximum tokens per chunk")
    overlap_percentage: float = Field(0.2, description="Overlap percentage")


class ChunkResponse(BaseModel):
    """Chunking response."""

    success: bool
    chunks_count: int
    chunks: List[Dict[str, Any]]
    errors: List[str] = []


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings."""

    texts: List[str] = Field(..., description="List of texts to embed")


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    success: bool
    embeddings_count: int
    dimensions: int
    sample_embedding: Optional[List[float]] = None
    errors: List[str] = []


class AddDocumentRequest(BaseModel):
    """Request for adding documents."""

    text: Optional[str] = Field(None, description="Text content")
    collection_name: str = Field(
        TEST_COLLECTION_NAME, description="Collection name (uses test collection by default)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class AddDocumentResponse(BaseModel):
    """Add document response."""

    success: bool
    documents_processed: int
    total_chunks: int
    processing_time_ms: float
    errors: List[str] = []


class SearchRequest(BaseModel):
    """Request for searching/retrieving documents."""

    query: str = Field(..., description="Search query")
    collection_name: str = Field(
        TEST_COLLECTION_NAME, description="Collection name to search in"
    )
    top_k: int = Field(20, description="Number of results to return", ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Metadata filters (optional)"
    )


class RetrieveRequest(BaseModel):
    """Request for advanced retrieval with hybrid search."""

    query: str = Field(..., description="Search query")
    collection_name: str = Field(
        TEST_COLLECTION_NAME, description="Collection name to search in"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Metadata filters (e.g., user_id, template_id)"
    )
    top_k: int = Field(20, description="Number of results to return", ge=1, le=100)
    enable_reranking: bool = Field(
        True, description="Use BGE reranking (Phase 4 - enabled by default)"
    )
    enable_keyword_search: bool = Field(
        True, description="Include BM25 keyword search (Phase 2 - enabled by default)"
    )
    enable_hyde: bool = Field(
        True, description="Use HyDE query generation (Phase 2 - enabled by default)"
    )
    score_threshold: Optional[float] = Field(
        None,
        description="Minimum relevance score filter (use negative values for BGE reranker, e.g., -5.0)"
    )
    return_full_chunks: bool = Field(
        True, description="Return complete vs truncated content"
    )
    deduplicate: bool = Field(True, description="Remove duplicate chunks")


class SearchResponse(BaseModel):
    """Search/retrieval response."""

    success: bool
    query: str
    queries_generated: Optional[Dict[str, str]] = None
    chunks_count: int
    retrieval_time_ms: float
    chunks: List[Dict[str, Any]]
    sources: List[Dict[str, Any]] = []
    stats: Dict[str, Any]
    errors: List[str] = []


class CollectionInfoResponse(BaseModel):
    """Collection information response."""

    name: str
    vectors_count: int
    status: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG client on startup."""
    global rag_client
    try:
        config = RAGConfig.from_env()
        rag_client = RAGClient(config)
        print("✓ RAG Client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize RAG Client: {e}")
        rag_client = None


# Health Check Endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {
        "rag_client": "initialized" if rag_client else "not_initialized",
        "config": "loaded" if os.getenv("QDRANT_URL") else "missing",
        "embeddings": "configured" if os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") else "missing",
        "vector_db": "configured" if os.getenv("QDRANT_URL") else "missing",
    }

    all_healthy = all(v in ["initialized", "loaded", "configured"] for v in components.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        message="insta_rag Testing API is running",
        components=components,
    )


# Configuration Testing Endpoints
@app.get("/api/v1/test/config", response_model=ConfigResponse)
async def test_config():
    """Test configuration loading and validation."""
    try:
        config = RAGConfig.from_env()
        config.validate()

        return ConfigResponse(
            valid=True,
            config=config.to_dict(),
            errors=[],
        )
    except ConfigurationError as e:
        return ConfigResponse(
            valid=False,
            config={},
            errors=[str(e)],
        )
    except Exception as e:
        return ConfigResponse(
            valid=False,
            config={},
            errors=[f"Unexpected error: {str(e)}"],
        )


# Chunking Testing Endpoints
@app.post("/api/v1/test/chunking", response_model=ChunkResponse)
async def test_chunking(request: TextChunkRequest):
    """Test semantic chunking functionality."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Create chunker
        chunker = SemanticChunker(
            embedder=rag_client.embedder,
            max_chunk_size=request.max_chunk_size,
            overlap_percentage=request.overlap_percentage,
        )

        # Chunk the text
        metadata = {
            "document_id": "test_doc",
            "source": "api_test",
        }
        chunks = chunker.chunk(request.text, metadata)

        # Convert chunks to dict
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "token_count": chunk.metadata.token_count,
                "char_count": chunk.metadata.char_count,
                "chunking_method": chunk.metadata.chunking_method,
            })

        return ChunkResponse(
            success=True,
            chunks_count=len(chunks),
            chunks=chunks_data,
            errors=[],
        )

    except ChunkingError as e:
        return ChunkResponse(
            success=False,
            chunks_count=0,
            chunks=[],
            errors=[f"Chunking error: {str(e)}"],
        )
    except Exception as e:
        return ChunkResponse(
            success=False,
            chunks_count=0,
            chunks=[],
            errors=[f"Unexpected error: {str(e)}"],
        )


@app.get("/api/v1/test/chunking/utils")
async def test_chunking_utils():
    """Test chunking utility functions."""
    test_text = "This is a test sentence. This is another test sentence."

    try:
        token_count = count_tokens_accurate(test_text)
        is_valid = validate_chunk_quality(test_text)

        return JSONResponse(content={
            "success": True,
            "test_text": test_text,
            "token_count": token_count,
            "is_valid_quality": is_valid,
            "errors": [],
        })

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "errors": [str(e)],
        })


# Embedding Testing Endpoints
@app.post("/api/v1/test/embedding", response_model=EmbeddingResponse)
async def test_embedding(request: EmbeddingRequest):
    """Test embedding generation."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Generate embeddings
        embeddings = rag_client.embedder.embed(request.texts)

        return EmbeddingResponse(
            success=True,
            embeddings_count=len(embeddings),
            dimensions=rag_client.embedder.get_dimensions(),
            sample_embedding=embeddings[0][:10] if embeddings else None,  # First 10 dims
            errors=[],
        )

    except EmbeddingError as e:
        return EmbeddingResponse(
            success=False,
            embeddings_count=0,
            dimensions=0,
            errors=[f"Embedding error: {str(e)}"],
        )
    except Exception as e:
        return EmbeddingResponse(
            success=False,
            embeddings_count=0,
            dimensions=0,
            errors=[f"Unexpected error: {str(e)}"],
        )


# Vector Database Testing Endpoints
@app.get("/api/v1/test/vectordb/collections")
async def test_vectordb_collections():
    """Test vector database collection listing."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        collections = rag_client.list_collections()

        return JSONResponse(content={
            "success": True,
            "collections": collections,
            "count": len(collections),
            "errors": [],
        })

    except VectorDBError as e:
        return JSONResponse(content={
            "success": False,
            "collections": [],
            "count": 0,
            "errors": [f"VectorDB error: {str(e)}"],
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "collections": [],
            "count": 0,
            "errors": [f"Unexpected error: {str(e)}"],
        })


@app.get("/api/v1/test/vectordb/collection/{collection_name}", response_model=CollectionInfoResponse)
async def test_vectordb_collection_info(collection_name: str = TEST_COLLECTION_NAME):
    """Test getting collection information."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        info = rag_client.get_collection_info(collection_name)

        return CollectionInfoResponse(
            name=info["name"],
            vectors_count=info["vectors_count"],
            status=info["status"],
        )

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# PDF Processing Testing Endpoints
@app.post("/api/v1/test/pdf/upload")
async def test_pdf_upload(file: UploadFile = File(...)):
    """Test PDF text extraction."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Extract text
            text = extract_text_from_pdf(tmp_path)
            token_count = count_tokens_accurate(text)

            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "text_length": len(text),
                "token_count": token_count,
                "text_preview": text[:200],
                "errors": [],
            })

        finally:
            # Clean up temp file
            Path(tmp_path).unlink()

    except PDFEncryptedError as e:
        return JSONResponse(content={
            "success": False,
            "errors": [f"PDF is encrypted: {str(e)}"],
        })
    except PDFEmptyError as e:
        return JSONResponse(content={
            "success": False,
            "errors": [f"PDF is empty: {str(e)}"],
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "errors": [f"Unexpected error: {str(e)}"],
        })


# Complete Document Processing Test
@app.post("/api/v1/test/documents/add", response_model=AddDocumentResponse)
async def test_add_document(request: AddDocumentRequest):
    """Test complete document processing pipeline."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Create document input
        doc = DocumentInput.from_text(
            text=request.text or "Default test text for document processing.",
            metadata=request.metadata,
        )

        # Process document
        response = rag_client.add_documents(
            documents=[doc],
            collection_name=request.collection_name,
            metadata={"test_run": True},
        )

        return AddDocumentResponse(
            success=response.success,
            documents_processed=response.documents_processed,
            total_chunks=response.total_chunks,
            processing_time_ms=response.processing_stats.total_time_ms,
            errors=response.errors,
        )

    except Exception as e:
        return AddDocumentResponse(
            success=False,
            documents_processed=0,
            total_chunks=0,
            processing_time_ms=0,
            errors=[f"Unexpected error: {str(e)}"],
        )


@app.post("/api/v1/test/documents/add-file")
async def test_add_document_file(
    file: UploadFile = File(...),
    collection_name: str = TEST_COLLECTION_NAME,
):
    """Test complete document processing pipeline with file upload."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Create document input
            doc = DocumentInput.from_file(
                file_path=tmp_path,
                metadata={"filename": file.filename},
            )

            # Process document
            response = rag_client.add_documents(
                documents=[doc],
                collection_name=collection_name,
                metadata={"test_run": True, "source": "file_upload"},
            )

            return JSONResponse(content={
                "success": response.success,
                "filename": file.filename,
                "documents_processed": response.documents_processed,
                "total_chunks": response.total_chunks,
                "processing_time_ms": response.processing_stats.total_time_ms,
                "chunking_time_ms": response.processing_stats.chunking_time_ms,
                "embedding_time_ms": response.processing_stats.embedding_time_ms,
                "upload_time_ms": response.processing_stats.upload_time_ms,
                "errors": response.errors,
            })

        finally:
            # Clean up temp file
            Path(tmp_path).unlink()

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "errors": [f"Unexpected error: {str(e)}"],
        })


# Advanced Retrieval Endpoint (Phase 2 - HyDE + BM25)
@app.post("/api/v1/retrieve", response_model=SearchResponse)
async def retrieve_documents(request: RetrieveRequest):
    """
    Advanced retrieval with hybrid search (Phase 4 - HyDE + BM25 + Reranking).

    Phase 4 Features (ENABLED BY DEFAULT):
    - HyDE query generation using Azure OpenAI
    - Dual vector search (standard + HyDE queries)
    - BM25 keyword search for exact term matching
    - BGE reranking (BAAI/bge-reranker-v2-m3) for improved relevance
    - Smart deduplication across all results
    - MongoDB content fetching (if enabled)
    - Comprehensive performance stats

    Note on score_threshold:
    - BGE reranker produces negative scores (higher = more relevant)
    - Use negative thresholds like -5.0 for filtering
    - Set to null/None for no filtering (recommended)

    This endpoint provides more advanced retrieval than /api/v1/search,
    with hybrid semantic + keyword search + reranking for better results.
    """
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Call the new retrieve() method
        response = rag_client.retrieve(
            query=request.query,
            collection_name=request.collection_name,
            filters=request.filters,
            top_k=request.top_k,
            enable_reranking=request.enable_reranking,
            enable_keyword_search=request.enable_keyword_search,
            enable_hyde=request.enable_hyde,
            score_threshold=request.score_threshold,
            return_full_chunks=request.return_full_chunks,
            deduplicate=request.deduplicate,
        )

        # Convert to SearchResponse format
        chunks_data = [chunk.to_dict() for chunk in response.chunks]

        return SearchResponse(
            success=response.success,
            query=response.query_original,
            queries_generated=response.queries_generated,
            chunks_count=len(response.chunks),
            retrieval_time_ms=response.retrieval_stats.total_time_ms,
            chunks=chunks_data,
            sources=[source.to_dict() for source in response.sources],
            stats=response.retrieval_stats.to_dict(),
            errors=response.errors,
        )

    except Exception as e:
        return SearchResponse(
            success=False,
            query=request.query,
            chunks_count=0,
            retrieval_time_ms=0,
            chunks=[],
            stats={},
            errors=[f"Retrieve error: {str(e)}"],
        )


# Search/Retrieval Testing Endpoints
@app.post("/api/v1/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for relevant documents using vector similarity.

    This endpoint performs semantic search using:
    1. Query embedding generation
    2. Vector similarity search in Qdrant
    3. Content retrieval from MongoDB (if enabled)
    4. Results ranking and scoring
    """
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Perform search
        response = rag_client.search(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
            filters=request.filters,
        )

        # Convert chunks to dicts
        chunks_data = [chunk.to_dict() for chunk in response.chunks]

        return SearchResponse(
            success=response.success,
            query=response.query_original,
            chunks_count=len(response.chunks),
            retrieval_time_ms=response.retrieval_stats.total_time_ms,
            chunks=chunks_data,
            sources=[source.to_dict() for source in response.sources],
            stats=response.retrieval_stats.to_dict(),
            errors=response.errors,
        )

    except Exception as e:
        return SearchResponse(
            success=False,
            query=request.query,
            chunks_count=0,
            retrieval_time_ms=0,
            chunks=[],
            stats={},
            errors=[f"Search error: {str(e)}"],
        )


@app.post("/api/v1/test/retrieval/search", response_model=SearchResponse)
async def test_retrieval_search(request: SearchRequest):
    """Test retrieval/search functionality (alias for /api/v1/search)."""
    return await search_documents(request)


@app.get("/api/v1/test/retrieval/collection/{collection_name}/sample")
async def get_collection_sample(
    collection_name: str,
    limit: int = 5,
):
    """Get sample chunks from a collection for testing.

    This is useful for:
    - Verifying data was uploaded correctly
    - Inspecting chunk content and metadata
    - Testing before performing searches
    """
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Get collection info first
        info = rag_client.get_collection_info(collection_name)

        # Perform a dummy search to get sample results
        # Use a generic query
        sample_query = "sample"
        response = rag_client.search(
            query=sample_query,
            collection_name=collection_name,
            top_k=limit,
        )

        return JSONResponse(content={
            "success": True,
            "collection_name": collection_name,
            "total_vectors": info["vectors_count"],
            "sample_count": len(response.chunks),
            "samples": [chunk.to_dict() for chunk in response.chunks],
        })

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e),
        }, status_code=404)


# Integration Test Endpoint
@app.get("/api/v1/test/integration")
async def test_integration():
    """Run integration test of all components."""
    results = {
        "config": {"status": "unknown"},
        "chunking": {"status": "unknown"},
        "embedding": {"status": "unknown"},
        "vectordb": {"status": "unknown"},
        "document_processing": {"status": "unknown"},
    }

    # Test 1: Configuration
    try:
        config = RAGConfig.from_env()
        config.validate()
        results["config"] = {"status": "passed", "details": "Configuration loaded and validated"}
    except Exception as e:
        results["config"] = {"status": "failed", "error": str(e)}

    # Test 2: Chunking
    try:
        if rag_client:
            test_text = "This is a test. This is another sentence. And one more."
            chunker = SemanticChunker(
                embedder=rag_client.embedder,
                max_chunk_size=1000,
                overlap_percentage=0.2,
            )
            chunks = chunker.chunk(test_text, {"document_id": "test"})
            results["chunking"] = {"status": "passed", "chunks_created": len(chunks)}
        else:
            results["chunking"] = {"status": "failed", "error": "RAG client not initialized"}
    except Exception as e:
        results["chunking"] = {"status": "failed", "error": str(e)}

    # Test 3: Embedding
    try:
        if rag_client:
            embeddings = rag_client.embedder.embed(["test sentence"])
            results["embedding"] = {
                "status": "passed",
                "dimensions": len(embeddings[0]) if embeddings else 0,
            }
        else:
            results["embedding"] = {"status": "failed", "error": "RAG client not initialized"}
    except Exception as e:
        results["embedding"] = {"status": "failed", "error": str(e)}

    # Test 4: Vector DB
    try:
        if rag_client:
            collections = rag_client.list_collections()
            results["vectordb"] = {"status": "passed", "collections_count": len(collections)}
        else:
            results["vectordb"] = {"status": "failed", "error": "RAG client not initialized"}
    except Exception as e:
        results["vectordb"] = {"status": "failed", "error": str(e)}

    # Test 5: Complete Document Processing
    try:
        if rag_client:
            doc = DocumentInput.from_text("Integration test document content.")
            response = rag_client.add_documents([doc], TEST_COLLECTION_NAME)
            results["document_processing"] = {
                "status": "passed" if response.success else "failed",
                "chunks": response.total_chunks,
                "processing_time_ms": response.processing_stats.total_time_ms,
            }
        else:
            results["document_processing"] = {"status": "failed", "error": "RAG client not initialized"}
    except Exception as e:
        results["document_processing"] = {"status": "failed", "error": str(e)}

    # Overall status
    all_passed = all(r["status"] == "passed" for r in results.values())

    return JSONResponse(content={
        "overall_status": "passed" if all_passed else "failed",
        "tests": results,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
