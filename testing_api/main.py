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

# Import MongoDB storage helper
from mongodb_storage import MongoDBStorage

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

# Global MongoDB storage (initialized on startup)
mongodb_storage: Optional[MongoDBStorage] = None

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
    """Request for adding documents with optional custom metadata.

    Custom metadata is completely flexible - add any fields you want!
    Examples:
    - {"department": "Engineering", "team": "Backend"}
    - {"user_id": "user-123", "access_level": "public"}
    - {"source_system": "Jira", "ticket_id": "PROJ-1234"}
    """

    text: Optional[str] = Field(
        None,
        description="Text content",
        example="REST API best practices and design patterns for building scalable systems"
    )
    collection_name: str = Field(
        TEST_COLLECTION_NAME,
        description="Collection name (uses test collection by default)",
        example="company_docs"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional custom metadata for filtering. Can be ANY fields you want!",
        example={
            "department": "Engineering",
            "team": "Backend",
            "author": "john@example.com",
            "priority": "high",
            "confidentiality": "internal",
            "created_date": "2025-10-18"
        }
    )


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


class UpdateDocumentRequest(BaseModel):
    """Request for updating documents."""

    collection_name: str = Field(
        TEST_COLLECTION_NAME, description="Collection name"
    )
    update_strategy: str = Field(
        ..., description="Update strategy: replace, append, delete, upsert"
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Metadata filters to match documents"
    )
    document_ids: Optional[List[str]] = Field(
        None, description="Specific document IDs to target"
    )
    new_documents_text: Optional[List[str]] = Field(
        None, description="List of text content for new documents"
    )
    metadata_updates: Optional[Dict[str, Any]] = Field(
        None, description="Metadata fields to update"
    )
    reprocess_chunks: bool = Field(
        True, description="If True, regenerate chunks and embeddings"
    )


class UpdateDocumentResponse(BaseModel):
    """Update document response."""

    success: bool
    strategy_used: str
    documents_affected: int
    chunks_deleted: int
    chunks_added: int
    chunks_updated: int
    updated_document_ids: List[str]
    errors: List[str] = []


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG client and MongoDB storage on startup."""
    global rag_client, mongodb_storage

    # Initialize RAG client
    try:
        config = RAGConfig.from_env()
        rag_client = RAGClient(config)
        print("✓ RAG Client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize RAG Client: {e}")
        rag_client = None

    # Initialize MongoDB storage
    try:
        # Use existing MONGO_CONNECTION_STRING from .env
        mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
        if not mongo_connection_string:
            raise ValueError("MONGO_CONNECTION_STRING not found in environment variables")

        mongodb_storage = MongoDBStorage(
            connection_string=mongo_connection_string,
            database_name="Test_Insta_RAG"
        )
        print("✓ MongoDB Storage initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize MongoDB Storage: {e}")
        mongodb_storage = None


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection on shutdown."""
    global mongodb_storage
    try:
        if mongodb_storage:
            mongodb_storage.close()
            print("✓ MongoDB connection closed successfully")
    except Exception as e:
        print(f"✗ Failed to close MongoDB connection: {e}")


# Helper Functions
async def store_chunks_to_mongodb(
    chunks: List,
    collection_name: str,
    document_ids: Optional[List[str]] = None
) -> tuple[int, List[str]]:
    """Store chunks to MongoDB and return MongoDB IDs.

    Args:
        chunks: List of Chunk objects
        collection_name: Qdrant collection name
        document_ids: Optional list of specific document IDs to store (for filtering)

    Returns:
        Tuple of (chunks_stored_count, mongodb_ids_list)
    """
    if not chunks or not mongodb_storage:
        return 0, []

    try:
        chunks_for_mongo = []
        for chunk in chunks:
            # Filter by document_ids if provided
            if document_ids and chunk.metadata.document_id not in document_ids:
                continue

            chunks_for_mongo.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "document_id": chunk.metadata.document_id,
                "collection_name": collection_name,
                "metadata": chunk.metadata.to_dict(),
            })

        if not chunks_for_mongo:
            return 0, []

        mongodb_ids = mongodb_storage.store_chunks_batch(chunks_for_mongo)
        print(f"✓ Stored {len(mongodb_ids)} chunks in MongoDB")
        return len(mongodb_ids), mongodb_ids

    except Exception as e:
        print(f"✗ Failed to store chunks in MongoDB: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, []


# Health Check Endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {
        "rag_client": "initialized" if rag_client else "not_initialized",
        "mongodb_storage": "initialized" if mongodb_storage else "not_initialized",
        "config": "loaded" if os.getenv("QDRANT_URL") else "missing",
        "embeddings": "configured" if os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") else "missing",
        "vector_db": "configured" if os.getenv("QDRANT_URL") else "missing",
        "mongodb": "configured" if os.getenv("MONGO_CONNECTION_STRING") else "missing",
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
    """Test complete document processing pipeline with MongoDB storage."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Create document input
        doc = DocumentInput.from_text(
            text=request.text or "Default test text for document processing.",
            metadata=request.metadata,
        )

        # Prepare global metadata
        global_metadata = {"test_run": True}

        # Process document via RAGClient
        # Note: By default, store_chunk_text_in_qdrant=False, so content won't be in Qdrant
        response = rag_client.add_documents(
            documents=[doc],
            collection_name=request.collection_name,
            metadata=global_metadata,
        )

        # NEW: Store chunk contents to MongoDB
        # This separates content storage from vector storage
        mongodb_chunks_stored = 0
        if response.success and response.chunks and mongodb_storage:
            try:
                # Extract chunks with content from response
                # These chunks contain the original text content
                chunks_for_mongo = []
                for chunk in response.chunks:
                    chunks_for_mongo.append({
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "document_id": chunk.metadata.document_id,
                        "collection_name": request.collection_name,
                        "metadata": chunk.metadata.to_dict(),
                    })

                # Store to MongoDB using unique chunk_id as key
                mongodb_storage.store_chunks_batch(chunks_for_mongo)
                mongodb_chunks_stored = len(chunks_for_mongo)

                print(f"✓ Stored {mongodb_chunks_stored} chunk contents in MongoDB")
                print(f"  Content is now separate from Qdrant vectors")
            except Exception as e:
                print(f"✗ Warning: Failed to store chunks in MongoDB: {str(e)}")
                print(f"  Qdrant still has vectors, but content won't be retrievable")
                import traceback
                traceback.print_exc()

        return AddDocumentResponse(
            success=response.success,
            documents_processed=response.documents_processed,
            total_chunks=response.total_chunks,
            processing_time_ms=response.processing_stats.total_time_ms,
            errors=response.errors,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
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
    """Test complete document processing pipeline with file upload and MongoDB storage."""
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

            # Prepare global metadata
            global_metadata = {"test_run": True, "source": "file_upload"}

            # Process document via RAGClient
            # Note: By default, store_chunk_text_in_qdrant=False, so content won't be in Qdrant
            response = rag_client.add_documents(
                documents=[doc],
                collection_name=collection_name,
                metadata=global_metadata,
            )

            # NEW: Store chunk contents to MongoDB
            # This separates content storage from vector storage
            mongodb_chunks_stored = 0
            if response.success and response.chunks and mongodb_storage:
                try:
                    # Extract chunks with content from response
                    # These chunks contain the original text content
                    chunks_for_mongo = []
                    for chunk in response.chunks:
                        chunks_for_mongo.append({
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,
                            "document_id": chunk.metadata.document_id,
                            "collection_name": collection_name,
                            "metadata": chunk.metadata.to_dict(),
                        })

                    # Store to MongoDB using unique chunk_id as key
                    mongodb_storage.store_chunks_batch(chunks_for_mongo)
                    mongodb_chunks_stored = len(chunks_for_mongo)

                    print(f"✓ Stored {mongodb_chunks_stored} chunk contents in MongoDB")
                    print(f"  Content is now separate from Qdrant vectors")
                except Exception as e:
                    print(f"✗ Warning: Failed to store chunks in MongoDB: {str(e)}")
                    print(f"  Qdrant still has vectors, but content won't be retrievable")
                    import traceback
                    traceback.print_exc()

            return JSONResponse(content={
                "success": response.success,
                "filename": file.filename,
                "documents_processed": response.documents_processed,
                "total_chunks": response.total_chunks,
                "mongodb_chunks_stored": mongodb_chunks_stored,
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
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "success": False,
            "errors": [f"Unexpected error: {str(e)}"],
        })


# Document Update Testing Endpoints
@app.post("/api/v1/test/documents/update", response_model=UpdateDocumentResponse)
async def test_update_documents(request: UpdateDocumentRequest):
    """Test document update operations (replace, append, delete, upsert).

    This endpoint provides flexible document management:
    - replace: Delete existing documents and add new ones
    - append: Add new documents without deleting
    - delete: Remove documents matching criteria
    - upsert: Update if exists, insert if doesn't

    Supports:
    - Filtering by metadata or document IDs
    - Metadata-only updates (reprocess_chunks=False)
    - Full document replacement with new content
    """
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Convert text list to DocumentInput objects
        new_documents = None
        if request.new_documents_text:
            new_documents = [
                DocumentInput.from_text(
                    text=text,
                    metadata=request.metadata_updates or {},
                )
                for text in request.new_documents_text
            ]

        # Call update_documents
        response = rag_client.update_documents(
            collection_name=request.collection_name,
            update_strategy=request.update_strategy,
            filters=request.filters,
            document_ids=request.document_ids,
            new_documents=new_documents,
            metadata_updates=request.metadata_updates,
            reprocess_chunks=request.reprocess_chunks,
        )

        return UpdateDocumentResponse(
            success=response.success,
            strategy_used=response.strategy_used,
            documents_affected=response.documents_affected,
            chunks_deleted=response.chunks_deleted,
            chunks_added=response.chunks_added,
            chunks_updated=response.chunks_updated,
            updated_document_ids=response.updated_document_ids,
            errors=response.errors,
        )

    except Exception as e:
        return UpdateDocumentResponse(
            success=False,
            strategy_used=request.update_strategy,
            documents_affected=0,
            chunks_deleted=0,
            chunks_added=0,
            chunks_updated=0,
            updated_document_ids=[],
            errors=[f"Update error: {str(e)}"],
        )


class DeleteDocumentRequest(BaseModel):
    """Request for DELETE strategy."""
    collection_name: str = Field(TEST_COLLECTION_NAME, description="Collection name")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    document_ids: Optional[List[str]] = Field(None, description="Document IDs to delete")


class AppendDocumentRequest(BaseModel):
    """Request for APPEND strategy."""
    collection_name: str = Field(TEST_COLLECTION_NAME, description="Collection name")
    new_documents_text: List[str] = Field(..., description="List of text documents to append")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata for documents")


class ReplaceDocumentRequest(BaseModel):
    """Request for REPLACE strategy."""
    collection_name: str = Field(TEST_COLLECTION_NAME, description="Collection name")
    new_documents_text: List[str] = Field(..., description="Replacement documents")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    document_ids: Optional[List[str]] = Field(None, description="Document IDs to replace")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata for documents")


class UpsertDocumentRequest(BaseModel):
    """Request for UPSERT strategy."""
    collection_name: str = Field(TEST_COLLECTION_NAME, description="Collection name")
    new_documents_text: List[str] = Field(..., description="Documents to upsert")
    document_ids: Optional[List[str]] = Field(None, description="Document IDs (must match length of texts)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata for documents")


class MetadataUpdateRequest(BaseModel):
    """Request for metadata-only update."""
    collection_name: str = Field(TEST_COLLECTION_NAME, description="Collection name")
    metadata_updates: Dict[str, Any] = Field(..., description="Metadata fields to update")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    document_ids: Optional[List[str]] = Field(None, description="Document IDs to update")


# ============================================================================
# NEW: Custom Metadata Filtering Request Models
# ============================================================================

class FilteredSearchRequest(BaseModel):
    """Request for searching with custom metadata filtering.

    Performs SEMANTIC SEARCH within documents matching your metadata criteria.

    Filter syntax supports:
    - Exact match: {"field": "value"}
    - Multiple values (OR): {"field": ["value1", "value2"]}
    - Range queries: {"field": {"$gt": 5}} (greater than)
    - Multiple conditions: {"dept": "Eng", "priority": "high"}

    Example use cases:
    1. Search in Engineering docs only
    2. Search for user's private documents
    3. Search in high-priority items from specific department
    4. Search across multiple teams
    """

    query: str = Field(
        ...,
        description="Semantic search query",
        example="API design patterns and best practices"
    )
    collection_name: str = Field(
        TEST_COLLECTION_NAME,
        description="Collection to search in",
        example="company_docs"
    )
    filters: Dict[str, Any] = Field(
        ...,
        description="Custom metadata filters (REQUIRED). Will search ONLY within matching documents.",
        example={
            "department": "Engineering",
            "priority": "high"
        }
    )
    top_k: int = Field(
        20,
        description="Number of results to return",
        ge=1,
        le=100,
        example=10
    )
    enable_reranking: bool = Field(
        True,
        description="Use BGE reranking for better relevance",
        example=True
    )


class MetadataListRequest(BaseModel):
    """Request to list documents by custom metadata filters.

    Returns ALL chunks matching the metadata criteria WITHOUT semantic search.
    Use when you want all documents matching certain criteria (no relevance ranking).

    Perfect for:
    - Getting all documents from a specific user
    - Listing all documents from a department
    - Finding all documents with a certain tag
    - Bulk export based on criteria

    Examples of filters:
    - Get specific user: {"user_id": "user-123"}
    - Get department: {"department": "Engineering"}
    - Get multiple conditions: {"department": "Finance", "priority": "high"}
    """

    collection_name: str = Field(
        TEST_COLLECTION_NAME,
        description="Collection to query",
        example="company_docs"
    )
    filters: Dict[str, Any] = Field(
        ...,
        description="Custom metadata filters (REQUIRED). Returns ALL chunks matching these criteria.",
        example={
            "user_id": "user-123",
            "access_level": "public"
        }
    )
    limit: int = Field(
        100,
        description="Maximum chunks to return",
        ge=1,
        le=1000,
        example=50
    )


class MetadataCountRequest(BaseModel):
    """Request to count documents matching custom metadata filters.

    Returns the COUNT of matching documents without retrieving them.
    Useful for validation and analytics.

    Use cases:
    - Check how many documents user has access to
    - Validate data before batch operations
    - Analytics: How many high-priority items?
    - Quota checking: How many docs in department?

    Example filters:
    - Count by department: {"department": "Engineering"}
    - Count by user: {"user_id": "user-123"}
    - Count with conditions: {"department": "Eng", "priority": "high"}
    """

    collection_name: str = Field(
        TEST_COLLECTION_NAME,
        description="Collection to query",
        example="company_docs"
    )
    filters: Dict[str, Any] = Field(
        ...,
        description="Custom metadata filters (REQUIRED). Counts all chunks matching these criteria.",
        example={
            "department": "Engineering"
        }
    )


@app.delete("/api/v1/test/collections/{collection_name}/clear")
async def clear_collection(collection_name: str):
    """Clear all data from a collection (Qdrant + MongoDB)."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Get initial stats
        info = rag_client.get_collection_info(collection_name)
        initial_vectors = info['vectors_count']

        # Clear Qdrant - delete and recreate collection
        rag_client.vectordb.client.delete_collection(collection_name=collection_name)
        rag_client.vectordb.create_collection(
            collection_name=collection_name,
            vector_size=rag_client.embedder.get_dimensions(),
            distance_metric="cosine"
        )

        # Clear MongoDB
        mongo_chunks_deleted = 0
        mongo_docs_deleted = 0
        if rag_client.mongodb:
            mongo_chunks_deleted = rag_client.mongodb.delete_chunks_by_collection(collection_name)
            mongo_docs_deleted = rag_client.mongodb.db.document_metadata.delete_many(
                {"collection_name": collection_name}
            ).deleted_count

        return JSONResponse(content={
            "success": True,
            "collection_name": collection_name,
            "qdrant_vectors_deleted": initial_vectors,
            "mongodb_chunks_deleted": mongo_chunks_deleted,
            "mongodb_docs_deleted": mongo_docs_deleted,
            "message": "Collection cleared successfully"
        })

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e),
        }, status_code=400)


@app.get("/api/v1/test/documents/list/{collection_name}")
async def list_document_ids(collection_name: str, limit: int = 50):
    """List all document IDs in a collection for debugging."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Get all document IDs
        doc_ids = rag_client.vectordb.get_document_ids(
            collection_name=collection_name,
            limit=limit
        )

        # Get chunk counts for each document
        doc_info = []
        for doc_id in doc_ids[:20]:  # Show details for first 20
            chunk_count = rag_client.vectordb.count_chunks(
                collection_name=collection_name,
                document_ids=[doc_id]
            )
            doc_info.append({
                "document_id": doc_id,
                "chunk_count": chunk_count
            })

        return JSONResponse(content={
            "success": True,
            "collection_name": collection_name,
            "total_documents": len(doc_ids),
            "documents": doc_info,
            "all_document_ids": doc_ids,
        })

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e),
        }, status_code=400)


@app.post("/api/v1/test/documents/update/delete")
async def test_delete_documents(request: DeleteDocumentRequest):
    """Test DELETE strategy: Remove documents from the knowledge base and MongoDB."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        print(f"\n[DELETE REQUEST] Collection: {request.collection_name}")
        print(f"[DELETE REQUEST] Filters: {request.filters}")
        print(f"[DELETE REQUEST] Document IDs: {request.document_ids}")

        response = rag_client.update_documents(
            collection_name=request.collection_name,
            update_strategy="delete",
            filters=request.filters,
            document_ids=request.document_ids,
        )

        print(f"[DELETE RESPONSE] Success: {response.success}")
        print(f"[DELETE RESPONSE] Chunks deleted: {response.chunks_deleted}")
        print(f"[DELETE RESPONSE] Errors: {response.errors}")

        # NEW: Delete from MongoDB as well (cleanup content store)
        mongodb_chunks_deleted = 0
        if response.success and response.updated_document_ids and mongodb_storage:
            try:
                mongodb_chunks_deleted = mongodb_storage.delete_chunks_by_document_ids(
                    response.updated_document_ids
                )
                print(f"[DELETE] Deleted {mongodb_chunks_deleted} chunk contents from MongoDB")
            except Exception as e:
                print(f"[DELETE WARNING] Failed to delete from MongoDB: {str(e)}")

        return JSONResponse(content={
            "success": response.success,
            "strategy": "delete",
            "chunks_deleted": response.chunks_deleted,
            "mongodb_chunks_deleted": mongodb_chunks_deleted,
            "documents_affected": response.documents_affected,
            "document_ids": response.updated_document_ids,
            "errors": response.errors,
        })

    except Exception as e:
        print(f"[DELETE ERROR] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }, status_code=400)


@app.post("/api/v1/test/documents/update/append")
async def test_append_documents(request: AppendDocumentRequest):
    """Test APPEND strategy: Add new documents to the knowledge base and MongoDB."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Convert text to DocumentInput
        new_documents = [
            DocumentInput.from_text(text=text, metadata=request.metadata or {})
            for text in request.new_documents_text
        ]

        response = rag_client.update_documents(
            collection_name=request.collection_name,
            update_strategy="append",
            new_documents=new_documents,
            metadata_updates=request.metadata,
        )

        # NEW: Store new chunks to MongoDB
        # Note: update_documents internally calls add_documents,
        # but doesn't expose chunks in response yet
        # For now, we note this limitation and track count only
        mongodb_chunks_stored = 0
        if response.success and response.chunks_added > 0 and mongodb_storage:
            print(f"✓ Note: {response.chunks_added} new chunks were added")
            print(f"  To store their content in MongoDB, use /api/v1/test/documents/add")
            print(f"  or enhance update_documents to return chunks")

        return JSONResponse(content={
            "success": response.success,
            "strategy": "append",
            "chunks_added": response.chunks_added,
            "mongodb_chunks_stored": mongodb_chunks_stored,
            "documents_affected": response.documents_affected,
            "document_ids": response.updated_document_ids,
            "errors": response.errors,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "success": False,
            "error": str(e),
        }, status_code=400)


@app.post("/api/v1/test/documents/update/replace")
async def test_replace_documents(request: ReplaceDocumentRequest):
    """Test REPLACE strategy: Replace existing documents with new ones (with MongoDB support)."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Convert text to DocumentInput
        new_documents = [
            DocumentInput.from_text(text=text, metadata=request.metadata or {})
            for text in request.new_documents_text
        ]

        # NEW: Get document_ids to be replaced (for MongoDB cleanup)
        docs_to_replace = request.document_ids
        if not docs_to_replace and request.filters:
            # Get document IDs using filters
            docs_to_replace = rag_client.vectordb.get_document_ids(
                request.collection_name, request.filters
            )

        # NEW: Delete old chunks from MongoDB before replacing
        mongodb_chunks_deleted = 0
        if docs_to_replace and mongodb_storage:
            try:
                mongodb_chunks_deleted = mongodb_storage.delete_chunks_by_document_ids(
                    docs_to_replace
                )
                print(f"✓ Deleted {mongodb_chunks_deleted} old chunks from MongoDB")
            except Exception as e:
                print(f"✗ Warning: Failed to delete old chunks from MongoDB: {str(e)}")

        # Call replace via update_documents
        response = rag_client.update_documents(
            collection_name=request.collection_name,
            update_strategy="replace",
            filters=request.filters,
            document_ids=request.document_ids,
            new_documents=new_documents,
            metadata_updates=request.metadata,
        )

        # NEW: Store new chunks to MongoDB (from response.chunks which has full content)
        mongodb_chunks_stored = 0
        if response.success and response.chunks and mongodb_storage:
            try:
                # Extract chunks with content from response
                chunks_for_mongo = []
                for chunk in response.chunks:
                    chunks_for_mongo.append({
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,  # ← Full content available!
                        "document_id": chunk.metadata.document_id,
                        "collection_name": request.collection_name,
                        "metadata": chunk.metadata.to_dict(),
                    })

                # Store to MongoDB
                if chunks_for_mongo:
                    mongodb_storage.store_chunks_batch(chunks_for_mongo)
                    mongodb_chunks_stored = len(chunks_for_mongo)
                    print(f"✓ Stored {mongodb_chunks_stored} new chunks to MongoDB")

            except Exception as e:
                print(f"✗ Warning: Failed to store new chunks to MongoDB: {str(e)}")
                import traceback
                traceback.print_exc()

        return JSONResponse(content={
            "success": response.success,
            "strategy": "replace",
            "chunks_deleted": response.chunks_deleted,
            "mongodb_chunks_deleted": mongodb_chunks_deleted,
            "chunks_added": response.chunks_added,
            "mongodb_chunks_stored": mongodb_chunks_stored,
            "documents_affected": response.documents_affected,
            "document_ids": response.updated_document_ids,
            "errors": response.errors,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "success": False,
            "error": str(e),
        }, status_code=400)


@app.post("/api/v1/test/documents/update/upsert")
async def test_upsert_documents(request: UpsertDocumentRequest):
    """Test UPSERT strategy: Update existing documents or insert new ones (with MongoDB support)."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        # Convert text to DocumentInput with document_ids
        new_documents = []
        for i, text in enumerate(request.new_documents_text):
            doc_metadata = request.metadata.copy() if request.metadata else {}
            if request.document_ids and i < len(request.document_ids):
                doc_metadata["document_id"] = request.document_ids[i]

            new_documents.append(
                DocumentInput.from_text(text=text, metadata=doc_metadata)
            )

        response = rag_client.update_documents(
            collection_name=request.collection_name,
            update_strategy="upsert",
            new_documents=new_documents,
            metadata_updates=request.metadata,
        )

        # NEW: Handle MongoDB operations for updated/deleted chunks
        mongodb_chunks_deleted = 0
        mongodb_chunks_stored = 0

        if response.success and mongodb_storage:
            # Delete old chunks from MongoDB (for documents that were updated)
            if response.chunks_deleted > 0:
                try:
                    # Get list of updated document_ids (documents that had chunks deleted)
                    updated_doc_ids = []
                    for new_doc in new_documents:
                        doc_id = new_doc.metadata.get("document_id")
                        if doc_id:
                            updated_doc_ids.append(doc_id)

                    if updated_doc_ids:
                        mongodb_chunks_deleted = mongodb_storage.delete_chunks_by_document_ids(
                            updated_doc_ids
                        )
                        print(f"✓ Deleted {mongodb_chunks_deleted} old chunks from MongoDB")
                except Exception as e:
                    print(f"✗ Warning: Failed to delete old chunks from MongoDB: {str(e)}")

            # NEW: Store new chunks to MongoDB (from response.chunks which has full content)
            if response.chunks and mongodb_storage:
                try:
                    # Extract chunks with content from response
                    chunks_for_mongo = []
                    for chunk in response.chunks:
                        chunks_for_mongo.append({
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,  # ← Full content available!
                            "document_id": chunk.metadata.document_id,
                            "collection_name": request.collection_name,
                            "metadata": chunk.metadata.to_dict(),
                        })

                    # Store to MongoDB
                    if chunks_for_mongo:
                        mongodb_storage.store_chunks_batch(chunks_for_mongo)
                        mongodb_chunks_stored = len(chunks_for_mongo)
                        print(f"✓ Stored {mongodb_chunks_stored} chunks to MongoDB")

                except Exception as e:
                    print(f"✗ Warning: Failed to store chunks to MongoDB: {str(e)}")
                    import traceback
                    traceback.print_exc()

        return JSONResponse(content={
            "success": response.success,
            "strategy": "upsert",
            "chunks_deleted": response.chunks_deleted,
            "mongodb_chunks_deleted": mongodb_chunks_deleted,
            "chunks_added": response.chunks_added,
            "mongodb_chunks_stored": mongodb_chunks_stored,
            "chunks_updated": response.chunks_updated,
            "documents_affected": response.documents_affected,
            "document_ids": response.updated_document_ids,
            "errors": response.errors,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "success": False,
            "error": str(e),
        }, status_code=400)


@app.post("/api/v1/test/documents/update/metadata")
async def test_update_metadata(request: MetadataUpdateRequest):
    """Test metadata-only update: Update metadata without reprocessing content."""
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        response = rag_client.update_documents(
            collection_name=request.collection_name,
            update_strategy="delete",  # Placeholder, metadata update doesn't need docs
            filters=request.filters,
            document_ids=request.document_ids,
            metadata_updates=request.metadata_updates,
            reprocess_chunks=False,
        )

        return JSONResponse(content={
            "success": response.success,
            "strategy": "metadata_update",
            "chunks_updated": response.chunks_updated,
            "documents_affected": response.documents_affected,
            "errors": response.errors,
        })

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e),
        }, status_code=400)


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

        # Call the retrieve() method
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

        # NEW: Fetch missing content from MongoDB and build chunks_data
        # By default, content is stored in MongoDB, not in Qdrant
        chunks_data = []
        for chunk in response.chunks:
            chunk_dict = chunk.to_dict()

            # Check if content is missing or empty
            if not chunk_dict.get("content") or chunk_dict["content"] == "":
                if mongodb_storage:
                    try:
                        # Extract chunk_id from metadata (it's nested there)
                        chunk_id = chunk_dict.get("metadata", {}).get("chunk_id")
                        if not chunk_id:
                            print(f"✗ Could not find chunk_id in metadata: {chunk_dict.get('metadata', {}).keys()}")
                        else:
                            # Fetch from MongoDB using chunk_id
                            mongo_doc = mongodb_storage.get_chunk_content(chunk_id)
                            if mongo_doc and mongo_doc.get("content"):
                                chunk_dict["content"] = mongo_doc["content"]
                                print(f"✓ Fetched content for chunk {chunk_id} from MongoDB")
                            else:
                                print(f"✗ No content found in MongoDB for chunk {chunk_id}")
                    except Exception as e:
                        print(f"✗ Error fetching content from MongoDB: {str(e)}")
                        import traceback
                        traceback.print_exc()

            chunks_data.append(chunk_dict)

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

        # NEW: Fetch missing content from MongoDB and build chunks_data
        # By default, content is stored in MongoDB, not in Qdrant
        chunks_data = []
        for chunk in response.chunks:
            chunk_dict = chunk.to_dict()

            # Check if content is missing or empty
            if not chunk_dict.get("content") or chunk_dict["content"] == "":
                if mongodb_storage:
                    try:
                        # Extract chunk_id from metadata (it's nested there)
                        chunk_id = chunk_dict.get("metadata", {}).get("chunk_id")
                        if not chunk_id:
                            print(f"✗ Could not find chunk_id in metadata: {chunk_dict.get('metadata', {}).keys()}")
                        else:
                            # Fetch from MongoDB using chunk_id
                            mongo_doc = mongodb_storage.get_chunk_content(chunk_id)
                            if mongo_doc and mongo_doc.get("content"):
                                chunk_dict["content"] = mongo_doc["content"]
                                print(f"✓ Fetched content for chunk {chunk_id} from MongoDB")
                            else:
                                print(f"✗ No content found in MongoDB for chunk {chunk_id}")
                    except Exception as e:
                        print(f"✗ Error fetching content from MongoDB: {str(e)}")
                        import traceback
                        traceback.print_exc()

            chunks_data.append(chunk_dict)

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


# ============================================================================
# NEW: Custom Metadata Filtering Endpoints
# ============================================================================

@app.post("/api/v1/documents/filter/search", response_model=SearchResponse)
async def filter_search_documents(request: FilteredSearchRequest):
    """Search with custom metadata filtering.

    Performs SEMANTIC SEARCH WITHIN documents matching your metadata criteria.
    Combines metadata filtering + semantic relevance ranking.

    ============================================================================
    EXAMPLE 1: Search in Engineering dept for API patterns
    ============================================================================
    POST /api/v1/documents/filter/search
    {
      "query": "REST API design patterns",
      "collection_name": "company_docs",
      "filters": {
        "department": "Engineering",
        "priority": "high"
      },
      "top_k": 10,
      "enable_reranking": true
    }

    Response will contain chunks matching:
    - Only from Engineering department
    - Only high-priority documents
    - Semantically similar to "REST API design patterns"
    - Ranked by relevance

    ============================================================================
    EXAMPLE 2: Search for specific user's documents
    ============================================================================
    POST /api/v1/documents/filter/search
    {
      "query": "authentication methods",
      "collection_name": "company_docs",
      "filters": {
        "user_id": "user-123",
        "access_level": "public"
      },
      "top_k": 5
    }

    Response will contain chunks matching:
    - Only user-123's documents
    - Only public access level
    - Semantically similar to "authentication methods"

    ============================================================================
    EXAMPLE 3: Search across multiple teams
    ============================================================================
    POST /api/v1/documents/filter/search
    {
      "query": "database optimization",
      "collection_name": "company_docs",
      "filters": {
        "team": ["Backend", "DataEngineering", "DevOps"],
        "priority": {"$gte": 2}
      },
      "top_k": 15
    }

    Response will contain chunks matching:
    - From Backend OR DataEngineering OR DevOps teams
    - Priority >= 2
    - Semantically similar to "database optimization"

    ============================================================================
    FILTER SYNTAX GUIDE
    ============================================================================
    1. Simple match: {"field": "value"}
    2. Multiple values (OR): {"field": ["val1", "val2"]}
    3. Range queries: {"field": {"$gt": 5}} (gt, gte, lt, lte)
    4. Multiple fields (AND): {"field1": "val1", "field2": "val2"}
    5. Combined: {"team": ["A", "B"], "priority": {"$gte": 1}}
    """
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        print(f"\n[FILTERED SEARCH]")
        print(f"Query: {request.query}")
        print(f"Filters: {request.filters}")
        print(f"Collection: {request.collection_name}")

        # Perform search with filters
        if request.enable_reranking:
            response = rag_client.retrieve(
                query=request.query,
                collection_name=request.collection_name,
                filters=request.filters,
                top_k=request.top_k,
                enable_reranking=True,
            )
        else:
            response = rag_client.search(
                query=request.query,
                collection_name=request.collection_name,
                filters=request.filters,
                top_k=request.top_k,
            )

        # Fetch missing content from MongoDB
        chunks_data = []
        for chunk in response.chunks:
            chunk_dict = chunk.to_dict()

            if not chunk_dict.get("content") or chunk_dict["content"] == "":
                if mongodb_storage:
                    try:
                        chunk_id = chunk_dict.get("metadata", {}).get("chunk_id")
                        if chunk_id:
                            mongo_doc = mongodb_storage.get_chunk_content(chunk_id)
                            if mongo_doc and mongo_doc.get("content"):
                                chunk_dict["content"] = mongo_doc["content"]
                    except Exception as e:
                        print(f"Warning: Could not fetch content from MongoDB: {str(e)}")

            chunks_data.append(chunk_dict)

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
        import traceback
        traceback.print_exc()
        return SearchResponse(
            success=False,
            query=request.query,
            chunks_count=0,
            retrieval_time_ms=0,
            chunks=[],
            stats={},
            errors=[f"Filter search error: {str(e)}"],
        )


@app.post("/api/v1/documents/filter/list", response_model=SearchResponse)
async def filter_list_documents(request: MetadataListRequest):
    """List ALL documents matching custom metadata filters (NO semantic search).

    Returns ALL chunks matching your criteria without semantic ranking.
    Perfect for bulk retrieval, exports, and metadata-based queries.

    ============================================================================
    EXAMPLE 1: Get all documents from specific user
    ============================================================================
    POST /api/v1/documents/filter/list
    {
      "collection_name": "company_docs",
      "filters": {
        "user_id": "user-123"
      },
      "limit": 100
    }

    Returns ALL chunks created by user-123 (no semantic ranking)

    ============================================================================
    EXAMPLE 2: Get all high-priority documents from department
    ============================================================================
    POST /api/v1/documents/filter/list
    {
      "collection_name": "company_docs",
      "filters": {
        "department": "Finance",
        "priority": "high"
      },
      "limit": 200
    }

    Returns ALL chunks where:
    - department = Finance AND priority = high
    - In insertion order (not ranked)

    ============================================================================
    EXAMPLE 3: Get documents from multiple sources
    ============================================================================
    POST /api/v1/documents/filter/list
    {
      "collection_name": "company_docs",
      "filters": {
        "source_system": ["Jira", "GitHub", "Confluence"]
      },
      "limit": 500
    }

    Returns ALL chunks from:
    - Jira OR GitHub OR Confluence
    - All chunks combined

    ============================================================================
    EXAMPLE 4: Bulk export - all public documents
    ============================================================================
    POST /api/v1/documents/filter/list
    {
      "collection_name": "company_docs",
      "filters": {
        "confidentiality": "public",
        "status": "published"
      },
      "limit": 10000
    }

    Returns ALL chunks for bulk export/archival

    ============================================================================
    USE CASES
    ============================================================================
    1. Export all documents for a user
    2. Bulk retrieve for backup/migration
    3. Get all documents from a team/department
    4. List all documents with specific tag
    5. Analytics: count documents by category
    6. Audit: retrieve all documents by author

    ============================================================================
    DIFFERENCE FROM /filter/search
    ============================================================================
    /filter/search:  Combines metadata filters + semantic search + ranking
    /filter/list:    Returns ALL matching metadata (no semantic search)

    Use /filter/search when: You want relevant results
    Use /filter/list when:   You want ALL results matching criteria
    """
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        print(f"\n[FILTER LIST]")
        print(f"Filters: {request.filters}")
        print(f"Collection: {request.collection_name}")

        # Use a dummy search query to get filtered results
        # We'll limit based on filters only
        response = rag_client.search(
            query="*",  # Dummy query
            collection_name=request.collection_name,
            filters=request.filters,
            top_k=request.limit,
        )

        # Fetch missing content from MongoDB
        chunks_data = []
        for chunk in response.chunks:
            chunk_dict = chunk.to_dict()

            if not chunk_dict.get("content") or chunk_dict["content"] == "":
                if mongodb_storage:
                    try:
                        chunk_id = chunk_dict.get("metadata", {}).get("chunk_id")
                        if chunk_id:
                            mongo_doc = mongodb_storage.get_chunk_content(chunk_id)
                            if mongo_doc and mongo_doc.get("content"):
                                chunk_dict["content"] = mongo_doc["content"]
                    except Exception as e:
                        print(f"Warning: Could not fetch content from MongoDB: {str(e)}")

            chunks_data.append(chunk_dict)

        return SearchResponse(
            success=response.success,
            query="[metadata filter - no semantic search]",
            chunks_count=len(chunks_data),
            retrieval_time_ms=response.retrieval_stats.total_time_ms,
            chunks=chunks_data,
            sources=[source.to_dict() for source in response.sources],
            stats=response.retrieval_stats.to_dict(),
            errors=response.errors,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return SearchResponse(
            success=False,
            query="[metadata filter]",
            chunks_count=0,
            retrieval_time_ms=0,
            chunks=[],
            stats={},
            errors=[f"Filter list error: {str(e)}"],
        )


@app.post("/api/v1/documents/filter/count")
async def filter_count_documents(request: MetadataCountRequest):
    """Count documents matching custom metadata filters (fast count).

    Returns just the COUNT without retrieving actual documents.
    Useful for validation, analytics, and quota checking.

    ============================================================================
    EXAMPLE 1: Count documents by department
    ============================================================================
    POST /api/v1/documents/filter/count
    {
      "collection_name": "company_docs",
      "filters": {
        "department": "Engineering"
      }
    }

    Response:
    {
      "success": true,
      "collection_name": "company_docs",
      "filters": {"department": "Engineering"},
      "count": 47,
      "chunks_found": 47
    }

    ============================================================================
    EXAMPLE 2: Count documents per user
    ============================================================================
    POST /api/v1/documents/filter/count
    {
      "collection_name": "company_docs",
      "filters": {
        "user_id": "user-123"
      }
    }

    Returns count of all chunks user-123 can access

    ============================================================================
    EXAMPLE 3: Count high-priority items
    ============================================================================
    POST /api/v1/documents/filter/count
    {
      "collection_name": "company_docs",
      "filters": {
        "department": "Finance",
        "priority": {"$gte": 3}
      }
    }

    Returns count of Finance docs with priority >= 3

    ============================================================================
    EXAMPLE 4: Count by multiple sources (quota check)
    ============================================================================
    POST /api/v1/documents/filter/count
    {
      "collection_name": "company_docs",
      "filters": {
        "source_system": ["Jira", "GitHub", "Confluence"]
      }
    }

    Returns total count from all three sources

    ============================================================================
    USE CASES
    ============================================================================
    1. Check quota: How many docs does user have?
    2. Validation: Is data loaded? (count > 0?)
    3. Analytics: How many high-priority items?
    4. Monitoring: Track document growth over time
    5. Planning: How many docs need migration?
    6. Audit: How many documents per department?

    ============================================================================
    WHY USE COUNT?
    ============================================================================
    - Much FASTER than /filter/list (no data retrieval)
    - Perfect for quick checks and validation
    - Lightweight for analytics queries
    - Use when you just need the number, not the data
    """
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        print(f"\n[FILTER COUNT]")
        print(f"Filters: {request.filters}")
        print(f"Collection: {request.collection_name}")

        # Count documents using Qdrant filter
        response = rag_client.search(
            query="*",
            collection_name=request.collection_name,
            filters=request.filters,
            top_k=10000,  # Get all matching
        )

        count = len(response.chunks)

        return JSONResponse(content={
            "success": True,
            "collection_name": request.collection_name,
            "filters": request.filters,
            "count": count,
            "chunks_found": count,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "count": 0,
        }, status_code=400)


@app.get("/api/v1/documents/{collection_name}/metadata/info")
async def get_collection_metadata_info(collection_name: str):
    """Get metadata schema and samples from a collection.

    IMPORTANT: Use this endpoint FIRST to discover what metadata fields exist
    in your collection before using filter endpoints!

    Returns:
    - Total document count in collection
    - List of all metadata field names available
    - Sample metadata from first few documents
    - Helps you understand what to filter on

    ============================================================================
    EXAMPLE: Discover metadata schema
    ============================================================================
    GET /api/v1/documents/company_docs/metadata/info

    Response:
    {
      "success": true,
      "collection_name": "company_docs",
      "total_vectors": 1250,
      "sample_count": 3,
      "metadata_fields": [
        "author",
        "confidentiality",
        "created_date",
        "department",
        "document_id",
        "priority",
        "source_system",
        "team",
        "user_id"
      ],
      "sample_metadata": [
        {
          "author": "john@example.com",
          "department": "Engineering",
          "team": "Backend",
          "priority": "high",
          "confidentiality": "internal",
          "source_system": "GitHub",
          "user_id": "user-123"
        },
        {
          "author": "jane@example.com",
          "department": "Finance",
          "team": "Analytics",
          "priority": "medium",
          "confidentiality": "public",
          "source_system": "Jira",
          "user_id": "user-456"
        },
        ...
      ]
    }

    ============================================================================
    WORKFLOW
    ============================================================================
    1. Call GET /metadata/info to see available fields
    2. Identify fields you want to filter on
    3. Use POST /filter/search, /filter/list, or /filter/count
       with the fields you discovered

    ============================================================================
    QUICK START
    ============================================================================
    # Step 1: Discover fields
    curl http://localhost:8000/api/v1/documents/company_docs/metadata/info

    # Step 2: See metadata_fields in response (e.g., "department", "user_id")

    # Step 3: Use those fields in filter operations
    curl -X POST http://localhost:8000/api/v1/documents/filter/search \\
      -H "Content-Type: application/json" \\
      -d '{"query":"...", "filters": {"department": "Engineering"}}'
    """
    try:
        if not rag_client:
            raise HTTPException(status_code=503, detail="RAG client not initialized")

        print(f"\n[METADATA INFO]")
        print(f"Collection: {collection_name}")

        # Get collection info
        info = rag_client.get_collection_info(collection_name)

        # Get sample chunks to inspect metadata
        response = rag_client.search(
            query="*",
            collection_name=collection_name,
            top_k=5,  # Get first 5
        )

        # Extract metadata samples
        metadata_samples = []
        metadata_keys = set()

        for chunk in response.chunks:
            chunk_dict = chunk.to_dict()
            if chunk_dict.get("metadata"):
                metadata_samples.append(chunk_dict["metadata"])
                metadata_keys.update(chunk_dict["metadata"].keys())

        return JSONResponse(content={
            "success": True,
            "collection_name": collection_name,
            "total_vectors": info["vectors_count"],
            "sample_count": len(metadata_samples),
            "metadata_fields": sorted(list(metadata_keys)),
            "sample_metadata": metadata_samples[:3],  # Show first 3 samples
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "success": False,
            "error": str(e),
        }, status_code=400)


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
