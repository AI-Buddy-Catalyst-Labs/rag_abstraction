"""Response models for RAG operations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .chunk import Chunk


@dataclass
class ProcessingStats:
    """Performance metrics for document processing.

    Attributes:
        total_tokens: Total tokens processed
        embedding_time_ms: Time for embedding generation
        chunking_time_ms: Time for chunking
        upload_time_ms: Time for Qdrant upload
        total_time_ms: Total processing time
        failed_chunks: Count of failed chunks
    """

    total_tokens: int = 0
    embedding_time_ms: float = 0.0
    chunking_time_ms: float = 0.0
    upload_time_ms: float = 0.0
    total_time_ms: float = 0.0
    failed_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "embedding_time_ms": self.embedding_time_ms,
            "chunking_time_ms": self.chunking_time_ms,
            "upload_time_ms": self.upload_time_ms,
            "total_time_ms": self.total_time_ms,
            "failed_chunks": self.failed_chunks,
        }


@dataclass
class AddDocumentsResponse:
    """Result from adding documents.

    Attributes:
        success: Boolean status
        documents_processed: Count of documents
        total_chunks: Total chunks created
        chunks: List of Chunk objects
        processing_stats: ProcessingStats object
        errors: List of error messages
    """

    success: bool
    documents_processed: int
    total_chunks: int
    chunks: List[Chunk] = field(default_factory=list)
    processing_stats: ProcessingStats = field(default_factory=ProcessingStats)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "success": self.success,
            "documents_processed": self.documents_processed,
            "total_chunks": self.total_chunks,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "processing_stats": self.processing_stats.to_dict(),
            "errors": self.errors,
        }


@dataclass
class UpdateDocumentsResponse:
    """Result from update operations.

    Attributes:
        success: Boolean status
        strategy_used: Update strategy applied
        documents_affected: Count of affected documents
        chunks_deleted: Chunks removed
        chunks_added: Chunks added
        chunks_updated: Chunks modified
        updated_document_ids: List of affected IDs
        chunks: List of Chunk objects for added/updated chunks (for external storage)
        errors: Error list
    """

    success: bool
    strategy_used: str
    documents_affected: int
    chunks_deleted: int = 0
    chunks_added: int = 0
    chunks_updated: int = 0
    updated_document_ids: List[str] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)  # NEW: For external storage (e.g., MongoDB)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "success": self.success,
            "strategy_used": self.strategy_used,
            "documents_affected": self.documents_affected,
            "chunks_deleted": self.chunks_deleted,
            "chunks_added": self.chunks_added,
            "chunks_updated": self.chunks_updated,
            "updated_document_ids": self.updated_document_ids,
            "errors": self.errors,
        }


@dataclass
class RetrievalStats:
    """Performance and count metrics for retrieval.

    Attributes:
        total_chunks_retrieved: Initial retrieval count
        vector_search_chunks: From vector search
        keyword_search_chunks: From BM25 search
        chunks_after_dedup: After deduplication
        chunks_after_reranking: Final count
        query_generation_time_ms: Time for query generation
        vector_search_time_ms: Time for vector search
        keyword_search_time_ms: Time for keyword search
        reranking_time_ms: Time for reranking
        total_time_ms: Total retrieval time
    """

    total_chunks_retrieved: int = 0
    vector_search_chunks: int = 0
    keyword_search_chunks: int = 0
    chunks_after_dedup: int = 0
    chunks_after_reranking: int = 0
    query_generation_time_ms: float = 0.0
    vector_search_time_ms: float = 0.0
    keyword_search_time_ms: float = 0.0
    reranking_time_ms: float = 0.0
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_chunks_retrieved": self.total_chunks_retrieved,
            "vector_search_chunks": self.vector_search_chunks,
            "keyword_search_chunks": self.keyword_search_chunks,
            "chunks_after_dedup": self.chunks_after_dedup,
            "chunks_after_reranking": self.chunks_after_reranking,
            "query_generation_time_ms": self.query_generation_time_ms,
            "vector_search_time_ms": self.vector_search_time_ms,
            "keyword_search_time_ms": self.keyword_search_time_ms,
            "reranking_time_ms": self.reranking_time_ms,
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class SourceInfo:
    """Aggregated information per source document.

    Attributes:
        source: Source document name
        chunks_count: Chunks from this source
        avg_relevance: Average relevance score
    """

    source: str
    chunks_count: int
    avg_relevance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "chunks_count": self.chunks_count,
            "avg_relevance": self.avg_relevance,
        }


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval.

    Attributes:
        content: Chunk text
        metadata: ChunkMetadata
        relevance_score: Reranker score (0-1)
        vector_score: Cosine similarity score
        keyword_score: BM25 score (optional)
        rank: Position in results
    """

    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    vector_score: float
    rank: int
    keyword_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "rank": self.rank,
        }


@dataclass
class RetrievalResponse:
    """Result from retrieval operations.

    Attributes:
        success: Boolean status
        query_original: Original query string
        queries_generated: Dict with standard and HyDE queries
        chunks: List of RetrievedChunk objects
        retrieval_stats: RetrievalStats object
        sources: List of SourceInfo objects
        errors: List of error messages
    """

    success: bool
    query_original: str
    queries_generated: Dict[str, str] = field(default_factory=dict)
    chunks: List[RetrievedChunk] = field(default_factory=list)
    retrieval_stats: RetrievalStats = field(default_factory=RetrievalStats)
    sources: List[SourceInfo] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "success": self.success,
            "query_original": self.query_original,
            "queries_generated": self.queries_generated,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "retrieval_stats": self.retrieval_stats.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
            "errors": self.errors,
        }
