"""Data models for chunk representation and metadata."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ChunkMetadata:
    """Complete metadata for a document chunk.

    Attributes:
        document_id: Parent document identifier
        source: Original source file/URL
        chunk_index: Position in document (0-based)
        total_chunks: Total chunks in document
        token_count: Number of tokens in chunk
        char_count: Character count
        chunking_method: Method used (e.g., "semantic", "recursive")
        extraction_date: Timestamp of chunk creation
        custom_fields: Dictionary of additional metadata
    """

    document_id: str
    source: str
    chunk_index: int
    total_chunks: int
    token_count: int
    char_count: int
    chunking_method: str
    extraction_date: datetime = field(default_factory=datetime.utcnow)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format for storage."""
        return {
            "document_id": self.document_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "chunking_method": self.chunking_method,
            "extraction_date": self.extraction_date.isoformat(),
            **self.custom_fields,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        """Create metadata from dictionary."""
        extraction_date = data.get("extraction_date")
        if isinstance(extraction_date, str):
            extraction_date = datetime.fromisoformat(extraction_date)
        elif extraction_date is None:
            extraction_date = datetime.utcnow()

        # Separate known fields from custom fields
        known_fields = {
            "document_id",
            "source",
            "chunk_index",
            "total_chunks",
            "token_count",
            "char_count",
            "chunking_method",
            "extraction_date",
        }
        custom_fields = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            document_id=data["document_id"],
            source=data["source"],
            chunk_index=data["chunk_index"],
            total_chunks=data["total_chunks"],
            token_count=data["token_count"],
            char_count=data["char_count"],
            chunking_method=data["chunking_method"],
            extraction_date=extraction_date,
            custom_fields=custom_fields,
        )


@dataclass
class Chunk:
    """Represents a processed document chunk.

    Attributes:
        chunk_id: Unique internal identifier
        content: Chunk text content
        metadata: ChunkMetadata object
        vector_id: Qdrant point ID (set after storage)
        embedding: Vector embedding (optional, for temporary storage)
    """

    chunk_id: str
    content: str
    metadata: ChunkMetadata
    vector_id: Optional[str] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "vector_id": self.vector_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            metadata=ChunkMetadata.from_dict(data["metadata"]),
            vector_id=data.get("vector_id"),
        )
