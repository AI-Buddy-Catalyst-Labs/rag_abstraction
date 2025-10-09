"""Base interface for vector database providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorSearchResult:
    """Result from vector search."""

    def __init__(
        self,
        chunk_id: str,
        score: float,
        content: str,
        metadata: Dict[str, Any],
        vector_id: Optional[str] = None,
    ):
        self.chunk_id = chunk_id
        self.score = score
        self.content = content
        self.metadata = metadata
        self.vector_id = vector_id


class BaseVectorDB(ABC):
    """Abstract base class for all vector database providers."""

    @abstractmethod
    def create_collection(
        self, collection_name: str, vector_size: int, distance_metric: str = "cosine"
    ) -> None:
        """Create a new collection.

        Args:
            collection_name: Name of the collection
            vector_size: Dimensionality of vectors
            distance_metric: Distance metric (cosine, euclidean, dot_product)
        """
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists
        """
        pass

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        chunk_ids: List[str],
        vectors: List[List[float]],
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Insert or update vectors in collection.

        Args:
            collection_name: Name of the collection
            chunk_ids: List of chunk IDs
            vectors: List of embedding vectors
            contents: List of chunk contents
            metadatas: List of metadata dictionaries
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors.

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            filters: Metadata filters

        Returns:
            List of VectorSearchResult objects
        """
        pass

    @abstractmethod
    def delete(
        self,
        collection_name: str,
        filters: Optional[Dict[str, Any]] = None,
        chunk_ids: Optional[List[str]] = None,
    ) -> int:
        """Delete vectors from collection.

        Args:
            collection_name: Name of the collection
            filters: Metadata filters for deletion
            chunk_ids: Specific chunk IDs to delete

        Returns:
            Number of vectors deleted
        """
        pass

    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection information
        """
        pass
