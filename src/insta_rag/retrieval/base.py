"""Base interface for reranking providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseReranker(ABC):
    """Abstract base class for all reranking providers."""

    @abstractmethod
    def rerank(
        self, query: str, chunks: List[Tuple[str, Dict[str, Any]]], top_k: int
    ) -> List[Tuple[int, float]]:
        """Rerank chunks based on relevance to query.

        Args:
            query: Query string
            chunks: List of (content, metadata) tuples
            top_k: Number of top results to return

        Returns:
            List of (original_index, relevance_score) tuples, sorted by relevance
        """
        pass
