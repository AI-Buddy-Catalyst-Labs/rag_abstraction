"""Reranking implementations for improving retrieval results."""

import time
from typing import Any, Dict, List, Tuple
import requests

from .base import BaseReranker


class BGEReranker(BaseReranker):
    """BGE (BAAI) reranker using BAAI/bge-reranker-v2-m3 model.

    This reranker uses a remote API endpoint that hosts the BGE reranker model.
    The model is designed to rerank search results based on semantic relevance.

    API Endpoint: http://118.67.212.45:8000/rerank
    Model: BAAI/bge-reranker-v2-m3

    Important: BGE reranker produces negative scores where:
    - Higher (less negative) scores = more relevant (e.g., -0.96 is better than -6.99)
    - Typical score range: -10.0 to +10.0
    - Most relevant results: -3.0 to +5.0
    - Use negative thresholds when filtering (e.g., score_threshold=-5.0)
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = "http://118.67.212.45:8000/rerank",
        normalize: bool = False,
        timeout: int = 30
    ):
        """Initialize BGE reranker.

        Args:
            api_key: API key for authentication
            api_url: Reranking API endpoint URL
            normalize: Whether to normalize scores (default: False)
            timeout: Request timeout in seconds (default: 30)
        """
        self.api_key = api_key
        self.api_url = api_url
        self.normalize = normalize
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        chunks: List[Tuple[str, Dict[str, Any]]],
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Rerank chunks based on relevance to query using BGE reranker.

        Args:
            query: Query string
            chunks: List of (content, metadata) tuples
            top_k: Number of top results to return

        Returns:
            List of (original_index, relevance_score) tuples, sorted by relevance

        Raises:
            Exception: If API request fails
        """
        if not chunks:
            return []

        # Extract just the content from chunks
        documents = [chunk[0] for chunk in chunks]

        # Prepare API request
        request_data = {
            "query": query,
            "documents": documents,
            "top_k": min(top_k, len(documents)),  # Don't request more than available
            "normalize": self.normalize
        }

        headers = {
            "accept": "application/json",
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

        try:
            # Make API request
            response = requests.post(
                self.api_url,
                json=request_data,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Parse response
            result = response.json()

            # Extract results: list of {document, score, index}
            reranked_results = []
            for item in result.get("results", []):
                original_index = item["index"]
                score = item["score"]
                reranked_results.append((original_index, score))

            return reranked_results

        except requests.exceptions.RequestException as e:
            raise Exception(f"BGE reranker API request failed: {str(e)}")
        except (KeyError, ValueError) as e:
            raise Exception(f"Failed to parse reranker response: {str(e)}")


class CohereReranker(BaseReranker):
    """Cohere reranker implementation (legacy support).

    Note: This is a placeholder for Cohere reranking support.
    The actual implementation would require the Cohere SDK.
    """

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        """Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            model: Cohere reranking model name
        """
        self.api_key = api_key
        self.model = model

    def rerank(
        self,
        query: str,
        chunks: List[Tuple[str, Dict[str, Any]]],
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Rerank chunks using Cohere API.

        Args:
            query: Query string
            chunks: List of (content, metadata) tuples
            top_k: Number of top results to return

        Returns:
            List of (original_index, relevance_score) tuples, sorted by relevance
        """
        raise NotImplementedError(
            "Cohere reranking not yet implemented. Use BGE reranker instead."
        )
