"""Reranking implementations for improving retrieval results."""

import time
import json
from typing import Any, Dict, List, Tuple
import requests
from openai import OpenAI

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


class LLMReranker(BaseReranker):
    """LLM-based reranker using gpt-oss-120b as fallback for BGE reranker.

    This reranker uses a language model to evaluate and rank chunks based on
    their relevance to a query. It's designed as a fallback when the primary
    BGE reranker service is unavailable.

    The LLM analyzes each chunk's content and assigns a relevance score,
    providing robust reranking capabilities even when dedicated reranking
    services are down.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "gpt-oss-120b",
        timeout: int = 60
    ):
        """Initialize LLM reranker.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the OpenAI-compatible endpoint
            model: Model deployment name (default: gpt-oss-120b)
            timeout: Request timeout in seconds (default: 60)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )

    def rerank(
        self,
        query: str,
        chunks: List[Tuple[str, Dict[str, Any]]],
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Rerank chunks based on relevance to query using LLM.

        Args:
            query: Query string
            chunks: List of (content, metadata) tuples
            top_k: Number of top results to return

        Returns:
            List of (original_index, relevance_score) tuples, sorted by relevance

        Raises:
            Exception: If API request fails or response parsing fails
        """
        if not chunks:
            return []

        # Extract just the content from chunks
        documents = [chunk[0] for chunk in chunks]

        # Limit to top_k chunks to avoid token limits
        num_chunks_to_evaluate = min(top_k * 3, len(documents))  # Evaluate 3x top_k
        documents_to_rank = documents[:num_chunks_to_evaluate]

        # Prepare the prompt for LLM
        chunks_text = ""
        for idx, doc in enumerate(documents_to_rank):
            # Truncate very long chunks to avoid token limits
            truncated_doc = doc[:500] if len(doc) > 500 else doc
            chunks_text += f"\n[{idx}] {truncated_doc}\n"

        prompt = f"""You are a relevance scoring system. Given a query and a list of text chunks, score each chunk's relevance to the query on a scale from -10.0 to 10.0, where higher scores indicate more relevant chunks.

Query: {query}

Text Chunks:
{chunks_text}

Instructions:
1. Analyze each chunk's relevance to the query
2. Assign a relevance score from -10.0 (not relevant) to 10.0 (highly relevant)
3. Return ONLY a valid JSON array with objects containing "index" and "score"
4. Sort results by score in descending order (highest scores first)
5. Return the top {top_k} most relevant chunks

Example format:
[
  {{"index": 0, "score": 8.5}},
  {{"index": 3, "score": 7.2}},
  {{"index": 1, "score": 5.8}}
]

Return your response as a JSON array:"""

        try:
            # Make API request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a relevance scoring system that returns only valid JSON arrays. Never include explanations, only return the JSON array."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=2000
            )

            # Extract the response content
            response_text = response.choices[0].message.content.strip()

            # Try to find JSON array in the response
            # Sometimes LLM might add extra text, so we look for the JSON array
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in LLM response")

            json_str = response_text[start_idx:end_idx]

            # Parse JSON response
            rankings = json.loads(json_str)

            # Validate and convert to required format
            reranked_results = []
            for item in rankings:
                if isinstance(item, dict) and "index" in item and "score" in item:
                    original_index = int(item["index"])
                    score = float(item["score"])

                    # Validate index is within range
                    if 0 <= original_index < len(documents_to_rank):
                        reranked_results.append((original_index, score))

            # If we didn't get enough results, pad with remaining chunks
            if len(reranked_results) < top_k:
                existing_indices = {idx for idx, _ in reranked_results}
                for idx in range(len(documents_to_rank)):
                    if idx not in existing_indices and len(reranked_results) < top_k:
                        # Assign a low default score
                        reranked_results.append((idx, -5.0))

            # Limit to top_k
            reranked_results = reranked_results[:top_k]

            return reranked_results

        except json.JSONDecodeError as e:
            raise Exception(f"LLM reranker failed to parse JSON response: {str(e)}. Response: {response_text[:200]}")
        except Exception as e:
            raise Exception(f"LLM reranker API request failed: {str(e)}")