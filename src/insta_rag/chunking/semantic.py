"""Semantic chunking implementation."""

import uuid
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from ..utils.exceptions import ChunkingError
from ..models.chunk import Chunk, ChunkMetadata
from .base import BaseChunker
from .utils import (
    add_overlap_to_chunks,
    count_tokens_accurate,
    split_into_sentences,
    split_text_by_tokens,
    validate_chunk_quality,
)


class SemanticChunker(BaseChunker):
    """Semantic chunking using sentence similarity.

    This chunker:
    1. Checks if document fits in single chunk
    2. Splits into sentences and embeds them
    3. Finds semantic boundaries (low similarity points)
    4. Splits at these boundaries
    5. Enforces token limits
    6. Adds overlap between chunks
    """

    def __init__(
        self,
        embedder,  # BaseEmbedder instance
        max_chunk_size: int = 1000,
        overlap_percentage: float = 0.2,
        threshold_percentile: int = 95,
        min_chunk_size: int = 100,
    ):
        """Initialize semantic chunker.

        Args:
            embedder: Embedding provider for semantic analysis
            max_chunk_size: Maximum tokens per chunk
            overlap_percentage: Overlap between chunks (0-1)
            threshold_percentile: Percentile for semantic breakpoint detection
            min_chunk_size: Minimum tokens per chunk
        """
        self.embedder = embedder
        self.max_chunk_size = max_chunk_size
        self.overlap_percentage = overlap_percentage
        self.threshold_percentile = threshold_percentile
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into semantic chunks.

        Args:
            text: Input text to chunk
            metadata: Base metadata to attach to all chunks

        Returns:
            List of Chunk objects
        """
        try:
            # Step 1: Check if single chunk is sufficient
            total_tokens = count_tokens_accurate(text)

            if total_tokens <= self.max_chunk_size:
                # Return as single chunk
                return self._create_chunks([text], metadata, "semantic_single")

            # Step 2: Try semantic chunking
            chunks = self._semantic_chunk(text)

            # Step 3: If semantic chunking fails, fall back to simple splitting
            if not chunks:
                chunks = split_text_by_tokens(
                    text,
                    self.max_chunk_size,
                    int(self.max_chunk_size * self.overlap_percentage),
                )

                return self._create_chunks(chunks, metadata, "semantic_fallback")

            # Step 4: Add overlap between chunks
            chunks = add_overlap_to_chunks(chunks, self.overlap_percentage)

            return self._create_chunks(chunks, metadata, "semantic")

        except Exception as e:
            raise ChunkingError(f"Semantic chunking failed: {str(e)}") from e

    def _semantic_chunk(self, text: str) -> List[str]:
        """Perform semantic chunking using sentence embeddings.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        try:
            # Split into sentences
            sentences = split_into_sentences(text)

            if len(sentences) <= 1:
                return [text]

            # Embed sentences
            embeddings = self.embedder.embed(sentences)

            # Calculate similarities between consecutive sentences
            similarities = self._calculate_similarities(embeddings)

            # Find breakpoints (low similarity points)
            breakpoints = self._find_breakpoints(similarities)

            # Split into chunks at breakpoints
            chunks = self._split_at_breakpoints(sentences, breakpoints)

            # Enforce token limits
            chunks = self._enforce_token_limits(chunks)

            return chunks

        except Exception as e:
            # Return empty list to trigger fallback
            print(f"Semantic chunking error (will fallback): {e}")
            return []

    def _calculate_similarities(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate cosine similarities between consecutive sentences.

        Args:
            embeddings: List of sentence embeddings

        Returns:
            List of similarity scores
        """
        similarities = []

        for i in range(len(embeddings) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i + 1])

            # Cosine similarity
            similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )
            similarities.append(float(similarity))

        return similarities

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """Find semantic breakpoints using percentile threshold.

        Args:
            similarities: List of similarity scores

        Returns:
            List of sentence indices to break at
        """
        if not similarities:
            return []

        # Calculate threshold (low similarity = topic change)
        threshold = np.percentile(similarities, 100 - self.threshold_percentile)

        # Find indices where similarity is below threshold
        breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

        return breakpoints

    def _split_at_breakpoints(
        self, sentences: List[str], breakpoints: List[int]
    ) -> List[str]:
        """Split sentences into chunks at breakpoints.

        Args:
            sentences: List of sentences
            breakpoints: Indices to split at

        Returns:
            List of text chunks
        """
        if not breakpoints:
            return [" ".join(sentences)]

        chunks = []
        start_idx = 0

        for breakpoint in breakpoints:
            chunk_sentences = sentences[start_idx:breakpoint]
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))
            start_idx = breakpoint

        # Add remaining sentences
        if start_idx < len(sentences):
            chunks.append(" ".join(sentences[start_idx:]))

        return chunks

    def _enforce_token_limits(self, chunks: List[str]) -> List[str]:
        """Enforce maximum token limits on chunks.

        Args:
            chunks: List of text chunks

        Returns:
            List of chunks with enforced limits
        """
        final_chunks = []

        for chunk in chunks:
            token_count = count_tokens_accurate(chunk)

            if token_count <= self.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split oversized chunk
                sub_chunks = split_text_by_tokens(
                    chunk,
                    self.max_chunk_size,
                    int(self.max_chunk_size * self.overlap_percentage),
                )
                final_chunks.extend(sub_chunks)

        return final_chunks

    def _create_chunks(
        self, text_chunks: List[str], base_metadata: Dict[str, Any], method: str
    ) -> List[Chunk]:
        """Create Chunk objects from text chunks.

        Args:
            text_chunks: List of text chunks
            base_metadata: Base metadata for all chunks
            method: Chunking method used

        Returns:
            List of Chunk objects
        """
        chunks = []
        document_id = base_metadata.get("document_id", str(uuid.uuid4()))
        source = base_metadata.get("source", "unknown")
        total_chunks = len(text_chunks)

        for idx, text in enumerate(text_chunks):
            # Validate chunk quality
            if not validate_chunk_quality(text):
                continue

            # Create metadata
            metadata = ChunkMetadata(
                document_id=document_id,
                source=source,
                chunk_index=idx,
                total_chunks=total_chunks,
                token_count=count_tokens_accurate(text),
                char_count=len(text),
                chunking_method=method,
                extraction_date=datetime.utcnow(),
                custom_fields={**base_metadata},
            )

            # Create chunk
            chunk = Chunk(
                chunk_id=f"{document_id}_chunk_{idx}",
                content=text,
                metadata=metadata,
            )

            chunks.append(chunk)

        return chunks

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate chunking configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        max_chunk_size = config.get("max_chunk_size", 1000)
        overlap_percentage = config.get("overlap_percentage", 0.2)

        if max_chunk_size <= 0:
            raise ChunkingError("max_chunk_size must be positive")

        if not 0 <= overlap_percentage < 1:
            raise ChunkingError("overlap_percentage must be between 0 and 1")

        return True
