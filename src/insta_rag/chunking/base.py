"""Base interface for chunking strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..models.chunk import Chunk


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into chunks.

        Args:
            text: Input text to chunk
            metadata: Base metadata to attach to all chunks

        Returns:
            List of Chunk objects
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate chunking configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass
