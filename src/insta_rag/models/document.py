"""Data models for document input specifications."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union


class SourceType(Enum):
    """Type of document source."""

    FILE = "file"
    TEXT = "text"
    BINARY = "binary"


@dataclass
class DocumentInput:
    """Represents an input document for processing.

    Attributes:
        source: File path, text string, or binary content
        source_type: Type of source (file, text, or binary)
        metadata: Optional document-specific metadata
        custom_chunking: Optional chunking override settings
    """

    source: Union[str, Path, bytes]
    source_type: SourceType
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    custom_chunking: Optional[Dict[str, Any]] = None

    @classmethod
    def from_file(
        cls, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
    ) -> "DocumentInput":
        """Create document input from file path.

        Args:
            file_path: Path to the document file
            metadata: Optional document-specific metadata

        Returns:
            DocumentInput instance
        """
        return cls(
            source=Path(file_path),
            source_type=SourceType.FILE,
            metadata=metadata or {},
        )

    @classmethod
    def from_text(
        cls, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "DocumentInput":
        """Create document input from raw text.

        Args:
            text: Raw text content
            metadata: Optional document-specific metadata

        Returns:
            DocumentInput instance
        """
        return cls(
            source=text,
            source_type=SourceType.TEXT,
            metadata=metadata or {},
        )

    @classmethod
    def from_binary(
        cls, content: bytes, metadata: Optional[Dict[str, Any]] = None
    ) -> "DocumentInput":
        """Create document input from binary content.

        Args:
            content: Binary document content
            metadata: Optional document-specific metadata

        Returns:
            DocumentInput instance
        """
        return cls(
            source=content,
            source_type=SourceType.BINARY,
            metadata=metadata or {},
        )

    def get_source_path(self) -> Optional[Path]:
        """Get source as Path if it's a file, None otherwise."""
        if self.source_type == SourceType.FILE:
            return (
                Path(self.source) if not isinstance(self.source, Path) else self.source
            )
        return None

    def get_source_text(self) -> Optional[str]:
        """Get source as text if it's text type, None otherwise."""
        if self.source_type == SourceType.TEXT:
            return str(self.source)
        return None

    def get_source_binary(self) -> Optional[bytes]:
        """Get source as binary if it's binary type, None otherwise."""
        if self.source_type == SourceType.BINARY:
            return bytes(self.source)
        return None
