"""insta_rag - A modular RAG library for document processing and retrieval."""

__version__ = "0.0.1"

from .core.client import RAGClient
from .core.config import RAGConfig
from .models.document import DocumentInput
from .models.response import AddDocumentsResponse

__all__ = [
    "RAGClient",
    "RAGConfig",
    "DocumentInput",
    "AddDocumentsResponse",
]
