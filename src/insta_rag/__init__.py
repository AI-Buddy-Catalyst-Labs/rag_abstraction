"""insta_rag - A modular RAG library for document processing and retrieval."""

import importlib.metadata

__version__ = importlib.metadata.version("insta_rag")

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
