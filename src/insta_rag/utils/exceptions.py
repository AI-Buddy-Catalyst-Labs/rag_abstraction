"""Custom exceptions for insta_rag library."""


class InstaRAGError(Exception):
    """Base exception for all insta_rag errors."""

    pass


# PDF Processing Errors
class PDFError(InstaRAGError):
    """Base exception for PDF processing errors."""

    pass


class PDFEncryptedError(PDFError):
    """Raised when a PDF is password-protected."""

    pass


class PDFCorruptedError(PDFError):
    """Raised when a PDF file is invalid or damaged."""

    pass


class PDFEmptyError(PDFError):
    """Raised when a PDF has no extractable text content."""

    pass


# Chunking Errors
class ChunkingError(InstaRAGError):
    """Raised when chunking operations fail."""

    pass


# Embedding Errors
class EmbeddingError(InstaRAGError):
    """Raised when embedding generation fails."""

    pass


# Vector Database Errors
class VectorDBError(InstaRAGError):
    """Base exception for vector database errors."""

    pass


class CollectionNotFoundError(VectorDBError):
    """Raised when target collection doesn't exist."""

    pass


class NoDocumentsFoundError(VectorDBError):
    """Raised when no documents match filters/IDs."""

    pass


# Retrieval Errors
class RetrievalError(InstaRAGError):
    """Base exception for retrieval errors."""

    pass


class QueryGenerationError(RetrievalError):
    """Raised when LLM query generation fails."""

    pass


class RerankingError(RetrievalError):
    """Raised when reranking operation fails."""

    pass


# Configuration Errors
class ConfigurationError(InstaRAGError):
    """Raised when configuration is invalid."""

    pass


class ValidationError(InstaRAGError):
    """Raised when input validation fails."""

    pass
