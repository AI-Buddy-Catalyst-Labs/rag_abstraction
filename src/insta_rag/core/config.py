"""Configuration management for insta_rag library."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..exceptions import ConfigurationError


@dataclass
class VectorDBConfig:
    """Vector database configuration."""

    url: str
    api_key: str
    provider: str = "qdrant"
    timeout: int = 30
    prefer_grpc: bool = False  # Changed to False to avoid connection issues
    https: Optional[bool] = None  # Auto-detect from URL if None
    verify_ssl: bool = False  # Set to False for self-signed certificates

    def validate(self) -> None:
        """Validate vector database configuration."""
        if not self.url:
            raise ConfigurationError("Vector database URL is required")
        if not self.api_key:
            raise ConfigurationError("Vector database API key is required")


@dataclass
class EmbeddingConfig:
    """Embedding provider configuration."""

    provider: str = "openai"  # openai, azure_openai, cohere
    model: str = "text-embedding-3-large"
    dimensions: int = 3072
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None  # For Azure OpenAI
    batch_size: int = 100

    def validate(self) -> None:
        """Validate embedding configuration."""
        if self.provider == "openai":
            if not self.api_key:
                raise ConfigurationError("OpenAI API key is required")
        elif self.provider == "azure_openai":
            if not self.api_key or not self.api_base or not self.deployment_name:
                raise ConfigurationError(
                    "Azure OpenAI requires api_key, api_base, and deployment_name"
                )
        elif self.provider == "cohere":
            if not self.api_key:
                raise ConfigurationError("Cohere API key is required")


@dataclass
class RerankingConfig:
    """Reranking configuration."""

    provider: str = "cohere"  # cohere, cross_encoder
    model: str = "rerank-v3.5"
    api_key: Optional[str] = None
    top_k: int = 20
    enabled: bool = True

    def validate(self) -> None:
        """Validate reranking configuration."""
        if self.enabled and self.provider == "cohere" and not self.api_key:
            raise ConfigurationError("Cohere API key is required for reranking")


@dataclass
class LLMConfig:
    """LLM configuration for query generation."""

    provider: str = "openai"  # openai, azure_openai, anthropic
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None  # For Azure OpenAI
    temperature: float = 0.0

    def validate(self) -> None:
        """Validate LLM configuration."""
        if self.provider == "openai":
            if not self.api_key:
                raise ConfigurationError("OpenAI API key is required")
        elif self.provider == "azure_openai":
            if not self.api_key or not self.api_base or not self.deployment_name:
                raise ConfigurationError(
                    "Azure OpenAI requires api_key, api_base, and deployment_name"
                )


@dataclass
class ChunkingConfig:
    """Chunking strategy configuration."""

    method: str = "semantic"  # semantic, recursive, fixed
    max_chunk_size: int = 1000  # tokens
    overlap_percentage: float = 0.2
    semantic_threshold_percentile: int = 95
    min_chunk_size: int = 100  # tokens

    def validate(self) -> None:
        """Validate chunking configuration."""
        if self.max_chunk_size <= 0:
            raise ConfigurationError("max_chunk_size must be positive")
        if not 0 <= self.overlap_percentage < 1:
            raise ConfigurationError("overlap_percentage must be between 0 and 1")


@dataclass
class PDFConfig:
    """PDF processing configuration."""

    parser: str = "pdfplumber"  # pdfplumber, pypdf2, chunkr, unstructured
    extract_images: bool = False
    extract_tables: bool = False
    validate_text: bool = True

    def validate(self) -> None:
        """Validate PDF configuration."""
        valid_parsers = {"pdfplumber", "pypdf2", "chunkr", "unstructured"}
        if self.parser not in valid_parsers:
            raise ConfigurationError(
                f"Invalid parser: {self.parser}. Must be one of {valid_parsers}"
            )


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""

    vector_search_limit: int = 25
    keyword_search_limit: int = 50
    enable_hyde: bool = True
    enable_keyword_search: bool = True
    final_top_k: int = 20
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product
    score_threshold: Optional[float] = None

    def validate(self) -> None:
        """Validate retrieval configuration."""
        if self.vector_search_limit <= 0:
            raise ConfigurationError("vector_search_limit must be positive")
        if self.keyword_search_limit <= 0:
            raise ConfigurationError("keyword_search_limit must be positive")
        if self.final_top_k <= 0:
            raise ConfigurationError("final_top_k must be positive")


@dataclass
class MongoDBConfig:
    """MongoDB configuration."""

    connection_string: str
    database_name: str = "Test_Insta_RAG"
    enabled: bool = True

    def validate(self) -> None:
        """Validate MongoDB configuration."""
        if self.enabled and not self.connection_string:
            raise ConfigurationError("MongoDB connection string is required when enabled")


@dataclass
class RAGConfig:
    """Main RAG system configuration."""

    vectordb: VectorDBConfig
    embedding: EmbeddingConfig
    mongodb: Optional[MongoDBConfig] = None
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.vectordb.validate()
        self.embedding.validate()
        if self.mongodb:
            self.mongodb.validate()
        self.reranking.validate()
        self.llm.validate()
        self.chunking.validate()
        self.pdf.validate()
        self.retrieval.validate()

    @classmethod
    def from_env(cls, **kwargs) -> "RAGConfig":
        """Create configuration from environment variables.

        Environment variables:
            QDRANT_URL: Qdrant database URL
            QDRANT_API_KEY: Qdrant API key
            OPENAI_API_KEY or AZURE_OPENAI_API_KEY: OpenAI/Azure API key
            AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint
            AZURE_EMBEDDING_DEPLOYMENT: Azure deployment name
            COHERE_API_KEY: Cohere API key for reranking

        Args:
            **kwargs: Override specific configuration values

        Returns:
            RAGConfig instance
        """
        # Vector DB Config
        vectordb_config = VectorDBConfig(
            url=os.getenv("QDRANT_URL", ""),
            api_key=os.getenv("QDRANT_API_KEY", ""),
        )

        # Determine if using Azure or OpenAI
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if azure_endpoint and azure_key:
            # Azure OpenAI configuration
            embedding_config = EmbeddingConfig(
                provider="azure_openai",
                model="text-embedding-3-large",
                dimensions=3072,
                api_key=azure_key,
                api_base=azure_endpoint,
                api_version="2024-02-01",
                deployment_name=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
            )

            llm_config = LLMConfig(
                provider="azure_openai",
                model="gpt-4",
                api_key=azure_key,
                api_base=azure_endpoint,
                api_version="2024-02-01",
                deployment_name=os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4"),
            )
        else:
            # Standard OpenAI configuration
            embedding_config = EmbeddingConfig(
                provider="openai",
                model="text-embedding-3-large",
                dimensions=3072,
                api_key=openai_key,
            )

            llm_config = LLMConfig(
                provider="openai",
                model="gpt-4",
                api_key=openai_key,
            )

        # Reranking config
        reranking_config = RerankingConfig(
            provider="cohere",
            model="rerank-v3.5",
            api_key=os.getenv("COHERE_API_KEY"),
            enabled=bool(os.getenv("COHERE_API_KEY")),  # Auto-disable if no key
        )

        # MongoDB config
        mongo_connection = os.getenv("MONGO_CONNECTION_STRING")
        mongodb_config = None
        if mongo_connection:
            mongodb_config = MongoDBConfig(
                connection_string=mongo_connection,
                database_name=os.getenv("MONGO_DATABASE_NAME", "Test_Insta_RAG"),
                enabled=True,
            )

        config = cls(
            vectordb=vectordb_config,
            embedding=embedding_config,
            mongodb=mongodb_config,
            reranking=reranking_config,
            llm=llm_config,
        )

        # Apply any overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(config, key) and isinstance(value, dict):
                config_obj = getattr(config, key)
                for k, v in value.items():
                    setattr(config_obj, k, v)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (without sensitive data)."""
        config_dict = {
            "vectordb": {"provider": self.vectordb.provider, "url": self.vectordb.url},
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "dimensions": self.embedding.dimensions,
            },
            "reranking": {
                "provider": self.reranking.provider,
                "model": self.reranking.model,
                "enabled": self.reranking.enabled,
            },
            "llm": {"provider": self.llm.provider, "model": self.llm.model},
            "chunking": {
                "method": self.chunking.method,
                "max_chunk_size": self.chunking.max_chunk_size,
                "overlap_percentage": self.chunking.overlap_percentage,
            },
            "pdf": {"parser": self.pdf.parser},
            "retrieval": {
                "vector_search_limit": self.retrieval.vector_search_limit,
                "keyword_search_limit": self.retrieval.keyword_search_limit,
                "enable_hyde": self.retrieval.enable_hyde,
                "enable_keyword_search": self.retrieval.enable_keyword_search,
                "final_top_k": self.retrieval.final_top_k,
            },
        }

        if self.mongodb:
            config_dict["mongodb"] = {
                "database_name": self.mongodb.database_name,
                "enabled": self.mongodb.enabled,
            }

        return config_dict
