"""OpenAI and Azure OpenAI embedding provider."""

from typing import List, Optional

from insta_rag.utils.exceptions import EmbeddingError
from .base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding provider supporting both OpenAI and Azure OpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        batch_size: int = 100,
    ):
        """Initialize OpenAI embedder.

        Args:
            api_key: OpenAI or Azure OpenAI API key
            model: Model name
            dimensions: Embedding dimensions
            api_base: Azure OpenAI endpoint (for Azure only)
            api_version: Azure OpenAI API version (for Azure only)
            deployment_name: Azure deployment name (for Azure only)
            batch_size: Batch size for embedding generation
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.api_base = api_base
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.batch_size = batch_size

        # Determine if using Azure or standard OpenAI
        self.is_azure = api_base is not None

        # Initialize client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            if self.is_azure:
                from openai import AzureOpenAI

                self.client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version=self.api_version or "2024-02-01",
                    azure_endpoint=self.api_base,
                )
            else:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)

        except ImportError as e:
            raise EmbeddingError(
                "OpenAI library not installed. Install with: pip install openai"
            ) from e
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI client: {str(e)}") from e

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            all_embeddings = []

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                # Call OpenAI API
                if self.is_azure:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.deployment_name or self.model,
                    )
                else:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model,
                        dimensions=self.dimensions,
                    )

                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}") from e

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        try:
            embeddings = self.embed([query])
            return embeddings[0] if embeddings else []
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {str(e)}") from e

    def get_dimensions(self) -> int:
        """Get the dimensionality of the embedding vectors.

        Returns:
            Number of dimensions in embedding vectors
        """
        return self.dimensions
