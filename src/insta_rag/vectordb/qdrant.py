"""Qdrant vector database implementation."""

import uuid
from typing import Any, Dict, List, Optional

from ..exceptions import CollectionNotFoundError, VectorDBError
from .base import BaseVectorDB, VectorSearchResult


class QdrantVectorDB(BaseVectorDB):
    """Qdrant vector database implementation."""

    def __init__(
        self,
        url: str,
        api_key: str,
        timeout: int = 60,  # Increased timeout
        prefer_grpc: bool = False,  # Disabled gRPC by default
        https: Optional[bool] = None,  # Auto-detect from URL if None
        verify_ssl: bool = False,  # Set to False for self-signed certificates
    ):
        """Initialize Qdrant client.

        Args:
            url: Qdrant instance URL
            api_key: Qdrant API key
            timeout: Request timeout in seconds
            prefer_grpc: Use gRPC for better performance (disabled by default for compatibility)
            https: Force HTTPS connection (auto-detect from URL if None)
            verify_ssl: Verify SSL certificates (set to False for self-signed certs)
        """
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.prefer_grpc = prefer_grpc
        self.https = https
        self.verify_ssl = verify_ssl

        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            import urllib.parse
            import ssl

            # Store for later use
            self.Distance = Distance
            self.VectorParams = VectorParams

            # Auto-detect HTTPS from URL if not explicitly set
            https = self.https
            if https is None:
                https = self.url.startswith("https://")

            # Parse the URL to get host and port
            parsed = urllib.parse.urlparse(self.url)
            host = parsed.hostname or parsed.netloc
            port = parsed.port or (443 if https else 6333)

            # Create SSL context if needed
            grpc_options = None
            if not self.verify_ssl:
                # Disable SSL verification for self-signed certificates
                grpc_options = {
                    "grpc.ssl_target_name_override": host,
                    "grpc.default_authority": host,
                }

            # Initialize client with SSL verification options
            # Note: verify parameter doesn't exist in older versions, so we use grpc_options
            try:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    prefer_grpc=False,  # Force disable gRPC
                    https=https,
                    grpc_options=grpc_options,
                    check_compatibility=False,  # Skip version check to avoid warnings
                )
            except TypeError:
                # Fallback for older qdrant-client versions without grpc_options or check_compatibility
                try:
                    self.client = QdrantClient(
                        host=host,
                        port=port,
                        api_key=self.api_key,
                        timeout=self.timeout,
                        prefer_grpc=False,
                        https=https,
                        check_compatibility=False,
                    )
                except TypeError:
                    # Fallback for very old versions
                    self.client = QdrantClient(
                        host=host,
                        port=port,
                        api_key=self.api_key,
                        timeout=self.timeout,
                        prefer_grpc=False,
                        https=https,
                    )

        except ImportError as e:
            raise VectorDBError(
                "Qdrant client not installed. Install with: pip install qdrant-client"
            ) from e
        except Exception as e:
            raise VectorDBError(f"Failed to initialize Qdrant client: {str(e)}") from e

    def create_collection(
        self, collection_name: str, vector_size: int, distance_metric: str = "cosine"
    ) -> None:
        """Create a new collection.

        Args:
            collection_name: Name of the collection
            vector_size: Dimensionality of vectors
            distance_metric: Distance metric (cosine, euclidean, dot_product)
        """
        try:
            from qdrant_client.models import Distance, VectorParams

            # Map distance metric names
            distance_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot_product": Distance.DOT,
            }

            distance = distance_map.get(distance_metric.lower(), Distance.COSINE)

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )

        except Exception as e:
            raise VectorDBError(f"Failed to create collection: {str(e)}") from e

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists
        """
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            raise VectorDBError(
                f"Failed to check collection existence: {str(e)}"
            ) from e

    def upsert(
        self,
        collection_name: str,
        chunk_ids: List[str],
        vectors: List[List[float]],
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Insert or update vectors in collection.

        Args:
            collection_name: Name of the collection
            chunk_ids: List of chunk IDs
            vectors: List of embedding vectors
            contents: List of chunk contents
            metadatas: List of metadata dictionaries
        """
        try:
            from qdrant_client.models import PointStruct

            # Verify collection exists
            if not self.collection_exists(collection_name):
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' does not exist"
                )

            # Create points
            points = []
            for i, (chunk_id, vector, content, metadata) in enumerate(
                zip(chunk_ids, vectors, contents, metadatas)
            ):
                # Combine content and metadata for payload
                payload = {"content": content, **metadata}

                # Create point with deterministic UUID from chunk_id
                point = PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
                    vector=vector,
                    payload=payload,
                )
                points.append(point)

            # Upsert points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(collection_name=collection_name, points=batch)

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(f"Failed to upsert vectors: {str(e)}") from e

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors.

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            filters: Metadata filters

        Returns:
            List of VectorSearchResult objects
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Verify collection exists
            if not self.collection_exists(collection_name):
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' does not exist"
                )

            # Build filter query
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    # Skip empty or None values
                    if value is not None and value != "" and value != {}:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )

                if conditions:
                    query_filter = Filter(must=conditions)

            # Perform search using query_points (new recommended method)
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )

            # Convert to VectorSearchResult objects
            results = []
            for hit in search_result.points:
                result = VectorSearchResult(
                    chunk_id=hit.payload.get("chunk_id", str(hit.id)),
                    score=hit.score,
                    content=hit.payload.get("content", ""),
                    metadata={
                        k: v for k, v in hit.payload.items() if k != "content"
                    },
                    vector_id=str(hit.id),
                )
                results.append(result)

            return results

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(f"Failed to search vectors: {str(e)}") from e

    def delete(
        self,
        collection_name: str,
        filters: Optional[Dict[str, Any]] = None,
        chunk_ids: Optional[List[str]] = None,
    ) -> int:
        """Delete vectors from collection.

        Args:
            collection_name: Name of the collection
            filters: Metadata filters for deletion
            chunk_ids: Specific chunk IDs to delete

        Returns:
            Number of vectors deleted
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Verify collection exists
            if not self.collection_exists(collection_name):
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' does not exist"
                )

            # Delete by chunk IDs
            if chunk_ids:
                point_ids = [
                    str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
                    for chunk_id in chunk_ids
                ]
                self.client.delete(
                    collection_name=collection_name, points_selector=point_ids
                )
                return len(chunk_ids)

            # Delete by filters
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

                if conditions:
                    query_filter = Filter(must=conditions)
                    # Get count before deletion
                    count_result = self.client.count(
                        collection_name=collection_name, count_filter=query_filter
                    )
                    count = count_result.count

                    # Perform deletion
                    self.client.delete(
                        collection_name=collection_name, points_selector=query_filter
                    )
                    return count

            return 0

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(f"Failed to delete vectors: {str(e)}") from e

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection information
        """
        try:
            if not self.collection_exists(collection_name):
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' does not exist"
                )

            info = self.client.get_collection(collection_name=collection_name)

            return {
                "name": collection_name,
                "vectors_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status,
            }

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(
                f"Failed to get collection info: {str(e)}"
            ) from e
