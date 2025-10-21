"""Qdrant vector database implementation."""

import uuid
from typing import Any, Dict, List, Optional
from insta_rag.utils.exceptions import CollectionNotFoundError, VectorDBError
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
        store_content: bool = False,  # NEW: Flag to control content storage
    ) -> None:
        """Insert or update vectors in collection.

        Args:
            collection_name: Name of the collection
            chunk_ids: List of chunk IDs
            vectors: List of embedding vectors
            contents: List of chunk contents
            metadatas: List of metadata dictionaries
            store_content: Whether to store chunk content in Qdrant payload (default: False)
                          If False, content is managed externally (e.g., in MongoDB)
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
                # Build payload with chunk_id and metadata
                # IMPORTANT: chunk_id must be in payload for update operations to work
                payload = {"chunk_id": chunk_id, **metadata}

                # Conditionally add content based on flag
                # NEW: If store_content is True, include content; otherwise, external storage handles it
                if store_content:
                    payload["content"] = content

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
                    metadata={k: v for k, v in hit.payload.items() if k != "content"},
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

    def delete_by_document_ids(
        self,
        collection_name: str,
        document_ids: List[str],
    ) -> int:
        """Delete all chunks belonging to specific documents using filter-based deletion.

        This is more efficient than getting chunk IDs first and then deleting by IDs.

        Args:
            collection_name: Name of the collection
            document_ids: List of document IDs to delete

        Returns:
            Number of chunks deleted
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny

            # Verify collection exists
            if not self.collection_exists(collection_name):
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' does not exist"
                )

            if not document_ids:
                return 0

            # Build filter for document_ids
            query_filter = Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchAny(any=document_ids))
                ]
            )

            # Get count before deletion
            count_result = self.client.count(
                collection_name=collection_name, count_filter=query_filter
            )
            count = count_result.count

            # Perform deletion using filter
            self.client.delete(
                collection_name=collection_name, points_selector=query_filter
            )

            return count

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(f"Failed to delete by document IDs: {str(e)}") from e

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
            raise VectorDBError(f"Failed to get collection info: {str(e)}") from e

    def get_document_ids(
        self,
        collection_name: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """Get unique document IDs from collection.

        Args:
            collection_name: Name of the collection
            filters: Metadata filters to match documents
            limit: Maximum number of document IDs to return

        Returns:
            List of unique document IDs
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
                    if value is not None and value != "" and value != {}:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                if conditions:
                    query_filter = Filter(must=conditions)

            # Scroll through all points to get document IDs
            document_ids = set()
            offset = None
            scroll_limit = limit if limit else 1000  # Batch size for scrolling

            while True:
                # Use scroll to get points
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=query_filter,
                    limit=scroll_limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points, next_offset = scroll_result

                # Extract document IDs
                for point in points:
                    doc_id = point.payload.get("document_id")
                    if doc_id:
                        document_ids.add(doc_id)
                        # Stop if we've reached the limit
                        if limit and len(document_ids) >= limit:
                            return list(document_ids)[:limit]

                # Check if we've reached the end
                if next_offset is None or len(points) == 0:
                    break

                offset = next_offset

            return list(document_ids)

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(f"Failed to get document IDs: {str(e)}") from e

    def count_chunks(
        self,
        collection_name: str,
        filters: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> int:
        """Count chunks matching criteria.

        Args:
            collection_name: Name of the collection
            filters: Metadata filters
            document_ids: Specific document IDs to count chunks for

        Returns:
            Number of chunks matching criteria
        """
        try:
            from qdrant_client.models import (
                Filter,
                FieldCondition,
                MatchValue,
                MatchAny,
            )

            # Verify collection exists
            if not self.collection_exists(collection_name):
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' does not exist"
                )

            # Build filter query
            conditions = []

            # Add document_ids filter if provided
            if document_ids:
                conditions.append(
                    FieldCondition(key="document_id", match=MatchAny(any=document_ids))
                )

            # Add other filters
            if filters:
                for key, value in filters.items():
                    if value is not None and value != "" and value != {}:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )

            query_filter = Filter(must=conditions) if conditions else None

            # Use count operation
            count_result = self.client.count(
                collection_name=collection_name,
                count_filter=query_filter,
                exact=True,
            )

            return count_result.count

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(f"Failed to count chunks: {str(e)}") from e

    def get_chunk_ids_by_documents(
        self,
        collection_name: str,
        document_ids: List[str],
    ) -> List[str]:
        """Get all chunk IDs belonging to specific documents.

        Args:
            collection_name: Name of the collection
            document_ids: List of document IDs

        Returns:
            List of chunk IDs
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny

            # Verify collection exists
            if not self.collection_exists(collection_name):
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' does not exist"
                )

            if not document_ids:
                return []

            # Build filter for document_ids
            query_filter = Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchAny(any=document_ids))
                ]
            )

            # Scroll through all matching points
            chunk_ids = []
            offset = None

            while True:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=query_filter,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points, next_offset = scroll_result

                # Extract chunk IDs
                for point in points:
                    chunk_id = point.payload.get("chunk_id")
                    if chunk_id:
                        chunk_ids.append(chunk_id)

                # Check if we've reached the end
                if next_offset is None or len(points) == 0:
                    break

                offset = next_offset

            return chunk_ids

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(f"Failed to get chunk IDs: {str(e)}") from e

    def update_metadata(
        self,
        collection_name: str,
        filters: Optional[Dict[str, Any]] = None,
        chunk_ids: Optional[List[str]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Update metadata for existing chunks without reprocessing content.

        Args:
            collection_name: Name of the collection
            filters: Metadata filters to match chunks
            chunk_ids: Specific chunk IDs to update
            metadata_updates: Dictionary of metadata fields to update

        Returns:
            Number of chunks updated
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Verify collection exists
            if not self.collection_exists(collection_name):
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' does not exist"
                )

            if not metadata_updates:
                return 0

            # If chunk_ids provided, update by IDs
            if chunk_ids:
                # Convert chunk_ids to point_ids
                point_ids = [
                    str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
                    for chunk_id in chunk_ids
                ]

                # Update payload for each point
                for point_id in point_ids:
                    self.client.set_payload(
                        collection_name=collection_name,
                        payload=metadata_updates,
                        points=[point_id],
                    )

                return len(chunk_ids)

            # Otherwise use filters
            if filters:
                # Build filter
                conditions = []
                for key, value in filters.items():
                    if value is not None and value != "" and value != {}:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )

                if not conditions:
                    return 0

                query_filter = Filter(must=conditions)

                # Count affected chunks first
                count_result = self.client.count(
                    collection_name=collection_name,
                    count_filter=query_filter,
                    exact=True,
                )

                # Update payload using filter
                self.client.set_payload(
                    collection_name=collection_name,
                    payload=metadata_updates,
                    points=query_filter,
                )

                return count_result.count

            return 0

        except CollectionNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(f"Failed to update metadata: {str(e)}") from e
