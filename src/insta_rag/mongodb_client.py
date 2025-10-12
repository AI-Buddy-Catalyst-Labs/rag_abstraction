"""MongoDB client for storing document contents."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId

from .exceptions import VectorDBError


class MongoDBClient:
    """MongoDB client for storing document chunks and content."""

    def __init__(self, connection_string: str, database_name: str = "Test_Insta_RAG"):
        """Initialize MongoDB client.

        Args:
            connection_string: MongoDB connection string
            database_name: Database name (default: Test_Insta_RAG)
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self._initialize_client()

    def _initialize_client(self):
        """Initialize MongoDB client and database."""
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure

            self.client = MongoClient(self.connection_string)
            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]

            # Create indexes
            self._create_indexes()

        except ImportError as e:
            raise VectorDBError(
                "pymongo not installed. Install with: pip install pymongo"
            ) from e
        except ConnectionFailure as e:
            raise VectorDBError(f"Failed to connect to MongoDB: {str(e)}") from e
        except Exception as e:
            raise VectorDBError(f"Failed to initialize MongoDB: {str(e)}") from e

    def _create_indexes(self):
        """Create necessary indexes for collections."""
        # Document contents collection
        self.db.document_contents.create_index("chunk_id", unique=True)
        self.db.document_contents.create_index("document_id")
        self.db.document_contents.create_index("collection_name")
        self.db.document_contents.create_index("created_at")

        # Document metadata collection (optional, for aggregated info)
        self.db.document_metadata.create_index("document_id", unique=True)
        self.db.document_metadata.create_index("collection_name")

    def store_chunk_content(
        self,
        chunk_id: str,
        content: str,
        document_id: str,
        collection_name: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Store chunk content in MongoDB.

        Args:
            chunk_id: Unique chunk identifier
            content: Full text content of the chunk
            document_id: Parent document identifier
            collection_name: Qdrant collection name
            metadata: Additional metadata

        Returns:
            MongoDB document ID (as string)
        """
        try:
            document = {
                "chunk_id": chunk_id,
                "content": content,
                "document_id": document_id,
                "collection_name": collection_name,
                "metadata": metadata,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            result = self.db.document_contents.insert_one(document)
            return str(result.inserted_id)

        except Exception as e:
            raise VectorDBError(f"Failed to store chunk in MongoDB: {str(e)}") from e

    def store_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[str]:
        """Store multiple chunks in MongoDB.

        Args:
            chunks: List of chunk documents with keys:
                - chunk_id: str
                - content: str
                - document_id: str
                - collection_name: str
                - metadata: dict

        Returns:
            List of MongoDB document IDs (as strings)
        """
        try:
            # Add timestamps
            for chunk in chunks:
                chunk["created_at"] = datetime.utcnow()
                chunk["updated_at"] = datetime.utcnow()

            result = self.db.document_contents.insert_many(chunks)
            return [str(oid) for oid in result.inserted_ids]

        except Exception as e:
            raise VectorDBError(
                f"Failed to store chunks batch in MongoDB: {str(e)}"
            ) from e

    def get_chunk_content(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk content by chunk_id.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Document with content and metadata, or None if not found
        """
        try:
            doc = self.db.document_contents.find_one({"chunk_id": chunk_id})
            if doc:
                doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
            return doc

        except Exception as e:
            raise VectorDBError(
                f"Failed to retrieve chunk from MongoDB: {str(e)}"
            ) from e

    def get_chunk_content_by_mongo_id(self, mongo_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk content by MongoDB _id.

        Args:
            mongo_id: MongoDB document ID

        Returns:
            Document with content and metadata, or None if not found
        """
        try:
            doc = self.db.document_contents.find_one({"_id": ObjectId(mongo_id)})
            if doc:
                doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
            return doc

        except Exception as e:
            raise VectorDBError(
                f"Failed to retrieve chunk by MongoDB ID: {str(e)}"
            ) from e

    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of chunk documents
        """
        try:
            chunks = list(
                self.db.document_contents.find({"document_id": document_id}).sort(
                    "metadata.chunk_index", 1
                )
            )

            # Convert ObjectIds to strings
            for chunk in chunks:
                chunk["_id"] = str(chunk["_id"])

            return chunks

        except Exception as e:
            raise VectorDBError(
                f"Failed to retrieve chunks by document: {str(e)}"
            ) from e

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by chunk_id.

        Args:
            chunk_id: Chunk identifier

        Returns:
            True if deleted, False if not found
        """
        try:
            result = self.db.document_contents.delete_one({"chunk_id": chunk_id})
            return result.deleted_count > 0

        except Exception as e:
            raise VectorDBError(f"Failed to delete chunk from MongoDB: {str(e)}") from e

    def delete_chunks_by_ids(self, chunk_ids: List[str]) -> int:
        """Delete multiple chunks by chunk_ids.

        Args:
            chunk_ids: List of chunk identifiers

        Returns:
            Number of chunks deleted
        """
        try:
            if not chunk_ids:
                return 0

            result = self.db.document_contents.delete_many({"chunk_id": {"$in": chunk_ids}})
            return result.deleted_count

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete chunks by IDs: {str(e)}"
            ) from e

    def delete_chunks_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Document identifier

        Returns:
            Number of chunks deleted
        """
        try:
            result = self.db.document_contents.delete_many({"document_id": document_id})
            return result.deleted_count

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete chunks by document: {str(e)}"
            ) from e

    def delete_chunks_by_document_ids(self, document_ids: List[str]) -> int:
        """Delete all chunks for multiple documents.

        Args:
            document_ids: List of document identifiers

        Returns:
            Number of chunks deleted
        """
        try:
            if not document_ids:
                return 0

            result = self.db.document_contents.delete_many({"document_id": {"$in": document_ids}})
            return result.deleted_count

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete chunks by document IDs: {str(e)}"
            ) from e

    def delete_chunks_by_collection(self, collection_name: str) -> int:
        """Delete all chunks in a collection.

        Args:
            collection_name: Collection name

        Returns:
            Number of chunks deleted
        """
        try:
            result = self.db.document_contents.delete_many(
                {"collection_name": collection_name}
            )
            return result.deleted_count

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete chunks by collection: {str(e)}"
            ) from e

    def store_document_metadata(
        self,
        document_id: str,
        collection_name: str,
        total_chunks: int,
        metadata: Dict[str, Any],
    ) -> str:
        """Store aggregated document metadata.

        Args:
            document_id: Document identifier
            collection_name: Collection name
            total_chunks: Total number of chunks
            metadata: Document metadata

        Returns:
            MongoDB document ID
        """
        try:
            document = {
                "document_id": document_id,
                "collection_name": collection_name,
                "total_chunks": total_chunks,
                "metadata": metadata,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            result = self.db.document_metadata.update_one(
                {"document_id": document_id}, {"$set": document}, upsert=True
            )

            return str(result.upserted_id) if result.upserted_id else document_id

        except Exception as e:
            raise VectorDBError(
                f"Failed to store document metadata: {str(e)}"
            ) from e

    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata.

        Args:
            document_id: Document identifier

        Returns:
            Document metadata or None
        """
        try:
            doc = self.db.document_metadata.find_one({"document_id": document_id})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc

        except Exception as e:
            raise VectorDBError(
                f"Failed to retrieve document metadata: {str(e)}"
            ) from e

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection.

        Args:
            collection_name: Collection name

        Returns:
            Dictionary with collection statistics
        """
        try:
            total_chunks = self.db.document_contents.count_documents(
                {"collection_name": collection_name}
            )

            total_documents = self.db.document_metadata.count_documents(
                {"collection_name": collection_name}
            )

            # Get total content size
            pipeline = [
                {"$match": {"collection_name": collection_name}},
                {"$project": {"content_length": {"$strLenCP": "$content"}}},
                {"$group": {"_id": None, "total_size": {"$sum": "$content_length"}}},
            ]

            size_result = list(self.db.document_contents.aggregate(pipeline))
            total_size = size_result[0]["total_size"] if size_result else 0

            return {
                "collection_name": collection_name,
                "total_chunks": total_chunks,
                "total_documents": total_documents,
                "total_content_size_bytes": total_size,
                "database": self.database_name,
            }

        except Exception as e:
            raise VectorDBError(
                f"Failed to get collection stats: {str(e)}"
            ) from e

    def close(self):
        """Close MongoDB connection."""
        if hasattr(self, "client"):
            self.client.close()
