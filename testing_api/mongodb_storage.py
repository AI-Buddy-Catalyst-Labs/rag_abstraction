"""MongoDB storage helper for testing_api - stores chunk content separately from vectors."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId


class MongoDBStorage:
    """MongoDB storage for chunk content in testing_api.

    This class handles storing chunk content separately from Qdrant vectors.
    The mongodb_id is stored in Qdrant metadata to link them together.
    """

    def __init__(self, connection_string: str, database_name: str = "Test_Insta_RAG"):
        """Initialize MongoDB storage.

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
            raise Exception(
                "pymongo not installed. Install with: pip install pymongo"
            ) from e
        except ConnectionFailure as e:
            raise Exception(f"Failed to connect to MongoDB: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Failed to initialize MongoDB: {str(e)}") from e

    def _create_indexes(self):
        """Create necessary indexes for collections."""
        # Chunk contents collection
        self.db.chunk_contents.create_index("chunk_id", unique=True)
        self.db.chunk_contents.create_index("document_id")
        self.db.chunk_contents.create_index("collection_name")
        self.db.chunk_contents.create_index("created_at")

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

            result = self.db.chunk_contents.insert_one(document)
            return str(result.inserted_id)

        except Exception as e:
            raise Exception(f"Failed to store chunk in MongoDB: {str(e)}") from e

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

            result = self.db.chunk_contents.insert_many(chunks)
            return [str(oid) for oid in result.inserted_ids]

        except Exception as e:
            raise Exception(
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
            doc = self.db.chunk_contents.find_one({"chunk_id": chunk_id})
            if doc:
                doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
            return doc

        except Exception as e:
            raise Exception(
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
            doc = self.db.chunk_contents.find_one({"_id": ObjectId(mongo_id)})
            if doc:
                doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
            return doc

        except Exception as e:
            raise Exception(
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
                self.db.chunk_contents.find({"document_id": document_id}).sort(
                    "metadata.chunk_index", 1
                )
            )

            # Convert ObjectIds to strings
            for chunk in chunks:
                chunk["_id"] = str(chunk["_id"])

            return chunks

        except Exception as e:
            raise Exception(
                f"Failed to retrieve chunks by document: {str(e)}"
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

            result = self.db.chunk_contents.delete_many({"document_id": {"$in": document_ids}})
            return result.deleted_count

        except Exception as e:
            raise Exception(
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
            result = self.db.chunk_contents.delete_many(
                {"collection_name": collection_name}
            )
            return result.deleted_count

        except Exception as e:
            raise Exception(
                f"Failed to delete chunks by collection: {str(e)}"
            ) from e

    def close(self):
        """Close MongoDB connection."""
        if hasattr(self, "client"):
            self.client.close()
