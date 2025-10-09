"""Main RAGClient - entry point for all RAG operations."""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..chunking.semantic import SemanticChunker
from ..embedding.openai import OpenAIEmbedder
from ..exceptions import ConfigurationError, ValidationError, VectorDBError
from ..models.document import DocumentInput, SourceType
from ..models.response import AddDocumentsResponse, ProcessingStats
from ..mongodb_client import MongoDBClient
from ..pdf_processing import extract_text_from_pdf
from ..vectordb.qdrant import QdrantVectorDB
from .config import RAGConfig


class RAGClient:
    """Main RAG client for document operations.

    This client orchestrates all RAG operations including:
    - Document ingestion and processing
    - Semantic chunking
    - Embedding generation
    - Vector storage
    - Hybrid retrieval
    """

    def __init__(self, config: RAGConfig):
        """Initialize RAG client.

        Args:
            config: RAG configuration object
        """
        self.config = config

        # Validate configuration
        self.config.validate()

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all RAG components."""
        # Initialize embedding provider
        self.embedder = OpenAIEmbedder(
            api_key=self.config.embedding.api_key,
            model=self.config.embedding.model,
            dimensions=self.config.embedding.dimensions,
            api_base=self.config.embedding.api_base,
            api_version=self.config.embedding.api_version,
            deployment_name=self.config.embedding.deployment_name,
            batch_size=self.config.embedding.batch_size,
        )

        # Initialize vector database
        self.vectordb = QdrantVectorDB(
            url=self.config.vectordb.url,
            api_key=self.config.vectordb.api_key,
            timeout=self.config.vectordb.timeout,
            prefer_grpc=self.config.vectordb.prefer_grpc,
        )

        # Initialize MongoDB client (if configured)
        self.mongodb = None
        if self.config.mongodb and self.config.mongodb.enabled:
            self.mongodb = MongoDBClient(
                connection_string=self.config.mongodb.connection_string,
                database_name=self.config.mongodb.database_name,
            )
            print(f"✓ MongoDB connected: {self.config.mongodb.database_name}")

        # Initialize chunker
        self.chunker = SemanticChunker(
            embedder=self.embedder,
            max_chunk_size=self.config.chunking.max_chunk_size,
            overlap_percentage=self.config.chunking.overlap_percentage,
            threshold_percentile=self.config.chunking.semantic_threshold_percentile,
            min_chunk_size=self.config.chunking.min_chunk_size,
        )

    def add_documents(
        self,
        documents: List[DocumentInput],
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 100,
        validate_chunks: bool = True,
    ) -> AddDocumentsResponse:
        """Process and add documents to the knowledge base.

        This method implements the complete document processing pipeline:
        1. Document Loading
        2. Text Extraction
        3. Semantic Chunking
        4. Chunk Validation
        5. Batch Embedding
        6. Vector Storage

        Args:
            documents: List of DocumentInput objects
            collection_name: Target Qdrant collection name
            metadata: Global metadata for all chunks
            batch_size: Embedding batch size
            validate_chunks: Enable chunk quality validation

        Returns:
            AddDocumentsResponse with processing results
        """
        start_time = time.time()
        stats = ProcessingStats()
        all_chunks = []
        errors = []

        try:
            # PHASE 1 & 2: Document Loading and Text Extraction
            print(f"Processing {len(documents)} document(s)...")
            extracted_texts = []
            doc_metadata_list = []

            for i, doc in enumerate(documents):
                try:
                    text, doc_meta = self._load_and_extract_document(doc, metadata)
                    extracted_texts.append(text)
                    doc_metadata_list.append(doc_meta)
                except Exception as e:
                    errors.append(f"Document {i}: {str(e)}")
                    print(f"Error processing document {i}: {e}")

            if not extracted_texts:
                return AddDocumentsResponse(
                    success=False,
                    documents_processed=0,
                    total_chunks=0,
                    processing_stats=stats,
                    errors=errors,
                )

            # PHASE 3: Semantic Chunking
            print("Chunking documents...")
            chunking_start = time.time()

            for text, doc_meta in zip(extracted_texts, doc_metadata_list):
                try:
                    chunks = self.chunker.chunk(text, doc_meta)
                    all_chunks.extend(chunks)
                except Exception as e:
                    errors.append(f"Chunking error: {str(e)}")
                    print(f"Chunking error: {e}")

            stats.chunking_time_ms = (time.time() - chunking_start) * 1000

            if not all_chunks:
                return AddDocumentsResponse(
                    success=False,
                    documents_processed=len(extracted_texts),
                    total_chunks=0,
                    processing_stats=stats,
                    errors=errors,
                )

            print(f"Created {len(all_chunks)} chunks")

            # PHASE 4: Chunk Validation (already done in chunker)
            # Count total tokens
            stats.total_tokens = sum(chunk.metadata.token_count for chunk in all_chunks)

            # PHASE 5: Batch Embedding
            print("Generating embeddings...")
            embedding_start = time.time()

            chunk_texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedder.embed(chunk_texts)

            # Attach embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding

            stats.embedding_time_ms = (time.time() - embedding_start) * 1000

            # PHASE 6: Content Storage & Vector Storage
            print(f"Storing content and vectors in collection '{collection_name}'...")
            upload_start = time.time()

            # Ensure collection exists
            if not self.vectordb.collection_exists(collection_name):
                print(f"Creating collection '{collection_name}'...")
                self.vectordb.create_collection(
                    collection_name=collection_name,
                    vector_size=self.embedder.get_dimensions(),
                    distance_metric=self.config.retrieval.distance_metric,
                )

            # If MongoDB is enabled, store content there and reference in Qdrant
            if self.mongodb:
                print("Storing content in MongoDB...")

                # Prepare MongoDB documents
                mongo_docs = []
                for chunk in all_chunks:
                    mongo_docs.append({
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "document_id": chunk.metadata.document_id,
                        "collection_name": collection_name,
                        "metadata": chunk.metadata.to_dict(),
                    })

                # Store in MongoDB batch
                mongo_ids = self.mongodb.store_chunks_batch(mongo_docs)

                # Prepare Qdrant data with MongoDB references (no content)
                chunk_ids = [chunk.chunk_id for chunk in all_chunks]
                vectors = [chunk.embedding for chunk in all_chunks]
                contents = []  # Empty - content is in MongoDB
                metadatas = []

                for i, chunk in enumerate(all_chunks):
                    # Store MongoDB reference in metadata instead of content
                    meta = chunk.metadata.to_dict()
                    meta["mongodb_id"] = mongo_ids[i]
                    meta["content_storage"] = "mongodb"
                    metadatas.append(meta)
                    contents.append("")  # Empty content placeholder

                print(f"✓ Stored {len(mongo_ids)} chunks in MongoDB")

                # Store aggregated document metadata
                for doc_id in set(chunk.metadata.document_id for chunk in all_chunks):
                    doc_chunks = [c for c in all_chunks if c.metadata.document_id == doc_id]
                    self.mongodb.store_document_metadata(
                        document_id=doc_id,
                        collection_name=collection_name,
                        total_chunks=len(doc_chunks),
                        metadata=doc_chunks[0].metadata.to_dict() if doc_chunks else {},
                    )

            else:
                # Store content directly in Qdrant (original behavior)
                chunk_ids = [chunk.chunk_id for chunk in all_chunks]
                vectors = [chunk.embedding for chunk in all_chunks]
                contents = [chunk.content for chunk in all_chunks]
                metadatas = [chunk.metadata.to_dict() for chunk in all_chunks]

            # Upload to Qdrant
            self.vectordb.upsert(
                collection_name=collection_name,
                chunk_ids=chunk_ids,
                vectors=vectors,
                contents=contents,
                metadatas=metadatas,
            )

            stats.upload_time_ms = (time.time() - upload_start) * 1000

            # Calculate total time
            stats.total_time_ms = (time.time() - start_time) * 1000

            print(f"Successfully processed {len(extracted_texts)} documents into {len(all_chunks)} chunks")
            print(f"Total time: {stats.total_time_ms:.2f}ms")

            return AddDocumentsResponse(
                success=True,
                documents_processed=len(extracted_texts),
                total_chunks=len(all_chunks),
                chunks=all_chunks,
                processing_stats=stats,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Fatal error: {str(e)}")
            stats.total_time_ms = (time.time() - start_time) * 1000

            return AddDocumentsResponse(
                success=False,
                documents_processed=len(documents),
                total_chunks=len(all_chunks),
                chunks=all_chunks,
                processing_stats=stats,
                errors=errors,
            )

    def _load_and_extract_document(
        self, document: DocumentInput, global_metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[str, Dict[str, Any]]:
        """Load and extract text from a document.

        Args:
            document: DocumentInput object
            global_metadata: Global metadata to merge

        Returns:
            Tuple of (extracted_text, document_metadata)
        """
        # Generate document ID
        document_id = str(uuid.uuid4())

        # Merge metadata
        doc_metadata = {
            "document_id": document_id,
            **(global_metadata or {}),
            **document.metadata,
        }

        # Extract text based on source type
        if document.source_type == SourceType.FILE:
            file_path = document.get_source_path()
            doc_metadata["source"] = str(file_path)

            # Check file extension
            if file_path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(file_path, self.config.pdf.parser)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                text = file_path.read_text(encoding="utf-8")
            else:
                raise ValidationError(
                    f"Unsupported file type: {file_path.suffix}. "
                    "Supported types: .pdf, .txt, .md"
                )

        elif document.source_type == SourceType.TEXT:
            text = document.get_source_text()
            doc_metadata["source"] = "text_input"

        elif document.source_type == SourceType.BINARY:
            # For binary content, try to decode as PDF
            raise NotImplementedError(
                "Binary PDF processing not yet implemented. Use file path instead."
            )

        else:
            raise ValidationError(f"Unknown source type: {document.source_type}")

        return text, doc_metadata

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection information
        """
        return self.vectordb.get_collection_info(collection_name)

    def list_collections(self) -> List[str]:
        """List all available collections.

        Returns:
            List of collection names
        """
        collections = self.vectordb.client.get_collections().collections
        return [c.name for c in collections]
