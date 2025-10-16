"""Main RAGClient - entry point for all RAG operations."""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..chunking.semantic import SemanticChunker
from ..embedding.openai import OpenAIEmbedder
from ..exceptions import ConfigurationError, ValidationError, VectorDBError
from ..models.document import DocumentInput, SourceType
from ..models.response import AddDocumentsResponse, ProcessingStats, UpdateDocumentsResponse
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
            https=self.config.vectordb.https,
            verify_ssl=self.config.vectordb.verify_ssl,
        )

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

            # PHASE 6: Vector Storage
            print(f"Storing vectors and content in collection '{collection_name}'...")
            upload_start = time.time()

            # Ensure collection exists
            if not self.vectordb.collection_exists(collection_name):
                print(f"Creating collection '{collection_name}'...")
                self.vectordb.create_collection(
                    collection_name=collection_name,
                    vector_size=self.embedder.get_dimensions(),
                    distance_metric=self.config.retrieval.distance_metric,
                )

            # Prepare data for Qdrant storage
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

    def update_documents(
        self,
        collection_name: str,
        update_strategy: str,
        filters: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
        new_documents: Optional[List[DocumentInput]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
        reprocess_chunks: bool = True,
    ) -> UpdateDocumentsResponse:
        """Update, replace, or delete existing documents in the knowledge base.

        This method provides flexible document management operations:
        - replace: Delete existing documents and add new ones
        - append: Add new documents without deleting
        - delete: Remove documents matching criteria
        - upsert: Update if exists, insert if doesn't

        Args:
            collection_name: Target Qdrant collection
            update_strategy: Operation type ("replace", "append", "delete", "upsert")
            filters: Metadata-based selection criteria (e.g., {"user_id": "123"})
            document_ids: Specific document IDs to target
            new_documents: Replacement or additional documents (for replace/append/upsert)
            metadata_updates: Metadata field updates (metadata-only updates)
            reprocess_chunks: If True, regenerate chunks and embeddings; if False, metadata-only updates

        Returns:
            UpdateDocumentsResponse with operation results

        Raises:
            ValidationError: Invalid parameters
            CollectionNotFoundError: Collection doesn't exist
            NoDocumentsFoundError: No documents match criteria (for delete/replace)
            VectorDBError: Qdrant operation failures
        """
        from ..exceptions import CollectionNotFoundError, NoDocumentsFoundError

        start_time = time.time()
        errors = []
        chunks_deleted = 0
        chunks_added = 0
        chunks_updated = 0
        updated_document_ids = []

        try:
            # ===================================================================
            # VALIDATION
            # ===================================================================
            # Validate update strategy
            valid_strategies = ["replace", "append", "delete", "upsert"]
            if update_strategy not in valid_strategies:
                raise ValidationError(
                    f"Invalid update_strategy: '{update_strategy}'. "
                    f"Must be one of: {', '.join(valid_strategies)}"
                )

            # Validate collection exists
            if not self.vectordb.collection_exists(collection_name):
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' does not exist"
                )

            # Validate strategy-specific requirements
            if update_strategy in ["replace", "delete"]:
                if not filters and not document_ids:
                    raise ValidationError(
                        f"'{update_strategy}' strategy requires either 'filters' or 'document_ids'"
                    )

            if update_strategy in ["replace", "append", "upsert"]:
                if not new_documents:
                    raise ValidationError(
                        f"'{update_strategy}' strategy requires 'new_documents'"
                    )

            if not reprocess_chunks and not metadata_updates:
                raise ValidationError(
                    "When reprocess_chunks=False, metadata_updates must be provided"
                )

            print(f"\n{'='*60}")
            print(f"UPDATE OPERATION: {update_strategy.upper()}")
            print(f"{'='*60}")
            print(f"Collection: {collection_name}")
            if filters:
                print(f"Filters: {filters}")
            if document_ids:
                print(f"Document IDs: {document_ids}")

            # ===================================================================
            # STRATEGY EXECUTION
            # ===================================================================

            if update_strategy == "delete":
                # DELETE STRATEGY: Remove documents
                print(f"\nExecuting DELETE strategy...")

                # Determine document IDs to delete
                if document_ids:
                    updated_document_ids = document_ids
                else:
                    # Get document IDs using filters
                    updated_document_ids = self.vectordb.get_document_ids(collection_name, filters)

                if not updated_document_ids:
                    raise NoDocumentsFoundError(
                        "No documents found matching the specified criteria"
                    )

                print(f"Deleting chunks for {len(updated_document_ids)} document(s)")

                # Delete from Qdrant using filter-based deletion (more efficient)
                chunks_deleted = self.vectordb.delete_by_document_ids(
                    collection_name=collection_name,
                    document_ids=updated_document_ids,
                )

                print(f"✓ Deleted {chunks_deleted} chunks")

            elif update_strategy == "append":
                # APPEND STRATEGY: Just add new documents
                print(f"\nExecuting APPEND strategy...")
                print(f"Adding {len(new_documents)} new document(s)...")

                # Use existing add_documents pipeline
                add_response = self.add_documents(
                    documents=new_documents,
                    collection_name=collection_name,
                    metadata=metadata_updates or {},
                )

                if not add_response.success:
                    errors.extend(add_response.errors)
                    raise VectorDBError(f"Failed to add documents: {add_response.errors}")

                chunks_added = add_response.total_chunks
                updated_document_ids = [
                    chunk.metadata.document_id
                    for chunk in add_response.chunks
                ]
                updated_document_ids = list(set(updated_document_ids))  # Unique IDs

                print(f"✓ Added {chunks_added} new chunks from {len(updated_document_ids)} document(s)")

            elif update_strategy == "replace":
                # REPLACE STRATEGY: Delete existing + add new
                print(f"\nExecuting REPLACE strategy...")

                # Step 1: Determine documents to replace
                if document_ids:
                    docs_to_replace = document_ids
                else:
                    docs_to_replace = self.vectordb.get_document_ids(collection_name, filters)

                if not docs_to_replace:
                    raise NoDocumentsFoundError(
                        "No documents found matching the specified criteria"
                    )

                print(f"Replacing {len(docs_to_replace)} document(s)")

                # Step 2: Delete existing chunks using filter-based deletion
                chunks_deleted = self.vectordb.delete_by_document_ids(
                    collection_name=collection_name,
                    document_ids=docs_to_replace,
                )

                print(f"✓ Deleted {chunks_deleted} old chunks")

                # Step 3: Add new documents
                print(f"Adding {len(new_documents)} replacement document(s)...")
                add_response = self.add_documents(
                    documents=new_documents,
                    collection_name=collection_name,
                    metadata=metadata_updates or {},
                )

                if not add_response.success:
                    errors.extend(add_response.errors)
                    raise VectorDBError(f"Failed to add replacement documents: {add_response.errors}")

                chunks_added = add_response.total_chunks
                updated_document_ids = [
                    chunk.metadata.document_id
                    for chunk in add_response.chunks
                ]
                updated_document_ids = list(set(updated_document_ids))

                print(f"✓ Added {chunks_added} new chunks from {len(updated_document_ids)} document(s)")

            elif update_strategy == "upsert":
                # UPSERT STRATEGY: Update if exists, insert if not
                print(f"\nExecuting UPSERT strategy...")
                print(f"Processing {len(new_documents)} document(s) for upsert...")

                docs_to_insert = []
                docs_to_update = []

                # Check each document to see if it exists
                for doc in new_documents:
                    # Extract document_id from metadata
                    doc_id = doc.metadata.get("document_id")
                    if not doc_id:
                        # Generate new ID if not provided
                        doc_id = str(uuid.uuid4())
                        doc.metadata["document_id"] = doc_id
                        docs_to_insert.append(doc)
                    else:
                        # Check if document exists
                        existing_chunks = self.vectordb.count_chunks(
                            collection_name=collection_name,
                            document_ids=[doc_id],
                        )
                        if existing_chunks > 0:
                            docs_to_update.append(doc)
                        else:
                            docs_to_insert.append(doc)

                print(f"Documents to insert: {len(docs_to_insert)}")
                print(f"Documents to update: {len(docs_to_update)}")

                # Process updates (replace existing)
                if docs_to_update:
                    for doc in docs_to_update:
                        doc_id = doc.metadata["document_id"]
                        print(f"  Updating document: {doc_id}")

                        # Delete existing chunks using filter-based deletion
                        deleted = self.vectordb.delete_by_document_ids(
                            collection_name=collection_name,
                            document_ids=[doc_id],
                        )
                        chunks_deleted += deleted

                    # Add updated documents
                    update_response = self.add_documents(
                        documents=docs_to_update,
                        collection_name=collection_name,
                        metadata=metadata_updates or {},
                    )
                    if update_response.success:
                        chunks_updated += update_response.total_chunks
                        updated_document_ids.extend([
                            chunk.metadata.document_id
                            for chunk in update_response.chunks
                        ])
                    else:
                        errors.extend(update_response.errors)

                # Process inserts (new documents)
                if docs_to_insert:
                    insert_response = self.add_documents(
                        documents=docs_to_insert,
                        collection_name=collection_name,
                        metadata=metadata_updates or {},
                    )
                    if insert_response.success:
                        chunks_added += insert_response.total_chunks
                        updated_document_ids.extend([
                            chunk.metadata.document_id
                            for chunk in insert_response.chunks
                        ])
                    else:
                        errors.extend(insert_response.errors)

                updated_document_ids = list(set(updated_document_ids))
                print(f"✓ Upserted {chunks_updated + chunks_added} chunks total")
                print(f"  - Updated: {chunks_updated} chunks")
                print(f"  - Inserted: {chunks_added} chunks")

            # Handle metadata-only updates (when reprocess_chunks=False)
            if not reprocess_chunks and metadata_updates:
                print(f"\nPerforming metadata-only update...")

                # Update metadata without reprocessing content
                if document_ids:
                    chunk_ids = self.vectordb.get_chunk_ids_by_documents(
                        collection_name, document_ids
                    )
                    updated_count = self.vectordb.update_metadata(
                        collection_name=collection_name,
                        chunk_ids=chunk_ids,
                        metadata_updates=metadata_updates,
                    )
                elif filters:
                    updated_count = self.vectordb.update_metadata(
                        collection_name=collection_name,
                        filters=filters,
                        metadata_updates=metadata_updates,
                    )
                else:
                    updated_count = 0

                chunks_updated = updated_count
                print(f"✓ Updated metadata for {chunks_updated} chunks")

            # Calculate total time
            total_time = (time.time() - start_time) * 1000

            # Print summary
            print(f"\n{'='*60}")
            print(f"UPDATE COMPLETE")
            print(f"{'='*60}")
            print(f"Strategy: {update_strategy}")
            print(f"Chunks deleted: {chunks_deleted}")
            print(f"Chunks added: {chunks_added}")
            print(f"Chunks updated: {chunks_updated}")
            print(f"Documents affected: {len(updated_document_ids)}")
            print(f"Total time: {total_time:.2f}ms")
            print(f"{'='*60}\n")

            return UpdateDocumentsResponse(
                success=True,
                strategy_used=update_strategy,
                documents_affected=len(updated_document_ids),
                chunks_deleted=chunks_deleted,
                chunks_added=chunks_added,
                chunks_updated=chunks_updated,
                updated_document_ids=updated_document_ids,
                errors=errors,
            )

        except (ValidationError, CollectionNotFoundError, NoDocumentsFoundError) as e:
            # Expected errors - re-raise
            raise

        except Exception as e:
            # Unexpected errors
            errors.append(f"Update operation failed: {str(e)}")
            print(f"\n✗ Update failed: {e}")

            return UpdateDocumentsResponse(
                success=False,
                strategy_used=update_strategy,
                documents_affected=len(updated_document_ids),
                chunks_deleted=chunks_deleted,
                chunks_added=chunks_added,
                chunks_updated=chunks_updated,
                updated_document_ids=updated_document_ids,
                errors=errors,
            )

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

    def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """Search for relevant chunks using query.

        This method performs vector similarity search to find the most relevant
        chunks for the given query.

        Args:
            query: Search query text
            collection_name: Collection to search in
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            RetrievalResponse with search results
        """
        from ..models.response import RetrievalResponse, RetrievedChunk, RetrievalStats

        start_time = time.time()
        stats = RetrievalStats()

        try:
            # Generate query embedding
            embedding_start = time.time()
            query_embedding = self.embedder.embed_query(query)
            stats.query_generation_time_ms = (time.time() - embedding_start) * 1000

            # Perform vector search
            search_start = time.time()
            search_results = self.vectordb.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                filters=filters,
            )
            stats.vector_search_time_ms = (time.time() - search_start) * 1000
            stats.vector_search_chunks = len(search_results)

            # Convert to RetrievedChunk objects
            retrieved_chunks = []
            for rank, result in enumerate(search_results):
                chunk = RetrievedChunk(
                    content=result.content,
                    metadata=result.metadata,
                    relevance_score=result.score,
                    vector_score=result.score,
                    rank=rank,
                )
                retrieved_chunks.append(chunk)

            stats.chunks_after_reranking = len(retrieved_chunks)
            stats.total_chunks_retrieved = len(retrieved_chunks)
            stats.total_time_ms = (time.time() - start_time) * 1000

            # Calculate source statistics
            from ..models.response import SourceInfo
            from collections import defaultdict

            source_stats = defaultdict(lambda: {"count": 0, "total_score": 0.0})
            for chunk in retrieved_chunks:
                source = chunk.metadata.get("source", "unknown")
                source_stats[source]["count"] += 1
                source_stats[source]["total_score"] += chunk.relevance_score

            sources = [
                SourceInfo(
                    source=source,
                    chunks_count=data["count"],
                    avg_relevance=data["total_score"] / data["count"],
                )
                for source, data in source_stats.items()
            ]

            return RetrievalResponse(
                success=True,
                query_original=query,
                queries_generated={"original": query},
                chunks=retrieved_chunks,
                retrieval_stats=stats,
                sources=sources,
                errors=[],
            )

        except Exception as e:
            stats.total_time_ms = (time.time() - start_time) * 1000
            return RetrievalResponse(
                success=False,
                query_original=query,
                retrieval_stats=stats,
                errors=[f"Search error: {str(e)}"],
            )

    def retrieve(
        self,
        query: str,
        collection_name: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        enable_reranking: bool = True,  # Phase 4 - BGE reranking ENABLED BY DEFAULT
        enable_keyword_search: bool = True,  # Phase 2 - ENABLED BY DEFAULT
        enable_hyde: bool = True,  # Phase 2 - ENABLED BY DEFAULT
        score_threshold: Optional[float] = None,
        return_full_chunks: bool = True,
        deduplicate: bool = True,
    ):
        """
        Advanced hybrid retrieval method (Phase 4 - with HyDE + BM25 + Reranking).

        Phase 4 implements:
        - HyDE query generation using Azure OpenAI
        - Dual vector search (standard + HyDE queries)
        - BM25 keyword search for exact term matching
        - Smart deduplication
        - BGE reranking (BAAI/bge-reranker-v2-m3)
        - MongoDB content fetching (if enabled)

        Args:
            query: User's search question
            collection_name: Target Qdrant collection
            filters: Metadata filters (e.g., {"user_id": "123", "template_id": "456"})
            top_k: Final number of chunks to return (default: 20)
            enable_reranking: Use BGE reranking (Phase 4 - default: True)
            enable_keyword_search: Include BM25 keyword search (default: True)
            enable_hyde: Use HyDE query generation (default: True)
            score_threshold: Minimum relevance score filter (optional)
                Note: BGE reranker produces negative scores (higher = more relevant)
                Use negative thresholds like -5.0 for BGE, or 0.5 for normalized scores
            return_full_chunks: Return complete vs truncated content (default: True)
            deduplicate: Remove duplicate chunks (default: True)

        Returns:
            RetrievalResponse with search results and performance statistics

        Example:
            >>> response = client.retrieve(
            ...     query="What is semantic chunking?",
            ...     collection_name="knowledge_base",
            ...     top_k=10
            ... )
            >>> for chunk in response.chunks:
            ...     print(f"Score: {chunk.relevance_score:.4f}")
            ...     print(f"Content: {chunk.content[:100]}...")
        """
        from ..models.response import RetrievalResponse, RetrievedChunk, RetrievalStats, SourceInfo
        from collections import defaultdict

        start_time = time.time()
        stats = RetrievalStats()
        queries_generated = {"original": query}

        try:
            # ===================================================================
            # STEP 1: QUERY GENERATION (Phase 2: HyDE)
            # ===================================================================
            query_gen_start = time.time()

            if enable_hyde:
                # Use HyDE query generator
                from ..retrieval.query_generator import HyDEQueryGenerator

                try:
                    hyde_generator = HyDEQueryGenerator(self.config.llm)
                    generated = hyde_generator.generate_queries(query)
                    standard_query = generated["standard"]
                    hyde_query = generated["hyde"]
                    queries_generated["standard"] = standard_query
                    queries_generated["hyde"] = hyde_query
                except Exception as e:
                    print(f"   Warning: HyDE generation failed: {e}")
                    # Fallback to original query
                    standard_query = query
                    hyde_query = query
                    queries_generated["standard"] = standard_query
            else:
                # Use original query
                standard_query = query
                hyde_query = None
                queries_generated["standard"] = standard_query

            stats.query_generation_time_ms = (time.time() - query_gen_start) * 1000

            # Print generated queries for visibility
            print(f"\n📝 Query Generation ({stats.query_generation_time_ms:.2f}ms):")
            print(f"   Original Query: {query}")
            if enable_hyde:
                print(f"   Standard Query: {standard_query}")
                if hyde_query and hyde_query != standard_query:
                    print(f"   HyDE Query: {hyde_query[:200]}{'...' if len(hyde_query) > 200 else ''}")
                else:
                    print(f"   HyDE Query: (same as original - generation may have failed)")
            else:
                print(f"   HyDE: Disabled")

            # ===================================================================
            # STEP 2: DUAL VECTOR SEARCH (Phase 2: Standard + HyDE)
            # ===================================================================
            print(f"\n🔍 Vector Search:")
            vector_search_start = time.time()
            all_vector_results = []

            # Search 1: Standard query (25 chunks)
            print(f"   Search 1: Standard query → ", end="")
            embedding_1 = self.embedder.embed_query(standard_query)
            results_1 = self.vectordb.search(
                collection_name=collection_name,
                query_vector=embedding_1,
                limit=25,
                filters=filters,
            )
            all_vector_results.extend(results_1)
            print(f"{len(results_1)} chunks")

            # Search 2: HyDE query (25 chunks) if enabled
            if enable_hyde and hyde_query and hyde_query != standard_query:
                print(f"   Search 2: HyDE query → ", end="")
                embedding_2 = self.embedder.embed_query(hyde_query)
                results_2 = self.vectordb.search(
                    collection_name=collection_name,
                    query_vector=embedding_2,
                    limit=25,
                    filters=filters,
                )
                all_vector_results.extend(results_2)
                print(f"{len(results_2)} chunks")

            stats.vector_search_time_ms = (time.time() - vector_search_start) * 1000
            stats.vector_search_chunks = len(all_vector_results)
            print(f"   ✓ Total vector results: {len(all_vector_results)} chunks ({stats.vector_search_time_ms:.2f}ms)")

            # ===================================================================
            # STEP 3: KEYWORD SEARCH (Phase 2: BM25)
            # ===================================================================
            keyword_results = []
            if enable_keyword_search:
                print(f"\n🔎 Keyword Search (BM25):")
                print(f"   Using query: {query}")
                keyword_search_start = time.time()

                try:
                    from ..retrieval.keyword_search import BM25Searcher

                    # Build BM25 searcher (caches corpus)
                    bm25_searcher = BM25Searcher(self, collection_name)

                    # Perform BM25 search using original query (not HyDE)
                    bm25_results = bm25_searcher.search(
                        query=query, limit=50, filters=filters
                    )

                    # Convert to VectorSearchResult-like objects
                    for result in bm25_results:
                        # Create a simple result object
                        class BM25Result:
                            def __init__(self, data):
                                self.chunk_id = data["chunk_id"]
                                self.score = data["score"]
                                self.content = data["content"]
                                self.metadata = data["metadata"]
                                self.metadata["chunk_id"] = data["chunk_id"]

                        keyword_results.append(BM25Result(result))

                    stats.keyword_search_time_ms = (
                        time.time() - keyword_search_start
                    ) * 1000
                    stats.keyword_search_chunks = len(keyword_results)
                    print(f"   ✓ BM25 results: {len(keyword_results)} chunks ({stats.keyword_search_time_ms:.2f}ms)")

                except Exception as e:
                    print(f"   ⚠️ Warning: BM25 search failed: {e}")
                    stats.keyword_search_time_ms = 0.0
                    stats.keyword_search_chunks = 0
            else:
                print(f"\n🔎 Keyword Search: Disabled")
                stats.keyword_search_chunks = 0
                stats.keyword_search_time_ms = 0.0

            # ===================================================================
            # STEP 4: COMBINE & DEDUPLICATE (Vector + Keyword results)
            # ===================================================================
            print(f"\n🔀 Combining & Deduplicating:")
            all_chunks = all_vector_results + keyword_results
            stats.total_chunks_retrieved = len(all_chunks)
            print(f"   Combined: {len(all_chunks)} total chunks")

            if deduplicate:
                # Deduplicate by chunk_id, keep highest score
                chunk_dict = {}
                for chunk in all_chunks:
                    chunk_id = chunk.chunk_id
                    if chunk_id not in chunk_dict or chunk.score > chunk_dict[chunk_id].score:
                        chunk_dict[chunk_id] = chunk
                unique_chunks = list(chunk_dict.values())
                print(f"   After deduplication: {len(unique_chunks)} unique chunks")
            else:
                unique_chunks = all_chunks
                print(f"   Deduplication: Disabled")

            stats.chunks_after_dedup = len(unique_chunks)

            # ===================================================================
            # STEP 5: RERANKING (Phase 4 - BGE Reranking with LLM Fallback)
            # ===================================================================
            print(f"\n🎯 Reranking:")
            reranking_start = time.time()

            if enable_reranking and self.config.reranking.enabled:
                try:
                    print(f"   Reranking {len(unique_chunks)} chunks using {self.config.reranking.provider}...")

                    # Initialize reranker
                    if self.config.reranking.provider == "bge":
                        from ..retrieval.reranker import BGEReranker

                        reranker = BGEReranker(
                            api_key=self.config.reranking.api_key,
                            api_url=self.config.reranking.api_url,
                            normalize=self.config.reranking.normalize,
                            timeout=self.config.reranking.timeout,
                        )
                    elif self.config.reranking.provider == "cohere":
                        from ..retrieval.reranker import CohereReranker

                        reranker = CohereReranker(
                            api_key=self.config.reranking.api_key,
                            model=self.config.reranking.model,
                        )
                    else:
                        raise ValueError(f"Unknown reranker provider: {self.config.reranking.provider}")

                    # Prepare chunks for reranking: list of (content, metadata) tuples
                    chunks_for_reranking = [
                        (chunk.content, chunk.metadata) for chunk in unique_chunks
                    ]

                    # Rerank - returns list of (original_index, relevance_score) tuples
                    reranked_results = reranker.rerank(
                        query=query,
                        chunks=chunks_for_reranking,
                        top_k=min(self.config.reranking.top_k, len(unique_chunks))
                    )

                    # Apply reranking scores and reorder chunks
                    ranked_chunks = []
                    for original_index, rerank_score in reranked_results:
                        chunk = unique_chunks[original_index]
                        # Update the score with reranking score
                        chunk.score = rerank_score
                        ranked_chunks.append(chunk)

                    stats.reranking_time_ms = (time.time() - reranking_start) * 1000
                    print(f"   ✓ Reranked to {len(ranked_chunks)} chunks ({stats.reranking_time_ms:.2f}ms)")
                    print(f"   ✓ Score range: {ranked_chunks[-1].score:.4f} to {ranked_chunks[0].score:.4f}")

                except Exception as e:
                    print(f"   ⚠️ Warning: {self.config.reranking.provider.upper()} reranking failed: {e}")

                    # Try LLM fallback if enabled
                    if self.config.reranking.fallback_enabled and self.config.reranking.fallback_endpoint and self.config.reranking.fallback_api_key:
                        try:
                            print(f"   🔄 Attempting LLM fallback using {self.config.reranking.fallback_model}...")
                            from ..retrieval.reranker import LLMReranker

                            llm_reranker = LLMReranker(
                                api_key=self.config.reranking.fallback_api_key,
                                base_url=self.config.reranking.fallback_endpoint,
                                model=self.config.reranking.fallback_model,
                                timeout=self.config.reranking.fallback_timeout,
                            )

                            # Prepare chunks for reranking
                            chunks_for_reranking = [
                                (chunk.content, chunk.metadata) for chunk in unique_chunks
                            ]

                            print(f"   📤 Sending {len(chunks_for_reranking)} chunks to LLM reranker...")

                            # Rerank using LLM
                            reranked_results = llm_reranker.rerank(
                                query=query,
                                chunks=chunks_for_reranking,
                                top_k=min(self.config.reranking.top_k, len(unique_chunks))
                            )

                            # Apply reranking scores and reorder chunks
                            ranked_chunks = []
                            for original_index, rerank_score in reranked_results:
                                chunk = unique_chunks[original_index]
                                chunk.score = rerank_score
                                ranked_chunks.append(chunk)

                            stats.reranking_time_ms = (time.time() - reranking_start) * 1000
                            print(f"   ✅ LLM fallback successful! Reranked to {len(ranked_chunks)} chunks ({stats.reranking_time_ms:.2f}ms)")
                            print(f"   ✓ Score range: {ranked_chunks[-1].score:.4f} to {ranked_chunks[0].score:.4f}")

                        except Exception as fallback_error:
                            print(f"   ⚠️ Warning: LLM fallback also failed: {fallback_error}")
                            print(f"   Falling back to vector score sorting...")
                            # Fallback to vector score sorting
                            ranked_chunks = sorted(unique_chunks, key=lambda x: x.score, reverse=True)
                            stats.reranking_time_ms = (time.time() - reranking_start) * 1000
                    else:
                        print(f"   LLM fallback not enabled or not configured")
                        print(f"   Falling back to vector score sorting...")
                        # Fallback to vector score sorting
                        ranked_chunks = sorted(unique_chunks, key=lambda x: x.score, reverse=True)
                        stats.reranking_time_ms = (time.time() - reranking_start) * 1000
            else:
                # Reranking disabled - sort by vector score
                ranked_chunks = sorted(unique_chunks, key=lambda x: x.score, reverse=True)
                stats.reranking_time_ms = 0.0
                print(f"   Reranking: Disabled")

            # ===================================================================
            # STEP 6: SELECTION & FORMATTING
            # ===================================================================
            print(f"   Step 6: Selecting top-{top_k} chunks from {len(ranked_chunks)} ranked chunks")

            # Select top_k chunks
            final_chunks = ranked_chunks[:top_k]
            print(f"   After top-k selection: {len(final_chunks)} chunks")

            # Apply score threshold if specified
            if score_threshold is not None:
                filtered_count = len(final_chunks)
                final_chunks = [c for c in final_chunks if c.score >= score_threshold]
                print(f"   After score threshold ({score_threshold}): {len(final_chunks)} chunks (filtered out: {filtered_count - len(final_chunks)})")

            stats.chunks_after_reranking = len(final_chunks)
            print(f"   ✓ Final chunks to return: {len(final_chunks)}")

            # Convert to RetrievedChunk objects
            retrieved_chunks = []
            empty_content_count = 0
            for rank, result in enumerate(final_chunks):
                # Truncate content if needed
                content = result.content if return_full_chunks else result.content[:500]

                # Track empty content
                if not content or len(content.strip()) == 0:
                    empty_content_count += 1

                chunk = RetrievedChunk(
                    content=content,
                    metadata=result.metadata,
                    relevance_score=result.score,
                    vector_score=result.score,
                    rank=rank,
                    keyword_score=None,  # Updated in Phase 2 if BM25 used
                )
                retrieved_chunks.append(chunk)

            if empty_content_count > 0:
                print(f"   ⚠️ Warning: {empty_content_count} chunks have empty content!")

            print(f"   ✓ Returning {len(retrieved_chunks)} chunks with content")

            # Calculate source statistics
            source_stats = defaultdict(lambda: {"count": 0, "total_score": 0.0})
            for chunk in retrieved_chunks:
                source = chunk.metadata.get("source", "unknown")
                source_stats[source]["count"] += 1
                source_stats[source]["total_score"] += chunk.relevance_score

            sources = [
                SourceInfo(
                    source=source,
                    chunks_count=data["count"],
                    avg_relevance=data["total_score"] / data["count"] if data["count"] > 0 else 0.0,
                )
                for source, data in source_stats.items()
            ]

            # Calculate total time
            stats.total_time_ms = (time.time() - start_time) * 1000

            # Print retrieval summary
            print(f"\n{'='*60}")
            print(f"✅ RETRIEVAL COMPLETE")
            print(f"{'='*60}")
            print(f"📊 Summary:")
            print(f"   Query: '{query}'")
            if enable_hyde and queries_generated.get("standard"):
                print(f"   Searches performed:")
                print(f"     1. Standard query: '{queries_generated['standard']}'")
                if queries_generated.get("hyde") and queries_generated["hyde"] != queries_generated["standard"]:
                    print(f"     2. HyDE query: '{queries_generated['hyde'][:80]}...'")
                if enable_keyword_search and stats.keyword_search_chunks > 0:
                    print(f"     3. BM25 keyword search")
            print(f"   Total chunks retrieved: {stats.total_chunks_retrieved}")
            print(f"   After deduplication: {stats.chunks_after_dedup}")
            print(f"   Final results returned: {len(retrieved_chunks)}")
            print(f"   Total time: {stats.total_time_ms:.2f}ms")
            print(f"{'='*60}\n")

            return RetrievalResponse(
                success=True,
                query_original=query,
                queries_generated=queries_generated,
                chunks=retrieved_chunks,
                retrieval_stats=stats,
                sources=sources,
                errors=[],
            )

        except Exception as e:
            stats.total_time_ms = (time.time() - start_time) * 1000
            return RetrievalResponse(
                success=False,
                query_original=query,
                queries_generated=queries_generated,
                retrieval_stats=stats,
                errors=[f"Retrieval error: {str(e)}"],
            )