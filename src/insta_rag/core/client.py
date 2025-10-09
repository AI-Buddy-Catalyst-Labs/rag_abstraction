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

            # If MongoDB is enabled, fetch content
            if self.mongodb:
                for result in search_results:
                    if result.metadata.get("content_storage") == "mongodb":
                        mongodb_id = result.metadata.get("mongodb_id")
                        if mongodb_id:
                            # Fetch content from MongoDB
                            mongo_doc = self.mongodb.get_chunk_content_by_mongo_id(str(mongodb_id))
                            if mongo_doc:
                                result.content = mongo_doc.get("content", "")

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
        enable_reranking: bool = False,  # Phase 3 - Cohere reranking
        enable_keyword_search: bool = True,  # Phase 2 - ENABLED BY DEFAULT
        enable_hyde: bool = True,  # Phase 2 - ENABLED BY DEFAULT
        score_threshold: Optional[float] = None,
        return_full_chunks: bool = True,
        deduplicate: bool = True,
    ):
        """
        Advanced hybrid retrieval method (Phase 2 - with HyDE + BM25).

        Phase 2 implements:
        - HyDE query generation using Azure OpenAI
        - Dual vector search (standard + HyDE queries)
        - BM25 keyword search for exact term matching
        - Smart deduplication
        - MongoDB content fetching (if enabled)

        Future phases:
        - Phase 3: Cohere reranking

        Args:
            query: User's search question
            collection_name: Target Qdrant collection
            filters: Metadata filters (e.g., {"user_id": "123", "template_id": "456"})
            top_k: Final number of chunks to return (default: 20)
            enable_reranking: Use Cohere reranking (Phase 3 - not yet implemented)
            enable_keyword_search: Include BM25 keyword search (default: True)
            enable_hyde: Use HyDE query generation (default: True)
            score_threshold: Minimum relevance score filter (optional)
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

            # ===================================================================
            # STEP 2: DUAL VECTOR SEARCH (Phase 2: Standard + HyDE)
            # ===================================================================
            vector_search_start = time.time()
            all_vector_results = []

            # Search 1: Standard query (25 chunks)
            embedding_1 = self.embedder.embed_query(standard_query)
            results_1 = self.vectordb.search(
                collection_name=collection_name,
                query_vector=embedding_1,
                limit=25,
                filters=filters,
            )
            all_vector_results.extend(results_1)

            # Search 2: HyDE query (25 chunks) if enabled
            if enable_hyde and hyde_query and hyde_query != standard_query:
                embedding_2 = self.embedder.embed_query(hyde_query)
                results_2 = self.vectordb.search(
                    collection_name=collection_name,
                    query_vector=embedding_2,
                    limit=25,
                    filters=filters,
                )
                all_vector_results.extend(results_2)

            stats.vector_search_time_ms = (time.time() - vector_search_start) * 1000
            stats.vector_search_chunks = len(all_vector_results)

            # ===================================================================
            # STEP 3: KEYWORD SEARCH (Phase 2: BM25)
            # ===================================================================
            keyword_results = []
            if enable_keyword_search:
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

                except Exception as e:
                    print(f"   Warning: BM25 search failed: {e}")
                    stats.keyword_search_time_ms = 0.0
                    stats.keyword_search_chunks = 0
            else:
                stats.keyword_search_chunks = 0
                stats.keyword_search_time_ms = 0.0

            # ===================================================================
            # STEP 4: COMBINE & DEDUPLICATE (Vector + Keyword results)
            # ===================================================================
            all_chunks = all_vector_results + keyword_results
            stats.total_chunks_retrieved = len(all_chunks)

            if deduplicate:
                # Deduplicate by chunk_id, keep highest score
                chunk_dict = {}
                for chunk in all_chunks:
                    chunk_id = chunk.chunk_id
                    if chunk_id not in chunk_dict or chunk.score > chunk_dict[chunk_id].score:
                        chunk_dict[chunk_id] = chunk
                unique_chunks = list(chunk_dict.values())
            else:
                unique_chunks = all_chunks

            stats.chunks_after_dedup = len(unique_chunks)

            # Fetch content from MongoDB if enabled
            mongodb_fetch_count = 0
            if self.mongodb:
                for result in unique_chunks:
                    if result.metadata.get("content_storage") == "mongodb":
                        mongodb_id = result.metadata.get("mongodb_id")
                        if mongodb_id:
                            mongo_doc = self.mongodb.get_chunk_content_by_mongo_id(
                                str(mongodb_id)
                            )
                            if mongo_doc:
                                result.content = mongo_doc.get("content", "")
                                mongodb_fetch_count += 1

                if mongodb_fetch_count > 0:
                    print(f"   ✓ Fetched content for {mongodb_fetch_count} chunks from MongoDB")

            # ===================================================================
            # STEP 5: RERANKING (Phase 3 - Skipped for MVP)
            # ===================================================================
            # Sort by vector score (no reranking yet)
            ranked_chunks = sorted(unique_chunks, key=lambda x: x.score, reverse=True)
            stats.reranking_time_ms = 0.0

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
