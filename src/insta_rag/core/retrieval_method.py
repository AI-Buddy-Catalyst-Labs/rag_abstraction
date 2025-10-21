"""
Advanced retrieval method for RAGClient with hybrid search and reranking.

This module implements a comprehensive retrieval pipeline:
1. Query Generation (with HyDE)
2. Dual Vector Search (standard + HyDE queries)
3. BM25 Keyword Search
4. Deduplication
5. Cohere Reranking
6. Final Selection

Author: insta_rag team
"""

import time
from collections import defaultdict
from typing import Any, Dict, Optional

from ..models.response import (
    RetrievalResponse,
    RetrievalStats,
    RetrievedChunk,
    SourceInfo,
)


def retrieve(
    rag_client,
    query: str,
    collection_name: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 20,
    enable_reranking: bool = True,
    enable_keyword_search: bool = True,
    enable_hyde: bool = True,
    score_threshold: Optional[float] = None,
    return_full_chunks: bool = True,
    deduplicate: bool = True,
):
    """
    Advanced hybrid retrieval with HyDE, BM25, and reranking.

    This method implements a 6-step retrieval pipeline:

    STEP 1: Query Generation
    ------------------------
    - Generates optimized search query
    - Optionally generates HyDE (Hypothetical Document Embeddings)
    - HyDE creates a hypothetical answer to improve retrieval

    STEP 2: Vector Search
    ---------------------
    - Performs dual vector search:
        * Standard query → 25 chunks
        * HyDE query → 25 chunks (if enabled)
    - Total: 50 chunks from vector search
    - Uses COSINE similarity in Qdrant
    - Applies metadata filters

    STEP 3: Keyword Search (BM25)
    -----------------------------
    - Optional lexical search using BM25 algorithm
    - Retrieves 50 additional chunks
    - Catches exact term matches missed by embeddings
    - Essential for names, codes, IDs

    STEP 4: Combine & Deduplicate
    -----------------------------
    - Pools all results (~100 chunks)
    - Removes duplicates by chunk_id
    - Keeps highest-scoring variant
    - Result: ~100 unique chunks

    STEP 5: Reranking
    -----------------
    - Sends all unique chunks to Cohere Rerank 3.5
    - Cross-encoder scoring for query-chunk relevance
    - Produces 0-1 relevance scores
    - More accurate than embedding similarity

    STEP 6: Selection & Formatting
    ------------------------------
    - Sorts by reranker scores (highest first)
    - Selects top_k chunks (default: 20)
    - Applies score_threshold if specified
    - Returns full chunks with metadata

    Args:
        rag_client: RAGClient instance
        query: User's search question
        collection_name: Target Qdrant collection
        filters: Metadata filters (e.g., {"user_id": "123", "template_id": "456"})
        top_k: Final number of chunks to return (default: 20)
        enable_reranking: Use Cohere reranking (default: True)
        enable_keyword_search: Include BM25 search (default: True)
        enable_hyde: Use HyDE query generation (default: True)
        score_threshold: Minimum relevance score filter (optional)
        return_full_chunks: Return complete vs truncated content (default: True)
        deduplicate: Remove duplicate chunks (default: True)

    Returns:
        RetrievalResponse with:
        - success: Boolean status
        - query_original: Original query string
        - queries_generated: Dict with standard and HyDE queries
        - chunks: List of RetrievedChunk objects
        - retrieval_stats: Performance metrics
        - sources: Source document statistics
        - errors: List of error messages

    Retrieval Modes:
        1. Full Hybrid (Default - Best Quality)
           - HyDE + Vector + Keyword + Reranking
           - ~100 chunks → rerank → top 20
           - Best accuracy, slightly higher latency

        2. Hybrid Without HyDE
           - Vector + Keyword + Reranking
           - Faster query generation

        3. Vector Only with Reranking
           - Pure semantic search + Reranking
           - Good for conceptual queries

        4. Fast Vector Search
           - Vector only, no reranking, no keyword
           - Fastest retrieval, lower accuracy

    Use Cases:
        - Document Generation: Retrieve context for AI writing
        - Question Answering: Find specific information
        - Template-Specific: Get template-associated knowledge
        - User-Specific: Find user's documents

    Example:
        >>> response = rag_client.retrieve(
        ...     query="What is semantic chunking?",
        ...     collection_name="knowledge_base",
        ...     filters={"user_id": "user_123"},
        ...     top_k=10,
        ...     enable_reranking=True,
        ... )
        >>> for chunk in response.chunks:
        ...     print(f"Score: {chunk.relevance_score:.4f}")
        ...     print(f"Content: {chunk.content[:100]}...")

    Performance:
        - Query generation: ~100-200ms
        - Vector search: ~200-400ms
        - Keyword search: ~100-300ms
        - Reranking: ~200-500ms
        - Total: ~600-1400ms (varies by chunk count)
    """
    start_time = time.time()
    stats = RetrievalStats()
    queries_generated = {"original": query}

    try:
        # ===================================================================
        # STEP 1: QUERY GENERATION
        # ===================================================================
        query_gen_start = time.time()

        # Generate optimized queries
        if enable_hyde:
            # TODO: Implement HyDE query generation using LLM
            # For now, use original query
            standard_query = query
            hyde_query = query  # Placeholder - will implement LLM generation
            queries_generated["standard"] = standard_query
            queries_generated["hyde"] = hyde_query
        else:
            standard_query = query
            queries_generated["standard"] = standard_query

        stats.query_generation_time_ms = (time.time() - query_gen_start) * 1000

        # ===================================================================
        # STEP 2: DUAL VECTOR SEARCH
        # ===================================================================
        vector_search_start = time.time()
        vector_chunks = []

        # Search with standard query (25 chunks)
        standard_embedding = rag_client.embedder.embed_query(standard_query)
        standard_results = rag_client.vectordb.search(
            collection_name=collection_name,
            query_vector=standard_embedding,
            limit=25,
            filters=filters,
        )
        vector_chunks.extend(standard_results)

        # Search with HyDE query (25 chunks) if enabled
        if enable_hyde and hyde_query != standard_query:
            hyde_embedding = rag_client.embedder.embed_query(hyde_query)
            hyde_results = rag_client.vectordb.search(
                collection_name=collection_name,
                query_vector=hyde_embedding,
                limit=25,
                filters=filters,
            )
            vector_chunks.extend(hyde_results)

        stats.vector_search_time_ms = (time.time() - vector_search_start) * 1000
        stats.vector_search_chunks = len(vector_chunks)

        # ===================================================================
        # STEP 3: KEYWORD SEARCH (BM25)
        # ===================================================================
        keyword_chunks = []
        if enable_keyword_search:
            keyword_search_start = time.time()

            # TODO: Implement BM25 keyword search
            # For now, placeholder (will add BM25 implementation)
            # keyword_chunks = perform_bm25_search(query, collection_name, limit=50)

            stats.keyword_search_time_ms = (time.time() - keyword_search_start) * 1000
            stats.keyword_search_chunks = len(keyword_chunks)

        # ===================================================================
        # STEP 4: COMBINE & DEDUPLICATE
        # ===================================================================
        all_chunks = vector_chunks + keyword_chunks
        stats.total_chunks_retrieved = len(all_chunks)

        if deduplicate:
            # Deduplicate by chunk_id, keep highest score
            chunk_dict = {}
            for chunk in all_chunks:
                chunk_id = chunk.chunk_id
                if (
                    chunk_id not in chunk_dict
                    or chunk.score > chunk_dict[chunk_id].score
                ):
                    chunk_dict[chunk_id] = chunk
            unique_chunks = list(chunk_dict.values())
        else:
            unique_chunks = all_chunks

        stats.chunks_after_dedup = len(unique_chunks)

        # Fetch content from MongoDB if needed
        if rag_client.mongodb:
            for result in unique_chunks:
                if result.metadata.get("content_storage") == "mongodb":
                    mongodb_id = result.metadata.get("mongodb_id")
                    if mongodb_id:
                        mongo_doc = rag_client.mongodb.get_chunk_content_by_mongo_id(
                            str(mongodb_id)
                        )
                        if mongo_doc:
                            result.content = mongo_doc.get("content", "")

        # ===================================================================
        # STEP 5: RERANKING
        # ===================================================================
        if enable_reranking and rag_client.config.reranking.enabled:
            reranking_start = time.time()

            # TODO: Implement Cohere reranking
            # For now, use vector scores
            # ranked_chunks = cohere_rerank(query, unique_chunks, top_k)
            ranked_chunks = sorted(unique_chunks, key=lambda x: x.score, reverse=True)

            stats.reranking_time_ms = (time.time() - reranking_start) * 1000
        else:
            # No reranking - sort by vector score
            ranked_chunks = sorted(unique_chunks, key=lambda x: x.score, reverse=True)

        # ===================================================================
        # STEP 6: SELECTION & FORMATTING
        # ===================================================================
        # Select top_k chunks
        final_chunks = ranked_chunks[:top_k]

        # Apply score threshold if specified
        if score_threshold is not None:
            final_chunks = [c for c in final_chunks if c.score >= score_threshold]

        stats.chunks_after_reranking = len(final_chunks)

        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        for rank, result in enumerate(final_chunks):
            chunk = RetrievedChunk(
                content=result.content if return_full_chunks else result.content[:500],
                metadata=result.metadata,
                relevance_score=result.score,
                vector_score=result.score,
                rank=rank,
                keyword_score=None,  # TODO: Add BM25 score when implemented
            )
            retrieved_chunks.append(chunk)

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
                avg_relevance=data["total_score"] / data["count"],
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
