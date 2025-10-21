"""BM25 keyword search implementation."""

from typing import Any, Dict, List, Optional


class BM25Searcher:
    """
    BM25 (Best Matching 25) keyword search.

    BM25 is a ranking function used for information retrieval.
    It's particularly good at finding documents with exact term matches,
    which complements semantic vector search.

    This implementation uses the rank-bm25 library for simplicity.
    """

    def __init__(self, rag_client, collection_name: str):
        """
        Initialize BM25 searcher.

        Args:
            rag_client: RAGClient instance
            collection_name: Collection to build corpus from
        """
        self.rag_client = rag_client
        self.collection_name = collection_name
        self.corpus = []
        self.chunk_metadata = []
        self.bm25 = None
        self._build_corpus()

    def _build_corpus(self):
        """
        Build BM25 corpus from collection.

        This fetches all chunks from the collection and builds a BM25 index.
        For large collections, this may take time and use memory.
        """
        try:
            from rank_bm25 import BM25Okapi

            print(f"   Building BM25 corpus for collection '{self.collection_name}'...")

            # Fetch all chunks from Qdrant using scroll
            # Note: For very large collections, implement pagination
            all_points = []
            offset = None

            # Scroll through all points in collection
            while True:
                batch, offset = self.rag_client.vectordb.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # Don't need vectors for BM25
                )

                all_points.extend(batch)

                if offset is None:
                    break

            print(f"   Fetched {len(all_points)} points from Qdrant")

            # Build corpus
            self.corpus = []
            self.chunk_metadata = []
            mongodb_fetch_count = 0
            skipped_count = 0

            for point in all_points:
                # Get content from payload
                content = point.payload.get("content", "")

                # If content is in MongoDB, fetch it
                if not content and point.payload.get("content_storage") == "mongodb":
                    mongodb_id = point.payload.get("mongodb_id")
                    if mongodb_id and self.rag_client.mongodb:
                        try:
                            mongo_doc = (
                                self.rag_client.mongodb.get_chunk_content_by_mongo_id(
                                    str(mongodb_id)
                                )
                            )
                            if mongo_doc:
                                content = mongo_doc.get("content", "")
                                mongodb_fetch_count += 1
                        except Exception as e:
                            print(
                                f"   Warning: Failed to fetch content from MongoDB for chunk {point.payload.get('chunk_id')}: {e}"
                            )
                            skipped_count += 1
                            continue

                # Skip if still no content
                if not content:
                    skipped_count += 1
                    continue

                # Tokenize content (simple whitespace + lowercase)
                tokens = content.lower().split()

                self.corpus.append(tokens)
                self.chunk_metadata.append(
                    {
                        "id": str(point.id),
                        "chunk_id": point.payload.get("chunk_id"),
                        "content": content,
                        "metadata": {
                            k: v
                            for k, v in point.payload.items()
                            if k not in ["content", "chunk_id"]
                        },
                    }
                )

            if mongodb_fetch_count > 0:
                print(
                    f"   Fetched content for {mongodb_fetch_count} chunks from MongoDB"
                )
            if skipped_count > 0:
                print(f"   Skipped {skipped_count} chunks without content")

            # Build BM25 index
            if self.corpus:
                self.bm25 = BM25Okapi(self.corpus)
                print(f"   ✓ BM25 corpus built: {len(self.corpus)} documents indexed")
            else:
                self.bm25 = None
                print("   ⚠️ BM25 corpus is empty - no documents indexed")

        except ImportError:
            print(
                "   Warning: rank-bm25 not installed. Install with: pip install rank-bm25"
            )
            self.bm25 = None
        except Exception as e:
            print(f"   Warning: BM25 corpus building failed: {e}")
            self.bm25 = None

    def search(
        self,
        query: str,
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Metadata filters to apply

        Returns:
            List of results with scores and metadata
        """
        if not self.bm25 or not self.corpus:
            print("   Warning: BM25 index not available, skipping keyword search")
            return []

        try:
            # Tokenize query
            query_tokens = query.lower().split()

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Create results with scores
            results = []
            for idx, score in enumerate(scores):
                if score > 0:  # Only include non-zero scores
                    chunk_data = self.chunk_metadata[idx]

                    # Apply filters if specified
                    if filters:
                        match = True
                        for key, value in filters.items():
                            if value is not None and value != "" and value != {}:
                                if chunk_data["metadata"].get(key) != value:
                                    match = False
                                    break
                        if not match:
                            continue

                    results.append(
                        {
                            "chunk_id": chunk_data["chunk_id"],
                            "score": float(score),
                            "content": chunk_data["content"],
                            "metadata": chunk_data["metadata"],
                        }
                    )

            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)

            # Return top results
            return results[:limit]

        except Exception as e:
            print(f"   Warning: BM25 search failed: {e}")
            return []
