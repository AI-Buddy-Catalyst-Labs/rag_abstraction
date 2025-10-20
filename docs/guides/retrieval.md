# Guide: Advanced Retrieval

The `retrieve()` method in `insta_rag` is designed to find the most relevant information for a user's query using a sophisticated hybrid search pipeline. This guide breaks down how it works.

## The Retrieval Pipeline

The retrieval process is a multi-step pipeline designed to maximize both semantic understanding (finding conceptually similar results) and lexical matching (finding exact keywords).

```mermaid
graph TD
    A[User Query] --> B{Step 1: Query Generation (HyDE)};
    B --> C{Step 2: Dual Vector Search};
    B --> D{Step 3: Keyword Search (BM25)};
    C --> E{Step 4: Combine & Deduplicate};
    D --> E;
    E --> F{Step 5: Reranking};
    F --> G{Step 6: Selection & Formatting};
    G --> H[Top-k Relevant Chunks];
```

### Step 1: Query Generation (HyDE)
*   **Goal**: To overcome the challenge of matching a short user query with long, detailed document chunks.
*   **Process**: The user's query is sent to an LLM (e.g., GPT-4) which generates two things:
    1.  **Optimized Query**: A rewritten, clearer version of the original query.
    2.  **Hypothetical Document Embedding (HyDE)**: A hypothetical answer or document that would perfectly answer the user's query.
*   **Benefit**: Searching with the embedding of the hypothetical answer is often more effective at finding relevant chunks than searching with the embedding of the short original query.

### Step 2: Dual Vector Search
*   **Goal**: To find semantically relevant chunks.
*   **Process**: Two parallel vector searches are performed in Qdrant:
    1.  Search with the embedding of the **optimized query**.
    2.  Search with the embedding of the **HyDE query**.
*   **Output**: A list of candidate chunks from both searches (e.g., 25 chunks from each, for a total of 50).

### Step 3: Keyword Search (BM25)
*   **Goal**: To find chunks containing exact keyword matches, which semantic search might miss.
*   **Process**: The original query is tokenized, and a BM25 (Best Match 25) algorithm is used to find chunks with high lexical overlap.
*   **Benefit**: Crucial for finding specific names, codes, acronyms, or direct quotes.
*   **Output**: A list of candidate chunks based on keyword relevance (e.g., 50 chunks).

### Step 4: Combine & Deduplicate
*   **Goal**: To create a single, unified pool of candidate chunks.
*   **Process**: The results from vector search and keyword search are combined. Duplicate chunks (which may have been found by both methods) are removed, keeping the instance with the highest score.
*   **Output**: A single list of unique candidate chunks (e.g., ~70-80 chunks).

### Step 5: Reranking
*   **Goal**: To intelligently re-order the candidate chunks for maximum relevance.
*   **Process**: The combined list of chunks is sent to a powerful cross-encoder model (like Cohere's Reranker or BAAI's BGE-Reranker). Unlike vector similarity, a cross-encoder directly compares the user's query against each candidate chunk's full text, providing a much more accurate relevance score.
*   **Benefit**: This is the most computationally intensive but also the most impactful step for improving the final quality of the results.

### Step 6: Selection & Formatting
*   **Goal**: To prepare the final response for the user.
*   **Process**:
    1.  The results are sorted by their new reranker scores.
    2.  A final `score_threshold` can be applied to filter out low-quality results.
    3.  The top `k` chunks are selected.
    4.  If using hybrid storage, the full content is fetched from MongoDB.

## Controlling Retrieval Features

All advanced features are enabled by default, but you can easily disable them to trade quality for speed.

```python
# High-quality (default)
response = client.retrieve(
    query="...",
    collection_name="...",
    enable_hyde=True,
    enable_keyword_search=True,
    enable_reranking=True
)

# Fast mode (vector search only)
response = client.retrieve(
    query="...",
    collection_name="...",
    enable_hyde=False,
    enable_keyword_search=False,
    enable_reranking=False
)
```

## Understanding Reranker Scores

The reranker model you use determines the range and interpretation of the `relevance_score`.

### BGE Reranker (e.g., `BAAI/bge-reranker-v2-m3`)

This model produces scores that are often negative. **Higher is better**.

*   **Range**: Typically -10.0 to +10.0
*   **Interpretation**:
    *   `> 0.0`: Very good relevance.
    *   `-3.0 to 0.0`: Good relevance.
    *   `-5.0 to -3.0`: Moderate relevance.
    *   `< -5.0`: Low relevance.
*   **Thresholding**: Use negative values for the `score_threshold`.
    ```python
    # Keep only moderately to highly relevant results
    client.retrieve(query="...", collection_name="...", score_threshold=-3.0)
    ```

### Cohere Reranker

Cohere's reranker produces normalized scores between 0 and 1. **Higher is better**.

*   **Range**: 0.0 to 1.0
*   **Interpretation**:
    *   `> 0.8`: Highly relevant.
    *   `0.5 to 0.8`: Moderately relevant.
    *   `< 0.5`: Low relevance.
*   **Thresholding**: Use positive float values.
    ```python
    # Keep only moderately to highly relevant results
    client.retrieve(query="...", collection_name="...", score_threshold=0.5)
    ```
