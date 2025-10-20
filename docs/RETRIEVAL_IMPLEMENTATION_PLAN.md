# Advanced Retrieval Method - Implementation Plan

## 🎯 Objective

Implement a comprehensive `retrieve()` method for RAGClient that uses hybrid search (vector + keyword) with HyDE query generation and Cohere reranking.

______________________________________________________________________

## 📊 Current State Analysis

### ✅ What Already Exists

1. **Vector Search (Qdrant)**

   - `QdrantVectorDB.search()` - WORKING
   - Returns `VectorSearchResult` objects
   - Supports metadata filters
   - Uses `query_points()` method (updated API)

1. **Embeddings (Azure OpenAI)**

   - `OpenAIEmbedder.embed()` - batch embedding
   - `OpenAIEmbedder.embed_query()` - single query embedding
   - 3072-dimensional vectors

1. **MongoDB Integration**

   - `MongoDBClient.get_chunk_content_by_mongo_id()` - fetch content
   - Hybrid storage working

1. **Response Models**

   - `RetrievalResponse` - complete response structure
   - `RetrievedChunk` - individual result
   - `RetrievalStats` - performance metrics
   - `SourceInfo` - source aggregation

1. **Basic Search Method**

   - `RAGClient.search()` - simple vector search
   - Already implemented and working

### ❌ What Needs to be Built

1. **HyDE Query Generation**

   - LLM call to generate hypothetical answer
   - Structured output for standard + HyDE queries
   - Error handling for LLM failures

1. **BM25 Keyword Search**

   - BM25 algorithm implementation OR integration
   - Document corpus indexing
   - Query tokenization
   - Metadata filtering support

1. **Deduplication Logic**

   - Hash-based or ID-based dedup
   - Score preservation (keep highest)
   - Efficient merging of results

1. **Cohere Reranking Integration**

   - Cohere API client setup
   - Batch reranking calls
   - Score normalization
   - Error handling / fallback

1. **Advanced Retrieve Method**

   - Orchestrate all 6 steps
   - Comprehensive error handling
   - Performance tracking
   - Flexible mode switching

1. **API Endpoints**

   - `/api/v1/retrieve` endpoint
   - Request/response models
   - Testing endpoints

______________________________________________________________________

## 🏗️ Architecture Design

### Component Hierarchy

```
RAGClient
├── retrieve()  [NEW - Main orchestration method]
│   │
│   ├── Step 1: Query Generation
│   │   └── HyDEQueryGenerator [NEW]
│   │       ├── Uses LLMConfig
│   │       └── Generates standard + HyDE queries
│   │
│   ├── Step 2: Vector Search (Dual)
│   │   ├── OpenAIEmbedder.embed_query() [EXISTS]
│   │   └── QdrantVectorDB.search() [EXISTS]
│   │
│   ├── Step 3: Keyword Search
│   │   └── BM25Searcher [NEW]
│   │       ├── Uses MongoDB/Qdrant for corpus
│   │       └── Returns scored chunks
│   │
│   ├── Step 4: Deduplication
│   │   └── deduplicate_chunks() [NEW]
│   │       └── Merge & remove duplicates
│   │
│   ├── Step 5: Reranking
│   │   └── CohereReranker [NEW]
│   │       ├── Uses RerankingConfig
│   │       └── Cross-encoder scoring
│   │
│   └── Step 6: Selection & Formatting
│       └── format_results() [NEW]
│           └── Apply threshold, limit, format
│
└── search()  [EXISTS - Keep as simple alternative]
```

______________________________________________________________________

## 📝 Detailed Implementation Plan

### Phase 1: Core Infrastructure (Priority: HIGH)

#### 1.1 HyDE Query Generator

**File**: `src/insta_rag/retrieval/query_generator.py`

**Status**: File exists, needs review and potential updates

**Tasks**:

- [ ] Review existing implementation
- [ ] Add HyDE generation using Azure OpenAI
- [ ] Use structured output (JSON mode)
- [ ] Single LLM call for both queries
- [ ] Error handling with fallback to original query

**Implementation**:

```python
class HyDEQueryGenerator:
    def generate_queries(self, query: str) -> Dict[str, str]:
        """
        Generate standard + HyDE queries.

        Returns:
            {"standard": "optimized query", "hyde": "hypothetical answer"}
        """
        # LLM prompt:
        # "Given the query: {query}
        #  Generate:
        #  1. An optimized search query
        #  2. A hypothetical answer to the query"

        # Use structured output for parsing
        # Fallback to original query if LLM fails
```

**Dependencies**: LLMConfig, Azure OpenAI client

______________________________________________________________________

#### 1.2 BM25 Keyword Search

**File**: `src/insta_rag/retrieval/keyword_search.py`

**Status**: File exists, needs review

**Options**:

1. **Use rank_bm25 library** (Python implementation)
1. **Use Qdrant's payload search** (if available)
1. **Custom implementation**

**Recommended**: rank_bm25 library (easiest)

**Tasks**:

- [ ] Install rank_bm25: `pip install rank-bm25`
- [ ] Build document corpus from collection
- [ ] Implement BM25Searcher class
- [ ] Support metadata filtering
- [ ] Cache corpus for performance

**Implementation**:

```python
from rank_bm25 import BM25Okapi


class BM25Searcher:
    def __init__(self, rag_client, collection_name):
        self.rag_client = rag_client
        self.collection_name = collection_name
        self.corpus = []  # List of documents
        self.chunk_metadata = []  # Corresponding metadata
        self._build_corpus()

    def _build_corpus(self):
        # Fetch all chunks from collection
        # Tokenize content
        # Build BM25 index
        pass

    def search(self, query: str, limit: int, filters: Dict) -> List:
        # Tokenize query
        # Get BM25 scores
        # Apply filters
        # Return top results
        pass
```

**Challenges**:

- Building corpus from Qdrant (need to fetch all docs)
- Keeping index updated when docs are added
- Memory usage for large collections

**Alternative Approach** (if BM25 is too complex for now):

- Skip keyword search initially
- Set `enable_keyword_search=False` by default
- Implement later as enhancement

______________________________________________________________________

#### 1.3 Cohere Reranker

**File**: `src/insta_rag/retrieval/reranker.py`

**Status**: File exists, needs review

**Tasks**:

- [ ] Review existing implementation
- [ ] Add Cohere client integration
- [ ] Implement rerank() method
- [ ] Handle API errors gracefully
- [ ] Add fallback (use vector scores if rerank fails)

**Implementation**:

```python
import cohere


class CohereReranker(BaseReranker):
    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        self.client = cohere.Client(api_key)
        self.model = model

    def rerank(
        self, query: str, chunks: List[Tuple[str, Dict]], top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Rerank chunks using Cohere.

        Args:
            query: Search query
            chunks: List of (content, metadata) tuples
            top_k: Number to return

        Returns:
            List of (original_index, relevance_score) tuples
        """
        try:
            # Prepare documents
            documents = [chunk[0] for chunk in chunks]

            # Call Cohere Rerank API
            results = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k,
            )

            # Return (index, score) pairs
            return [(r.index, r.relevance_score) for r in results.results]

        except Exception as e:
            # Fallback: return original order with dummy scores
            return [(i, 1.0 - (i * 0.01)) for i in range(min(top_k, len(chunks)))]
```

**Dependencies**:

- Cohere API key (from config)
- `cohere` Python package

______________________________________________________________________

### Phase 2: Helper Functions (Priority: MEDIUM)

#### 2.1 Deduplication

**File**: `src/insta_rag/retrieval/utils.py`

**Tasks**:

- [ ] Create utility functions
- [ ] Hash-based deduplication
- [ ] Keep highest score

**Implementation**:

```python
def deduplicate_chunks(
    chunks: List[VectorSearchResult], key_func=lambda x: x.chunk_id
) -> List[VectorSearchResult]:
    """
    Remove duplicate chunks, keeping highest score.

    Args:
        chunks: List of search results
        key_func: Function to extract unique key

    Returns:
        Deduplicated list
    """
    chunk_dict = {}
    for chunk in chunks:
        key = key_func(chunk)
        if key not in chunk_dict or chunk.score > chunk_dict[key].score:
            chunk_dict[key] = chunk
    return list(chunk_dict.values())
```

______________________________________________________________________

#### 2.2 Result Formatting

**File**: `src/insta_rag/retrieval/utils.py`

**Tasks**:

- [ ] Convert VectorSearchResult to RetrievedChunk
- [ ] Apply score threshold
- [ ] Truncate content if needed
- [ ] Calculate source statistics

**Implementation**:

```python
def format_retrieval_results(
    search_results: List,
    query: str,
    return_full_chunks: bool,
    score_threshold: Optional[float],
    mongodb_client: Optional[MongoDBClient],
) -> List[RetrievedChunk]:
    """Format search results into RetrievedChunk objects."""
    # Fetch MongoDB content if needed
    # Apply score threshold
    # Truncate if not return_full_chunks
    # Add rank positions
    pass
```

______________________________________________________________________

### Phase 3: Main Retrieve Method (Priority: HIGH)

#### 3.1 RAGClient.retrieve()

**File**: `src/insta_rag/core/client.py`

**Tasks**:

- [ ] Add retrieve() method to RAGClient
- [ ] Orchestrate all 6 steps
- [ ] Add comprehensive error handling
- [ ] Track timing for each step
- [ ] Support all modes (full hybrid, vector-only, etc.)

**Implementation Structure**:

```python
def retrieve(
    self,
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
) -> RetrievalResponse:
    """
    Advanced hybrid retrieval with HyDE, BM25, and reranking.

    [Full docstring with all details]
    """

    # Step 1: Query Generation
    # Step 2: Dual Vector Search
    # Step 3: Keyword Search (BM25)
    # Step 4: Combine & Deduplicate
    # Step 5: Reranking
    # Step 6: Selection & Formatting

    # Return RetrievalResponse
```

______________________________________________________________________

### Phase 4: API Endpoints (Priority: MEDIUM)

#### 4.1 Retrieve Endpoint

**File**: `testing_api/main.py`

**Tasks**:

- [ ] Add RetrieveRequest model
- [ ] Add /api/v1/retrieve endpoint
- [ ] Support all parameters
- [ ] Add to OpenAPI spec

**Implementation**:

```python
class RetrieveRequest(BaseModel):
    query: str
    collection_name: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 20
    enable_reranking: bool = True
    enable_keyword_search: bool = True
    enable_hyde: bool = True
    score_threshold: Optional[float] = None


@app.post("/api/v1/retrieve")
async def retrieve_documents(request: RetrieveRequest):
    """Advanced retrieval with hybrid search."""
    response = rag_client.retrieve(**request.dict())
    return response.to_dict()
```

______________________________________________________________________

### Phase 5: Testing & Documentation (Priority: HIGH)

#### 5.1 Unit Tests

**Tasks**:

- [ ] Test HyDE query generation
- [ ] Test BM25 search
- [ ] Test deduplication
- [ ] Test reranking
- [ ] Test full pipeline

#### 5.2 Integration Tests

**Tasks**:

- [ ] End-to-end retrieval test
- [ ] Test with MongoDB hybrid storage
- [ ] Test all retrieval modes
- [ ] Performance benchmarking

#### 5.3 Documentation

**Tasks**:

- [ ] API documentation
- [ ] Usage examples
- [ ] Performance guidelines
- [ ] Troubleshooting guide

______________________________________________________________________

## 🚀 Implementation Phases & Timeline

### Phase 1: MVP (Minimum Viable Product)

**Goal**: Basic retrieve() method working

**Components**:

1. ✅ Basic retrieve() method structure (DONE - created retrieval_method.py)
1. ⏳ Simple query generation (no HyDE initially)
1. ⏳ Dual vector search using existing methods
1. ⏳ Basic deduplication
1. ⏳ NO keyword search (skip for MVP)
1. ⏳ NO reranking (skip for MVP)
1. ⏳ Basic API endpoint

**Output**: Functional retrieve() that does dual vector search + dedup

______________________________________________________________________

### Phase 2: HyDE Integration

**Goal**: Add query generation

**Components**:

1. Review/implement HyDEQueryGenerator
1. LLM-based query optimization
1. Structured output parsing
1. Error handling

**Output**: retrieve() with HyDE query generation

______________________________________________________________________

### Phase 3: Reranking Integration

**Goal**: Add Cohere reranking

**Components**:

1. Review/implement CohereReranker
1. API integration
1. Error handling & fallback
1. Performance optimization

**Output**: retrieve() with reranking for better results

______________________________________________________________________

### Phase 4: BM25 Integration (Optional)

**Goal**: Add keyword search

**Components**:

1. Implement BM25Searcher OR use library
1. Corpus building
1. Index management
1. Hybrid fusion

**Output**: Full hybrid search (vector + keyword)

______________________________________________________________________

## 📋 Dependencies & Requirements

### Python Packages Needed

```bash
# Already installed (verify):
- qdrant-client>=1.7.0
- openai>=1.12.0
- pymongo>=4.6.0

# Need to add to pyproject.toml:
- cohere>=4.47.0  # ✅ Already in dependencies
- rank-bm25>=0.2.2  # ❌ Need to add
```

### API Keys Needed

```env
# Already configured:
AZURE_OPENAI_API_KEY ✅
QDRANT_API_KEY ✅
MONGO_CONNECTION_STRING ✅

# Need to verify:
COHERE_API_KEY=your_cohere_api_key_here
```

### Configuration Updates

**File**: `src/insta_rag/core/config.py`

- ✅ RerankingConfig exists
- ✅ LLMConfig exists
- ⏳ Verify all fields are present

______________________________________________________________________

## 🎯 Decision Points

### Decision 1: BM25 Implementation

**Options**:
A. Use rank-bm25 library (simple, fast to implement)
B. Custom implementation (more control)
C. Skip for now (focus on vector + reranking first)

**Recommendation**: Start with **Option C** (skip), add later as **Option A**

**Rationale**: Vector search + reranking provides 80% of value, BM25 adds 20%

______________________________________________________________________

### Decision 2: HyDE Implementation

**Options**:
A. Full LLM-based HyDE generation
B. Simple query expansion (synonyms, etc.)
C. Skip for MVP

**Recommendation**: **Option A** for Phase 2, **Option C** for Phase 1 MVP

**Rationale**: HyDE provides significant improvements, but not critical for MVP

______________________________________________________________________

### Decision 3: Reranking Fallback

**Options**:
A. Fail if Cohere API fails
B. Fall back to vector scores
C. Cache rerank results

**Recommendation**: **Option B** (fallback)

**Rationale**: System should work even if reranking fails

______________________________________________________________________

## 📊 Success Criteria

### Phase 1 MVP Success

- [ ] retrieve() method callable
- [ ] Dual vector search working
- [ ] Deduplication working
- [ ] Returns RetrievalResponse
- [ ] API endpoint working
- [ ] No errors with test data

### Full Implementation Success

- [ ] All 6 steps working
- [ ] HyDE improves results
- [ ] Reranking improves relevance
- [ ] BM25 catches exact matches
- [ ] Performance < 2 seconds average
- [ ] Comprehensive error handling
- [ ] Full documentation

______________________________________________________________________

## 🔧 Implementation Order (Recommended)

**Week 1: Foundation**

1. Review existing retrieval code
1. Create MVP retrieve() method (no HyDE, no BM25, no reranking)
1. Add basic API endpoint
1. Test with existing collections

**Week 2: Core Features**
5\. Add HyDE query generation
6\. Add Cohere reranking
7\. Test performance improvements

**Week 3: Advanced Features**
8\. Add BM25 keyword search (if needed)
9\. Optimize performance
10\. Add comprehensive tests

**Week 4: Polish**
11\. Documentation
12\. Error handling refinement
13\. Production readiness

______________________________________________________________________

## 🚨 Risks & Mitigation

### Risk 1: BM25 Corpus Building

**Issue**: Building BM25 corpus requires fetching all documents
**Impact**: High memory usage, slow initialization
**Mitigation**:

- Use lazy loading
- Cache corpus
- Implement incremental updates
- OR skip BM25 for now

### Risk 2: Cohere API Rate Limits

**Issue**: Reranking API may have rate limits
**Impact**: Failed retrievals
**Mitigation**:

- Implement retry logic
- Fallback to vector scores
- Batch processing

### Risk 3: Performance Degradation

**Issue**: 6-step pipeline may be slow
**Impact**: Poor user experience
**Mitigation**:

- Parallel execution where possible
- Caching
- Mode switching (fast vs accurate)

______________________________________________________________________

## 📁 File Structure

```
src/insta_rag/
├── core/
│   ├── client.py  [UPDATE - add retrieve() method]
│   └── retrieval_method.py  [NEW - orchestration logic]
│
├── retrieval/
│   ├── __init__.py
│   ├── base.py  [EXISTS - review]
│   ├── query_generator.py  [EXISTS - review/update]
│   ├── keyword_search.py  [EXISTS - review/update]
│   ├── reranker.py  [EXISTS - review/update]
│   ├── vector_search.py  [EXISTS - review]
│   └── utils.py  [NEW - helper functions]
│
└── models/
    └── response.py  [EXISTS - all models ready]

testing_api/
├── main.py  [UPDATE - add /api/v1/retrieve endpoint]
└── openapi.yaml  [UPDATE - add retrieve spec]
```

______________________________________________________________________

## ✅ Next Immediate Steps

1. **Review existing retrieval modules**

   - Check what's in `src/insta_rag/retrieval/`
   - Identify what works vs needs updates

1. **Build Phase 1 MVP**

   - Start with simple retrieve() (dual vector search)
   - No HyDE, no BM25, no reranking
   - Get it working end-to-end

1. **Test MVP**

   - Upload test documents
   - Call retrieve()
   - Verify results

1. **Iterate**

   - Add HyDE
   - Add reranking
   - Add BM25 (if needed)

______________________________________________________________________

This plan provides a clear roadmap from current state to full implementation, with flexibility to adjust based on priorities and challenges discovered during implementation.
