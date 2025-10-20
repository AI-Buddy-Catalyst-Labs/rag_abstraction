# Document Upload Flow to Qdrant - Complete Analysis

## Overview

The insta_rag library implements a sophisticated 6-phase pipeline to process documents and store them in Qdrant vector database. Here's how it works:

______________________________________________________________________

## 📊 High-Level Architecture

```
DocumentInput → Text Extraction → Semantic Chunking → Embedding → Vector Storage (Qdrant)
                                                                  ↓
                                                          Content Storage (MongoDB - Optional)
```

______________________________________________________________________

## 🔄 Complete Pipeline Flow

### Entry Point: `RAGClient.add_documents()`

**Location:** `src/insta_rag/core/client.py:83-283`

```python
client.add_documents(
    documents=[DocumentInput.from_file("document.pdf")],
    collection_name="my_documents",
    metadata={"category": "research"},
    batch_size=100,
)
```

______________________________________________________________________

## Phase-by-Phase Breakdown

### **PHASE 1: Document Loading**

**Handler:** `_load_and_extract_document()` (client.py:285-336)

**What happens:**

1. Generate unique `document_id` using UUID
1. Merge global metadata with document-specific metadata
1. Determine source type (FILE, TEXT, or BINARY)

**Input:**

```python
DocumentInput(
    source=Path("example.pdf"), source_type=SourceType.FILE, metadata={"author": "John"}
)
```

**Output:**

```python
document_id = "123e4567-e89b-12d3-a456-426614174000"
doc_metadata = {
    "document_id": "123e4567...",
    "source": "/path/to/example.pdf",
    "author": "John",
}
```

______________________________________________________________________

### **PHASE 2: Text Extraction**

**Handler:** `extract_text_from_pdf()` (pdf_processing.py:9-48)

**What happens:**

1. Validate PDF file exists
1. Try primary parser (pdfplumber by default)
1. If primary fails, fallback to PyPDF2
1. Handle encryption and corruption errors
1. Extract text page-by-page
1. Join pages with double newlines

**Code Flow:**

```python
# pdf_processing.py:51-75
with pdfplumber.open(pdf_path) as pdf:
    # Check encryption
    if pdf.metadata.get("Encrypt"):
        raise PDFEncryptedError()

    # Extract page by page
    text_parts = []
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n\n".join(text_parts)
```

**Output:**

```
"This is the full text extracted from the PDF.
It contains all pages joined together with
double newlines between pages..."
```

______________________________________________________________________

### **PHASE 3: Semantic Chunking**

**Handler:** `SemanticChunker.chunk()` (chunking/semantic.py:56-93)

**What happens:**

1. Count total tokens in document
1. If ≤ max_chunk_size (1000 tokens), return as single chunk
1. Otherwise, perform semantic chunking:
   - Split text into sentences
   - Generate embeddings for each sentence
   - Calculate cosine similarity between consecutive sentences
   - Find breakpoints (low similarity = topic change)
   - Split at breakpoints
   - Enforce token limits
   - Add overlap between chunks (20% by default)

**Detailed Semantic Chunking Process:**

```python
# Step 1: Split into sentences
sentences = split_into_sentences(text)
# Result: ["First sentence.", "Second sentence.", ...]

# Step 2: Embed sentences
embeddings = self.embedder.embed(sentences)
# Result: [[0.1, 0.2, ...], [0.15, 0.19, ...], ...]

# Step 3: Calculate similarities
similarities = []
for i in range(len(embeddings) - 1):
    vec1 = np.array(embeddings[i])
    vec2 = np.array(embeddings[i + 1])
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    similarities.append(similarity)
# Result: [0.95, 0.92, 0.45, 0.88, ...]  # 0.45 = topic change!

# Step 4: Find breakpoints using percentile threshold (95th percentile)
threshold = np.percentile(similarities, 100 - 95)  # = 5th percentile (low values)
breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
# Result: [3, 7, 12]  # Split at these indices

# Step 5: Split at breakpoints
chunks = []
start = 0
for bp in breakpoints:
    chunk = " ".join(sentences[start:bp])
    chunks.append(chunk)
    start = bp
# Result: ["Chunk 1 text...", "Chunk 2 text...", "Chunk 3 text..."]

# Step 6: Enforce token limits and add overlap
chunks = self._enforce_token_limits(chunks)
chunks = add_overlap_to_chunks(chunks, overlap_percentage=0.2)
```

**Chunk Object Creation:**

```python
# chunking/semantic.py:231-276
for idx, text in enumerate(text_chunks):
    metadata = ChunkMetadata(
        document_id=document_id,
        source="/path/to/file.pdf",
        chunk_index=idx,
        total_chunks=len(text_chunks),
        token_count=count_tokens_accurate(text),
        char_count=len(text),
        chunking_method="semantic",
        extraction_date=datetime.utcnow(),
        custom_fields={...},
    )

    chunk = Chunk(
        chunk_id=f"{document_id}_chunk_{idx}",
        content=text,
        metadata=metadata,
        embedding=None,  # Will be filled in Phase 5
    )
```

**Output:**

```python
[
    Chunk(
        chunk_id="123e4567..._chunk_0",
        content="First chunk of text about topic A...",
        metadata=ChunkMetadata(...),
        embedding=None,
    ),
    Chunk(
        chunk_id="123e4567..._chunk_1",
        content="...overlap from previous chunk. Second chunk about topic B...",
        metadata=ChunkMetadata(...),
        embedding=None,
    ),
    ...,
]
```

______________________________________________________________________

### **PHASE 4: Chunk Validation**

**Handler:** `validate_chunk_quality()` (chunking/utils.py)

**What happens:**

- Minimum length check (>= 10 characters)
- Quality checks built into chunking process
- Token count validation

**Note:** Most validation happens during chunk creation in Phase 3

______________________________________________________________________

### **PHASE 5: Batch Embedding Generation**

**Handler:** `OpenAIEmbedder.embed()` (embedding/openai.py:70-109)

**What happens:**

1. Extract content text from all chunks
1. Process in batches (default: 100 chunks per batch)
1. Call Azure OpenAI or OpenAI API
1. Attach embeddings back to chunk objects

**Code Flow:**

```python
# client.py:169-180
chunk_texts = [chunk.content for chunk in all_chunks]
# Result: ["Chunk 1 text...", "Chunk 2 text...", ...]

embeddings = self.embedder.embed(chunk_texts)

# Attach embeddings to chunks
for chunk, embedding in zip(all_chunks, embeddings):
    chunk.embedding = embedding
```

**Inside the Embedder:**

```python
# embedding/openai.py:86-105
all_embeddings = []

# Process in batches
for i in range(0, len(texts), batch_size):  # batch_size = 100
    batch = texts[i : i + batch_size]

    # Call Azure OpenAI API
    response = self.client.embeddings.create(
        input=batch,
        model="text-embedding-3-large",  # Deployment name
    )

    # Extract embeddings
    batch_embeddings = [item.embedding for item in response.data]
    all_embeddings.extend(batch_embeddings)

return all_embeddings
```

**Output:**
Each chunk now has:

```python
chunk.embedding = [0.023, -0.012, 0.045, ..., 0.019]  # 3072 dimensions
```

______________________________________________________________________

### **PHASE 6: Vector & Content Storage**

**Handler:** `QdrantVectorDB.upsert()` (vectordb/qdrant.py:108-159)

**What happens:**

#### **6A: Collection Setup**

```python
# client.py:186-193
if not self.vectordb.collection_exists(collection_name):
    print(f"Creating collection '{collection_name}'...")
    self.vectordb.create_collection(
        collection_name=collection_name,
        vector_size=3072,  # From embedder.get_dimensions()
        distance_metric="cosine",
    )
```

**Qdrant Collection Created With:**

- **Vector size:** 3072 dimensions (text-embedding-3-large)
- **Distance metric:** COSINE similarity
- **Status:** Ready to receive vectors

______________________________________________________________________

#### **6B: Two Storage Modes**

##### **Mode 1: MongoDB Enabled (Hybrid Storage)**

**Used when:** `config.mongodb.enabled = True`

```python
# client.py:196-238

# Store full content in MongoDB
mongo_docs = []
for chunk in all_chunks:
    mongo_docs.append(
        {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,  # Full text stored here
            "document_id": chunk.metadata.document_id,
            "collection_name": collection_name,
            "metadata": chunk.metadata.to_dict(),
        }
    )

mongo_ids = self.mongodb.store_chunks_batch(mongo_docs)

# Store vectors in Qdrant with MongoDB references
chunk_ids = [chunk.chunk_id for chunk in all_chunks]
vectors = [chunk.embedding for chunk in all_chunks]
contents = []  # Empty - content is in MongoDB
metadatas = []

for i, chunk in enumerate(all_chunks):
    meta = chunk.metadata.to_dict()
    meta["mongodb_id"] = mongo_ids[i]  # Reference to MongoDB
    meta["content_storage"] = "mongodb"
    metadatas.append(meta)
    contents.append("")  # Empty placeholder
```

**Storage Architecture (MongoDB Mode):**

```
MongoDB (Content Storage):
{
    "_id": ObjectId("..."),
    "chunk_id": "123e4567..._chunk_0",
    "content": "Full chunk text stored here...",
    "document_id": "123e4567...",
    "collection_name": "my_documents",
    "metadata": {...}
}

Qdrant (Vector + Metadata):
{
    "id": "uuid-deterministic",
    "vector": [0.023, -0.012, ..., 0.019],  # 3072 dims
    "payload": {
        "chunk_id": "123e4567..._chunk_0",
        "document_id": "123e4567...",
        "source": "/path/to/file.pdf",
        "chunk_index": 0,
        "total_chunks": 5,
        "token_count": 850,
        "mongodb_id": "ObjectId(...)",  # Reference
        "content_storage": "mongodb"
    }
}
```

##### **Mode 2: Qdrant Only (Direct Storage)**

**Used when:** `config.mongodb.enabled = False`

```python
# client.py:239-244
chunk_ids = [chunk.chunk_id for chunk in all_chunks]
vectors = [chunk.embedding for chunk in all_chunks]
contents = [chunk.content for chunk in all_chunks]  # Content in Qdrant
metadatas = [chunk.metadata.to_dict() for chunk in all_chunks]
```

**Storage Architecture (Qdrant-Only Mode):**

```
Qdrant (Vector + Content + Metadata):
{
    "id": "uuid-deterministic",
    "vector": [0.023, -0.012, ..., 0.019],  # 3072 dims
    "payload": {
        "content": "Full chunk text stored directly...",  # Content here!
        "chunk_id": "123e4567..._chunk_0",
        "document_id": "123e4567...",
        "source": "/path/to/file.pdf",
        "chunk_index": 0,
        "total_chunks": 5,
        "token_count": 850,
        "char_count": 4200,
        "chunking_method": "semantic",
        "extraction_date": "2025-10-09T10:00:00"
    }
}
```

______________________________________________________________________

#### **6C: Qdrant Upload Process**

**Handler:** `QdrantVectorDB.upsert()` (vectordb/qdrant.py:108-159)

```python
# vectordb/qdrant.py:135-154
points = []
for chunk_id, vector, content, metadata in zip(chunk_ids, vectors, contents, metadatas):
    # Combine content and metadata for payload
    payload = {
        "content": content,  # Empty if MongoDB mode
        **metadata,  # Spread all metadata fields
    }

    # Create deterministic UUID from chunk_id
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    point = PointStruct(
        id=point_id,
        vector=vector,  # 3072-dimensional embedding
        payload=payload,
    )
    points.append(point)

# Upsert in batches (100 points per batch)
batch_size = 100
for i in range(0, len(points), batch_size):
    batch = points[i : i + batch_size]
    self.client.upsert(collection_name=collection_name, points=batch)
```

**Key Details:**

- **Point ID:** Deterministic UUID generated from `chunk_id` using `uuid.uuid5()`
  - Same chunk_id always produces same UUID
  - Allows for idempotent upserts (re-uploading overwrites)
- **Batch Size:** 100 points per batch to optimize network calls
- **Upsert:** Updates if exists, inserts if new

______________________________________________________________________

## 📈 Performance Statistics

The library tracks timing for each phase:

```python
ProcessingStats(
    chunking_time_ms=1250.5,  # Phase 3
    embedding_time_ms=3420.8,  # Phase 5
    upload_time_ms=890.2,  # Phase 6
    total_time_ms=5561.5,  # All phases
    total_tokens=12500,  # Total tokens processed
)
```

______________________________________________________________________

## 🔍 Example: Complete Flow for 1 PDF

**Input:**

```python
doc = DocumentInput.from_file("research_paper.pdf")
client.add_documents([doc], collection_name="papers")
```

**Processing:**

1. **Phase 1-2:** Extract 50 pages → 25,000 words
1. **Phase 3:** Semantic chunking → 18 chunks (avg 1,389 words each)
1. **Phase 4:** Validation → All 18 pass
1. **Phase 5:** Embedding → 18 API calls (batched) → 18 × 3072-dim vectors
1. **Phase 6:**
   - MongoDB: Store 18 full text chunks
   - Qdrant: Store 18 vectors + metadata references
   - Total: 18 points in Qdrant collection

**Result in Qdrant:**

```
Collection: "papers"
├── Vector 0: [0.023, -0.012, ...] + metadata
├── Vector 1: [0.019, 0.045, ...] + metadata
├── Vector 2: [0.031, -0.008, ...] + metadata
...
└── Vector 17: [0.012, 0.028, ...] + metadata

Total: 18 points, 3072 dimensions, COSINE distance
```

______________________________________________________________________

## 🎯 Key Design Decisions

### 1. **Semantic Chunking**

- **Why:** Preserves topical coherence better than fixed-size chunks
- **How:** Analyzes sentence-to-sentence similarity using embeddings
- **Benefit:** Chunks align with natural topic boundaries

### 2. **Deterministic UUIDs**

```python
uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)
```

- **Why:** Same chunk_id always produces same Qdrant point ID
- **Benefit:** Idempotent uploads, easy to update/replace

### 3. **Hybrid Storage (MongoDB + Qdrant)**

- **Qdrant:** Fast vector search (optimized for embeddings)
- **MongoDB:** Full text storage (cheaper, better for large content)
- **Benefit:** Best of both worlds - fast retrieval + cost efficiency

### 4. **Batch Processing**

- **Embeddings:** 100 chunks per API call
- **Qdrant Upload:** 100 points per upsert
- **Benefit:** Reduced API calls, better performance

### 5. **Overlap Between Chunks**

- **Default:** 20% overlap
- **Why:** Prevents information loss at chunk boundaries
- **Example:** Last 200 tokens of chunk N = first 200 tokens of chunk N+1

______________________________________________________________________

## 🔧 Configuration Impact

Your `.env` settings affect the flow:

```bash
# Embedding Configuration
AZURE_OPENAI_ENDPOINT=https://...      # API endpoint
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large  # Model
# → Affects: Phase 5 (3072 dimensions)

# Qdrant Configuration
QDRANT_URL=https://8cb410af...         # Cloud instance
QDRANT_API_KEY=eyJhbGci...             # Authentication
# → Affects: Phase 6 (where vectors go)

# MongoDB Configuration (Optional)
MONGO_CONNECTION_STRING=mongodb://...  # If enabled
# → Affects: Phase 6 (hybrid vs direct storage)

# Chunking Configuration (in code)
max_chunk_size=1000                    # Token limit per chunk
overlap_percentage=0.2                 # 20% overlap
semantic_threshold_percentile=95       # Sensitivity to topic changes
# → Affects: Phase 3 (how chunks are created)
```

______________________________________________________________________

## 🚀 Usage Recommendations

### For Best Results:

1. **Document Size:**

   - Small (< 1000 tokens): Stored as single chunk
   - Medium (1K-100K tokens): Semantic chunking shines
   - Large (> 100K tokens): May need batch processing

1. **Collection Naming:**

   - Use descriptive names: `research_papers`, `user_manuals`
   - One collection per document type/domain

1. **Metadata:**

   - Add meaningful metadata: `{"category": "AI", "year": 2024}`
   - Used for filtering during search

1. **Batch Size:**

   - Default 100 works well for most cases
   - Reduce if hitting API rate limits
   - Increase for faster processing (if API allows)

______________________________________________________________________

## 🔬 Search Flow (Reverse Process)

When you search:

```python
results = client.search("What is semantic chunking?", collection_name="papers")
```

1. **Query Embedding:** Your question → 3072-dim vector
1. **Qdrant Search:** Find similar vectors using COSINE distance
1. **MongoDB Retrieval:** (If enabled) Fetch full content using mongodb_id
1. **Reranking:** (If enabled with Cohere) Re-order by relevance
1. **Return:** Top K most relevant chunks with content + metadata

______________________________________________________________________

## 📝 Summary

**The upload flow is a 6-phase pipeline:**

```
PDF → Text → Semantic Chunks → Embeddings → Qdrant (vectors) + MongoDB (content)
```

**Key transformations:**

- **Document** (1 PDF) → **Text** (25K words) → **Chunks** (18 semantic pieces) → **Vectors** (18 × 3072 floats) → **Stored** (18 Qdrant points)

**Your setup specifically:**

- ✅ Azure OpenAI for embeddings (text-embedding-3-large, 3072 dims)
- ✅ Qdrant Cloud for vector storage (COSINE similarity)
- ✅ MongoDB for content storage (hybrid mode)
- ✅ Semantic chunking with 20% overlap
- ✅ Cohere for reranking

This architecture provides:

- **Fast search** via Qdrant's vector similarity
- **Rich context** via semantic chunks
- **Cost efficiency** via MongoDB content storage
- **Scalability** via batch processing
