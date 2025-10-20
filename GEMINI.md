# Gemini Context: insta_rag Project

## Project Overview

This project, `insta_rag`, is a modular and extensible Python library designed for building Retrieval-Augmented Generation (RAG) pipelines. It abstracts the complexity of RAG into three primary operations: adding, updating, and retrieving documents.

**Key Technologies & Architecture:**

- **Core Client:** The main entry point is the `RAGClient`, which orchestrates all operations.
- **Embeddings & LLMs:** Utilizes OpenAI (`text-embedding-3-large`, GPT-4) or Azure OpenAI for generating embeddings and hypothetical answers (HyDE).
- **Vector Database:** Uses Qdrant for efficient vector storage and search.
- **Reranking:** Integrates Cohere for cross-encoder reranking to improve the relevance of search results.
- **Architecture:** The library is built on an interface-based design, allowing for plug-and-play components. Core modules for `chunking`, `embedding`, `vectordb`, and `retrieval` each have a `base.py` defining an abstract interface, making it easy to extend with new implementations (e.g., adding Pinecone as a vector DB).
- **Data Models:** Pydantic is used for robust data validation and clear data structures for documents, chunks, and API responses.

The primary goal is to provide a complete, configuration-driven RAG system that is both easy to use and easy to extend.

## Documentation

The project documentation has been reorganized for clarity and is located in the `/docs` directory.

- **[README.md](./docs/README.md):** Main landing page with links to all other documents.
- **[installation.md](./docs/installation.md):** Detailed installation instructions.
- **[quickstart.md](./docs/quickstart.md):** A hands-on guide to get started quickly.
- **Guides (`/docs/guides`):**
  - **[document-management.md](./docs/guides/document-management.md):** Covers adding, updating, and deleting documents.
  - **[retrieval.md](./docs/guides/retrieval.md):** Explains the advanced hybrid retrieval pipeline.
  - **[storage-backends.md](./docs/guides/storage-backends.md):** Details on configuring Qdrant-only vs. hybrid Qdrant+MongoDB storage.
  - **[local-development.md](./docs/guides/local-development.md):** Instructions for setting up a local Qdrant instance.

## Building and Running

### 1. Installation

The project uses `uv` for package management.

```bash
# Install the package in editable mode with all dependencies
uv pip install -e .
```

Alternatively, using `pip` and a virtual environment:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

### 2. Environment Setup

The client is configured via a `.env` file. Create one in the project root with the variables listed in `docs/installation.md`.

### 3. Running the Example

The `examples/basic_usage.py` script demonstrates the core functionality of the library.

```bash
# Run the basic usage example
python examples/basic_usage.py
```

### 4. Running Tests

The project contains a `tests/` directory. Tests can be run using `pytest`.

```bash
# TODO: Verify if this is the correct test command.
pytest
```

## Development Conventions

This project has a strong focus on code quality and consistency, enforced by several tools.

### 1. Linting and Formatting

- **Tool:** `Ruff` is used for both linting and formatting.

- **Usage:**

  ```bash
  # Check for linting errors and auto-fix them
  ruff check . --fix

  # Format the codebase
  ruff format .
  ```

### 2. Pre-commit Hooks

- **Framework:** `pre-commit` is used to run checks before each commit.

- **Setup:** First-time contributors must install the hooks:

  ```bash
  pre-commit install
  ```

### 3. Commit Messages

- **Standard:** The project follows the **Conventional Commits** specification, enforced by `commitizen`.
