# Gemini Code Assistant Context

This document provides context for the `insta_rag` project, a Python-based Retrieval-Augmented Generation (RAG) library.

## Project Overview

`insta_rag` is a modular and extensible library for building RAG pipelines. It abstracts the complexity of document processing, chunking, embedding, and retrieval into a simple-to-use client. The library is designed with a plug-and-play architecture, allowing developers to easily swap components like embedding models, vector databases, and rerankers.

### Key Technologies

-   **Programming Language:** Python 3.9+
-   **Core Dependencies:**
    -   `openai`: For generating embeddings and powering HyDE (Hypothetical Document Embeddings).
    -   `qdrant-client`: For vector storage and search.
    -   `cohere`: For reranking search results.
    -   `pdfplumber` & `PyPDF2`: For PDF text extraction.
    -   `pymongo`: For metadata storage.
    -   `fastapi`: For the testing and example API.
-   **Architecture:**
    -   **Modular:** The library is divided into distinct modules for chunking, embedding, vector database interaction, and retrieval.
    -   **Interface-Based:** Core components are built around abstract base classes, making it easy to add new implementations.
    -   **Configuration-Driven:** A central `RAGConfig` object controls the behavior of the entire library.

## Building and Running

### Installation

1.  **Install Python:** Ensure you have Python 3.9 or higher installed.
2.  **Install Dependencies:** It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    ```

### Running the Testing API

The project includes a FastAPI-based testing API in the `testing_api` directory. This is the best way to test the library's functionality.

1.  **Set up Environment Variables:** Create a `.env` file in the root of the project with the following variables:

    ```env
    QDRANT_URL="your_qdrant_url"
    QDRANT_API_KEY="your_qdrant_api_key"
    AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
    AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
    AZURE_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
    ```

2.  **Run the API:**

    ```bash
    cd testing_api
    python main.py
    ```

    The API will be available at `http://localhost:8000`. You can access the Swagger UI for interactive documentation at `http://localhost:8000/docs`.

## Development Conventions

### Code Style

-   The project uses `ruff` for linting and formatting. The configuration is in `pyproject.toml`.
-   **Line Length:** 88 characters.
-   **Quotes:** Double quotes (`"`).
-   **Indentation:** 4 spaces.

### Testing

-   The `testing_api` directory contains a comprehensive suite of endpoints for testing all components of the `insta_rag` library.
-   To run the tests, start the testing API and use a tool like `curl` or the Swagger UI to send requests to the various endpoints.

### Commits

-   The project uses [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
-   `commitizen` is used to format commit messages.

### Key Files

-   `src/insta_rag/core/client.py`: The main entry point for the RAG library.
-   `src/insta_rag/core/config.py`: Defines the configuration for the RAG client.
-   `src/insta_rag/retrieval/reranker.py`: Implements reranking logic.
-   `testing_api/main.py`: The FastAPI application for testing the library.
-   `README.md`: Provides a detailed overview of the library's architecture and usage.
-   `pyproject.toml`: Defines project dependencies and tool configurations.
