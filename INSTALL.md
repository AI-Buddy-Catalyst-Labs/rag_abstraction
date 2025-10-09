# Installation Guide for insta_rag

## Prerequisites

- Python 3.9 or higher
- pip or uv package manager

## Installation Options

### Option 1: Using uv (Recommended)

If you're already using uv (as indicated by your project setup):

```bash
# Install the package with all dependencies
uv pip install -e .

# Or sync from lock file
uv sync
```

### Option 2: Using pip with Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install the package
pip install -e .
```

### Option 3: Install from requirements file

```bash
# Activate your virtual environment first, then:
pip install -r requirements-rag.txt
```

## Required Dependencies

The following packages will be installed automatically:

- **openai** (>=1.12.0) - OpenAI and Azure OpenAI API client
- **qdrant-client** (>=1.7.0) - Qdrant vector database client
- **pdfplumber** (>=0.10.3) - PDF text extraction (primary)
- **PyPDF2** (>=3.0.1) - PDF text extraction (fallback)
- **tiktoken** (>=0.5.2) - Token counting for OpenAI models
- **numpy** (>=1.24.0) - Numerical operations for semantic chunking
- **python-dotenv** (>=1.0.0) - Environment variable management
- **cohere** (>=4.47.0) - Cohere API for reranking (optional)
- **pydantic** (>=2.5.0) - Data validation

## Verify Installation

After installation, verify it works:

```python
from insta_rag import RAGClient, RAGConfig

print("✓ insta_rag installed successfully!")
```

Or run the test script:

```bash
python examples/simple_test.py
```

## Environment Setup

Create a `.env` file with your API keys:

```env
# Required: Qdrant Vector Database
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# Required: OpenAI/Azure OpenAI
# Option 1: Azure OpenAI (recommended)
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Option 2: Standard OpenAI
OPENAI_API_KEY=your_openai_api_key

# Optional: Cohere (for reranking)
COHERE_API_KEY=your_cohere_api_key
```

## Troubleshooting

### "externally-managed-environment" Error

If you see this error, you need to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Import Errors

If you get import errors, make sure all dependencies are installed:

```bash
# Check installed packages
pip list | grep -E "(openai|qdrant|pdfplumber|tiktoken)"

# Reinstall if needed
pip install -e .
```

### PDF Processing Issues

If PDF extraction fails:

```bash
# Ensure pdfplumber is properly installed
pip install --upgrade pdfplumber PyPDF2
```

### Qdrant Connection Issues

1. Verify your `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
2. Test connection:
   ```python
   from qdrant_client import QdrantClient
   client = QdrantClient(url="your_url", api_key="your_key")
   print(client.get_collections())
   ```

## Development Installation

For development with testing tools:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

This includes:
- ruff (linting and formatting)
- pre-commit (git hooks)
- commitizen (conventional commits)

## Updating Dependencies

To update all dependencies:

```bash
# With pip
pip install --upgrade -r requirements-rag.txt

# With uv
uv pip install --upgrade -r requirements-rag.txt
```

## Uninstallation

```bash
pip uninstall insta_rag
```

## Support

If you encounter any installation issues:

1. Check that Python version is 3.9+: `python --version`
2. Ensure pip is up to date: `pip install --upgrade pip`
3. Try installing in a fresh virtual environment
4. Check the GitHub Issues: https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/issues
