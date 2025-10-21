import pytest
from unittest.mock import MagicMock
from insta_rag import RAGClient, RAGConfig, DocumentInput


@pytest.fixture
def mock_rag_config():
    """Fixture to create a mock RAGConfig."""
    return RAGConfig(
        vectordb=MagicMock(),
        embedding=MagicMock(),
        reranking=MagicMock(),
        llm=MagicMock(),
        chunking=MagicMock(),
        pdf=MagicMock(),
        retrieval=MagicMock(),
    )


def test_rag_client_initialization(mocker, mock_rag_config):
    """Test that the RAGClient initializes its components correctly."""
    mocker.patch("insta_rag.core.client.OpenAIEmbedder")
    mocker.patch("insta_rag.core.client.QdrantVectorDB")
    mocker.patch("insta_rag.core.client.SemanticChunker")

    client = RAGClient(mock_rag_config)

    assert client.config == mock_rag_config
    assert client.embedder is not None
    assert client.vectordb is not None
    assert client.chunker is not None


def test_add_documents(mocker, mock_rag_config):
    """Test the add_documents method with mocked components."""
    # Arrange
    mock_embedder = mocker.patch("insta_rag.core.client.OpenAIEmbedder")
    mock_vectordb = mocker.patch("insta_rag.core.client.QdrantVectorDB")
    mock_chunker = mocker.patch("insta_rag.core.client.SemanticChunker")

    client = RAGClient(mock_rag_config)
    client.embedder = mock_embedder
    client.vectordb = mock_vectordb
    client.chunker = mock_chunker

    mock_chunker.chunk.return_value = [MagicMock(metadata=MagicMock(token_count=10))]
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_vectordb.collection_exists.return_value = True

    documents = [DocumentInput.from_text("This is a test document.")]
    collection_name = "test_collection"

    # Act
    response = client.add_documents(documents, collection_name=collection_name)

    # Assert
    assert response.success
    assert response.total_chunks == 1
    mock_chunker.chunk.assert_called_once()
    mock_embedder.embed.assert_called_once()
    mock_vectordb.upsert.assert_called_once()


def test_retrieve_documents(mocker, mock_rag_config):
    """Test the retrieve method with mocked components."""
    # Arrange
    mock_embedder = mocker.patch("insta_rag.core.client.OpenAIEmbedder")
    mock_vectordb = mocker.patch("insta_rag.core.client.QdrantVectorDB")
    mocker.patch(
        "insta_rag.core.client.SemanticChunker"
    )  # not used in retrieve, but initialized

    client = RAGClient(mock_rag_config)
    client.embedder = mock_embedder
    client.vectordb = mock_vectordb

    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_vectordb.search.return_value = [
        MagicMock(
            content="This is a test document.", metadata={"source": "test"}, score=0.9
        )
    ]

    query = "What is this document about?"
    collection_name = "test_collection"

    # Act
    response = client.retrieve(query, collection_name=collection_name)

    # Assert
    assert response.success
    assert len(response.chunks) == 1
    assert "test document" in response.chunks[0].content
    mock_embedder.embed_query.assert_called_once_with(query)
    mock_vectordb.search.assert_called_once()
