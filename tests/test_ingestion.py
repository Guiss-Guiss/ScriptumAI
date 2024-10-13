import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from backend.ingest_component import IngestComponent
from backend.embedding_component import EmbeddingComponent

@pytest.fixture
def mock_chroma_client():
    with patch('chromadb.PersistentClient') as mock_client:
        mock_instance = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_instance
        yield mock_client

@pytest.fixture
def mock_embedding_component():
    return Mock(spec=EmbeddingComponent)

@pytest.fixture
def ingest_component(mock_chroma_client, mock_embedding_component):
    return IngestComponent(mock_embedding_component)

def test_ingest_component_initialization(ingest_component):
    assert ingest_component is not None
    assert ingest_component.embedding_component is not None
    assert ingest_component.collection is not None

def test_ingest_file(ingest_component, tmp_path):
    # Create a temporary test file
    test_file = tmp_path / "test_document.txt"
    test_file.write_text("This is a test document for ingestion.")

    # Mock the necessary methods
    ingest_component.embedding_component.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    ingest_component.collection.add.return_value = None

    result = ingest_component.ingest_file(str(test_file))
    assert isinstance(result, dict)
    assert "filename" in result
    assert result["filename"] == "test_document.txt"

def test_ingest_directory(ingest_component, tmp_path):
    # Create temporary test files
    (tmp_path / "doc1.txt").write_text("This is document 1.")
    (tmp_path / "doc2.txt").write_text("This is document 2.")

    # Mock the necessary methods
    ingest_component.embedding_component.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ingest_component.collection.add.return_value = None

    results = ingest_component.ingest_directory(str(tmp_path))
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(result, dict) for result in results)

def test_get_collection_stats(ingest_component):
    stats = ingest_component.get_collection_stats()
    assert isinstance(stats, dict)
    assert "total_documents" in stats
    assert isinstance(stats["total_documents"], int)

if __name__ == '__main__':
    pytest.main()