import pytest
import torch
from unittest.mock import Mock, patch
from backend.embedding_component import EmbeddingComponent

@pytest.fixture
def mock_ollama_client():
    with patch('ollama.Client') as mock_client:
        mock_instance = Mock()
        mock_instance.embeddings.return_value = {
            'embeddings': [
                {'embedding': [0.1, 0.2, 0.3]},
                {'embedding': [0.4, 0.5, 0.6]}
            ]
        }
        mock_client.return_value = mock_instance
        yield mock_client

@pytest.fixture
def embedding_component(mock_ollama_client):
    return EmbeddingComponent()

def test_embedding_component_initialization(embedding_component):
    assert embedding_component is not None
    assert embedding_component.model is not None
    assert embedding_component.device is not None

def test_embed_query(embedding_component):
    query = "What is RAG?"
    embedding = embedding_component.embed_query(query)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.dim() == 1  # Should be a 1D tensor
    assert embedding.shape[0] == 3  # Based on our mock data

def test_embed_documents(embedding_component):
    documents = ["This is document 1.", "This is document 2."]
    embeddings = embedding_component.embed_documents(documents)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.dim() == 2  # Should be a 2D tensor
    assert embeddings.shape == (2, 3)  # 2 documents, 3-dimensional embeddings

def test_cosine_similarity(embedding_component):
    vec1 = torch.tensor([1.0, 0.0, 0.0])
    vec2 = torch.tensor([0.0, 1.0, 0.0])
    similarity = embedding_component.cosine_similarity(vec1, vec2)
    assert isinstance(similarity, torch.Tensor)
    assert similarity.item() == pytest.approx(0.0)  # Orthogonal vectors should have similarity 0

    vec3 = torch.tensor([1.0, 0.0, 0.0])
    similarity = embedding_component.cosine_similarity(vec1, vec3)
    assert similarity.item() == pytest.approx(1.0)  # Identical vectors should have similarity 1

if __name__ == '__main__':
    pytest.main()