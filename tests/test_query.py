import pytest
from unittest.mock import Mock, patch
from backend.query_component import QueryComponent
from backend.embedding_component import EmbeddingComponent
from backend.retrieval_component import RetrievalComponent

@pytest.fixture
def mock_embedding_component():
    mock = Mock(spec=EmbeddingComponent)
    mock.embed_query.return_value = [0.1, 0.2, 0.3]  # Mock embedding
    return mock

@pytest.fixture
def mock_retrieval_component():
    mock = Mock(spec=RetrievalComponent)
    mock.retrieve.return_value = [
        {"chunk": "RAG is Retrieval-Augmented Generation.", "similarity_score": 0.9, "metadata": {"source": "doc1"}},
        {"chunk": "RAG combines retrieval and generation.", "similarity_score": 0.8, "metadata": {"source": "doc2"}}
    ]
    return mock

@pytest.fixture
def query_component(mock_embedding_component, mock_retrieval_component):
    return QueryComponent(mock_embedding_component, mock_retrieval_component)

def test_query_component_initialization(query_component):
    assert query_component is not None
    assert query_component.embedding_component is not None
    assert query_component.retrieval_component is not None

@patch('backend.query_component.ollama.Client')
def test_process_query(mock_ollama_client, query_component):
    mock_ollama_instance = Mock()
    mock_ollama_instance.generate.return_value = {
        'response': 'RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant information with text generation.'
    }
    mock_ollama_client.return_value = mock_ollama_instance

    query = "What is RAG?"
    result = query_component.process_query(query)

    assert isinstance(result, dict)
    assert 'query' in result
    assert 'response' in result
    assert 'relevant_chunks' in result
    assert result['query'] == query
    assert "RAG" in result['response']
    assert len(result['relevant_chunks']) == 2

    # Check if the embedding and retrieval components were called
    query_component.embedding_component.embed_query.assert_called_once_with(query)
    query_component.retrieval_component.retrieve.assert_called_once()

    # Check if ollama.generate was called with the correct prompt
    mock_ollama_instance.generate.assert_called_once()
    generate_args = mock_ollama_instance.generate.call_args[1]
    assert 'prompt' in generate_args
    assert query in generate_args['prompt']
    assert 'RAG is Retrieval-Augmented Generation' in generate_args['prompt']

@patch('backend.query_component.ollama.Client')
def test_process_query_with_error(mock_ollama_client, query_component):
    mock_ollama_instance = Mock()
    mock_ollama_instance.generate.side_effect = Exception("API Error")
    mock_ollama_client.return_value = mock_ollama_instance

    query = "What is RAG?"
    result = query_component.process_query(query)

    assert isinstance(result, dict)
    assert 'query' in result
    assert 'response' in result
    assert 'error' in result
    assert result['query'] == query
    assert result['response'] is None
    assert "Error processing query" in result['error']

def test_semantic_search(query_component):
    query = "What is RAG?"
    results = query_component.semantic_search(query, n_results=2)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(result, dict) for result in results)
    assert all('chunk' in result and 'similarity_score' in result for result in results)

    # Check if the embedding and retrieval components were called
    query_component.embedding_component.embed_query.assert_called_once_with(query)
    query_component.retrieval_component.retrieve.assert_called_once_with(query, k=2)

@patch('backend.query_component.ollama.Client')
def test_process_query_with_custom_parameters(mock_ollama_client, query_component):
    mock_ollama_instance = Mock()
    mock_ollama_instance.generate.return_value = {'response': 'Custom response'}
    mock_ollama_client.return_value = mock_ollama_instance

    query = "Custom query"
    result = query_component.process_query(query, temperature=0.5, max_tokens=100)

    mock_ollama_instance.generate.assert_called_once()
    generate_args = mock_ollama_instance.generate.call_args[1]
    assert generate_args['options']['temperature'] == 0.5
    assert generate_args['options']['max_tokens'] == 100

if __name__ == '__main__':
    pytest.main()