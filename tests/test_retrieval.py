import pytest
from backend.retrieval_component import RetrievalComponent
from backend.embedding_component import EmbeddingComponent
from backend.ingest_component import IngestComponent

@pytest.fixture
def retrieval_component():
    embedding_component = EmbeddingComponent()
    ingest_component = IngestComponent(embedding_component)
    return RetrievalComponent(embedding_component)

@pytest.fixture
def sample_documents(retrieval_component):
    docs = [
        "RAG stands for Retrieval-Augmented Generation.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science."
    ]
    for doc in docs:
        retrieval_component.collection.add(
            documents=[doc],
            metadatas=[{"source": "test"}],
            ids=[f"doc_{docs.index(doc)}"]
        )
    return docs

def test_retrieval_component_initialization(retrieval_component):
    assert retrieval_component is not None
    assert retrieval_component.embedding_component is not None
    assert retrieval_component.collection is not None

def test_retrieve(retrieval_component, sample_documents):
    query = "What is RAG?"
    results = retrieval_component.retrieve(query, k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(result, dict) for result in results)
    assert all("chunk" in result and "similarity_score" in result for result in results)
    assert results[0]["chunk"] == sample_documents[0]  # The most relevant document should be the first one

def test_batch_retrieve(retrieval_component, sample_documents):
    queries = ["What is RAG?", "Tell me about machine learning"]
    results = retrieval_component.batch_retrieve(queries, k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(result, list) for result in results)
    assert all(len(result) == 2 for result in results)
    assert results[0][0]["chunk"] == sample_documents[0]  # The most relevant document for the first query
    assert results[1][0]["chunk"] == sample_documents[1]  # The most relevant document for the second query

def test_retrieve_by_id(retrieval_component, sample_documents):
    chunk_ids = ["doc_0", "doc_1"]
    results = retrieval_component.retrieve_by_id(chunk_ids)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(result, dict) for result in results)
    assert results[0]["chunk"] == sample_documents[0]
    assert results[1]["chunk"] == sample_documents[1]

def test_get_collection_stats(retrieval_component, sample_documents):
    stats = retrieval_component.get_collection_stats()
    assert isinstance(stats, dict)
    assert "total_chunks" in stats
    assert stats["total_chunks"] == len(sample_documents)
