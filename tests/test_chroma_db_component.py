import unittest
from unittest.mock import patch, MagicMock
from chroma_db_component import ChromaDBComponent

class TestChromaDBComponent(unittest.TestCase):
    @patch('chroma_db_component.chromadb.Client')
    def setUp(self, mock_client):
        self.mock_client = mock_client
        self.mock_collection = MagicMock()
        self.mock_client.return_value.get_or_create_collection.return_value = self.mock_collection
        self.chroma_db = ChromaDBComponent()

    def test_initialization(self):
        self.mock_client.assert_called_once()
        self.mock_client.return_value.get_or_create_collection.assert_called_once_with(name="document_collection")

    def test_add_documents(self):
        ids = ["1", "2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}]
        documents = ["content1", "content2"]

        self.chroma_db.add_documents(ids, embeddings, metadatas, documents)

        self.mock_collection.add.assert_called_once_with(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def test_query(self):
        query_embedding = [0.1, 0.2]
        n_results = 5
        expected_results = {
            'ids': [['1', '2']],
            'embeddings': [[[0.1, 0.2], [0.3, 0.4]]],
            'metadatas': [[{'source': 'doc1'}, {'source': 'doc2'}]],
            'documents': [['content1', 'content2']],
            'distances': [[0.1, 0.2]]
        }
        self.mock_collection.query.return_value = expected_results

        results = self.chroma_db.query(query_embedding, n_results)

        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        self.assertEqual(results, expected_results)

    def test_get_collection_stats(self):
        self.mock_collection.name = "test_collection"
        self.mock_collection.count.return_value = 10

        stats = self.chroma_db.get_collection_stats()

        self.assertEqual(stats, {"name": "test_collection", "count": 10})

if __name__ == '__main__':
    unittest.main()