import unittest
from unittest.mock import Mock, patch
import asyncio
from retrieval_component import RetrievalComponent

class TestRetrievalComponent(unittest.TestCase):
    def setUp(self):
        self.mock_chroma_db = Mock()
        self.mock_embedding_component = Mock()
        self.retrieval_component = RetrievalComponent(self.mock_chroma_db, self.mock_embedding_component)

    @patch('retrieval_component.logger')
    async def test_retrieve_similar_documents(self, mock_logger):
        # Mock the generate_embedding method
        self.mock_embedding_component.generate_embedding.return_value = [0.1, 0.2, 0.3]

        # Mock the similarity_search method
        mock_results = [
            {'id': '1', 'document': 'content1', 'metadata': {'source': 'doc1'}, 'distance': 0.1},
            {'id': '2', 'document': 'content2', 'metadata': {'source': 'doc2'}, 'distance': 0.2}
        ]
        self.mock_chroma_db.similarity_search.return_value = mock_results

        # Call the method
        results = await self.retrieval_component.retrieve_similar_documents("test query", n_results=2)

        # Assert the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['id'], '1')
        self.assertEqual(results[1]['id'], '2')

        # Assert that the methods were called correctly
        self.mock_embedding_component.generate_embedding.assert_called_once_with("test query")
        self.mock_chroma_db.similarity_search.assert_called_once_with([0.1, 0.2, 0.3], 2)

        # Assert that the logger was used
        mock_logger.info.assert_called()

    @patch('retrieval_component.logger')
    async def test_retrieve_similar_documents_error(self, mock_logger):
        # Mock the generate_embedding method to raise an exception
        self.mock_embedding_component.generate_embedding.side_effect = Exception("Embedding error")

        # Call the method
        results = await self.retrieval_component.retrieve_similar_documents("test query")

        # Assert that an empty list is returned on error
        self.assertEqual(results, [])

        # Assert that the error was logged
        mock_logger.error.assert_called()

if __name__ == '__main__':
    unittest.main()