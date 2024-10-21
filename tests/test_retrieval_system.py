import unittest
from unittest.mock import Mock, patch
import asyncio
from retrieval_system import RetrievalSystem

class TestRetrievalSystem(unittest.TestCase):
    def setUp(self):
        self.mock_retrieval_component = Mock()
        self.mock_query_component = Mock()
        self.retrieval_system = RetrievalSystem(self.mock_retrieval_component, self.mock_query_component)

    @patch('retrieval_system.logger')
    async def test_fetch_relevant_chunks(self, mock_logger):
        self.mock_query_component.process_query.return_value = {
            'processed_query': 'processed test query',
            'options': {}
        }

        mock_results = [
            {'content': 'content1', 'metadata': {'source': 'doc1'}, 'distance': 0.1},
            {'content': 'content2', 'metadata': {'source': 'doc2'}, 'distance': 0.2}
        ]
        self.mock_retrieval_component.retrieve_similar_documents.return_value = mock_results

        results = await self.retrieval_system.fetch_relevant_chunks("test query", n_results=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['content'], 'content1')
        self.assertEqual(results[1]['content'], 'content2')
        self.assertAlmostEqual(results[0]['similarity_score'], 0.9)  # 1 - distance
        self.assertAlmostEqual(results[1]['similarity_score'], 0.8)  # 1 - distance

        self.mock_query_component.process_query.assert_called_once_with("test query")
        self.mock_retrieval_component.retrieve_similar_documents.assert_called_once_with('processed test query', 2)

        mock_logger.info.assert_called()

    @patch('retrieval_system.logger')
    async def test_fetch_relevant_chunks_error(self, mock_logger):

        self.mock_query_component.process_query.side_effect = Exception("Query processing error")

        results = await self.retrieval_system.fetch_relevant_chunks("test query")

        self.assertEqual(results, [])

        mock_logger.error.assert_called()

if __name__ == '__main__':
    unittest.main()
