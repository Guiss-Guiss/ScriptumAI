import unittest
from unittest.mock import patch, MagicMock
import asyncio
from app import main

class TestIntegration(unittest.TestCase):
    @patch('app.st')
    @patch('app.file_uploader')
    @patch('app.ChromaDBComponent')
    @patch('app.EmbeddingComponent')
    @patch('app.RetrievalComponent')
    @patch('app.QueryComponent')
    @patch('app.RetrievalSystem')
    @patch('app.RAGComponent')
    async def test_main_flow(self, mock_rag, mock_retrieval_system, mock_query, mock_retrieval, 
                             mock_embedding, mock_chroma, mock_file_uploader, mock_st):

        mock_st.file_uploader.return_value = ['test_file.txt']
        mock_st.button.return_value = True
        mock_st.text_input.return_value = "What is the capital of France?"

        mock_file_uploader.return_value = None

        mock_retrieval_system_instance = mock_retrieval_system.return_value
        mock_retrieval_system_instance.fetch_relevant_chunks.return_value = [
            {'content': 'Paris is the capital of France.', 'similarity_score': 0.9, 'metadata': {}}
        ]

        mock_rag_instance = mock_rag.return_value
        mock_rag_instance.generate_answer.return_value = "The capital of France is Paris."

        await main()

        mock_file_uploader.assert_called_once()

        mock_retrieval_system_instance.fetch_relevant_chunks.assert_called_once_with("What is the capital of France?")

        mock_rag_instance.generate_answer.assert_called_once()
d
        mock_st.write.assert_any_call("The capital of France is Paris.")

if __name__ == '__main__':
    unittest.main()
