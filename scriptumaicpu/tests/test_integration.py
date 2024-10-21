import unittest
from unittest.mock import patch, MagicMock
import asyncio
from app import main  # Assuming your main application logic is in a function called main in app.py

class TestIntegration(unittest.TestCase):
    @patch('app.st')  # Mock Streamlit
    @patch('app.file_uploader')
    @patch('app.ChromaDBComponent')
    @patch('app.EmbeddingComponent')
    @patch('app.RetrievalComponent')
    @patch('app.QueryComponent')
    @patch('app.RetrievalSystem')
    @patch('app.RAGComponent')
    async def test_main_flow(self, mock_rag, mock_retrieval_system, mock_query, mock_retrieval, 
                             mock_embedding, mock_chroma, mock_file_uploader, mock_st):
        # Set up mock behaviors
        mock_st.file_uploader.return_value = ['test_file.txt']
        mock_st.button.return_value = True
        mock_st.text_input.return_value = "What is the capital of France?"

        mock_file_uploader.return_value = None  # Simulate successful file upload

        mock_retrieval_system_instance = mock_retrieval_system.return_value
        mock_retrieval_system_instance.fetch_relevant_chunks.return_value = [
            {'content': 'Paris is the capital of France.', 'similarity_score': 0.9, 'metadata': {}}
        ]

        mock_rag_instance = mock_rag.return_value
        mock_rag_instance.generate_answer.return_value = "The capital of France is Paris."

        # Run the main function
        await main()

        # Assert that the file was uploaded
        mock_file_uploader.assert_called_once()

        # Assert that a query was processed
        mock_retrieval_system_instance.fetch_relevant_chunks.assert_called_once_with("What is the capital of France?")

        # Assert that an answer was generated
        mock_rag_instance.generate_answer.assert_called_once()

        # Assert that the answer was displayed
        mock_st.write.assert_any_call("The capital of France is Paris.")

if __name__ == '__main__':
    unittest.main()