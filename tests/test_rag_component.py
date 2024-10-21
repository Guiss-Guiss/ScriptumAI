import unittest
from unittest.mock import patch, MagicMock
import asyncio
from rag_component import RAGComponent

class TestRAGComponent(unittest.TestCase):
    def setUp(self):
        self.rag_component = RAGComponent()

    @patch('rag_component.ollama.achat')
    @patch('rag_component.logger')
    async def test_generate_answer(self, mock_logger, mock_achat):

        mock_achat.return_value = {
            'message': {
                'content': 'This is a generated answer.'
            }
        }

        query = "What is the capital of France?"
        context_chunks = [
            {'content': 'Paris is the capital of France.'},
            {'content': 'France is a country in Western Europe.'}
        ]

        answer = await self.rag_component.generate_answer(query, context_chunks)

        self.assertEqual(answer, 'This is a generated answer.')

        expected_prompt = """Given the following context, answer the question. If the answer is not in the context, say "I don't have enough information to answer that question."

Context:
Paris is the capital of France.
France is a country in Western Europe.

Question: What is the capital of France?

Answer:"""
        mock_achat.assert_called_once_with(model='llama2', messages=[
            {
                'role': 'user',
                'content': expected_prompt
            }
        ])

        mock_logger.info.assert_called()

    @patch('rag_component.ollama.achat')
    @patch('rag_component.logger')
    async def test_generate_answer_no_context(self, mock_logger, mock_achat):
        query = "What is the capital of France?"
        context_chunks = []

        answer = await self.rag_component.generate_answer(query, context_chunks)

        self.assertEqual(answer, "I'm sorry, I couldn't generate an answer at this time.")

        mock_achat.assert_not_called()

        mock_logger.warning.assert_called()

    @patch('rag_component.ollama.achat')
    @patch('rag_component.logger')
    async def test_generate_answer_api_error(self, mock_logger, mock_achat):

        mock_achat.side_effect = Exception("API Error")

        query = "What is the capital of France?"
        context_chunks = [{'content': 'Paris is the capital of France.'}]

        answer = await self.rag_component.generate_answer(query, context_chunks)

        self.assertEqual(answer, "An error occurred while generating the answer.")

        mock_logger.error.assert_called()

if __name__ == '__main__':
    unittest.main()
