import pytest
from unittest.mock import patch, MagicMock
from embedding_component import EmbeddingComponent

class TestEmbeddingComponent:
    @pytest.fixture
    def embedding_component(self):
        return EmbeddingComponent()

    @pytest.mark.asyncio
    async def test_generate_embedding(self, embedding_component):
        with patch('embedding_component.ollama.completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = {"embedding": [0.1, 0.2, 0.3]}
            result = await embedding_component.generate_embedding("test text")
            assert result == [0.1, 0.2, 0.3]
            mock_completion.assert_called_once_with(model="llama2", prompt="test text")

    @pytest.mark.asyncio
    async def test_generate_embedding_error(self, embedding_component):
        with patch('embedding_component.ollama.completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.side_effect = Exception("API Error")
            result = await embedding_component.generate_embedding("test text")
            assert result is None

class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)