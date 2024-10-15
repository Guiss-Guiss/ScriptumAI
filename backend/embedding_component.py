import ollama
import torch
import torch.multiprocessing as mp
from typing import List, Union
from config import OLLAMA_BASE_URL, EMBEDDING_MODEL, BATCH_SIZE, EMBEDDING_DEVICE
from loguru import logger

class EmbeddingComponent:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.model = EMBEDDING_MODEL
        self.device = torch.device(EMBEDDING_DEVICE if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized EmbeddingComponent with model: {self.model} on device: {self.device}")

    def get_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            embeddings = pool.map(self._get_single_embedding, texts)

        return torch.tensor(embeddings, device=self.device)

    def _get_single_embedding(self, text: str) -> List[float]:
        try:
            logger.debug(f"Sending request to Ollama API for text: {text[:50]}...")
            logger.debug(f"Using model: {self.model}")
            logger.debug(f"Ollama base URL: {OLLAMA_BASE_URL}")

            response = self.client.embeddings(model=self.model, prompt=text)
            logger.debug(f"Raw API response: {response}")

            if not isinstance(response, dict):
                logger.error(f"Unexpected response type: {type(response)}")
                raise TypeError(f"Expected dict, got {type(response)}")

            if 'embedding' not in response:
                logger.error(f"'embedding' key not found in response. Keys present: {list(response.keys())}")
                raise KeyError("'embedding' key not found in API response")

            return response['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding for text: {str(e)}", exc_info=True)
            raise

    def embed_documents(self, documents: List[str]) -> torch.Tensor:
        logger.info(f"Embedding {len(documents)} documents")
        return self.get_embeddings(documents)

    def embed_query(self, query: str) -> torch.Tensor:
        logger.info(f"Embedding query: {query[:50]}...")
        return self.get_embeddings(query)[0]

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)

    def batch_embed(self, texts: List[str], batch_size: int = BATCH_SIZE) -> torch.Tensor:
        logger.info(f"Batch embedding {len(texts)} texts with batch size {batch_size}")
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.get_embeddings(batch)
            all_embeddings.append(batch_embeddings)
        return torch.cat(all_embeddings, dim=0)

