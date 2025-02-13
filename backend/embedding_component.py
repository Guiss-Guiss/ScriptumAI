import ollama
import torch
from typing import List, Union
from config import OLLAMA_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSION, BATCH_SIZE, EMBEDDING_DEVICE
from loguru import logger

class EmbeddingComponent:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.model = EMBEDDING_MODEL
        self.dimension = EMBEDDING_DIMENSION
        self.device = torch.device(EMBEDDING_DEVICE if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized EmbeddingComponent with model: {self.model}, dimension: {self.dimension} on device: {self.device}")

    def get_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for text in texts:
            try:
                logger.debug(f"Sending request to Ollama API for text: {text[:50]}...")
                logger.debug(f"Using model: {self.model}")
                logger.debug(f"Ollama base URL: {OLLAMA_BASE_URL}")

                response = self.client.embeddings(model=self.model, prompt=text)
                logger.debug(f"Raw API response: {response}")

                if isinstance(response, dict):
                    embedding = response.get("embedding")
                elif hasattr(response, "embedding"):  
                    embedding = response.embedding  # Extract the embedding attribute from the object
                elif isinstance(response, list) and len(response) > 0 and isinstance(response[0], float):
                    embedding = response  # If response is directly a list of floats, use it
                else:
                    logger.error(f"Unexpected response type: {type(response)}")
                    raise TypeError(f"Expected dict or EmbeddingsResponse object, got {type(response)}")

                # Validate that embedding is a list
                if not isinstance(embedding, list):
                    logger.error(f"Invalid embedding format: {embedding}")
                    raise ValueError("Ollama API did not return a valid embedding list")

                # Handle dimension mismatch
                if len(embedding) != self.dimension:
                    logger.warning(f"Embedding dimension mismatch. Expected: {self.dimension}, Got: {len(embedding)}")
                    if len(embedding) < self.dimension:
                        embedding += [0] * (self.dimension - len(embedding))  # Pad with zeros
                    else:
                        embedding = embedding[:self.dimension]  # Trim excess dimensions

                all_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error getting embedding for text: {str(e)}", exc_info=True)
                raise

        embeddings_tensor = torch.tensor(all_embeddings, device=self.device)
        logger.info(f"Created embeddings tensor of shape {embeddings_tensor.shape}")
        return embeddings_tensor


    def embed_documents(self, documents: List[str]) -> torch.Tensor:
        logger.info(f"Embedding {len(documents)} documents")
        return self.get_embeddings(documents)

    def embed_query(self, query: str) -> torch.Tensor:
        logger.info(f"Embedding query: {query[:50]}...")
        return self.get_embeddings(query)[0]

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two tensors.

        Args:
            a (torch.Tensor): First tensor
            b (torch.Tensor): Second tensor

        Returns:
            torch.Tensor: Cosine similarity
        """
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)

    def get_embedding_dim(self) -> int:
        return self.dimension