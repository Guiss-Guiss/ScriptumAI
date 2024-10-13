import ollama
import torch
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

        all_embeddings = []
        for text in texts:
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

                embedding = response['embedding']
                all_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error getting embedding for text: {str(e)}", exc_info=True)
                raise

        return torch.tensor(all_embeddings, device=self.device)

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

if __name__ == "__main__":
    # Test the EmbeddingComponent
    embedder = EmbeddingComponent()

    # Test document embedding
    docs = ["This is a test document.", "This is another test document."]
    doc_embeddings = embedder.embed_documents(docs)
    print(f"Document embeddings shape: {doc_embeddings.shape}")

    # Test query embedding
    query = "What is RAG?"
    query_embedding = embedder.embed_query(query)
    print(f"Query embedding shape: {query_embedding.shape}")

    # Test cosine similarity
    similarity = embedder.cosine_similarity(doc_embeddings[0].unsqueeze(0), query_embedding.unsqueeze(0))
    print(f"Cosine similarity between first document and query: {similarity.item()}")