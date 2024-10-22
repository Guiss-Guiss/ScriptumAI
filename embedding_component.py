import logging
import ollama
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)

def get_optimal_thread_count():
    num_cores = multiprocessing.cpu_count()
    return max(2, num_cores * 2)  # Optimiser pour le CPU

class EmbeddingComponent:
    def __init__(self):
        self.EMBEDDING_DIMENSION = 768
        self.EMBED_MODEL = "nomic-embed-text"
        self.num_threads = get_optimal_thread_count()
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        logger.info(f"EmbeddingComponent initialized with dimension: {self.EMBEDDING_DIMENSION} and {self.num_threads} threads")

    def _generate_embedding_sync(self, text: str) -> List[float]:
        try:
            logger.info(f"Generating embedding for text of length {len(text)}")
            response = ollama.embeddings(model=self.EMBED_MODEL, prompt=text)
            embedding = response['embedding']
            if embedding is not None:
                if len(embedding) != self.EMBEDDING_DIMENSION:
                    raise ValueError(f"Generated embedding dimension {len(embedding)} does not match expected {self.EMBEDDING_DIMENSION}")
                logger.info(f"Embedding generated successfully. Dimension: {len(embedding)}")
                return embedding
            else:
                logger.warning("No embedding generated")
                return None
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    async def generate_embedding(self, text: str) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._generate_embedding_sync, text)

    async def generate_embeddings_for_chunks(self, chunks: List[str], batch_size=5) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.num_threads} threads in batches of {batch_size}")
        embeddings = []
        
        # Traiter les chunks en batchs pour réduire la charge sur CPU
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            tasks = [self.generate_embedding(chunk) for chunk in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            embeddings.extend([emb for emb in batch_embeddings if emb is not None])
        
        logger.info(f"Generated {len(embeddings)} embeddings out of {len(chunks)} chunks")
        return embeddings

    def get_thread_count(self):
        return self.num_threads
