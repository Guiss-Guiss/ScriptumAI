import torch
import torch.multiprocessing as mp
from typing import List, Dict, Any
from loguru import logger
from backend.utils import initialize_chroma_client
import langdetect
from functools import partial

from config import (
    TOP_K_RESULTS,
    EMBEDDING_DEVICE,
    CHROMA_COLLECTION_NAME,
    SUPPORTED_LANGUAGES
)
from backend.embedding_component import EmbeddingComponent

class RetrievalComponent:
    def __init__(self, embedding_component: EmbeddingComponent):
        self.embedding_component = embedding_component
        self.device = torch.device(EMBEDDING_DEVICE if torch.cuda.is_available() else "cpu")
        try:
            self.chroma_client, self.collections = self._initialize_collections()
            logger.info(f"Initialized RetrievalComponent with collections: {[col.name for col in self.collections.values()]} on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client and collections: {e}", exc_info=True)
            raise

    def _initialize_collections(self):
        client, _ = initialize_chroma_client()
        collections = {}
        for lang in SUPPORTED_LANGUAGES:
            collection_name = f"{CHROMA_COLLECTION_NAME}_{lang}"
            collections[lang] = client.get_or_create_collection(name=collection_name)
        return client, collections

    def _detect_language(self, text: str) -> str:
        try:
            return langdetect.detect(text)
        except:
            return SUPPORTED_LANGUAGES[0]

    def find_similar_chunks(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embedding_component.embed_query(query)
            query_lang = self._detect_language(query)

            with mp.Pool(processes=mp.cpu_count()) as pool:
                all_results = pool.map(partial(self._search_collection, query_embedding=query_embedding.tolist(), k=k), 
                                       self.collections.items())

            all_results = [item for sublist in all_results for item in sublist]
            all_results.sort(key=lambda x: x['similarity_score'], reverse=True)

            prioritized_results = [r for r in all_results if r['language'] == query_lang] + \
                                  [r for r in all_results if r['language'] != query_lang]

            return prioritized_results[:k]

        except Exception as e:
            logger.error(f"Error finding similar chunks: {str(e)}", exc_info=True)
            return []

    def _search_collection(self, collection_item, query_embedding, k):
        lang, collection = collection_item
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["metadatas", "documents", "distances"]
            )
            return [
                {
                    'chunk_id': results['ids'][0][i],
                    'chunk': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],
                    'language': lang
                }
                for i in range(len(results['ids'][0]))
            ]
        except Exception as e:
            logger.error(f"Error searching collection {lang}: {str(e)}")
            return []

    @torch.no_grad()
    def retrieve(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        return self.find_similar_chunks(query, k)

    def batch_retrieve(self, queries: List[str], k: int = TOP_K_RESULTS) -> List[List[Dict[str, Any]]]:
        logger.info(f"Batch retrieving top {k} results for {len(queries)} queries")
        query_embeddings = self.embedding_component.embed_documents(queries)
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            batch_retrieved_chunks = pool.map(partial(self.find_similar_chunks, k=k), queries)

        return batch_retrieved_chunks

    def retrieve_by_id(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving {len(chunk_ids)} chunks by ID")

        with mp.Pool(processes=mp.cpu_count()) as pool:
            retrieved_chunks = pool.map(self._retrieve_single_chunk, chunk_ids)

        return [chunk for chunk in retrieved_chunks if chunk is not None]

    def _retrieve_single_chunk(self, chunk_id: str) -> Dict[str, Any]:
        for lang, collection in self.collections.items():
            try:
                results = collection.get(
                    ids=[chunk_id],
                    include=["documents", "metadatas"]
                )
                if results['ids']:
                    return {
                        "chunk_id": results['ids'][0],
                        "chunk": results['documents'][0],
                        "metadata": results['metadatas'][0],
                        "language": lang
                    }
            except Exception as e:
                logger.error(f"Error retrieving chunk {chunk_id} from collection {lang}: {str(e)}")
        return None

    def get_collection_stats(self) -> Dict[str, int]:
        try:
            stats = {}
            for lang, collection in self.collections.items():
                stats[f"total_chunks_{lang}"] = collection.count()
            stats["total_chunks"] = sum(stats.values())
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}", exc_info=True)
            raise