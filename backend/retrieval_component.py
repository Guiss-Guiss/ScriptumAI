import torch
from typing import List, Dict, Any
from loguru import logger
from backend.utils import initialize_chroma_client
import langdetect

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

    def find_similar_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embedding_component.embed_query(query)
            query_lang = self._detect_language(query)

            all_results = []
            for lang, collection in self.collections.items():
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=k,
                    include=["metadatas", "documents", "distances"]
                )
                for i in range(len(results['ids'][0])):
                    all_results.append({
                        'chunk_id': results['ids'][0][i],
                        'chunk': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],
                        'language': lang
                    })

            all_results.sort(key=lambda x: x['similarity_score'], reverse=True)

            prioritized_results = [r for r in all_results if r['language'] == query_lang] + \
                                  [r for r in all_results if r['language'] != query_lang]

            return prioritized_results[:k]

        except Exception as e:
            logger.error(f"Error finding similar chunks: {str(e)}", exc_info=True)
            return []

    @torch.no_grad()
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.find_similar_chunks(query, k)

    def batch_retrieve(self, queries: List[str], k: int = TOP_K_RESULTS) -> List[List[Dict[str, Any]]]:
        logger.info(f"Batch retrieving top {k} results for {len(queries)} queries")
        query_embeddings = self.embedding_component.embed_documents(queries)
        
        batch_retrieved_chunks = []
        for i, query in enumerate(queries):
            query_lang = self._detect_language(query)
            all_results = []
            
            for lang, collection in self.collections.items():
                results = collection.query(
                    query_embeddings=[query_embeddings[i].cpu().numpy().tolist()],
                    n_results=k,
                    include=["documents", "metadatas", "distances"]
                )
                for j in range(len(results['ids'][0])):
                    all_results.append({
                        "chunk_id": results['ids'][0][j],
                        "chunk": results['documents'][0][j],
                        "metadata": results['metadatas'][0][j],
                        "similarity_score": 1 - results['distances'][0][j],
                        "language": lang
                    })
            
            all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            prioritized_results = [r for r in all_results if r['language'] == query_lang] + \
                                  [r for r in all_results if r['language'] != query_lang]
            
            batch_retrieved_chunks.append(prioritized_results[:k])

        return batch_retrieved_chunks

    def retrieve_by_id(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving {len(chunk_ids)} chunks by ID")

        retrieved_chunks = []
        for lang, collection in self.collections.items():
            results = collection.get(
                ids=chunk_ids,
                include=["documents", "metadatas"]
            )
            for i in range(len(results['ids'])):
                retrieved_chunks.append({
                    "chunk_id": results['ids'][i],
                    "chunk": results['documents'][i],
                    "metadata": results['metadatas'][i],
                    "language": lang
                })

        return retrieved_chunks

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

