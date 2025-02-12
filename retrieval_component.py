import logging
from typing import List, Dict, Any
from chroma_db_component import ChromaDBComponent
from embedding_component import EmbeddingComponent
from caching import cache_manager

logger = logging.getLogger(__name__)

class RetrievalComponent:
    def __init__(self, chroma_db: ChromaDBComponent, embedding_component: EmbeddingComponent):
        self.chroma_db = chroma_db
        self.embedding_component = embedding_component
        logger.info("RetrievalComponent initialized")

    @cache_manager.cache_query
    async def retrieve_similar_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            # Generate embedding for the query
            query_embedding = await self.embedding_component.generate_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []

            # Perform similarity search
            results = self.chroma_db.similarity_search(query_embedding, n_results)
            
            if not results or 'ids' not in results:
                logger.warning("No results found in similarity search")
                return []

            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': float(results['distances'][0][i]) if 'distances' in results else 1.0
                }
                formatted_results.append(result)

            logger.info(f"Retrieved {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving similar documents: {str(e)}", exc_info=True)
            return []
