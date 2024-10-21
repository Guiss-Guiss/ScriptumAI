import logging
from typing import List, Dict, Any
from retrieval_component import RetrievalComponent
from query_component import QueryComponent
from language_utils import get_translation, get_current_language

logger = logging.getLogger(__name__)

class RetrievalSystem:
    def __init__(self, retrieval_component: RetrievalComponent, query_component: QueryComponent):
        self.retrieval_component = retrieval_component
        self.query_component = query_component
        logger.info(get_translation("retrieval_system_initialized"))

    async def fetch_relevant_chunks(self, query: str, n_results: int = 5, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        lang = get_current_language()
        try:
            logger.info(get_translation("fetching_relevant_chunks").format(query=query))
            processed_query = self.query_component.process_query(query)
            
            # Récupérer les documents similaires
            results = await self.retrieval_component.retrieve_similar_documents(processed_query['processed_query'], n_results)
            
            # Filtrer les documents en fonction du seuil de similarité (confidence_threshold)
            filtered_chunks = [
                chunk for chunk in results if chunk['similarity_score'] >= confidence_threshold
            ]

            logger.info(get_translation("retrieved_relevant_chunks").format(
                chunk_count=len(filtered_chunks),
                confidence_threshold=confidence_threshold
            ))
            return filtered_chunks
        except Exception as e:
            logger.error(get_translation("error_fetching_chunks").format(error=str(e)), exc_info=True)
            return []