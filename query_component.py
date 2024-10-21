import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryComponent:
    def __init__(self):
        logger.info("QueryComponent initialized")

    def process_query(self, query: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process the user query and prepare it for retrieval.

        :param query: The user's input query
        :param options: Optional parameters for query processing
        :return: A dictionary containing the processed query and any additional metadata
        """
        try:
            logger.info(f"Processing query: {query}")

            processed_query = query.strip().lower()

            result = {
                "original_query": query,
                "processed_query": processed_query,
                "options": options or {}
            }

            logger.info(f"Query processed successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise
