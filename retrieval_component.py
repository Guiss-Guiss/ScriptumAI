import logging
from chroma_db_component import ChromaDBComponent
from embedding_component import EmbeddingComponent

logger = logging.getLogger(__name__)

def normalize_distance(distance, max_distance=1.0, min_distance=0.0):
    """Normalise la distance pour obtenir une similarité."""
    return (distance - min_distance) / (max_distance - min_distance)

class RetrievalComponent:
    def __init__(self, chroma_db: ChromaDBComponent, embedding_component: EmbeddingComponent):
        self.chroma_db = chroma_db
        self.embedding_component = embedding_component
        logger.info("RetrievalComponent initialized")

    async def retrieve_similar_documents(self, query, n_results=5):
        try:
            logger.info(f"Retrieving similar documents for query: {query}")
            query_embedding = await self.embedding_component.generate_embedding(query)

            if query_embedding is None:
                logger.error("Failed to generate embedding for the query")
                return []

            # Exécution de la recherche de similarité
            results = self.chroma_db.similarity_search(query_embedding, n_results)

            # Normalisation des distances et ajout de la similarité dans les résultats
            normalized_distances = [normalize_distance(dist) for dist in results['distances'][0]]
            
            # Ajouter la similarité aux résultats pour le filtrage basé sur le seuil de confiance
            documents_with_similarity = []
            for i, chunk in enumerate(results['documents'][0]):
                chunk_similarity = 1 - normalized_distances[i]  # Convertir distance en similarité
                documents_with_similarity.append({
                    'content': chunk,
                    'similarity_score': chunk_similarity,
                    'metadata': results['metadatas'][0][i]
                })

            logger.info(f"Retrieved {len(documents_with_similarity)} similar documents")
            return documents_with_similarity
        except Exception as e:
            logger.error(f"Error retrieving similar documents: {str(e)}", exc_info=True)
            return []
