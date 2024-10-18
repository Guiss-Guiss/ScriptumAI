import numpy as np
from backend.embedding_component import EmbeddingComponent
from backend.ingest_component import IngestComponent
from config import EMBEDDING_MODEL, CHROMA_COLLECTION_NAME, SUPPORTED_LANGUAGES
from loguru import logger

def check_embedding_consistency():
    embedding_component = EmbeddingComponent()
    ingest_component = IngestComponent(embedding_component)

    # Get the actual embedding dimension
    embedding_dim = embedding_component.get_embedding_dim()
    logger.info(f"Actual embedding dimension for model {EMBEDDING_MODEL}: {embedding_dim}")

    # Test embedding generation
    test_text = "This is a test sentence for embedding."
    test_embedding = embedding_component.get_embeddings(test_text)
    logger.info(f"Test embedding dimension: {test_embedding.shape[1]}")

    if test_embedding.shape[1] != embedding_dim:
        logger.error(f"Embedding dimension mismatch! Expected {embedding_dim}, got {test_embedding.shape[1]}")
    else:
        logger.info("Embedding dimension is correct.")

    # Check collections
    for lang in SUPPORTED_LANGUAGES:
        collection_name = f"{CHROMA_COLLECTION_NAME}_{lang}"
        collection = ingest_component.collections.get(lang)
        if collection:
            try:
                count = collection.count()
                logger.info(f"Collection {collection_name} count: {count}")
                
                if count > 0:
                    sample = collection.get(limit=1, include=["embeddings"])
                    if 'embeddings' in sample and len(sample['embeddings']) > 0:
                        sample_embedding = sample['embeddings'][0]
                        if isinstance(sample_embedding, (np.ndarray, list)):
                            sample_dimension = len(sample_embedding)
                            logger.info(f"Sample embedding dimension in {collection_name}: {sample_dimension}")
                            if sample_dimension != embedding_dim:
                                logger.error(f"Dimension mismatch in {collection_name}! Expected {embedding_dim}, got {sample_dimension}")
                            logger.info(f"Sample embedding type: {type(sample_embedding)}")
                            logger.info(f"Sample embedding first few values: {sample_embedding[:5]}")
                        else:
                            logger.error(f"Unexpected embedding type in {collection_name}: {type(sample_embedding)}")
                    else:
                        logger.warning(f"No embeddings found in sample from {collection_name}")
            except Exception as e:
                logger.error(f"Error checking collection {collection_name}: {str(e)}", exc_info=True)
        else:
            logger.warning(f"Collection {collection_name} not found")

    # Test search functionality
    try:
        fr_collection = ingest_component.collections.get('fr')
        if fr_collection:
            search_results = fr_collection.query(
                query_texts=["test query"],
                n_results=1
            )
            logger.info(f"Search functionality test successful. Results: {search_results}")
            
            if not search_results['ids'][0]:
                logger.warning("Search returned no results. Checking collection content...")
                sample_docs = fr_collection.get(limit=5, include=["documents", "metadatas"])
                logger.info(f"Sample documents from collection: {sample_docs}")
        else:
            logger.warning("French collection not found for search test")
    except Exception as e:
        logger.error(f"Error during search test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    check_embedding_consistency()