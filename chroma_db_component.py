import chromadb
import logging

logger = logging.getLogger(__name__)

class ChromaDBComponent:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.EMBEDDING_DIMENSION = 768
        self._initialize_client()
        self._create_collection()

    def _initialize_client(self):
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            self._log_and_raise_error("Error initializing ChromaDB client", e)

    def _create_collection(self, collection_name="document_collection"):
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None
            )
            logger.info(f"Collection '{collection_name}' created or retrieved successfully")
            
            collection_info = self.collection.get()
            if collection_info['embeddings']:
                dim = len(collection_info['embeddings'][0])
                logger.info(f"Collection dimensionality: {dim}")
                if dim != self.EMBEDDING_DIMENSION:
                    raise ValueError(f"Collection dimensionality {dim} does not match expected {self.EMBEDDING_DIMENSION}")
            else:
                logger.info("Collection is empty")
        except Exception as e:
            self._log_and_raise_error("Error creating/retrieving collection", e)

    def add_documents(self, ids, embeddings, metadatas, documents):
        try:
            logger.info(f"Adding {len(documents)} documents, {len(embeddings)} embeddings, and {len(metadatas)} metadata entries")

            if len(documents) != len(embeddings) or len(documents) != len(metadatas) or len(ids) != len(documents):
                raise ValueError("Mismatch in the number of documents, embeddings, metadatas, or ids")

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Successfully added {len(ids)} documents to the collection")
        except Exception as e:
            logger.error(f"Error adding documents to collection: {str(e)}", exc_info=True)
            raise


    def list_all_documents(self):
        try:
            return self.collection.get()
        except Exception as e:
            logger.error(f"Error listing all documents: {str(e)}")
            raise

    def query(self, query_embedding, n_results=5):
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            logger.info(f"Query executed successfully, returned {len(results['ids'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}")
            raise

    def get_collection_stats(self):
        try:
            return {
                "name": self.collection.name,
                "count": self.collection.count()
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise

    def list_all_documents(self):
        try:
            all_docs = self.collection.get()
            logger.info(f"Retrieved {len(all_docs['ids'])} documents from the collection")
            return all_docs
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise

    def similarity_search(self, query_embedding, n_results=5):
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            if 'ids' not in results or 'documents' not in results:
                logger.error("Unexpected structure in results from ChromaDB")
                return []

            logger.info(f"Results: {results}")

            logger.info(f"Similarity search executed successfully, returned {len(results['ids'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Error with similarity search: {str(e)}")
            raise
