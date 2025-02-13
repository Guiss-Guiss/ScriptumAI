import os
import logging
from typing import List, Dict, Any

from config import SUPPORTED_FILE_TYPES, LLM_MODEL
from backend.embedding_component import EmbeddingComponent
from backend.ingest_component import IngestComponent
from backend.retrieval_component import RetrievalComponent
from backend.query_component import QueryComponent

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Starting main.py")
logger.debug("Current working directory: %s", os.getcwd())

class RAGApplication:
    def __init__(self):
        logger.info("Initializing RAG Application")
        self.embedding_component = EmbeddingComponent()
        self.ingest_component = IngestComponent(self.embedding_component)
        self.retrieval_component = RetrievalComponent(self.embedding_component)
        self.query_component = QueryComponent(self.embedding_component, self.retrieval_component)

    def ingest_document(self, file_path: str):
        """Handles document ingestion."""
        try:
            self.ingest_component.ingest_file(file_path)
            logger.info(f"File ingested successfully: {file_path}")
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}", exc_info=True)
            raise Exception(f"Error ingesting document: {str(e)}")

    def ingest_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Handles batch ingestion of documents in a directory."""
        logger.info(f"Ingesting documents from directory: {directory_path}")
        return self.ingest_component.ingest_directory(directory_path)

    def process_query(self, query: str) -> Dict[str, Any]:
        """Processes a user query and retrieves relevant information."""
        logger.info(f"Processing query: {query}")
        try:
            result = self.query_component.process_query(query)
            if result.get('error'):
                logger.error(f"Error processing query: {result['error']}", exc_info=True)
                print(f"Error: {result['error']}")
            else:
                print(f"Response: {result['response']}")
                print("\nRelevant chunks:")
                for i, chunk in enumerate(result['relevant_chunks'], 1):
                    print(f"{i}. {chunk['chunk'][:100]}... (Score: {chunk['similarity_score']:.4f})")
        except Exception as e:
            logger.error(f"Unhandled error processing query: {e}", exc_info=True)
            print(f"An unexpected error occurred while processing the query.")
        return result

    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Performs semantic search to retrieve similar documents."""
        logger.info(f"Performing semantic search for query: {query}")
        return self.retrieval_component.retrieve(query, k)

    def get_stats(self) -> Dict[str, Any]:
        """Returns system statistics for document ingestion and model usage."""
        return {
            "total_documents": self.ingest_component.get_collection_stats().get("total_documents", 0),
            "embedding_model": self.embedding_component.model,
            "llm_model": LLM_MODEL,
            "supported_file_types": SUPPORTED_FILE_TYPES,
        }

def print_menu():
    """Displays a CLI menu for the RAG application."""
    print("\nRAG Application Menu:")
    print("1. Ingest a document")
    print("2. Ingest a directory")
    print("3. Process a query")
    print("4. Perform semantic search")
    print("5. Get system statistics")
    print("6. Exit")

def main():

    while True:
        print_menu()
        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            file_path = input("Enter the path to the document: ")
            try:
                rag_app.ingest_document(file_path)
                print(f"Ingested document successfully.")
            except Exception as e:
                logger.error(f"Error ingesting document: {e}", exc_info=True)
                print(f"Error ingesting document: {str(e)}")

        elif choice == '2':
            dir_path = input("Enter the path to the directory: ")
            try:
                results = rag_app.ingest_directory(dir_path)
                print(f"Ingested {len(results)} documents")
            except Exception as e:
                logger.error(f"Error ingesting directory: {e}", exc_info=True)
                print(f"Error ingesting directory: {str(e)}")

        elif choice == '3':
            query = input("Enter your query: ")
            rag_app.process_query(query)

        elif choice == '4':
            query = input("Enter your search query: ")
            try:
                results = rag_app.semantic_search(query)
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['chunk'][:100]}... (Score: {result['similarity_score']:.4f})")
            except Exception as e:
                logger.error(f"Error performing semantic search: {e}", exc_info=True)
                print(f"Error performing semantic search: {str(e)}")

        elif choice == '5':
            try:
                stats = rag_app.get_stats()
                for key, value in stats.items():
                    print(f"{key}: {value}")
            except Exception as e:
                logger.error(f"Error getting system statistics: {e}", exc_info=True)
                print(f"Error getting system statistics: {str(e)}")

        elif choice == '6':
            print("Exiting RAG Application. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

rag_app = RAGApplication()
