import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
import torch
import torch.multiprocessing as mp
import magic
import langdetect
import json

from config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    SUPPORTED_FILE_TYPES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_DEVICE,
    BATCH_SIZE,
    SUPPORTED_LANGUAGES
)
from backend.embedding_component import EmbeddingComponent
from backend.utils import (
    read_file,
    chunk_text,
    get_file_metadata,
    initialize_chroma_client
)

INGESTED_FILES_PATH = "backend/ingested_files.json"

class IngestComponent:
    def __init__(self, embedding_component: EmbeddingComponent):
        self.embedding_component = embedding_component
        self.device = torch.device(EMBEDDING_DEVICE if torch.cuda.is_available() else "cpu")
        try:
            self.chroma_client, self.collections = self._initialize_collections()
            logger.info(f"Initialized IngestComponent with collections: {[col.name for col in self.collections.values()]} on device: {self.device}")
            self.ingested_files = self._load_ingested_files()
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

    def _load_ingested_files(self) -> List[str]:
        if os.path.exists(INGESTED_FILES_PATH):
            with open(INGESTED_FILES_PATH, "r") as f:
                return json.load(f)
        return []

    def _save_ingested_files(self):
        with open(INGESTED_FILES_PATH, "w") as f:
            json.dump(self.ingested_files, f)

    def _detect_language(self, text: str) -> str:
        try:
            return langdetect.detect(text)
        except:
            return SUPPORTED_LANGUAGES[0]

    def get_ingested_files(self) -> List[str]:
        return self.ingested_files

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        try:
            logger.debug(f"Ingesting file {file_path}")

            content = read_file(file_path)
            chunks = chunk_text(content, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            metadata = get_file_metadata(file_path)
            lang = self._detect_language(content)

            file_name = Path(file_path).name
            ids = [f"{file_name}_{i}" for i in range(len(chunks))]
            metadatas = [{**metadata, "chunk_index": i, "language": lang, "filename": file_name} for i in range(len(chunks))]

            embeddings = self.embedding_component.get_embeddings(chunks)

            collection = self.collections.get(lang, self.collections[SUPPORTED_LANGUAGES[0]])
            collection.add(
                ids=ids,
                embeddings=embeddings.cpu().numpy().tolist(),
                documents=chunks,
                metadatas=metadatas
            )

            if file_name not in self.ingested_files:
                self.ingested_files.append(file_name)
                self._save_ingested_files()

            logger.debug(f"Successfully ingested file {file_path}")
            return {"status": "success", "file": str(file_path), "language": lang}
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {str(e)}", exc_info=True)
            return {"status": "error", "file": str(file_path), "error": str(e)}

    def remove_file(self, file_name: str) -> bool:
        try:
            logger.debug(f"Attempting to remove file: {file_name}")
            
            for collection in self.collections.values():
                collection.delete(where={"filename": file_name})
            
            if file_name in self.ingested_files:
                self.ingested_files.remove(file_name)
                self._save_ingested_files()
            
            logger.info(f"Successfully removed file: {file_name}")
            return True
        except Exception as e:
            logger.error(f"Error removing file {file_name}: {str(e)}", exc_info=True)
            return False

    def ingest_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        logger.info(f"Ingesting documents from directory: {directory_path}")

        file_paths = [file_path for file_path in directory_path.rglob("*") 
                      if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FILE_TYPES]

        results = []
        for file_path in file_paths:
            result = self.ingest_file(str(file_path))
            results.append(result)

        successful = [result for result in results if result.get('status') == 'success']
        failed = [result for result in results if result.get('status') == 'error']

        logger.info(f"Successfully ingested {len(successful)} documents from the directory.")
        if failed:
            logger.warning(f"Failed to ingest {len(failed)} files.")

        return results

    def check_chroma_health(self):
        try:
            self.chroma_client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"Error checking Chroma health: {e}")
            return False

    def get_collection_stats(self):
        try:
            stats = {}
            total_documents = 0
            for lang, collection in self.collections.items():
                count = collection.count()
                stats[f"total_chunks_{lang}"] = count
                total_documents += count
            
            stats["total_documents"] = total_documents
            stats["total_chunks"] = total_documents  # For backwards compatibility if needed
            
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise