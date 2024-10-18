from typing import List, Dict, Any
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import magic
import langdetect
import shutil
import numpy as np

from config import (
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

class IngestComponent:
    def __init__(self, embedding_component: EmbeddingComponent):
        self.embedding_component = embedding_component
        self.device = torch.device(EMBEDDING_DEVICE if torch.cuda.is_available() else "cpu")
        try:
            self.chroma_client, self.collections = self._initialize_collections()
            logger.info(f"Initialized IngestComponent with collections: {[col.name for col in self.collections.values()]} on device: {self.device}")
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

    def _batch_embed(self, chunks: List[str]) -> List[List[float]]:
        logger.debug(f"Starting batch embedding of {len(chunks)} chunks")
        all_embeddings = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            logger.debug(f"Embedding batch {i//BATCH_SIZE + 1} of {len(chunks)//BATCH_SIZE + 1}")
            embeddings = self.embedding_component.embed_documents(batch)
            # Ensure embeddings are on CPU and converted to Python lists
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy().tolist()
            elif isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            all_embeddings.extend(embeddings)
        logger.debug(f"Batch embedding complete. Total embeddings: {len(all_embeddings)}")
        return all_embeddings

    def _batch_ingest(self, chunks, ids, metadatas):
        logger.debug(f"Starting batch ingest of {len(chunks)} chunks")
        embeddings = self._batch_embed(chunks)
        try:
            lang = metadatas[0]["language"]
            collection = self.collections.get(lang, self.collections[SUPPORTED_LANGUAGES[0]])
            logger.debug(f"Adding to collection {collection.name}")
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            logger.info(f"Successfully ingested batch of {len(chunks)} chunks into {lang} collection")
        except Exception as e:
            logger.error(f"Failed to ingest batch: {e}", exc_info=True)
            raise
        finally:
            self.clear_cache()

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        logger.info(f"Ingesting file {file_path}")

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")

            file_type = magic.from_file(str(file_path), mime=True)
            if file_type not in SUPPORTED_FILE_TYPES:
                logger.error(f"Unsupported file type: {file_type}")
                raise ValueError(f"Unsupported file type: {file_type}")

            content = read_file(file_path)
            logger.debug(f"File content read. Length: {len(content)}")
            chunks = chunk_text(content, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            logger.debug(f"Text chunked into {len(chunks)} parts")
            metadata = get_file_metadata(file_path)
            lang = self._detect_language(content)
            logger.debug(f"Detected language: {lang}")

            ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]
            metadatas = [{**metadata, "chunk_index": i, "language": lang} for i in range(len(chunks))]

            logger.debug("Starting batch ingest")
            self._batch_ingest(chunks, ids, metadatas)

            logger.info(f"Successfully ingested file {file_path}")
            return {**metadata, "language": lang, "chunks_count": len(chunks)}
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {str(e)}", exc_info=True)
            raise
        finally:
            self.clear_cache()

    def ingest_directory(self, directory_path: str, num_gpus: int = 1) -> List[Dict[str, Any]]:
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        logger.info(f"Ingesting files from directory: {directory_path}")

        file_paths = [file_path for file_path in directory_path.rglob("*") if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FILE_TYPES]
        logger.info(f"Found {len(file_paths)} files to ingest")

        if num_gpus > 1 and torch.cuda.device_count() > 1:
            # Multi-GPU processing
            with mp.Pool(num_gpus) as pool:
                results = list(tqdm(
                    pool.imap(self.ingest_file, file_paths),
                    total=len(file_paths),
                    desc="Ingesting files",
                    unit="file"
                ))
        else:
            results = []
            for file_path in tqdm(file_paths, desc="Ingesting files", unit="file"):
                try:
                    result = self.ingest_file(str(file_path))
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error ingesting file {file_path}: {e}", exc_info=True)
                    results.append({"status": "failed", "file": str(file_path), "error": str(e)})
                finally:
                    self.clear_cache()

        logger.info(f"Finished ingesting {len(file_paths)} files from {directory_path}")
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
            stats["total_chunks"] = total_documents
            
            logger.info(f"Collection stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise

    def clear_cache(self):
        for pycache_dir in Path(".").rglob("__pycache__"):
            try:
                shutil.rmtree(pycache_dir)
                logger.info(f"Cache {pycache_dir} successfully emptied.")
            except Exception as e:
                logger.error(f"Error deleting cache {pycache_dir}: {e}", exc_info=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared.")