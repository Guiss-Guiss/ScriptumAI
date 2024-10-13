import os
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import magic
import langdetect

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

    def _batch_embed(self, chunks: List[str], device: torch.device) -> torch.Tensor:
        all_embeddings = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            with torch.no_grad():
                # Déplacer les embeddings sur le GPU choisi
                embeddings = self.embedding_component.embed_documents(batch).to(device)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def _batch_ingest(self, chunks, ids, metadatas, device):
        embeddings = self._batch_embed(chunks, device)
        try:
            lang = metadatas[0]["language"]
            collection = self.collections.get(lang, self.collections[SUPPORTED_LANGUAGES[0]])
            collection.add(
                ids=ids,
                embeddings=embeddings.cpu().numpy().tolist(),
                documents=chunks,
                metadatas=metadatas
            )
            logger.debug(f"Successfully ingested batch of {len(chunks)} chunks into {lang} collection")
        except Exception as e:
            logger.error(f"Failed to ingest batch: {e}", exc_info=True)

    def ingest_file(self, file_path: str, device: torch.device = None) -> Dict[str, Any]:
        if device is None:
            device = self.device
        logger.debug(f"Ingesting file {file_path} on device {device}")

        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = magic.from_file(str(file_path), mime=True)
        if file_type not in SUPPORTED_FILE_TYPES:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")

        content = read_file(file_path)
        chunks = chunk_text(content, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        metadata = get_file_metadata(file_path)
        lang = self._detect_language(content)

        ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{**metadata, "chunk_index": i, "language": lang} for i in range(len(chunks))]

        # Ingestion
        self._batch_ingest(chunks, ids, metadatas, device)

        logger.debug(f"Successfully ingested file {file_path}")
        return {**metadata, "language": lang}

    def ingest_directory(self, directory_path: str, num_gpus: int) -> List[Dict[str, Any]]:
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        logger.debug(f"Ingesting files from directory: {directory_path}")

        file_paths = [file_path for file_path in directory_path.rglob("*") if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FILE_TYPES]

        file_groups = [file_paths[i::num_gpus] for i in range(num_gpus)]

        processes = []
        for gpu_id, files in enumerate(file_groups):
            p = mp.Process(target=self._process_files_on_gpu, args=(files, gpu_id))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        logger.debug(f"Successfully ingested files from {directory_path} using {num_gpus} GPUs")

    def _process_files_on_gpu(self, files: List[Path], gpu_id: int):
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else "cpu")
        logger.debug(f"Processing {len(files)} files on GPU {gpu_id} ({device})")

        all_chunks = []
        all_file_metadata = []
        all_ids = []

        for file_path in files:
            try:
                content = read_file(file_path)
                chunks = chunk_text(content, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                all_chunks.extend(chunks)

                metadata = get_file_metadata(file_path)
                lang = self._detect_language(content)
                all_ids.extend([f"{file_path.stem}_{i}" for i in range(len(chunks))])
                all_file_metadata.extend([{**metadata, "chunk_index": i, "language": lang} for i in range(len(chunks))])

                if len(all_chunks) >= BATCH_SIZE:
                    self._batch_ingest(all_chunks, all_ids, all_file_metadata, device)
                    all_chunks.clear()
                    all_ids.clear()
                    all_file_metadata.clear()

            except Exception as e:
                logger.error(f"Error ingesting file {file_path} on GPU {gpu_id}: {e}", exc_info=True)

        if all_chunks:
            self._batch_ingest(all_chunks, all_ids, all_file_metadata, device)

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

