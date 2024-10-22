import os
import streamlit as st
from document_processor import process_document, chunk_text
from progress_tracker import ProgressTracker
from embedding_component import EmbeddingComponent
from chroma_db_component import ChromaDBComponent
import asyncio
import logging
import uuid
from language_utils import get_current_language, get_translation

logger = logging.getLogger(__name__)

embedding_component = EmbeddingComponent()
chroma_db = ChromaDBComponent()

async def process_file(file_path, file_name, tracker):
    lang = get_current_language()
    try:
        logger.info(get_translation("processing_file", lang).format(file_name=file_name))
        
        # Traiter le document
        processed_content = process_document(file_path)
        if processed_content is None:
            logger.error(get_translation("failed_to_process", lang).format(file_name=file_name))
            return get_translation('file_processing_failed', lang).format(file_name=file_name)
        
        # Gérer le cas des PDF (générateur) et des autres types de fichiers (chaîne de caractères)
        if isinstance(processed_content, str):
            chunks = chunk_text([processed_content])
        else:  # C'est un générateur (pour les PDF)
            chunks = chunk_text(processed_content)
        
        # Convertir les chunks en liste pour pouvoir les compter et les utiliser
        chunks = list(chunks)
        logger.info(get_translation("text_chunked", lang).format(chunk_count=len(chunks)))
        
        # Générer les embeddings
        embeddings = await embedding_component.generate_embeddings_for_chunks(chunks)
        
        if embeddings and len(embeddings) > 0:
            logger.info(get_translation("embeddings_generated", lang).format(file_name=file_name))
            
            # Ajouter à ChromaDB
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{"source": file_name} for _ in chunks]
            chroma_db.add_documents(ids, embeddings, metadatas, chunks)
            
            tracker.update(file_name)
            return get_translation('file_processed_success', lang).format(file_name=file_name)
        else:
            logger.error(get_translation("no_embeddings_generated", lang).format(file_name=file_name))
            return get_translation('embedding_generation_failed', lang).format(file_name=file_name)
    except Exception as e:
        logger.exception(get_translation("error_processing_file", lang).format(file_name=file_name, error=str(e)))
        return get_translation('file_processing_error', lang).format(file_name=file_name, error=str(e))

def save_file(content, file_name, extension):
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    base_name = os.path.splitext(file_name)[0]
    file_path = os.path.join("uploads", f"{base_name}.{extension}")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(get_translation("saved_processed_content").format(file_path=file_path))

async def file_uploader(uploaded_files):
    lang = get_current_language()
    tracker = ProgressTracker(len(uploaded_files))
    tasks = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_path = os.path.join("uploads", file_name)
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        task = asyncio.create_task(process_file(file_path, file_name, tracker))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, Exception):
            st.error(get_translation("error_occurred", lang).format(error=str(result)))
        else:
            st.write(result)

    tracker.complete()
    return results