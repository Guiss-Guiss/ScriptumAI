import os
from datetime import datetime
import streamlit as st
from document_processor import DocumentProcessor
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
document_processor = DocumentProcessor()

async def process_file(file_path, file_name, tracker):
    lang = get_current_language()
    try:
        logger.info(f"Processing file: {file_name}")
        tracker.update(file_name, chunks_processed=0, total_chunks=0)
        
        processed_content = await document_processor.process_document(file_path)
        if processed_content is None:
            logger.error(f"Failed to process {file_name}")
            return f"Failed to process {file_name}"
        
        chunks = []
        async for chunk in document_processor.chunk_text(processed_content):
            chunks.append(chunk)
            tracker.update(
                file_name,
                chunks_processed=len(chunks),
                total_chunks=max(1, len(chunks))
            )
        
        logger.info(f"Text chunked into {len(chunks)} chunks")
        
        embeddings = []
        for i, chunk in enumerate(chunks, 1):
            embedding = await embedding_component.generate_embedding(chunk)
            embeddings.append(embedding)
            tracker.increment_embeddings()
            tracker.update(
                file_name,
                chunks_processed=i,
                total_chunks=len(chunks)
            )
        
        if embeddings and len(embeddings) > 0:
            logger.info(f"Embeddings generated successfully for {file_name}")
            
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{
                "source": file_name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            } for i in range(len(chunks))]
            
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))
                chroma_db.add_documents(
                    ids[i:batch_end],
                    embeddings[i:batch_end],
                    metadatas[i:batch_end],
                    chunks[i:batch_end]
                )

            processing_time = tracker.get_processing_time(file_name)
            
            st.success(f"""✨ {get_translation('processing_completed', lang)}

        • {get_translation('Document', lang)}: {file_name} 
    • {len(embeddings)} {get_translation('embeddings_in', lang)} {processing_time} {get_translation('seconds', lang)}""")
            
            return ""
            
        else:
            logger.error(f"No embeddings generated for {file_name}")
            return {"success": False, "error": "No embeddings generated"}
            
    except Exception as e:
        logger.exception(f"Error processing file {file_name}: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error removing temporary file {file_path}: {str(e)}")

async def file_uploader(uploaded_files):
    if not uploaded_files:
        return []

    tracker = ProgressTracker(len(uploaded_files))
    tasks = []
    temp_paths = []
    lang = get_current_language()

    try:
        os.makedirs("uploads", exist_ok=True)
        
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_path = os.path.join("uploads", file_name)
            temp_paths.append(file_path)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            task = asyncio.create_task(process_file(file_path, file_name, tracker))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        tracker.complete()
        return results

    except Exception as e:
        logger.error(f"Error during file processing: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
        return []
        
    finally:
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_path}: {str(e)}")

def save_processed_content(content, file_name, extension="txt"):
    try:
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        
        base_name = os.path.splitext(file_name)[0]
        file_path = os.path.join("uploads", f"{base_name}_processed.{extension}")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Saved processed content to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving processed content: {str(e)}")
        return None