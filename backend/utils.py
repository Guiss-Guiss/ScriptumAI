import os
import hashlib
from typing import List, Dict, Any
from pathlib import Path
import magic
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document
import markdown
from loguru import logger
import chromadb
import time
import sqlite3
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME, MAX_RETRIES, RETRY_DELAY

def read_file(file_path: Path) -> str:
    """
    Read the content of a file based on its type.

    Args:
        file_path (Path): Path to the file.

    Returns:
        str: Content of the file.
    """
    logger.debug(f"Attempting to read file: {file_path}")
    file_type = magic.from_file(str(file_path), mime=True)
    logger.debug(f"File type detected: {magic.from_file(str(file_path), mime=True)}")

    try:
        if file_type == 'text/plain':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_type == 'application/pdf':
            return read_pdf(file_path)
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return read_docx(file_path)
        elif file_type == 'text/html':
            return read_html(file_path)
        elif file_type == 'text/markdown':
            return read_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}", exc_info=True)
        raise

def read_pdf(file_path: Path) -> str:
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return ' '.join(page.extract_text() for page in reader.pages)

def read_docx(file_path: Path) -> str:
    doc = Document(file_path)
    return ' '.join(paragraph.text for paragraph in doc.paragraphs)

def read_html(file_path: Path) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        return soup.get_text()

def read_markdown(file_path: Path) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        md_text = file.read()
        html = markdown.markdown(md_text)
        return BeautifulSoup(html, 'html.parser').get_text()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap

    return chunks

def get_file_metadata(file_path: Path) -> Dict[str, Any]:

    stats = file_path.stat()
    return {
        "filename": file_path.name,
        "file_path": str(file_path),
        "file_type": magic.from_file(str(file_path), mime=True),
        "file_size": stats.st_size,
        "created_at": stats.st_ctime,
        "modified_at": stats.st_mtime,
        "file_hash": get_file_hash(file_path)
    }

def get_file_hash(file_path: Path) -> str:

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def clean_text(text: str) -> str:

    text = ' '.join(text.split())
    text = text.lower()

    return text

def initialize_chroma_client():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIRECTORY))
            collection = chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine", "name": CHROMA_COLLECTION_NAME}  # Add the name to metadata
            )
            logger.info(f"Successfully initialized ChromaDB client and collection: {CHROMA_COLLECTION_NAME}")
            return chroma_client, collection
        except Exception as e:
            retries += 1
            logger.error(f"Error initializing ChromaDB (attempt {retries}/{MAX_RETRIES}): {str(e)}", exc_info=True)
            if retries < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries reached. Unable to initialize ChromaDB.", exc_info=True)
                raise

