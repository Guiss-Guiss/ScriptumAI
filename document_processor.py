import os
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import markdown
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    try:
        if file_extension == '.txt':
            return process_txt(file_path)
        elif file_extension == '.pdf':
            return process_pdf(file_path)
        elif file_extension == '.docx':
            return process_docx(file_path)
        elif file_extension == '.html':
            return process_html(file_path)
        elif file_extension == '.md':
            return process_md(file_path)
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return None
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
        return None

def process_pdf(file_path, chunk_size=1000):
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            yield page.extract_text() or ""

def process_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def process_docx(file_path):
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def process_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        return soup.get_text()

def process_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        html_content = markdown.markdown(content)
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text()

def chunk_text(text_generator, chunk_size=1000, overlap=100):
    current_chunk = ""
    for text in text_generator:
        current_chunk += text
        while len(current_chunk) >= chunk_size:
            yield current_chunk[:chunk_size]
            current_chunk = current_chunk[chunk_size-overlap:]
    if current_chunk:
        yield current_chunk