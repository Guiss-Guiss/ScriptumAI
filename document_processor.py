import os
import asyncio
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import markdown
import logging
from typing import AsyncGenerator, Optional, Union, List

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.tracker = None

    async def process_document(self, file_path: str) -> Optional[AsyncGenerator[str, None]]:
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        try:
            if file_extension == '.txt':
                content = await self.process_txt(file_path)
                async def generator():
                    yield content
                return generator()
            elif file_extension == '.pdf':
                return self.process_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                content = await self.process_docx(file_path)
                async def generator():
                    yield content
                return generator()
            elif file_extension == '.html':
                content = await self.process_html(file_path)
                async def generator():
                    yield content
                return generator()
            elif file_extension == '.md':
                content = await self.process_md(file_path)
                async def generator():
                    yield content
                return generator()
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                return None
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            return None

    async def _read_file(self, file_path: str) -> str:
        try:
            def read():
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            return await asyncio.to_thread(read)
        except UnicodeDecodeError:
            def read_with_fallback():
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            return await asyncio.to_thread(read_with_fallback)

    async def process_pdf(self, file_path: str) -> AsyncGenerator[str, None]:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    yield page.extract_text() or ""
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}", exc_info=True)
            yield ""

    async def process_txt(self, file_path: str) -> str:
        return await self._read_file(file_path)

    async def process_docx(self, file_path: str) -> str:
        def _process_docx():
            doc = Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return await asyncio.to_thread(_process_docx)

    async def process_html(self, file_path: str) -> str:
        content = await self._read_file(file_path)
        def parse_html():
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        return await asyncio.to_thread(parse_html)

    async def process_md(self, file_path: str) -> str:
        content = await self._read_file(file_path)
        def parse_md():
            html_content = markdown.markdown(content)
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        return await asyncio.to_thread(parse_md)

    async def chunk_text(
        self, 
        text_generator: AsyncGenerator[str, None], 
        chunk_size: int = 1000, 
        overlap: int = 100
    ) -> AsyncGenerator[str, None]:
        current_chunk = ""
        
        async for text in text_generator:
            current_chunk += text
            while len(current_chunk) >= chunk_size:
                yield current_chunk[:chunk_size]
                current_chunk = current_chunk[chunk_size-overlap:]
        
        if current_chunk:
            yield current_chunk

    async def process_documents(self, file_paths: List[str]) -> bool:
        try:
            total_chunks = 0
            for file_path in file_paths:
                text_gen = await self.process_document(file_path)
                if text_gen:
                    chunks = [chunk async for chunk in self.chunk_text(text_gen)]
                    total_chunks += len(chunks)

            if self.tracker:
                self.tracker = self.tracker.__class__(total_chunks)

            processed_chunks = 0
            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                if self.tracker:
                    self.tracker.update(file_name)
                
                text_gen = await self.process_document(file_path)
                if text_gen:
                    async for chunk in self.chunk_text(text_gen):
                        processed_chunks += 1
                        if self.tracker:
                            self.tracker.update(
                                file_name,
                                chunks_processed=processed_chunks,
                                total_chunks=total_chunks
                            )
                            self.tracker.increment_embeddings()
            
            if self.tracker:
                self.tracker.complete()
            return True

        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}", exc_info=True)
            return False