import logging
import ollama
import asyncio
from concurrent.futures import ThreadPoolExecutor
from language_utils import get_translation
from langdetect import detect
from typing import Optional, Dict, Any, List, Tuple
import httpx
import streamlit as st
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class OllamaConnectionManager:
    """Manages connection to Ollama server"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=5.0)
    
    def check_connection(self) -> bool:
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            return False
    
    def get_models(self) -> List[str]:
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json().get('models', [])
                return [model.get('name') for model in models_data if model.get('name')]
            return []
        except Exception as e:
            logger.error(f"Failed to get models: {str(e)}")
            return []

@contextmanager
def loading_spinner(message: str):
    """Context manager for displaying a loading spinner"""
    with st.spinner(message):
        yield

class StreamingManager:
    """Manages streaming responses from Ollama"""
    
    def __init__(self, placeholder: Any):
        self.placeholder = placeholder
        self.current_response = ""
        self.chunks = []
    
    def update_stream(self, content: str):
        self.current_response += content
        self.chunks.append(content)
        self.placeholder.markdown(self.current_response + "▌")
    
    def finalize(self):
        if self.current_response:
            self.placeholder.markdown(self.current_response)
        return self.current_response

class RAGComponent:
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    GENERATION_TIMEOUT = 120
    CHUNK_SIZE = 1024

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the RAG component with extensive error checking"""
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.connection_manager = OllamaConnectionManager()
        
        # Verify Ollama connection first
        if not self.connection_manager.check_connection():
            raise RuntimeError(get_translation("ollama_connection_error"))
        
        self.model_name = self._validate_model(model_name)
        if not self.model_name:
            raise RuntimeError(get_translation("no_models_available"))
        
        logger.info(get_translation("rag_component_initialized").format(model=self.model_name))

    def _validate_model(self, model_name: Optional[str]) -> Optional[str]:
        """Validate model with retries and detailed error handling"""
        for attempt in range(self.MAX_RETRIES):
            try:
                available_models = self.connection_manager.get_models()
                
                if not available_models:
                    logger.error("No models found in Ollama")
                    time.sleep(self.RETRY_DELAY)
                    continue
                
                if model_name is None:
                    return available_models[0]
                
                if model_name in available_models:
                    return model_name
                
                logger.warning(f"Model {model_name} not found, using {available_models[0]}")
                return available_models[0]
                
            except Exception as e:
                logger.error(f"Error in model validation (attempt {attempt + 1}): {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                
        return None

    async def _make_request_with_retry(self, func, *args, **kwargs) -> Any:
        """Make request with retry logic and proper error handling"""
        last_error = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except (httpx.RemoteProtocolError, ConnectionError) as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self.RETRY_DELAY * (attempt + 1)
                    logger.warning(f"Connection error: {str(e)}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise last_error

    async def _verify_model_availability(self) -> Tuple[bool, Optional[str]]:
        """Verify model availability with proper error handling"""
        try:
            await self._make_request_with_retry(ollama.show, self.model_name)
            return True, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Model {self.model_name} not available: {error_msg}")
            return False, error_msg

    async def _stream_response(self, prompt: str, streaming_manager: StreamingManager) -> Optional[str]:
        """Handle streaming response with proper error handling"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True
            )
            
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    streaming_manager.update_stream(chunk['message']['content'])
            
            return streaming_manager.finalize()
            
        except Exception as e:
            logger.error(f"Error in stream_response: {str(e)}")
            raise

    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer with comprehensive error handling and streaming"""
        if not self.model_name:
            return get_translation("no_models_available")

        try:
            # Verify model availability
            model_available, error_msg = await self._verify_model_availability()
            if not model_available:
                return get_translation("model_not_found_error").format(
                    model=self.model_name,
                    error=error_msg
                )
            
            # Detect language and prepare prompt
            lang = detect(query)
            logger.info(get_translation("generating_answer_for_query").format(query=query))
            
            context = "\n".join(chunk['content'] for chunk in context_chunks)
            prompt = get_translation("rag_prompt").format(context=context, query=query)

            # Create streaming interface
            answer_container = st.empty()
            streaming_manager = StreamingManager(answer_container)

            with loading_spinner(get_translation("generating_response")):
                try:
                    answer = await asyncio.wait_for(
                        self._stream_response(prompt, streaming_manager),
                        timeout=self.GENERATION_TIMEOUT
                    )
                    
                    if answer:
                        logger.info(get_translation("answer_generated_successfully"))
                        return True
                    
                    logger.warning(get_translation("no_valid_response_from_model"))
                    return get_translation("couldnt_generate_answer")

                except asyncio.TimeoutError:
                    logger.error("Response generation timed out")
                    streaming_manager.placeholder.error(get_translation("response_timeout_error"))
                    return get_translation("response_timeout_error")
                
                except (httpx.RemoteProtocolError, ConnectionError) as e:
                    logger.error(f"Connection error during generation: {str(e)}")
                    return get_translation("connection_error_after_retries")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(
                get_translation("error_generating_answer").format(error=error_msg),
                exc_info=True
            )
            return get_translation("error_occurred_while_generating").format(error=error_msg)

    async def change_model(self, new_model_name: str) -> bool:
        """Change model with proper validation and error handling"""
        try:
            # Verify new model exists
            try:
                await self._make_request_with_retry(ollama.show, new_model_name)
            except Exception as e:
                logger.error(f"Model {new_model_name} not available: {str(e)}")
                return False

            self.model_name = new_model_name
            logger.info(get_translation("model_changed_successfully").format(model=new_model_name))
            return True
            
        except Exception as e:
            logger.error(f"Error changing model: {str(e)}")
            return False

    def get_current_model(self) -> Optional[str]:
        """Get current model name"""
        return self.model_name

    def cleanup(self):
        """Cleanup resources"""
        if self._executor:
            self._executor.shutdown(wait=False)
        if hasattr(self, 'connection_manager'):
            if hasattr(self.connection_manager, 'client'):
                self.connection_manager.client.close()

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()
