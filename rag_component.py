import logging
import ollama
import asyncio
from concurrent.futures import ThreadPoolExecutor
from language_utils import get_translation
from langdetect import detect
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class RAGComponent:
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the RAG component with a specific model name.
        If no model is specified, it will try to use the first available model.
        """
        self._executor = None
        self.model_name = self._validate_model(model_name)
        logger.info(get_translation("rag_component_initialized").format(model=self.model_name))

    def _validate_model(self, model_name: Optional[str]) -> str:
        """
        Validate that the specified model exists in Ollama.
        If no model is specified or the specified model doesn't exist,
        return the first available model or a default model.
        """
        try:
            available_models = ollama.list()
            model_names = [model['name'] for model in available_models['models']]
            
            if not model_names:
                logger.warning("No models found in Ollama. Using default model.")
                return "deepseek-r1:8b"
            
            if model_name is None:
                logger.info(f"No model specified. Using first available model: {model_names[0]}")
                return model_names[0]
            
            if model_name in model_names:
                logger.info(f"Using specified model: {model_name}")
                return model_name
            
            logger.warning(f"Specified model {model_name} not found. Using first available model: {model_names[0]}")
            return model_names[0]
            
        except Exception as e:
            logger.error(f"Error validating model: {str(e)}")
            return "deepseek-r1:8b"

    @property
    def executor(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="RAG_Worker")
        return self._executor

    async def _generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        try:
            # Detect query language for response
            lang = detect(query)
            logger.info(get_translation("generating_answer_for_query").format(query=query))
            
            # Prepare context from chunks
            context = "\n".join(chunk['content'] for chunk in context_chunks)
            prompt = get_translation("rag_prompt").format(context=context, query=query)

            # Generate response using selected model
            response = await asyncio.to_thread(
                lambda: ollama.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}]
                )
            )

            if response and 'message' in response:
                answer = response['message']['content']
                logger.info(get_translation("answer_generated_successfully"))
                return answer

            logger.warning(get_translation("no_valid_response_from_model"))
            return get_translation("couldnt_generate_answer")

        except Exception as e:
            error_msg = str(e)
            logger.error(
                get_translation("error_generating_answer").format(error=error_msg), 
                exc_info=True
            )
            
            if "model not found" in error_msg.lower():
                return get_translation("model_not_found_error").format(model=self.model_name)
            
            return get_translation("error_occurred_while_generating").format(error=error_msg)

    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate an answer for the given query using the provided context chunks.
        """
        return await self._generate_answer(query, context_chunks)

    def change_model(self, new_model_name: str) -> bool:
        """
        Change the model being used by the RAG component.
        Returns True if the change was successful, False otherwise.
        """
        try:
            validated_model = self._validate_model(new_model_name)
            if validated_model != self.model_name:
                self.model_name = validated_model
                logger.info(f"Model changed to: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error changing model: {str(e)}")
            return False

    def get_current_model(self) -> str:
        """
        Get the name of the currently used model.
        """
        return self.model_name

    def __del__(self):
        if self._executor:
            self._executor.shutdown(wait=False)