import logging
import ollama
import asyncio
from concurrent.futures import ThreadPoolExecutor
from language_utils import get_translation, get_current_language
from langdetect import detect

logger = logging.getLogger(__name__)

class RAGComponent:
    def __init__(self, model_name="llama3.1:latest"):
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=1)
        logger.info(get_translation("rag_component_initialized").format(model=self.model_name))

    def _generate_answer_sync(self, query, context_chunks):
        lang = detect(query)
        try:
            logger.info(get_translation("generating_answer_for_query").format(query=query))
            context = "\n".join([chunk['content'] for chunk in context_chunks])
            prompt = get_translation("rag_prompt").format(context=context, query=query)

            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])

            if response and 'message' in response:
                answer = response['message']['content']
                logger.info(get_translation("answer_generated_successfully"))
                return answer
            else:
                logger.warning(get_translation("no_valid_response_from_model"))
                return get_translation("couldnt_generate_answer")
        except Exception as e:
            logger.error(get_translation("error_generating_answer").format(error=str(e)), exc_info=True)
            return get_translation("error_occurred_while_generating").format(error=str(e))

    async def generate_answer(self, query, context_chunks):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._generate_answer_sync, query, context_chunks)