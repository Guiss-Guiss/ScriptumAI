import ollama
import torch
from typing import List, Dict, Any
from loguru import logger
import langdetect

from config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    TOP_K_RESULTS,
    EMBEDDING_DEVICE,
    SUPPORTED_LANGUAGES
)
from backend.embedding_component import EmbeddingComponent
from backend.retrieval_component import RetrievalComponent

class QueryComponent:
    def __init__(self, embedding_component: EmbeddingComponent, retrieval_component: RetrievalComponent):
        self.embedding_component = embedding_component
        self.retrieval_component = retrieval_component
        self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        self.device = torch.device(EMBEDDING_DEVICE if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized QueryComponent with LLM model: {LLM_MODEL} on device: {self.device}")

    def _detect_language(self, text: str) -> str:
        try:
            return langdetect.detect(text)
        except:
            return SUPPORTED_LANGUAGES[0]

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing query: {query}")
            
            query_lang = self._detect_language(query)
            logger.info(f"Detected query language: {query_lang}")

            relevant_chunks = self.retrieval_component.retrieve(query, k=TOP_K_RESULTS)

            response = self._generate_response(query, relevant_chunks, query_lang)

            return {
                "query": query,
                "response": response,
                "relevant_chunks": relevant_chunks,
                "query_language": query_lang,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "query": query,
                "response": None,
                "relevant_chunks": None,
                "query_language": None,
                "error": f"Error processing query: {str(e)}"
            }

    def _generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]], query_lang: str) -> str:
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"Content: {chunk['chunk']}")
            context_parts.append(f"Source: {chunk['metadata'].get('source', 'Unknown')}")
            context_parts.append(f"Relevance: {chunk['similarity_score']:.4f}")
            context_parts.append(f"Language: {chunk['language']}")
            context_parts.append("---")
        context = "\n".join(context_parts)

        prompt = f"""
        You are a helpful AI assistant. Use the following context to answer the user's question.
        If you cannot answer the question based on the context, say so and produce an answer that is as helpful as possible.
        The user's question is in {query_lang}. Please respond in the same language.

        Context:
        {context}

        User Question: {query}

        Assistant:
        """

        logger.debug(f"Generated prompt: {prompt}")

        try:
            logger.info(f"Sending request to Ollama API with model: {LLM_MODEL}")
            response = self.ollama_client.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={
                    "max_tokens": LLM_MAX_TOKENS,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            logger.debug(f"Received response from Ollama API: {response}")
            return response['response']
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while trying to generate a response."

    @torch.no_grad()
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        query_lang = self._detect_language(query)
        logger.info(f"Performing semantic search for query in {query_lang}")
        query_embedding = self.embedding_component.embed_query(query).to(self.device)
        return self.retrieval_component.retrieve(query, n_results)

