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
    SUPPORTED_LANGUAGES,
    TEMPERATURE,
    TOP_P
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

    def process_query(self, query: str) -> Dict[str, Any]:
        """Processes a query and retrieves relevant information."""
        try:
            logger.info(f"Processing query: {query}")
            relevant_chunks = self.retrieval_component.retrieve(query, k=TOP_K_RESULTS)
            response = self._generate_response(query, relevant_chunks)
            return {
                "query": query,
                "response": response,
                "relevant_chunks": relevant_chunks,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "query": query,
                "response": None,
                "relevant_chunks": None,
                "error": f"Error processing query: {str(e)}"
            }

    def _generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generates a response based on retrieved relevant chunks."""
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"Content: {chunk['chunk']}")
            context_parts.append(f"Source: {chunk['metadata'].get('source', 'Unknown')}")
            context_parts.append(f"Relevance: {chunk['similarity_score']:.4f}")
            context_parts.append("---")
        context = "\n".join(context_parts)

        prompt = f"""
        You are an AI assistant specializing in text analysis. Your task is to provide detailed and in-depth answers based on the given context.

            1. Carefully analyze all the provided excerpts.
            2. Synthesize the information from multiple excerpts to form a coherent and detailed answer.
            3. If the context contains information about specific characters, events, or concepts, elaborate on them.
            4. Provide examples or explanations to support your points whenever possible.
            5. If there are multiple perspectives or interpretations in the excerpts, discuss them.
            6. If the excerpts do not contain enough information to fully answer the question, clearly indicate what is known and what remains uncertain.
            7. Organize your response logically, using paragraphs to separate different points or aspects of the answer.
            8. Aim for an answer of at least 150 words, but expand further if the information and question justify it.

            Base your response primarily on the provided context. If you make inferences or connections beyond the given information, clearly state it.
            If the query langauge is not English, please provide the response in the detected language.

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
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
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

