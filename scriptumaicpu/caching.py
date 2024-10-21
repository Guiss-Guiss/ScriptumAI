import functools
from cachetools import TTLCache, LRUCache
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, embedding_cache_size=1000, embedding_ttl=3600, query_cache_size=100, query_ttl=300):
        self.embedding_cache = TTLCache(maxsize=embedding_cache_size, ttl=embedding_ttl)
        self.query_cache = TTLCache(maxsize=query_cache_size, ttl=query_ttl)

    def cache_embedding(self, func):
        @functools.wraps(func)
        async def wrapper(text):
            if text in self.embedding_cache:
                logger.info(f"Embedding cache hit for text: {text[:50]}...")
                return self.embedding_cache[text]
            embedding = await func(text)
            self.embedding_cache[text] = embedding
            logger.info(f"Embedding cache miss for text: {text[:50]}... Cached new embedding.")
            return embedding
        return wrapper

    def cache_query(self, func):
        @functools.wraps(func)
        async def wrapper(query, n_results):
            cache_key = (query, n_results)
            if cache_key in self.query_cache:
                logger.info(f"Query cache hit for query: {query}")
                return self.query_cache[cache_key]
            results = await func(query, n_results)
            self.query_cache[cache_key] = results
            logger.info(f"Query cache miss for query: {query}. Cached new results.")
            return results
        return wrapper

cache_manager = CacheManager()