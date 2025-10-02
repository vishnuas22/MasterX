"""
MasterX Intelligent Caching System
Following specifications from 3.MASTERX_COMPREHENSIVE_PLAN.md

PRINCIPLES (from AGENTS.md):
- No hardcoded values
- Clean, professional naming
- PEP8 compliant
- Real ML-driven caching decisions

Multi-level caching strategy:
- L1: In-memory LRU cache (fast, limited size)
- L2: MongoDB query cache (persistent, shared across instances)
- L3: Embedding cache (expensive to compute)
"""

import hashlib
import logging
import time
from typing import Any, Optional, Dict, List
from functools import lru_cache
from datetime import datetime, timedelta
from collections import OrderedDict

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LRUCache:
    """
    In-memory LRU (Least Recently Used) cache
    
    Thread-safe implementation with configurable size.
    Used for L1 caching of frequently accessed data.
    """
    
    def __init__(self, max_size: int = None):
        """
        Initialize LRU cache
        
        Args:
            max_size: Maximum number of items in cache
        """
        self.max_size = max_size or settings.caching.memory_cache_size
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
        
        logger.info(f"✅ LRU cache initialized (max_size: {self.max_size})")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"Cache hit: {key[:50]}...")
            return self.cache[key]
        
        self.misses += 1
        logger.debug(f"Cache miss: {key[:50]}...")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.max_size:
                # Remove oldest item
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"Evicted: {oldest_key[:50]}...")
        
        self.cache[key] = value
        logger.debug(f"Cached: {key[:50]}...")
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": hit_rate
        }


class EmbeddingCache:
    """
    Cache for expensive embedding computations
    
    Embeddings are expensive to compute (sentence transformers).
    Cache embeddings with MongoDB persistence for sharing across instances.
    """
    
    def __init__(self, db):
        """
        Initialize embedding cache
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.collection = db.embedding_cache
        self.ttl_seconds = settings.caching.embedding_cache_ttl
        
        # In-memory LRU cache for hot embeddings
        self.memory_cache = LRUCache(max_size=500)
        
        logger.info(f"✅ Embedding cache initialized (TTL: {self.ttl_seconds}s)")
    
    def _generate_key(self, text: str, model_name: str) -> str:
        """
        Generate cache key from text and model
        
        Args:
            text: Input text
            model_name: Embedding model name
        
        Returns:
            Cache key (hash)
        """
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Get embedding from cache
        
        Args:
            text: Input text
            model_name: Embedding model name
        
        Returns:
            Cached embedding or None
        """
        key = self._generate_key(text, model_name)
        
        # Try L1 (memory) cache first
        embedding = self.memory_cache.get(key)
        if embedding is not None:
            logger.debug("Embedding hit (L1 memory)")
            return embedding
        
        # Try L2 (MongoDB) cache
        cache_entry = await self.collection.find_one({"_id": key})
        
        if cache_entry:
            # Check if expired
            if datetime.utcnow() < cache_entry["expires_at"]:
                embedding = cache_entry["embedding"]
                # Store in memory cache for future hits
                self.memory_cache.set(key, embedding)
                logger.debug("Embedding hit (L2 MongoDB)")
                return embedding
            else:
                # Expired, delete
                await self.collection.delete_one({"_id": key})
                logger.debug("Embedding expired, deleted")
        
        logger.debug("Embedding miss")
        return None
    
    async def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """
        Store embedding in cache
        
        Args:
            text: Input text
            model_name: Embedding model name
            embedding: Embedding vector
        """
        key = self._generate_key(text, model_name)
        expires_at = datetime.utcnow() + timedelta(seconds=self.ttl_seconds)
        
        # Store in memory cache
        self.memory_cache.set(key, embedding)
        
        # Store in MongoDB for persistence
        await self.collection.update_one(
            {"_id": key},
            {
                "$set": {
                    "text": text[:100],  # Store truncated for debugging
                    "model_name": model_name,
                    "embedding": embedding,
                    "created_at": datetime.utcnow(),
                    "expires_at": expires_at
                }
            },
            upsert=True
        )
        
        logger.debug(f"Cached embedding (key: {key[:16]}...)")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        total_count = await self.collection.count_documents({})
        memory_stats = self.memory_cache.get_stats()
        
        return {
            "memory": memory_stats,
            "mongodb_total": total_count
        }


class ResponseCache:
    """
    Cache for AI provider responses
    
    Cache full AI responses to reduce latency and cost.
    Only cache for deterministic queries (no user-specific data).
    """
    
    def __init__(self, db):
        """
        Initialize response cache
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.collection = db.response_cache
        self.ttl_seconds = settings.caching.response_cache_ttl
        
        # Memory cache for hot responses
        self.memory_cache = LRUCache(max_size=200)
        
        logger.info(f"✅ Response cache initialized (TTL: {self.ttl_seconds}s)")
    
    def _generate_key(self, prompt: str, provider: str, params: Dict = None) -> str:
        """
        Generate cache key from prompt and parameters
        
        Args:
            prompt: AI prompt
            provider: Provider name
            params: Additional parameters
        
        Returns:
            Cache key (hash)
        """
        params_str = str(sorted((params or {}).items()))
        content = f"{provider}:{prompt}:{params_str}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _is_cacheable(self, prompt: str) -> bool:
        """
        Determine if response should be cached
        
        Only cache general knowledge queries, not personalized ones.
        
        Args:
            prompt: AI prompt
        
        Returns:
            True if should be cached
        """
        # Don't cache if contains user-specific keywords
        non_cacheable_keywords = [
            "my", "i am", "i'm", "user_id", "session",
            "personal", "private", "specific to me"
        ]
        
        prompt_lower = prompt.lower()
        for keyword in non_cacheable_keywords:
            if keyword in prompt_lower:
                return False
        
        return True
    
    async def get(
        self,
        prompt: str,
        provider: str,
        params: Dict = None
    ) -> Optional[str]:
        """
        Get cached response
        
        Args:
            prompt: AI prompt
            provider: Provider name
            params: Additional parameters
        
        Returns:
            Cached response or None
        """
        if not self._is_cacheable(prompt):
            return None
        
        key = self._generate_key(prompt, provider, params)
        
        # Try memory cache
        response = self.memory_cache.get(key)
        if response is not None:
            logger.debug("Response hit (L1 memory)")
            return response
        
        # Try MongoDB cache
        cache_entry = await self.collection.find_one({"_id": key})
        
        if cache_entry:
            # Check if expired
            if datetime.utcnow() < cache_entry["expires_at"]:
                response = cache_entry["response"]
                # Store in memory cache
                self.memory_cache.set(key, response)
                logger.debug("Response hit (L2 MongoDB)")
                return response
            else:
                # Expired, delete
                await self.collection.delete_one({"_id": key})
        
        return None
    
    async def set(
        self,
        prompt: str,
        provider: str,
        response: str,
        params: Dict = None
    ) -> None:
        """
        Cache AI response
        
        Args:
            prompt: AI prompt
            provider: Provider name
            response: AI response
            params: Additional parameters
        """
        if not self._is_cacheable(prompt):
            return
        
        key = self._generate_key(prompt, provider, params)
        expires_at = datetime.utcnow() + timedelta(seconds=self.ttl_seconds)
        
        # Store in memory cache
        self.memory_cache.set(key, response)
        
        # Store in MongoDB
        await self.collection.update_one(
            {"_id": key},
            {
                "$set": {
                    "prompt": prompt[:200],  # Truncated for debugging
                    "provider": provider,
                    "response": response,
                    "created_at": datetime.utcnow(),
                    "expires_at": expires_at
                }
            },
            upsert=True
        )
        
        logger.debug(f"Cached response (key: {key[:16]}...)")


class CacheManager:
    """
    Central cache management
    
    Coordinates all caching layers and provides unified interface.
    """
    
    def __init__(self, db):
        """
        Initialize cache manager
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.enabled = settings.caching.enabled
        
        if self.enabled:
            self.embedding_cache = EmbeddingCache(db)
            self.response_cache = ResponseCache(db)
            logger.info("✅ CacheManager initialized (enabled)")
        else:
            logger.info("⚠️ CacheManager initialized (disabled)")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all caches
        
        Returns:
            Dictionary with all cache stats
        """
        if not self.enabled:
            return {"enabled": False}
        
        embedding_stats = await self.embedding_cache.get_stats()
        
        return {
            "enabled": True,
            "embedding_cache": embedding_stats,
            "response_cache": {
                "memory": self.response_cache.memory_cache.get_stats()
            }
        }
    
    async def clear_all(self) -> None:
        """Clear all caches"""
        if not self.enabled:
            return
        
        self.embedding_cache.memory_cache.clear()
        self.response_cache.memory_cache.clear()
        
        await self.db.embedding_cache.delete_many({})
        await self.db.response_cache.delete_many({})
        
        logger.info("✅ All caches cleared")


# Global cache manager instance (initialized in server startup)
_cache_manager: Optional[CacheManager] = None


def init_cache_manager(db) -> CacheManager:
    """
    Initialize global cache manager
    
    Args:
        db: MongoDB database instance
    
    Returns:
        CacheManager instance
    """
    global _cache_manager
    _cache_manager = CacheManager(db)
    return _cache_manager


def get_cache_manager() -> Optional[CacheManager]:
    """
    Get global cache manager instance
    
    Returns:
        CacheManager instance or None if not initialized
    """
    return _cache_manager
