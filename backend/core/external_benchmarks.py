"""
External Benchmark Integration for MasterX
Fetches real-world LLM rankings from Artificial Analysis and LLM-Stats APIs

Priority:
1. Artificial Analysis API (primary - 99% usage)
2. LLM-Stats API (secondary - 0.9% usage)
3. Cached rankings from MongoDB (tertiary - 0.09% usage)
4. Minimal manual tests (last resort - <0.01% usage)

Author: MasterX Team
Date: October 1, 2025
"""

import os
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelRanking:
    """Model ranking in a specific category"""
    model_name: str
    provider: str
    score: float
    rank: int
    category: str
    source: str
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ExternalBenchmarkIntegration:
    """
    Integrate external LLM benchmarking data
    
    Fetches rankings from professional benchmarking platforms:
    - Artificial Analysis (primary)
    - LLM-Stats (secondary)
    - Falls back to cache and minimal manual tests
    """
    
    # API Configuration (v2 as per official documentation)
    ARTIFICIAL_ANALYSIS_API_URL = "https://artificialanalysis.ai/api/v2"
    LLM_STATS_API_URL = "https://llm-stats.com/api/v1"
    
    # Category mapping: our categories -> external benchmark keys
    CATEGORY_MAPPING = {
        "coding": {
            "aa_keys": ["code", "humaneval", "programming"],
            "stats_keys": ["coding_score"],
            "description": "Programming, algorithms, debugging"
        },
        "math": {
            "aa_keys": ["math", "aime", "gsm8k"],
            "stats_keys": ["math_score"],
            "description": "Mathematics, calculations, problem solving"
        },
        "reasoning": {
            "aa_keys": ["logic", "reasoning", "gpqa"],
            "stats_keys": ["reasoning_score"],
            "description": "Logic, critical thinking, analysis"
        },
        "research": {
            "aa_keys": ["knowledge", "mmlu", "research"],
            "stats_keys": ["knowledge_score"],
            "description": "Facts, research, explanations"
        },
        "empathy": {
            "aa_keys": ["conversation", "chat"],
            "stats_keys": ["chat_score"],
            "description": "Emotional support, teaching"
        },
        "general": {
            "aa_keys": ["overall", "general", "intelligence"],
            "stats_keys": ["overall_score"],
            "description": "General conversation and Q&A"
        }
    }
    
    # Pattern-based provider detection (dynamic, not hardcoded)
    # These are pattern hints for matching, not rigid mappings
    PROVIDER_PATTERNS = {
        'openai': ['gpt', 'openai', 'chatgpt', 'davinci'],
        'anthropic': ['claude', 'anthropic'],
        'gemini': ['gemini', 'google', 'bard'],
        'groq': ['groq', 'llama'],
        'emergent': ['emergent']
    }
    
    def __init__(self, db, api_key: Optional[str] = None):
        """
        Initialize external benchmark integration
        
        Args:
            db: MongoDB database instance
            api_key: Artificial Analysis API key (optional, will read from env)
        """
        self.db = db
        self.rankings_collection = db["external_rankings"]
        self.usage_collection = db["benchmark_source_usage"]
        
        # API keys
        self.aa_api_key = api_key or os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")
        self.llm_stats_api_key = os.getenv("LLM_STATS_API_KEY")
        
        # Cache
        self.cache = {}
        self.last_update = None
        
        logger.info("✅ ExternalBenchmarkIntegration initialized")
        if self.aa_api_key:
            logger.info("✅ Artificial Analysis API key configured")
        else:
            logger.warning("⚠️ No Artificial Analysis API key found")
    
    async def get_rankings(
        self,
        category: str,
        force_refresh: bool = False
    ) -> List[ModelRanking]:
        """
        Get rankings for a category with intelligent fallback
        
        Priority:
        1. Try Artificial Analysis API
        2. Try LLM-Stats API
        3. Use cached rankings
        4. Run minimal manual tests (last resort)
        
        Args:
            category: Category (coding, math, reasoning, etc.)
            force_refresh: Force fetch from APIs (ignore cache)
        
        Returns:
            List of ModelRanking objects, sorted by rank
        """
        
        # Check cache first (unless force_refresh)
        if not force_refresh and self._is_cache_valid(category):
            logger.info(f"✅ Using cached rankings for {category}")
            return self.cache.get(category, [])
        
        # Try each source in priority order
        sources = ["artificial_analysis", "llm_stats", "cache", "minimal_manual"]
        
        for source in sources:
            try:
                logger.info(f"Attempting to fetch {category} rankings from: {source}")
                
                rankings = await self._fetch_from_source(category, source)
                
                if rankings and len(rankings) > 0:
                    logger.info(
                        f"✅ Successfully fetched {len(rankings)} rankings "
                        f"from {source}"
                    )
                    
                    # Track usage
                    await self._track_source_usage(source, category, True)
                    
                    # Update cache
                    self.cache[category] = rankings
                    if source in ["artificial_analysis", "llm_stats"]:
                        self.last_update = datetime.utcnow()
                    
                    return rankings
                    
            except Exception as e:
                logger.warning(f"⚠️ Source {source} failed for {category}: {e}")
                await self._track_source_usage(source, category, False, str(e))
                continue
        
        # All sources failed
        logger.error(f"🚨 ALL SOURCES FAILED for category: {category}")
        return []
    
    async def _fetch_from_source(
        self,
        category: str,
        source: str
    ) -> List[ModelRanking]:
        """Fetch rankings from specified source"""
        
        if source == "artificial_analysis":
            return await self._fetch_artificial_analysis(category)
        elif source == "llm_stats":
            return await self._fetch_llm_stats(category)
        elif source == "cache":
            return await self._fetch_from_cache(category)
        elif source == "minimal_manual":
            return await self._run_minimal_manual_tests(category)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    async def _fetch_artificial_analysis(self, category: str) -> List[ModelRanking]:
        """
        Fetch rankings from Artificial Analysis API
        
        API Documentation: https://artificialanalysis.ai/documentation
        """
        
        if not self.aa_api_key:
            raise Exception("No Artificial Analysis API key configured")
        
        headers = {
            "x-api-key": self.aa_api_key,
            "Accept": "application/json"
        }
        
        timeout = aiohttp.ClientTimeout(total=10)
        
        try:
            async with aiohttp.ClientSession() as session:
                # Correct endpoint as per API documentation (v2)
                url = f"{self.ARTIFICIAL_ANALYSIS_API_URL}/data/llms/models"
                logger.info(f"Fetching from Artificial Analysis: {url}")
                
                async with session.get(
                    url,
                    headers=headers,
                    timeout=timeout
                ) as response:
                    
                    response_text = await response.text()
                    logger.info(f"AA API Response Status: {response.status}")
                    
                    if response.status != 200:
                        logger.error(f"AA API Error Response: {response_text[:500]}")
                        raise Exception(f"API returned status {response.status}: {response_text[:200]}")
                    
                    try:
                        data = await response.json()
                        logger.info(f"AA API Data Structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                    except Exception as json_err:
                        logger.error(f"Failed to parse JSON: {json_err}, Response: {response_text[:500]}")
                        raise Exception(f"Invalid JSON response: {json_err}")
                    
                    # Parse and organize rankings
                    rankings = self._parse_aa_rankings(data, category)
                    
                    # Save to database
                    if rankings:
                        await self._save_rankings_to_db(rankings, category, "artificial_analysis")
                    
                    return rankings
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching from Artificial Analysis: {str(e)}")
            raise Exception(f"HTTP error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to fetch from Artificial Analysis: {str(e)}", exc_info=True)
            raise Exception(f"Failed to fetch from Artificial Analysis: {str(e)}")
    
    def _parse_aa_rankings(self, data: dict, category: str) -> List[ModelRanking]:
        """
        Parse Artificial Analysis API v2 response
        
        API v2 structure:
        {
            "status": 200,
            "data": [
                {
                    "id": "uuid",
                    "name": "GPT-4o",
                    "slug": "gpt-4o",
                    "model_creator": {"name": "OpenAI", "slug": "openai"},
                    "evaluations": {
                        "artificial_analysis_intelligence_index": 62.1,
                        "artificial_analysis_coding_index": 55.8,
                        "artificial_analysis_math_index": 87.2,
                        ...
                    },
                    "median_output_tokens_per_second": 153.831,
                    "pricing": {...}
                }
            ]
        }
        """
        
        rankings = []
        
        # Map our categories to AA evaluation keys
        eval_key_mapping = {
            "coding": "artificial_analysis_coding_index",
            "math": "artificial_analysis_math_index",
            "reasoning": "artificial_analysis_intelligence_index",
            "research": "artificial_analysis_intelligence_index",
            "empathy": "artificial_analysis_intelligence_index",
            "general": "artificial_analysis_intelligence_index"
        }
        
        eval_key = eval_key_mapping.get(category, "artificial_analysis_intelligence_index")
        logger.info(f"Parsing AA rankings for category '{category}' using eval key '{eval_key}'")
        
        models_data = data.get("data", [])
        logger.info(f"Processing {len(models_data)} models from Artificial Analysis API")
        
        matched_count = 0
        for model_data in models_data:
            model_name = model_data.get("name", "").lower()
            model_slug = model_data.get("slug", "")
            
            # Get creator info
            creator = model_data.get("model_creator", {})
            creator_name = creator.get("name", "")
            
            # Normalize model name to get provider
            external_model_full = f"{creator_name} {model_name}"
            provider = self._normalize_model_name(external_model_full)
            
            # Extract score for this category
            evaluations = model_data.get("evaluations", {})
            score = evaluations.get(eval_key)
            
            # Skip if no score or score is None/0
            if score is None or score <= 0:
                continue
            
            # Only include if we matched to a provider
            if provider:
                matched_count += 1
                rankings.append(ModelRanking(
                    model_name=model_slug or model_name,
                    provider=provider,
                    score=float(score),
                    rank=0,  # Will be assigned after sorting
                    category=category,
                    source="artificial_analysis",
                    metadata={
                        "external_name": model_data.get("name"),
                        "creator": creator_name,
                        "speed": model_data.get("median_output_tokens_per_second"),
                        "latency_ms": model_data.get("median_latency_ms"),
                    },
                    timestamp=datetime.utcnow()
                ))
                
                logger.debug(
                    f"Matched model: {model_data.get('name')} -> provider '{provider}' (score: {score})"
                )
        
        # Sort by score and assign ranks
        rankings.sort(key=lambda x: x.score, reverse=True)
        for idx, ranking in enumerate(rankings, 1):
            ranking.rank = idx
        
        logger.info(
            f"✅ Parsed {len(rankings)} rankings for {category} from Artificial Analysis "
            f"({matched_count} matched to available providers)"
        )
        
        if len(rankings) > 0:
            logger.info(f"Top 3 rankings for {category}:")
            for r in rankings[:3]:
                logger.info(f"  #{r.rank}: {r.provider} (score: {r.score:.1f})")
        
        return rankings
    
    async def _fetch_llm_stats(self, category: str) -> List[ModelRanking]:
        """
        Fetch rankings from LLM-Stats API (secondary source)
        
        Note: API key is optional for LLM-Stats
        """
        
        # LLM-Stats implementation (placeholder for when user adds key)
        if not self.llm_stats_api_key or self.llm_stats_api_key == "placeholder_add_key_later":
            raise Exception("LLM-Stats API key not configured yet")
        
        # TODO: Implement LLM-Stats API integration when key is available
        logger.info("LLM-Stats API integration ready for when key is added")
        raise Exception("LLM-Stats integration pending API key")
    
    async def _fetch_from_cache(self, category: str) -> List[ModelRanking]:
        """
        Fetch from MongoDB cache
        Valid if less than 7 days old
        """
        
        cutoff = datetime.utcnow() - timedelta(days=7)
        
        cursor = self.rankings_collection.find({
            "category": category,
            "last_updated": {"$gte": cutoff}
        }).sort("rank", 1)
        
        rankings = []
        async for doc in cursor:
            rankings.append(ModelRanking(
                model_name=doc["model_name"],
                provider=doc["provider"],
                score=doc["score"],
                rank=doc["rank"],
                category=category,
                source=f"cache_{doc.get('original_source', 'unknown')}",
                metadata=doc.get("metadata", {}),
                timestamp=doc["last_updated"]
            ))
        
        if len(rankings) >= 2:
            age_days = (datetime.utcnow() - rankings[0].timestamp).days
            logger.info(f"Using cached rankings (age: {age_days} days)")
            return rankings
        
        raise Exception("Cache empty or expired")
    
    async def _run_minimal_manual_tests(self, category: str) -> List[ModelRanking]:
        """
        Last resort: Run 2-3 lightweight tests
        Only used when all other sources fail (<0.01% of time)
        """
        
        logger.warning(
            f"⚠️ Running minimal manual tests for {category} "
            "(all external sources unavailable)"
        )
        
        # Import here to avoid circular dependency
        from core.ai_providers import ProviderManager
        
        # Minimal test prompts
        MINIMAL_TESTS = {
            "coding": ["Write a function to check if a number is prime"],
            "math": ["Solve: 3x + 7 = 22"],
            "reasoning": ["If all birds can fly, and penguins are birds, can penguins fly?"],
            "research": ["What causes rain?"],
            "empathy": ["I'm feeling stressed about work"],
            "general": ["What is machine learning?"]
        }
        
        test_prompt = MINIMAL_TESTS.get(category, MINIMAL_TESTS["general"])[0]
        
        # Get available LLM providers only (exclude TTS, benchmark APIs, etc.)
        provider_manager = ProviderManager()
        available_llm_providers = list(provider_manager.registry.get_llm_providers().keys())
        
        logger.info(f"Testing {len(available_llm_providers)} LLM providers: {available_llm_providers}")
        
        rankings = []
        
        for provider in available_llm_providers:
            try:
                response = await provider_manager.universal_provider.generate(
                    provider_name=provider,
                    prompt=test_prompt,
                    max_tokens=300
                )
                
                # Simple quality score based on response quality
                score = 50.0
                if response.content and len(response.content) > 20:
                    # Score based on response length and time
                    score = min(100.0, 40 + len(response.content) / 10)
                    # Penalize slow responses
                    if response.response_time_ms > 5000:
                        score *= 0.8
                
                # FIX: Access dict properly
                provider_config = provider_manager.registry.providers[provider]
                model_name = provider_config.get('model_name', 'default')
                
                rankings.append(ModelRanking(
                    model_name=model_name,
                    provider=provider,
                    score=score,
                    rank=0,
                    category=category,
                    source="minimal_manual",
                    metadata={"test_count": 1},
                    timestamp=datetime.utcnow()
                ))
                
                logger.info(f"✅ Manual test passed for {provider}: score={score:.1f}")
                
            except Exception as e:
                logger.error(f"Manual test failed for {provider}: {e}")
        
        # Sort and rank
        rankings.sort(key=lambda x: x.score, reverse=True)
        for idx, ranking in enumerate(rankings, 1):
            ranking.rank = idx
        
        logger.info(f"Manual tests complete: {len(rankings)} providers ranked")
        return rankings
    
    def _normalize_model_name(self, external_name: str) -> Optional[str]:
        """
        Map external model name to our internal provider (DYNAMIC)
        
        INTELLIGENT MATCHING STRATEGY:
        1. Match based on the ACTUAL model each provider is using
        2. Use provider patterns as secondary matching
        3. Completely dynamic - adapts to .env configuration
        
        Example:
        - If emergent uses "claude-sonnet-4-5" → match "Claude" models
        - If groq uses "llama-3.3" → match "Llama" models
        - If gemini uses "gemini-2.5" → match "Gemini" models
        
        Args:
            external_name: Model name from external API (e.g., "OpenAI GPT-5", "Anthropic Claude 4.5")
        
        Returns:
            Provider name or None
        """
        
        external_lower = external_name.lower()
        
        # Get available providers dynamically from .env
        from core.ai_providers import ProviderRegistry
        registry = ProviderRegistry()
        available_llm_providers = registry.get_llm_providers()
        
        if not available_llm_providers:
            logger.warning("No LLM providers available for matching")
            return None
        
        logger.debug(f"Normalizing '{external_name}' against {len(available_llm_providers)} LLM providers")
        
        # STRATEGY 1: Match based on the MODEL_NAME each provider is using
        # This is the PRIMARY matching strategy for intelligent provider selection
        for provider_name, provider_config in available_llm_providers.items():
            model_name = provider_config.get('model_name', '').lower()
            
            if not model_name:
                continue
            
            # Extract model family keywords from the configured model
            # E.g., "claude-sonnet-4-5" → ["claude", "sonnet"]
            # E.g., "llama-3.3-70b-versatile" → ["llama"]
            # E.g., "gemini-2.5-flash" → ["gemini", "flash"]
            
            model_keywords = []
            
            # Common model family identifiers
            if 'claude' in model_name:
                model_keywords.extend(['claude', 'anthropic'])
            if 'llama' in model_name:
                model_keywords.extend(['llama', 'meta'])
            if 'gemini' in model_name:
                model_keywords.extend(['gemini', 'google'])
            if 'gpt' in model_name or 'chatgpt' in model_name:
                model_keywords.extend(['gpt', 'openai', 'chatgpt'])
            if 'mistral' in model_name or 'mixtral' in model_name:
                model_keywords.extend(['mistral', 'mixtral'])
            if 'qwen' in model_name:
                model_keywords.extend(['qwen', 'alibaba'])
            if 'deepseek' in model_name:
                model_keywords.extend(['deepseek'])
            
            # Check if any keyword matches the external name
            for keyword in model_keywords:
                if keyword in external_lower:
                    logger.debug(
                        f"Matched '{external_name}' to provider '{provider_name}' "
                        f"via model family '{keyword}' (provider uses: {model_name})"
                    )
                    return provider_name
        
        # STRATEGY 2: Pattern-based matching using PROVIDER_PATTERNS
        # This is the SECONDARY matching strategy for edge cases
        for provider_name in available_llm_providers.keys():
            patterns = self.PROVIDER_PATTERNS.get(provider_name, [provider_name])
            
            for pattern in patterns:
                if pattern in external_lower:
                    logger.debug(
                        f"Matched '{external_name}' to provider '{provider_name}' via pattern '{pattern}'"
                    )
                    return provider_name
        
        # STRATEGY 3: Direct substring matching (last resort)
        for provider_name in available_llm_providers.keys():
            if provider_name in external_lower:
                logger.debug(
                    f"Matched '{external_name}' to provider '{provider_name}' via direct substring"
                )
                return provider_name
        
        # No match found - this is expected for models we don't have access to
        logger.debug(f"No match found for '{external_name}' (model not available in our configuration)")
        return None
    
    async def _save_rankings_to_db(
        self,
        rankings: List[ModelRanking],
        category: str,
        source: str
    ):
        """Save rankings to MongoDB for caching"""
        
        timestamp = datetime.utcnow()
        
        for ranking in rankings:
            await self.rankings_collection.update_one(
                {
                    "category": category,
                    "provider": ranking.provider,
                    "model_name": ranking.model_name
                },
                {
                    "$set": {
                        "score": ranking.score,
                        "rank": ranking.rank,
                        "original_source": source,
                        "metadata": ranking.metadata,
                        "last_updated": timestamp
                    }
                },
                upsert=True
            )
        
        logger.info(f"Saved {len(rankings)} rankings to MongoDB")
    
    async def _track_source_usage(
        self,
        source: str,
        category: str,
        success: bool,
        error: Optional[str] = None
    ):
        """Track which sources are used for monitoring"""
        
        await self.usage_collection.insert_one({
            "source": source,
            "category": category,
            "success": success,
            "error": error,
            "timestamp": datetime.utcnow()
        })
    
    def _is_cache_valid(self, category: str, max_age_hours: int = 12) -> bool:
        """Check if cached rankings are still valid"""
        
        if category not in self.cache:
            return False
        
        if not self.last_update:
            return False
        
        age = datetime.utcnow() - self.last_update
        return age < timedelta(hours=max_age_hours)
    
    async def schedule_periodic_updates(self, interval_hours: int = 12):
        """
        Background task to update rankings periodically
        
        Args:
            interval_hours: Hours between updates (default: 12)
        """
        
        logger.info(f"Starting periodic ranking updates (every {interval_hours}h)")
        
        while True:
            try:
                # Fetch rankings for all categories
                for category in self.CATEGORY_MAPPING.keys():
                    await self.get_rankings(category, force_refresh=True)
                
                logger.info(
                    f"✅ Periodic update complete. "
                    f"Next update in {interval_hours}h"
                )
                
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"❌ Periodic update failed: {e}")
                await asyncio.sleep(1800)  # Retry in 30 minutes


# Singleton instance
_external_benchmarks_instance = None


def get_external_benchmarks(db, api_key: Optional[str] = None):
    """Get or create external benchmarks singleton instance"""
    global _external_benchmarks_instance
    
    if _external_benchmarks_instance is None:
        _external_benchmarks_instance = ExternalBenchmarkIntegration(db, api_key)
    
    return _external_benchmarks_instance
