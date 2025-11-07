"""
RAG (Retrieval-Augmented Generation) Engine for MasterX
Perplexity-Inspired Real-Time Knowledge Integration

PURPOSE:
- Real-time web search for current knowledge
- Emotion-aware source selection
- Difficulty-aware content filtering
- Citation tracking and transparency
- Graceful fallback handling

PRINCIPLES (AGENTS.md):
- Clean, production-ready code
- Comprehensive error handling
- Async/await patterns
- Type hints and docstrings
- No hardcoded values
"""

import os
import logging
import time
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import aiohttp
from pydantic import BaseModel, Field

from core.models import EmotionState, LearningReadiness
from utils.errors import MasterXError

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class SearchProvider(str, Enum):
    """Available search providers"""
    SERPER = "serper"
    BRAVE = "brave"
    FALLBACK = "fallback"


class SourceType(str, Enum):
    """Types of web sources"""
    EDUCATIONAL = "educational"
    NEWS = "news"
    ACADEMIC = "academic"
    DOCUMENTATION = "documentation"
    BLOG = "blog"
    FORUM = "forum"
    VIDEO = "video"
    GENERAL = "general"


class SearchResult(BaseModel):
    """Single search result"""
    title: str
    url: str
    snippet: str
    source_type: SourceType = SourceType.GENERAL
    credibility_score: float = Field(ge=0.0, le=1.0, default=0.5)
    difficulty_estimate: float = Field(ge=0.0, le=1.0, default=0.5)
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    published_date: Optional[datetime] = None
    
    def to_citation(self, index: int) -> str:
        """Format as citation for display"""
        return f"[{index}] {self.title} - {self.url}"


class RAGContext(BaseModel):
    """RAG-augmented context for AI response"""
    query: str
    sources: List[SearchResult]
    provider_used: SearchProvider
    search_time_ms: float
    total_results: int
    filtered_results: int
    context_text: str  # Formatted text for AI prompt
    citations: List[str]  # Formatted citations


# ============================================================================
# WEB SEARCH ENGINE
# ============================================================================

class WebSearchEngine:
    """
    Unified web search interface with multiple providers
    
    Providers:
    1. Serper API (Google search, primary)
    2. Brave Search (privacy-focused, fallback)
    3. Fallback (graceful degradation)
    """
    
    def __init__(
        self,
        serper_api_key: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        timeout: int = 10
    ):
        """
        Initialize web search engine
        
        Args:
            serper_api_key: Serper API key (from env: SERPER_API_KEY)
            brave_api_key: Brave Search API key (from env: BRAVE_API_KEY)
            timeout: Request timeout in seconds
        """
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY")
        self.timeout = timeout
        
        # Track which providers are available
        self.available_providers = []
        if self.serper_api_key:
            self.available_providers.append(SearchProvider.SERPER)
            logger.info("✅ Serper API configured")
        if self.brave_api_key:
            self.available_providers.append(SearchProvider.BRAVE)
            logger.info("✅ Brave Search API configured")
        
        if not self.available_providers:
            logger.warning("⚠️  No search API keys configured - RAG will use fallback mode")
        
        logger.info(f"WebSearchEngine initialized (providers: {len(self.available_providers)})")
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "search"
    ) -> Tuple[List[Dict[str, Any]], SearchProvider]:
        """
        Search using available providers with automatic fallback
        
        Args:
            query: Search query
            max_results: Maximum number of results
            search_type: Type of search (search, news, etc.)
        
        Returns:
            Tuple of (results list, provider used)
        """
        logger.info(f"🔍 Searching: '{query[:50]}...' (max_results={max_results})")
        
        # Try providers in priority order
        for provider in self.available_providers:
            try:
                if provider == SearchProvider.SERPER:
                    results = await self._search_serper(query, max_results, search_type)
                elif provider == SearchProvider.BRAVE:
                    results = await self._search_brave(query, max_results)
                else:
                    continue
                
                logger.info(f"✅ Search successful via {provider.value}: {len(results)} results")
                return results, provider
            
            except Exception as e:
                logger.warning(f"⚠️  {provider.value} search failed: {e}")
                continue
        
        # All providers failed - use fallback
        logger.warning("⚠️  All search providers failed, using fallback")
        fallback_results = self._search_fallback(query)
        return fallback_results, SearchProvider.FALLBACK
    
    async def _search_serper(
        self,
        query: str,
        max_results: int,
        search_type: str
    ) -> List[Dict[str, Any]]:
        """
        Search using Serper API (Google search results)
        
        API Docs: https://serper.dev/docs
        """
        url = "https://google.serper.dev/search"
        
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": max_results,
            "type": search_type
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Serper API error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Parse Serper response format
                results = []
                
                # Organic results
                if "organic" in data:
                    for item in data["organic"][:max_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                            "position": item.get("position", 0),
                            "date": item.get("date")
                        })
                
                # Knowledge graph (if available)
                if "knowledgeGraph" in data:
                    kg = data["knowledgeGraph"]
                    if kg.get("description"):
                        results.insert(0, {
                            "title": kg.get("title", "Knowledge Graph"),
                            "url": kg.get("website", ""),
                            "snippet": kg.get("description", ""),
                            "position": -1,
                            "type": "knowledge_graph"
                        })
                
                return results
    
    async def _search_brave(
        self,
        query: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Search using Brave Search API
        
        API Docs: https://brave.com/search/api/
        """
        url = "https://api.search.brave.com/res/v1/web/search"
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key
        }
        
        params = {
            "q": query,
            "count": max_results
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Brave Search API error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Parse Brave response format
                results = []
                
                if "web" in data and "results" in data["web"]:
                    for idx, item in enumerate(data["web"]["results"][:max_results]):
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("description", ""),
                            "position": idx + 1,
                            "age": item.get("age")
                        })
                
                return results
    
    def _search_fallback(self, query: str) -> List[Dict[str, Any]]:
        """
        Fallback search when no APIs available
        Returns educational placeholder guiding user about the topic
        """
        return [{
            "title": f"Educational Context: {query}",
            "url": "https://masterx.ai/learn",
            "snippet": (
                f"I'll help you learn about '{query}' using my training knowledge. "
                "Note: Real-time web search is currently unavailable, so I'm using "
                "my built-in knowledge base. This information may not include the "
                "very latest updates."
            ),
            "position": 0,
            "type": "fallback"
        }]


# ============================================================================
# SOURCE ANALYZER
# ============================================================================

class SourceAnalyzer:
    """
    Analyze and score search results for credibility and difficulty
    """
    
    # Domain credibility scores (configurable via database in production)
    TRUSTED_DOMAINS = {
        # Educational
        ".edu": 0.95,
        "khanacademy.org": 0.95,
        "coursera.org": 0.90,
        "edx.org": 0.90,
        "udacity.com": 0.85,
        
        # Academic
        "scholar.google.com": 0.95,
        "arxiv.org": 0.90,
        "nature.com": 0.95,
        "science.org": 0.95,
        
        # Documentation
        "docs.python.org": 0.95,
        "developer.mozilla.org": 0.95,
        "stackoverflow.com": 0.85,
        
        # News (reputable)
        "bbc.com": 0.85,
        "reuters.com": 0.90,
        "apnews.com": 0.90,
        
        # General
        "wikipedia.org": 0.80,
    }
    
    def __init__(self):
        """Initialize source analyzer"""
        logger.info("SourceAnalyzer initialized")
    
    def analyze(self, result: Dict[str, Any]) -> SearchResult:
        """
        Analyze search result and create SearchResult object
        
        Args:
            result: Raw search result from provider
        
        Returns:
            SearchResult with scoring
        """
        title = result.get("title", "")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        
        # Estimate credibility
        credibility = self._estimate_credibility(url)
        
        # Classify source type
        source_type = self._classify_source_type(url, title, snippet)
        
        # Estimate difficulty
        difficulty = self._estimate_difficulty(title, snippet)
        
        # Calculate relevance (based on position for now)
        position = result.get("position", 10)
        relevance = max(0.0, 1.0 - (position / 20.0))
        
        return SearchResult(
            title=title,
            url=url,
            snippet=snippet,
            source_type=source_type,
            credibility_score=credibility,
            difficulty_estimate=difficulty,
            relevance_score=relevance
        )
    
    def _estimate_credibility(self, url: str) -> float:
        """
        Estimate source credibility based on domain
        
        Args:
            url: Source URL
        
        Returns:
            Credibility score (0.0 - 1.0)
        """
        url_lower = url.lower()
        
        # Check trusted domains
        for domain, score in self.TRUSTED_DOMAINS.items():
            if domain in url_lower:
                return score
        
        # Check for educational TLD
        if ".edu" in url_lower:
            return 0.90
        
        # Check for government
        if ".gov" in url_lower:
            return 0.85
        
        # Check for HTTPS (basic security)
        if url.startswith("https://"):
            return 0.60
        
        # Default for unknown sources
        return 0.50
    
    def _classify_source_type(
        self,
        url: str,
        title: str,
        snippet: str
    ) -> SourceType:
        """
        Classify source type based on URL and content
        
        Args:
            url: Source URL
            title: Result title
            snippet: Result snippet
        
        Returns:
            SourceType classification
        """
        url_lower = url.lower()
        content_lower = f"{title} {snippet}".lower()
        
        # Educational
        if any(domain in url_lower for domain in [".edu", "khan", "coursera", "udacity"]):
            return SourceType.EDUCATIONAL
        
        # Academic
        if any(term in url_lower for term in ["scholar", "arxiv", "journal", "paper"]):
            return SourceType.ACADEMIC
        
        # Documentation
        if any(term in url_lower for term in ["docs.", "documentation", "api"]):
            return SourceType.DOCUMENTATION
        
        # News
        if any(term in url_lower for term in ["news", "bbc", "reuters", "cnn"]):
            return SourceType.NEWS
        
        # Forum
        if any(term in url_lower for term in ["stackoverflow", "reddit", "forum"]):
            return SourceType.FORUM
        
        # Video
        if any(term in url_lower for term in ["youtube", "vimeo", "video"]):
            return SourceType.VIDEO
        
        # Blog
        if any(term in url_lower for term in ["blog", "medium", "substack"]):
            return SourceType.BLOG
        
        return SourceType.GENERAL
    
    def _estimate_difficulty(self, title: str, snippet: str) -> float:
        """
        Estimate content difficulty based on text analysis
        
        Args:
            title: Result title
            snippet: Result snippet
        
        Returns:
            Difficulty estimate (0.0 = beginner, 1.0 = expert)
        """
        content = f"{title} {snippet}".lower()
        
        # Indicators of difficulty
        difficulty_score = 0.5  # Start at medium
        
        # Beginner indicators (decrease difficulty)
        beginner_terms = [
            "introduction", "beginner", "tutorial", "basics", "getting started",
            "simple", "easy", "guide", "101", "for beginners", "explained"
        ]
        for term in beginner_terms:
            if term in content:
                difficulty_score -= 0.05
        
        # Advanced indicators (increase difficulty)
        advanced_terms = [
            "advanced", "expert", "professional", "research", "theorem",
            "dissertation", "phd", "optimization", "algorithm", "complexity"
        ]
        for term in advanced_terms:
            if term in content:
                difficulty_score += 0.08
        
        # Technical jargon (increase difficulty)
        technical_terms = [
            "implementation", "architecture", "framework", "methodology",
            "quantitative", "empirical", "theoretical"
        ]
        for term in technical_terms:
            if term in content:
                difficulty_score += 0.03
        
        # Clamp to valid range
        return max(0.0, min(1.0, difficulty_score))


# ============================================================================
# RAG ENGINE
# ============================================================================

class RAGEngine:
    """
    Main RAG (Retrieval-Augmented Generation) Engine
    
    Features:
    - Real-time web search
    - Emotion-aware source selection
    - Difficulty-aware filtering
    - Citation management
    - Context formatting for AI
    """
    
    def __init__(
        self,
        serper_api_key: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        max_sources: int = 5,
        search_timeout: int = 10
    ):
        """
        Initialize RAG engine
        
        Args:
            serper_api_key: Serper API key
            brave_api_key: Brave Search API key
            max_sources: Maximum sources to include
            search_timeout: Search request timeout
        """
        self.search_engine = WebSearchEngine(
            serper_api_key=serper_api_key,
            brave_api_key=brave_api_key,
            timeout=search_timeout
        )
        self.source_analyzer = SourceAnalyzer()
        self.max_sources = max_sources
        
        logger.info(f"✅ RAGEngine initialized (max_sources={max_sources})")
    
    async def augment_query(
        self,
        query: str,
        emotion_state: Optional[EmotionState] = None,
        ability_level: float = 0.5,
        enable_search: bool = True
    ) -> Optional[RAGContext]:
        """
        Augment query with real-time web knowledge
        
        Args:
            query: User query
            emotion_state: Student's emotion state (for source selection)
            ability_level: Student's ability level (0.0-1.0)
            enable_search: Whether to enable web search
        
        Returns:
            RAGContext with sources and formatted text, or None if search disabled
        """
        if not enable_search:
            logger.debug("RAG search disabled for this query")
            return None
        
        start_time = time.time()
        
        try:
            logger.info(f"🌐 RAG augmentation started for: '{query[:50]}...'")
            
            # Step 1: Search the web
            raw_results, provider = await self.search_engine.search(
                query=query,
                max_results=self.max_sources * 2  # Get more, then filter
            )
            
            if not raw_results:
                logger.warning("No search results returned")
                return None
            
            logger.info(f"📊 Retrieved {len(raw_results)} raw results from {provider.value}")
            
            # Step 2: Analyze and score sources
            analyzed_sources = [
                self.source_analyzer.analyze(result)
                for result in raw_results
            ]
            
            # Step 3: Filter based on emotion and ability
            filtered_sources = self._filter_sources(
                sources=analyzed_sources,
                emotion_state=emotion_state,
                ability_level=ability_level
            )
            
            # Step 4: Rank and select top sources
            top_sources = self._rank_sources(filtered_sources)[:self.max_sources]
            
            logger.info(
                f"✅ Selected {len(top_sources)} sources "
                f"(filtered from {len(analyzed_sources)})"
            )
            
            # Step 5: Format context text for AI prompt
            context_text = self._format_context(top_sources)
            
            # Step 6: Generate citations
            citations = [
                source.to_citation(idx + 1)
                for idx, source in enumerate(top_sources)
            ]
            
            # Calculate timing
            search_time_ms = (time.time() - start_time) * 1000
            
            rag_context = RAGContext(
                query=query,
                sources=top_sources,
                provider_used=provider,
                search_time_ms=search_time_ms,
                total_results=len(analyzed_sources),
                filtered_results=len(filtered_sources),
                context_text=context_text,
                citations=citations
            )
            
            logger.info(
                f"✅ RAG augmentation complete: "
                f"{len(top_sources)} sources, "
                f"{search_time_ms:.0f}ms"
            )
            
            return rag_context
        
        except Exception as e:
            logger.error(f"❌ RAG augmentation failed: {e}")
            # Don't fail the entire request - graceful degradation
            return None
    
    def _filter_sources(
        self,
        sources: List[SearchResult],
        emotion_state: Optional[EmotionState],
        ability_level: float
    ) -> List[SearchResult]:
        """
        Filter sources based on emotion and ability
        
        Emotion-aware filtering:
        - Struggling students: Prefer beginner-friendly sources
        - Confident students: Can handle advanced sources
        - Anxious students: Prefer trusted, credible sources
        
        Args:
            sources: List of analyzed sources
            emotion_state: Student emotion state
            ability_level: Student ability (0.0-1.0)
        
        Returns:
            Filtered list of sources
        """
        if not sources:
            return []
        
        filtered = []
        
        # Determine difficulty tolerance
        if emotion_state:
            readiness = emotion_state.learning_readiness
            
            if readiness in [LearningReadiness.LOW_READINESS, LearningReadiness.NOT_READY]:
                # Struggling - only beginner/intermediate sources
                max_difficulty = 0.6
                min_credibility = 0.75  # Higher trust needed
            elif readiness == LearningReadiness.MODERATE_READINESS:
                max_difficulty = 0.75
                min_credibility = 0.65
            else:
                # Confident - can handle advanced
                max_difficulty = 1.0
                min_credibility = 0.55
        else:
            # No emotion data - use ability level
            max_difficulty = min(0.5 + ability_level * 0.5, 1.0)
            min_credibility = 0.60
        
        # Filter based on criteria
        for source in sources:
            # Check difficulty threshold
            if source.difficulty_estimate > max_difficulty:
                logger.debug(
                    f"Filtered (too difficult): {source.title[:50]} "
                    f"(difficulty={source.difficulty_estimate:.2f})"
                )
                continue
            
            # Check credibility threshold
            if source.credibility_score < min_credibility:
                logger.debug(
                    f"Filtered (low credibility): {source.title[:50]} "
                    f"(credibility={source.credibility_score:.2f})"
                )
                continue
            
            filtered.append(source)
        
        logger.debug(
            f"Filtering complete: {len(filtered)}/{len(sources)} sources passed "
            f"(max_difficulty={max_difficulty:.2f}, min_credibility={min_credibility:.2f})"
        )
        
        return filtered
    
    def _rank_sources(self, sources: List[SearchResult]) -> List[SearchResult]:
        """
        Rank sources by combined score (relevance + credibility + difficulty)
        
        Args:
            sources: List of filtered sources
        
        Returns:
            Sorted list (best first)
        """
        if not sources:
            return []
        
        # Calculate combined score for each source
        scored_sources = []
        for source in sources:
            # Weighted scoring
            combined_score = (
                source.relevance_score * 0.5 +      # Relevance most important
                source.credibility_score * 0.35 +   # Then credibility
                (1.0 - source.difficulty_estimate) * 0.15  # Prefer simpler (for most students)
            )
            scored_sources.append((combined_score, source))
        
        # Sort by score (descending)
        scored_sources.sort(key=lambda x: x[0], reverse=True)
        
        # Return just the sources
        return [source for _, source in scored_sources]
    
    def _format_context(self, sources: List[SearchResult]) -> str:
        """
        Format sources into context text for AI prompt
        
        Args:
            sources: List of top sources
        
        Returns:
            Formatted context text
        """
        if not sources:
            return ""
        
        context_parts = ["REAL-TIME WEB SOURCES (current as of today):"]
        context_parts.append("")
        
        for idx, source in enumerate(sources, 1):
            context_parts.append(f"[{idx}] {source.title}")
            context_parts.append(f"Source: {source.url}")
            context_parts.append(f"Content: {source.snippet}")
            context_parts.append("")
        
        return "\n".join(context_parts)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def create_rag_engine() -> RAGEngine:
    """
    Factory function to create RAG engine with environment configuration
    
    Returns:
        Configured RAGEngine instance
    """
    serper_key = os.getenv("SERPER_API_KEY")
    brave_key = os.getenv("BRAVE_API_KEY")
    
    if not serper_key and not brave_key:
        logger.warning(
            "⚠️  No search API keys found in environment. "
            "RAG will use fallback mode. Set SERPER_API_KEY or BRAVE_API_KEY."
        )
    
    return RAGEngine(
        serper_api_key=serper_key,
        brave_api_key=brave_key,
        max_sources=5
    )
