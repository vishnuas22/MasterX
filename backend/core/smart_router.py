"""
Smart Router - Intelligent Provider Selection
Following specifications from 4.DYNAMIC_AI_ROUTING_SYSTEM.md

Selects the best AI provider for each request based on:
1. Session continuity (stick with provider for same topic)
2. Task category detection
3. Latest benchmark scores
4. Emotion state (frustrated → empathy provider)
5. Circuit breaker status

No hardcoded rules - all decisions based on real-time data.
"""

import logging
from typing import Optional, Dict
from core.models import EmotionState
from core.benchmarking import BenchmarkEngine

logger = logging.getLogger(__name__)


# ============================================================================
# TASK CATEGORY DETECTOR
# ============================================================================

class TaskCategoryDetector:
    """
    Detect task category from user message
    
    Categories:
    - coding: Programming, algorithms, debugging
    - math: Mathematics, calculations, equations
    - research: Analysis, deep research, citations
    - language: Grammar, translation, writing
    - empathy: Emotional support, encouragement
    - general: Everything else
    """
    
    # Keyword mappings for category detection
    CATEGORY_KEYWORDS = {
        'coding': [
            'code', 'function', 'algorithm', 'programming', 'python', 'javascript',
            'java', 'c++', 'debug', 'error', 'syntax', 'compile', 'variable',
            'class', 'method', 'loop', 'array', 'recursion', 'api', 'database',
            'query', 'sql', 'html', 'css', 'react', 'node', 'git', 'docker'
        ],
        'math': [
            'solve', 'calculate', 'equation', 'derivative', 'integral', 'geometry',
            'algebra', 'calculus', 'trigonometry', 'probability', 'statistics',
            'theorem', 'proof', 'formula', 'graph', 'function', 'matrix',
            'vector', 'limit', 'sum', 'product', 'square root', 'exponent'
        ],
        'research': [
            'analyze', 'research', 'compare', 'study', 'evidence', 'paper',
            'theory', 'hypothesis', 'experiment', 'data', 'findings', 'conclusion',
            'methodology', 'literature', 'scholarly', 'academic', 'citation',
            'peer-reviewed', 'investigation', 'exploration', 'examination'
        ],
        'language': [
            'translate', 'grammar', 'correct', 'sentence', 'spanish', 'french',
            'german', 'italian', 'chinese', 'japanese', 'language', 'vocabulary',
            'pronunciation', 'conjugation', 'tense', 'preposition', 'article',
            'adjective', 'verb', 'noun', 'phrase', 'idiom', 'writing', 'essay'
        ]
    }
    
    def detect(self, message: str, emotion_state: Optional[EmotionState] = None) -> str:
        """
        Detect task category from message
        
        Args:
            message: User message
            emotion_state: Detected emotion state (if available)
        
        Returns:
            Category name (coding, math, research, language, empathy, general)
        """
        
        # Check emotion first - if frustrated/anxious, route to empathy
        if emotion_state:
            if emotion_state.primary_emotion in ['frustration', 'anxiety', 'overwhelmed', 'sadness']:
                logger.info(f"🎭 Category: empathy (detected {emotion_state.primary_emotion})")
                return 'empathy'
        
        # Keyword-based detection
        msg_lower = message.lower()
        
        # Count keyword matches per category
        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in msg_lower)
            if score > 0:
                scores[category] = score
        
        # Return category with highest score
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])[0]
            logger.info(f"🎯 Category detected: {best_category} (score: {scores[best_category]})")
            return best_category
        
        # Default to general
        logger.info("🎯 Category: general (no specific keywords found)")
        return 'general'


# ============================================================================
# SMART ROUTER
# ============================================================================

class SmartRouter:
    """
    Intelligent provider selection based on benchmarks
    
    Decision flow:
    1. Check session continuity (same topic = same provider)
    2. Detect task category from message
    3. Get latest benchmark scores for category
    4. Consider emotion state for empathy routing
    5. Filter by circuit breaker status
    6. Select top performer
    """
    
    def __init__(
        self,
        benchmark_engine: BenchmarkEngine,
        session_manager: 'SessionManager'  # Forward reference
    ):
        self.benchmark_engine = benchmark_engine
        self.session_manager = session_manager
        self.category_detector = TaskCategoryDetector()
        
        logger.info("✅ SmartRouter initialized")
    
    async def select_provider(
        self,
        message: str,
        emotion_state: Optional[EmotionState],
        session_id: str,
        circuit_breaker: Optional['CircuitBreaker'] = None
    ) -> tuple[str, str]:
        """
        Select best provider for this request
        
        Args:
            message: User message
            emotion_state: Detected emotion state
            session_id: Current session ID
            circuit_breaker: Circuit breaker for health checks (optional)
        
        Returns:
            Tuple of (provider_name, category)
        """
        
        # 1. Check session continuity
        session_info = await self.session_manager.get_session_info(session_id)
        
        if session_info and session_info.get('current_provider'):
            # Check if topic changed
            current_topic = session_info.get('current_topic')
            new_category = self.category_detector.detect(message, emotion_state)
            
            if current_topic == new_category:
                # Continue with same provider (topic unchanged)
                provider = session_info['current_provider']
                logger.info(f"📌 Continuing with {provider} (same topic: {new_category})")
                return provider, new_category
            else:
                logger.info(f"🔄 Topic changed: {current_topic} → {new_category}")
        
        # 2. Detect task category
        category = self.category_detector.detect(message, emotion_state)
        
        # 3. Get latest benchmarks for this category
        benchmarks = await self.benchmark_engine.get_latest_benchmarks(
            category=category,
            max_age_hours=24
        )
        
        if not benchmarks:
            # No recent benchmarks, use default
            logger.warning(f"⚠️ No benchmarks for {category}, using default provider")
            provider = self._get_default_provider()
            return provider, category
        
        # 4. Filter by circuit breaker (if provided)
        if circuit_breaker:
            available = [b for b in benchmarks if circuit_breaker.is_available(b.provider)]
            if not available:
                logger.error("❌ No providers available (circuit breaker)")
                provider = self._get_default_provider()
                return provider, category
        else:
            available = benchmarks
        
        # 5. Select top performer
        best = available[0]  # Already sorted by score in get_latest_benchmarks
        
        logger.info(
            f"🎯 Selected {best.provider} for {category} "
            f"(score: {best.final_score:.1f}, quality: {best.quality_score:.1f})"
        )
        
        return best.provider, category
    
    def _get_default_provider(self) -> str:
        """Fallback provider when no benchmarks available"""
        # Prefer groq (fastest) as default
        return 'groq'
    
    async def should_switch_provider(
        self,
        session_id: str,
        new_message: str,
        emotion_state: Optional[EmotionState]
    ) -> bool:
        """
        Determine if provider should switch for new message
        
        Returns True if topic has changed
        """
        
        session_info = await self.session_manager.get_session_info(session_id)
        
        if not session_info or not session_info.get('current_topic'):
            return True
        
        current_topic = session_info['current_topic']
        new_category = self.category_detector.detect(new_message, emotion_state)
        
        return current_topic != new_category
