"""
Emotion Engine - Main emotion detection and analysis orchestrator.

This module provides comprehensive emotion detection featuring:
- Transformer-based emotion recognition (BERT/RoBERTa)
- Behavioral pattern analysis
- Real-time learning state prediction
- ML-driven intervention recommendations
- Multimodal emotion fusion

Author: MasterX AI Team
Version: 1.0 (Enhanced from v9.0)
"""

import asyncio
import logging
import time
import uuid
import gc
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Import enhanced core components
from .emotion_core import (
    EmotionConstants,
    EmotionCategory,
    InterventionLevel,
    LearningReadiness,
    EmotionalTrajectory,
    EmotionMetrics,
    EmotionResult,
    BehavioralPattern
)

from .emotion_transformer import (
    EmotionTransformer,
    AdaptiveThresholdManager
)

# Import legacy components for backward compatibility
try:
    from .emotion_core import (
        AuthenticBehavioralAnalyzer,
        AuthenticPatternRecognitionEngine
    )
    LEGACY_ANALYZERS_AVAILABLE = True
except ImportError:
    LEGACY_ANALYZERS_AVAILABLE = False
    logger.warning("Legacy analyzers not available, using fallback")

# Import circuit breaker if available
try:
    from ...core.enhanced_database_models import (
        UltraEnterpriseCircuitBreaker,
        CircuitBreakerState
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False


# ============================================================================
# EMOTION ENGINE
# ============================================================================

class EmotionEngine:
    """
    Main emotion detection and analysis engine.
    
    Provides comprehensive emotion analysis including:
    - Transformer-based emotion recognition
    - Behavioral pattern analysis
    - Learning state prediction
    - Intervention recommendations
    - Trajectory prediction
    """
    
    def __init__(self):
        """Initialize emotion detection engine."""
        
        self.engine_id = str(uuid.uuid4())
        self.is_initialized = False
        
        # Core components
        self.transformer = EmotionTransformer()
        self.threshold_manager = AdaptiveThresholdManager()
        
        # Legacy analyzers (if available)
        if LEGACY_ANALYZERS_AVAILABLE:
            self.behavioral_analyzer = AuthenticBehavioralAnalyzer()
            self.pattern_engine = AuthenticPatternRecognitionEngine()
        else:
            self.behavioral_analyzer = None
            self.pattern_engine = None
        
        # Circuit breaker for fault tolerance
        if CIRCUIT_BREAKER_AVAILABLE:
            self.circuit_breaker = UltraEnterpriseCircuitBreaker(
                name="emotion_engine",
                failure_threshold=EmotionConstants.FAILURE_THRESHOLD,
                recovery_timeout=EmotionConstants.RECOVERY_TIMEOUT,
                success_threshold=EmotionConstants.SUCCESS_THRESHOLD
            )
        else:
            self.circuit_breaker = None
        
        # User patterns and adaptation
        self.user_patterns: Dict[str, BehavioralPattern] = {}
        self.global_patterns: Dict[str, Any] = {}
        
        # Caching for performance
        self.emotion_cache: Dict[str, EmotionResult] = {}
        self.pattern_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.analysis_metrics = deque(maxlen=10000)
        self.response_times = deque(maxlen=1000)
        
        # Concurrency control
        self.analysis_semaphore = asyncio.Semaphore(
            EmotionConstants.MAX_CONCURRENT_ANALYSES
        )
        
        # Background tasks
        self._learning_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        
        logger.info(f"✅ Emotion Engine initialized - ID: {self.engine_id}")
    
    async def initialize(self) -> bool:
        """Initialize all engine components."""
        start_time = time.time()
        
        try:
            logger.info("🚀 Initializing Emotion Engine components...")
            
            # Phase 1: Initialize transformer
            logger.info("Phase 1: Initializing transformer...")
            transformer_success = await self.transformer.initialize()
            if not transformer_success:
                logger.warning("⚠️ Transformer initialization incomplete, using fallback")
            
            # Phase 2: Initialize behavioral analyzer
            if self.behavioral_analyzer:
                logger.info("Phase 2: Initializing behavioral analyzer...")
                await self.behavioral_analyzer.initialize()
            
            # Phase 3: Initialize pattern engine
            if self.pattern_engine:
                logger.info("Phase 3: Initializing pattern recognition...")
                await self.pattern_engine.initialize()
            
            # Phase 4: Load user patterns
            logger.info("Phase 4: Loading user patterns...")
            await self._load_user_patterns()
            
            # Phase 5: Start background tasks
            logger.info("Phase 5: Starting background tasks...")
            await self._start_background_tasks()
            
            self.is_initialized = True
            init_time = (time.time() - start_time) * 1000
            
            logger.info(
                f"✅ Emotion Engine initialized successfully "
                f"(time: {init_time:.1f}ms, target: {EmotionConstants.TARGET_ANALYSIS_TIME_MS}ms)"
            )
            
            return True
            
        except Exception as e:
            init_time = (time.time() - start_time) * 1000
            logger.error(
                f"❌ Emotion Engine initialization failed: {str(e)} "
                f"(time: {init_time:.1f}ms)"
            )
            return False
    
    async def analyze_emotion(
        self,
        user_id: str,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        behavioral_data: Optional[Dict[str, Any]] = None
    ) -> EmotionResult:
        """
        Analyze emotion from text and behavioral data.
        
        Args:
            user_id: User identifier
            text: Text to analyze
            context: Optional context information
            behavioral_data: Optional behavioral indicators
            
        Returns:
            EmotionResult with comprehensive emotion analysis
        """
        analysis_id = str(uuid.uuid4())
        start_time = time.time()
        
        async with self.analysis_semaphore:
            try:
                if not self.is_initialized:
                    await self.initialize()
                
                logger.debug(f"🔍 Analyzing emotion for user {user_id}")
                
                # Execute analysis pipeline with circuit breaker
                if self.circuit_breaker:
                    result = await self.circuit_breaker(
                        self._execute_analysis_pipeline,
                        user_id, text, context, behavioral_data, start_time
                    )
                else:
                    result = await self._execute_analysis_pipeline(
                        user_id, text, context, behavioral_data, start_time
                    )
                
                # Update performance metrics
                analysis_time = (time.time() - start_time) * 1000
                self.response_times.append(analysis_time)
                
                # Update user patterns
                await self._update_user_patterns(user_id, result)
                
                logger.debug(
                    f"✅ Emotion analysis complete for user {user_id} "
                    f"(time: {analysis_time:.1f}ms, emotion: {result.metrics.primary_emotion})"
                )
                
                return result
                
            except Exception as e:
                logger.error(
                    f"❌ Emotion analysis failed for user {user_id}: {str(e)}"
                )
                return self._generate_fallback_result(user_id, text, e)
    
    async def _execute_analysis_pipeline(
        self,
        user_id: str,
        text: str,
        context: Optional[Dict[str, Any]],
        behavioral_data: Optional[Dict[str, Any]],
        start_time: float
    ) -> EmotionResult:
        """Execute the complete emotion analysis pipeline."""
        
        # Phase 1: Preprocess input
        phase_start = time.time()
        processed_input = await self._preprocess_input(
            text, context, behavioral_data, user_id
        )
        preprocess_time = (time.time() - phase_start) * 1000
        
        # Phase 2: Transformer-based emotion detection
        phase_start = time.time()
        transformer_result = await self.transformer.predict(
            text, user_id
        )
        transformer_time = (time.time() - phase_start) * 1000
        
        # Phase 3: Behavioral analysis
        phase_start = time.time()
        behavioral_result = await self._analyze_behavioral_patterns(
            user_id, behavioral_data or {}
        )
        behavioral_time = (time.time() - phase_start) * 1000
        
        # Phase 4: Pattern recognition
        phase_start = time.time()
        pattern_result = await self._analyze_patterns(
            user_id, text, context or {}
        )
        pattern_time = (time.time() - phase_start) * 1000
        
        # Phase 5: Multimodal fusion
        phase_start = time.time()
        fused_result = await self._fuse_multimodal_analysis(
            transformer_result,
            behavioral_result,
            pattern_result
        )
        fusion_time = (time.time() - phase_start) * 1000
        
        # Phase 6: Learning state analysis
        phase_start = time.time()
        learning_state = await self._analyze_learning_state(
            fused_result, user_id
        )
        learning_time = (time.time() - phase_start) * 1000
        
        # Phase 7: Intervention analysis
        phase_start = time.time()
        intervention = await self._analyze_intervention_needs(
            learning_state, user_id
        )
        intervention_time = (time.time() - phase_start) * 1000
        
        # Phase 8: Trajectory prediction
        phase_start = time.time()
        trajectory = await self._predict_trajectory(
            fused_result, learning_state, user_id
        )
        trajectory_time = (time.time() - phase_start) * 1000
        
        # Generate final result
        total_time = (time.time() - start_time) * 1000
        
        result = self._generate_emotion_result(
            user_id=user_id,
            text=text,
            fused_result=fused_result,
            learning_state=learning_state,
            intervention=intervention,
            trajectory=trajectory,
            analysis_time=total_time
        )
        
        logger.debug(
            f"📊 Pipeline timing - "
            f"preprocess: {preprocess_time:.1f}ms, "
            f"transformer: {transformer_time:.1f}ms, "
            f"behavioral: {behavioral_time:.1f}ms, "
            f"pattern: {pattern_time:.1f}ms, "
            f"fusion: {fusion_time:.1f}ms, "
            f"learning: {learning_time:.1f}ms, "
            f"intervention: {intervention_time:.1f}ms, "
            f"trajectory: {trajectory_time:.1f}ms, "
            f"total: {total_time:.1f}ms"
        )
        
        return result
    
    async def _preprocess_input(
        self,
        text: str,
        context: Optional[Dict[str, Any]],
        behavioral_data: Optional[Dict[str, Any]],
        user_id: str
    ) -> Dict[str, Any]:
        """Preprocess input with user context enrichment."""
        try:
            # Get user patterns for context
            user_pattern = self.user_patterns.get(user_id)
            
            processed = {
                'text': text.strip() if text else "",
                'context': context or {},
                'behavioral': behavioral_data or {},
                'user_context': {
                    'has_history': user_pattern is not None,
                    'interaction_count': user_pattern.total_interactions if user_pattern else 0,
                    'avg_engagement': user_pattern.avg_engagement if user_pattern else 0.5,
                    'avg_cognitive_load': user_pattern.avg_cognitive_load if user_pattern else 0.5
                }
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"❌ Input preprocessing failed: {e}")
            return {
                'text': text,
                'context': {},
                'behavioral': {},
                'user_context': {}
            }
    
    async def _analyze_behavioral_patterns(
        self,
        user_id: str,
        behavioral_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze behavioral patterns using legacy analyzer or fallback."""
        try:
            if self.behavioral_analyzer:
                # Use legacy behavioral analyzer
                engagement = await self.behavioral_analyzer.analyze_engagement_patterns(
                    user_id, behavioral_data
                )
                cognitive = await self.behavioral_analyzer.analyze_cognitive_load_indicators(
                    user_id, behavioral_data
                )
                
                return {
                    'engagement': engagement.get('overall_engagement', 0.5),
                    'cognitive_load': cognitive.get('overall_cognitive_load', 0.5),
                    'confidence': 0.7
                }
            else:
                # Fallback: extract basic behavioral indicators
                return self._extract_basic_behavioral_indicators(behavioral_data)
            
        except Exception as e:
            logger.error(f"❌ Behavioral analysis failed: {e}")
            return {
                'engagement': 0.5,
                'cognitive_load': 0.5,
                'confidence': 0.3
            }
    
    def _extract_basic_behavioral_indicators(
        self,
        behavioral_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract basic behavioral indicators as fallback."""
        engagement = behavioral_data.get('engagement_level', 0.5)
        cognitive_load = behavioral_data.get('cognitive_load', 0.5)
        
        # Extract from time spent, interactions, etc.
        time_spent = behavioral_data.get('time_spent_seconds', 0)
        if time_spent > 0:
            # Normalize time spent to engagement (0-1)
            # Assume 5 minutes is optimal engagement
            engagement = min(1.0, time_spent / 300)
        
        return {
            'engagement': engagement,
            'cognitive_load': cognitive_load,
            'confidence': 0.5
        }
    
    async def _analyze_patterns(
        self,
        user_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze emotional patterns using pattern engine or fallback."""
        try:
            if self.pattern_engine:
                # Use legacy pattern engine
                patterns = await self.pattern_engine.recognize_emotional_patterns(
                    text, user_id, context
                )
                return patterns
            else:
                # Fallback: basic pattern detection
                return self._detect_basic_patterns(text)
            
        except Exception as e:
            logger.error(f"❌ Pattern analysis failed: {e}")
            return {
                'emotional_signals': {},
                'pattern_confidence': 0.3
            }
    
    def _detect_basic_patterns(self, text: str) -> Dict[str, Any]:
        """Detect basic emotional patterns as fallback."""
        text_lower = text.lower()
        
        # Simple keyword detection
        struggle_keywords = ['confused', 'don\'t understand', 'frustrated', 'difficult']
        positive_keywords = ['understand', 'got it', 'makes sense', 'clear', 'easy']
        
        struggle_score = sum(1 for kw in struggle_keywords if kw in text_lower)
        positive_score = sum(1 for kw in positive_keywords if kw in text_lower)
        
        return {
            'emotional_signals': {
                'struggle_indicators': struggle_score,
                'positive_indicators': positive_score
            },
            'pattern_confidence': 0.5
        }
    
    async def _fuse_multimodal_analysis(
        self,
        transformer_result: Dict[str, Any],
        behavioral_result: Dict[str, Any],
        pattern_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fuse multiple analysis sources with adaptive weighting."""
        try:
            # Calculate confidence-based weights
            t_conf = transformer_result.get('confidence', 0.5)
            b_conf = behavioral_result.get('confidence', 0.5)
            p_conf = pattern_result.get('pattern_confidence', 0.5)
            
            total_conf = t_conf + b_conf + p_conf
            if total_conf > 0:
                weights = {
                    'transformer': t_conf / total_conf,
                    'behavioral': b_conf / total_conf,
                    'pattern': p_conf / total_conf
                }
            else:
                weights = {'transformer': 0.5, 'behavioral': 0.3, 'pattern': 0.2}
            
            # Fuse emotion predictions
            primary_emotion = transformer_result.get(
                'primary_emotion',
                EmotionCategory.NEUTRAL.value
            )
            
            # Fuse engagement and cognitive load
            engagement = (
                behavioral_result.get('engagement', 0.5) * weights['behavioral'] +
                0.5 * weights['transformer']  # Transformer contributes baseline
            )
            
            cognitive_load = (
                behavioral_result.get('cognitive_load', 0.5) * weights['behavioral'] +
                0.5 * weights['transformer']
            )
            
            # Overall confidence
            fusion_confidence = (
                t_conf * weights['transformer'] +
                b_conf * weights['behavioral'] +
                p_conf * weights['pattern']
            )
            
            return {
                'primary_emotion': primary_emotion,
                'confidence': fusion_confidence,
                'engagement': engagement,
                'cognitive_load': cognitive_load,
                'arousal': transformer_result.get('arousal', 0.5),
                'valence': transformer_result.get('valence', 0.5),
                'weights': weights
            }
            
        except Exception as e:
            logger.error(f"❌ Multimodal fusion failed: {e}")
            return {
                'primary_emotion': EmotionCategory.NEUTRAL.value,
                'confidence': 0.5,
                'engagement': 0.5,
                'cognitive_load': 0.5,
                'arousal': 0.5,
                'valence': 0.5
            }
    
    async def _analyze_learning_state(
        self,
        fused_result: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Analyze learning state with adaptive thresholds."""
        try:
            # Get user-specific thresholds
            thresholds = self.threshold_manager.get_thresholds(user_id)
            
            emotion = fused_result['primary_emotion']
            engagement = fused_result['engagement']
            cognitive_load = fused_result['cognitive_load']
            
            # Calculate readiness score
            readiness_score = self._calculate_readiness_score(
                engagement, cognitive_load, emotion, thresholds
            )
            
            # Determine learning readiness
            learning_readiness = self._determine_learning_readiness(
                readiness_score, emotion, cognitive_load
            )
            
            return {
                'learning_readiness': learning_readiness,
                'readiness_score': readiness_score,
                'engagement': engagement,
                'cognitive_load': cognitive_load
            }
            
        except Exception as e:
            logger.error(f"❌ Learning state analysis failed: {e}")
            return {
                'learning_readiness': LearningReadiness.MODERATE_READINESS.value,
                'readiness_score': 0.5,
                'engagement': 0.5,
                'cognitive_load': 0.5
            }
    
    def _calculate_readiness_score(
        self,
        engagement: float,
        cognitive_load: float,
        emotion: str,
        thresholds: Dict[str, float]
    ) -> float:
        """Calculate learning readiness score with adaptive thresholds."""
        factors = []
        
        # Engagement factor
        factors.append(engagement)
        
        # Cognitive load factor (optimal is moderate, not too high or low)
        optimal_load = thresholds.get('optimal_cognitive_load', 0.6)
        if cognitive_load <= optimal_load:
            load_factor = 1.0 - abs(cognitive_load - optimal_load) / optimal_load
        else:
            load_factor = max(0.0, 1.0 - (cognitive_load - optimal_load) / (1.0 - optimal_load))
        factors.append(load_factor)
        
        # Emotional factor
        positive_emotions = [
            EmotionCategory.JOY.value,
            EmotionCategory.EXCITEMENT.value,
            EmotionCategory.CURIOSITY.value,
            EmotionCategory.ENGAGEMENT.value,
            EmotionCategory.SATISFACTION.value
        ]
        negative_emotions = [
            EmotionCategory.FRUSTRATION.value,
            EmotionCategory.ANXIETY.value,
            EmotionCategory.CONFUSION.value,
            EmotionCategory.COGNITIVE_OVERLOAD.value
        ]
        
        if emotion in positive_emotions:
            emotional_factor = 0.8
        elif emotion in negative_emotions:
            emotional_factor = 0.3
        else:
            emotional_factor = 0.5
        
        factors.append(emotional_factor)
        
        # Weighted average
        weights = [0.4, 0.35, 0.25]  # engagement, cognitive, emotional
        score = sum(f * w for f, w in zip(factors, weights))
        
        return max(0.0, min(1.0, score))
    
    def _determine_learning_readiness(
        self,
        readiness_score: float,
        emotion: str,
        cognitive_load: float
    ) -> str:
        """Determine learning readiness state."""
        # Check for overload conditions
        if emotion == EmotionCategory.COGNITIVE_OVERLOAD.value or cognitive_load > 0.85:
            return LearningReadiness.NOT_READY.value
        
        # Check for optimal flow state
        if emotion == EmotionCategory.FLOW_STATE.value and readiness_score > 0.7:
            return LearningReadiness.OPTIMAL_READINESS.value
        
        # Use score thresholds
        if readiness_score >= 0.75:
            return LearningReadiness.OPTIMAL_READINESS.value
        elif readiness_score >= 0.6:
            return LearningReadiness.HIGH_READINESS.value
        elif readiness_score >= 0.4:
            return LearningReadiness.MODERATE_READINESS.value
        elif readiness_score >= 0.25:
            return LearningReadiness.LOW_READINESS.value
        else:
            return LearningReadiness.NOT_READY.value
    
    async def _analyze_intervention_needs(
        self,
        learning_state: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Analyze intervention needs with ML-driven recommendations."""
        try:
            readiness = learning_state['learning_readiness']
            readiness_score = learning_state['readiness_score']
            cognitive_load = learning_state['cognitive_load']
            
            # Calculate intervention need (not hardcoded, based on state)
            intervention_factors = []
            
            # Readiness-based intervention need
            readiness_mapping = {
                LearningReadiness.NOT_READY.value: 0.9,
                LearningReadiness.LOW_READINESS.value: 0.6,
                LearningReadiness.MODERATE_READINESS.value: 0.3,
                LearningReadiness.HIGH_READINESS.value: 0.1,
                LearningReadiness.OPTIMAL_READINESS.value: 0.0
            }
            intervention_factors.append(readiness_mapping.get(readiness, 0.3))
            
            # Cognitive load based intervention
            if cognitive_load > 0.8:
                intervention_factors.append(0.8)
            elif cognitive_load > 0.6:
                intervention_factors.append(0.4)
            else:
                intervention_factors.append(0.1)
            
            # Calculate overall need
            intervention_need = sum(intervention_factors) / len(intervention_factors)
            
            # Determine intervention level
            if intervention_need >= 0.8:
                level = InterventionLevel.CRITICAL.value
            elif intervention_need >= 0.6:
                level = InterventionLevel.SIGNIFICANT.value
            elif intervention_need >= 0.4:
                level = InterventionLevel.MODERATE.value
            elif intervention_need >= 0.2:
                level = InterventionLevel.MILD.value
            elif intervention_need >= 0.1:
                level = InterventionLevel.PREVENTIVE.value
            else:
                level = InterventionLevel.NONE.value
            
            # Generate recommendations
            recommendations = self._generate_intervention_recommendations(
                level, readiness, cognitive_load
            )
            
            return {
                'intervention_level': level,
                'intervention_need': intervention_need,
                'recommendations': recommendations,
                'intervention_type': self._determine_intervention_type(level)
            }
            
        except Exception as e:
            logger.error(f"❌ Intervention analysis failed: {e}")
            return {
                'intervention_level': InterventionLevel.NONE.value,
                'intervention_need': 0.0,
                'recommendations': [],
                'intervention_type': 'none'
            }
    
    def _generate_intervention_recommendations(
        self,
        level: str,
        readiness: str,
        cognitive_load: float
    ) -> List[str]:
        """Generate specific intervention recommendations."""
        recommendations = []
        
        if level == InterventionLevel.CRITICAL.value:
            recommendations.extend([
                "Take a 15-20 minute break to reduce cognitive load",
                "Switch to simpler review material",
                "Consider ending session and resuming when refreshed"
            ])
        elif level == InterventionLevel.SIGNIFICANT.value:
            recommendations.extend([
                "Take a 5-10 minute break",
                "Review foundational concepts before proceeding",
                "Try different learning approach or resources"
            ])
        elif level == InterventionLevel.MODERATE.value:
            recommendations.extend([
                "Take a brief 2-3 minute break",
                "Use visual aids or examples for better understanding",
                "Break down complex topics into smaller parts"
            ])
        elif level == InterventionLevel.MILD.value:
            recommendations.extend([
                "You're doing well! Small adjustments may help",
                "Consider varying learning methods for better retention"
            ])
        elif readiness == LearningReadiness.OPTIMAL_READINESS.value:
            recommendations.extend([
                "Excellent learning state! Keep up the momentum",
                "This is a great time for challenging material"
            ])
        
        return recommendations
    
    def _determine_intervention_type(self, level: str) -> str:
        """Determine type of intervention needed."""
        mapping = {
            InterventionLevel.CRITICAL.value: 'immediate_break',
            InterventionLevel.SIGNIFICANT.value: 'active_guidance',
            InterventionLevel.MODERATE.value: 'gentle_support',
            InterventionLevel.MILD.value: 'encouragement',
            InterventionLevel.PREVENTIVE.value: 'proactive_tips',
            InterventionLevel.NONE.value: 'positive_reinforcement'
        }
        return mapping.get(level, 'none')
    
    async def _predict_trajectory(
        self,
        emotion_data: Dict[str, Any],
        learning_state: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Predict emotional trajectory based on patterns."""
        try:
            # Get user's emotional history
            user_pattern = self.user_patterns.get(user_id)
            
            if not user_pattern or len(user_pattern.emotional_history) < 3:
                # Not enough data for prediction
                return {
                    'trajectory': EmotionalTrajectory.STABLE_NEUTRAL.value,
                    'confidence': 0.3
                }
            
            # Analyze recent emotional trend
            recent_emotions = user_pattern.emotional_history[-5:]
            valences = [self._calculate_emotion_valence(e) for e in recent_emotions]
            
            # Calculate trend
            if len(valences) >= 2:
                slope = (valences[-1] - valences[0]) / (len(valences) - 1)
            else:
                slope = 0.0
            
            # Determine trajectory
            if slope > 0.15:
                trajectory = EmotionalTrajectory.IMPROVING.value
            elif slope < -0.15:
                trajectory = EmotionalTrajectory.DECLINING.value
            elif valences[-1] > 0.6:
                trajectory = EmotionalTrajectory.STABLE_POSITIVE.value
            elif valences[-1] < 0.4:
                trajectory = EmotionalTrajectory.DECLINING.value
            else:
                trajectory = EmotionalTrajectory.STABLE_NEUTRAL.value
            
            # Calculate confidence based on data points
            confidence = min(0.9, 0.5 + len(recent_emotions) * 0.1)
            
            return {
                'trajectory': trajectory,
                'confidence': confidence,
                'slope': slope
            }
            
        except Exception as e:
            logger.error(f"❌ Trajectory prediction failed: {e}")
            return {
                'trajectory': EmotionalTrajectory.STABLE_NEUTRAL.value,
                'confidence': 0.3
            }
    
    def _calculate_emotion_valence(self, emotion: str) -> float:
        """Calculate emotional valence (0=negative, 1=positive)."""
        positive_emotions = {
            EmotionCategory.JOY.value: 0.9,
            EmotionCategory.EXCITEMENT.value: 0.85,
            EmotionCategory.SATISFACTION.value: 0.8,
            EmotionCategory.CURIOSITY.value: 0.75,
            EmotionCategory.CONFIDENCE.value: 0.8,
            EmotionCategory.FLOW_STATE.value: 0.9,
            EmotionCategory.BREAKTHROUGH_MOMENT.value: 0.95,
            EmotionCategory.MASTERY_JOY.value: 0.85
        }
        
        negative_emotions = {
            EmotionCategory.FRUSTRATION.value: 0.2,
            EmotionCategory.ANXIETY.value: 0.15,
            EmotionCategory.CONFUSION.value: 0.25,
            EmotionCategory.COGNITIVE_OVERLOAD.value: 0.1,
            EmotionCategory.ANGER.value: 0.1,
            EmotionCategory.FEAR.value: 0.15,
            EmotionCategory.SADNESS.value: 0.2
        }
        
        return positive_emotions.get(emotion) or negative_emotions.get(emotion) or 0.5
    
    def _generate_emotion_result(
        self,
        user_id: str,
        text: str,
        fused_result: Dict[str, Any],
        learning_state: Dict[str, Any],
        intervention: Dict[str, Any],
        trajectory: Dict[str, Any],
        analysis_time: float
    ) -> EmotionResult:
        """Generate comprehensive emotion result."""
        
        # Create emotion metrics
        metrics = EmotionMetrics(
            primary_emotion=fused_result['primary_emotion'],
            emotion_distribution={},
            confidence=fused_result['confidence'],
            arousal=fused_result.get('arousal', 0.5),
            valence=fused_result.get('valence', 0.5),
            dominance=0.5,
            learning_readiness=learning_state['learning_readiness'],
            cognitive_load=learning_state['cognitive_load'],
            engagement_level=learning_state['engagement'],
            intervention_level=intervention['intervention_level'],
            intervention_confidence=intervention['intervention_need'],
            emotional_trajectory=trajectory['trajectory'],
            analysis_time_ms=analysis_time,
            model_version="1.0",
            timestamp=datetime.utcnow()
        )
        
        # Determine intervention needs
        intervention_needed = intervention['intervention_level'] not in [
            InterventionLevel.NONE.value,
            InterventionLevel.PREVENTIVE.value
        ]
        
        # Calculate difficulty/pacing adjustments
        difficulty_adj = self._calculate_difficulty_adjustment(
            learning_state['learning_readiness'],
            learning_state['cognitive_load']
        )
        pacing_adj = self._calculate_pacing_adjustment(
            learning_state['engagement']
        )
        support_level = self._determine_support_level(
            intervention['intervention_level']
        )
        
        # Create result
        result = EmotionResult(
            metrics=metrics,
            user_id=user_id,
            text_analyzed=text,
            intervention_needed=intervention_needed,
            intervention_type=intervention['intervention_type'],
            intervention_suggestions=intervention['recommendations'],
            difficulty_adjustment=difficulty_adj,
            pacing_adjustment=pacing_adj,
            support_level=support_level,
            prediction_quality=fused_result['confidence'],
            model_type="transformer_multimodal"
        )
        
        return result
    
    def _calculate_difficulty_adjustment(
        self,
        readiness: str,
        cognitive_load: float
    ) -> float:
        """Calculate difficulty adjustment (-1 easier, +1 harder)."""
        if cognitive_load > 0.8 or readiness == LearningReadiness.NOT_READY.value:
            return -0.5  # Make easier
        elif cognitive_load > 0.6:
            return -0.2  # Slightly easier
        elif cognitive_load < 0.3 and readiness == LearningReadiness.OPTIMAL_READINESS.value:
            return 0.3  # Make harder (user is ready)
        else:
            return 0.0  # Keep same
    
    def _calculate_pacing_adjustment(self, engagement: float) -> float:
        """Calculate pacing adjustment (-1 slower, +1 faster)."""
        if engagement < 0.3:
            return -0.3  # Slow down
        elif engagement > 0.8:
            return 0.2  # Can go faster
        else:
            return 0.0  # Current pace is fine
    
    def _determine_support_level(self, intervention_level: str) -> str:
        """Determine support level needed."""
        mapping = {
            InterventionLevel.CRITICAL.value: 'intensive',
            InterventionLevel.SIGNIFICANT.value: 'enhanced',
            InterventionLevel.MODERATE.value: 'standard',
            InterventionLevel.MILD.value: 'standard',
            InterventionLevel.PREVENTIVE.value: 'minimal',
            InterventionLevel.NONE.value: 'minimal'
        }
        return mapping.get(intervention_level, 'standard')
    
    def _generate_fallback_result(
        self,
        user_id: str,
        text: str,
        error: Exception
    ) -> EmotionResult:
        """Generate fallback result on error."""
        logger.error(f"Generating fallback result due to error: {error}")
        
        metrics = EmotionMetrics(
            primary_emotion=EmotionCategory.NEUTRAL.value,
            confidence=0.3,
            learning_readiness=LearningReadiness.MODERATE_READINESS.value,
            cognitive_load=0.5,
            engagement_level=0.5,
            intervention_level=InterventionLevel.NONE.value
        )
        
        return EmotionResult(
            metrics=metrics,
            user_id=user_id,
            text_analyzed=text,
            intervention_needed=False,
            model_type="fallback"
        )
    
    async def _update_user_patterns(
        self,
        user_id: str,
        result: EmotionResult
    ) -> None:
        """Update user behavioral patterns."""
        try:
            if user_id not in self.user_patterns:
                self.user_patterns[user_id] = BehavioralPattern(user_id=user_id)
            
            pattern = self.user_patterns[user_id]
            pattern.update_from_result(result)
            
            # Update adaptive thresholds
            self.threshold_manager.update_thresholds(
                user_id,
                result.metrics.primary_emotion,
                result.metrics.confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to update user patterns: {e}")
    
    async def _load_user_patterns(self) -> None:
        """Load user patterns from storage."""
        # TODO: Load from database
        self.user_patterns = {}
        logger.info("✅ User patterns loaded")
    
    async def _start_background_tasks(self) -> None:
        """Start background learning and optimization tasks."""
        try:
            if not self._learning_task or self._learning_task.done():
                self._learning_task = asyncio.create_task(
                    self._continuous_learning_loop()
                )
            
            if not self._optimization_task or self._optimization_task.done():
                self._optimization_task = asyncio.create_task(
                    self._optimization_loop()
                )
            
            logger.info("✅ Background tasks started")
        except Exception as e:
            logger.error(f"❌ Background task startup failed: {e}")
    
    async def _continuous_learning_loop(self) -> None:
        """Continuous learning from user interactions."""
        while self.is_initialized:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._process_learning_updates()
            except Exception as e:
                logger.error(f"❌ Learning loop error: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_loop(self) -> None:
        """Continuous system optimization."""
        while self.is_initialized:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                await self._optimize_system()
            except Exception as e:
                logger.error(f"❌ Optimization loop error: {e}")
                await asyncio.sleep(60)
    
    async def _process_learning_updates(self) -> None:
        """Process learning updates."""
        try:
            # Clean up old cache entries
            if len(self.emotion_cache) > 1000:
                # Keep only recent 500
                keys_to_remove = list(self.emotion_cache.keys())[:-500]
                for key in keys_to_remove:
                    del self.emotion_cache[key]
            
            logger.debug("✅ Learning updates processed")
        except Exception as e:
            logger.error(f"❌ Learning updates failed: {e}")
    
    async def _optimize_system(self) -> None:
        """Optimize system performance."""
        try:
            # Garbage collection
            gc.collect()
            
            # Calculate average response time
            if self.response_times:
                avg_time = sum(self.response_times) / len(self.response_times)
                logger.info(f"📊 Avg response time: {avg_time:.1f}ms")
            
            logger.debug("✅ System optimization complete")
        except Exception as e:
            logger.error(f"❌ System optimization failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown engine and cleanup resources."""
        try:
            logger.info("🛑 Shutting down Emotion Engine...")
            
            self.is_initialized = False
            
            # Cancel background tasks
            if self._learning_task and not self._learning_task.done():
                self._learning_task.cancel()
            if self._optimization_task and not self._optimization_task.done():
                self._optimization_task.cancel()
            
            # Clear caches
            self.emotion_cache.clear()
            self.pattern_cache.clear()
            
            logger.info("✅ Emotion Engine shutdown complete")
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Global emotion engine instance (for backward compatibility)
emotion_engine = EmotionEngine()


__all__ = [
    'EmotionEngine',
    'emotion_engine'
]
