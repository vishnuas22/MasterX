"""
MasterX Engine - Main Orchestrator
Following specifications from 5.DEVELOPMENT_HANDOFF_GUIDE.md

Phase 3: COMPLETE INTEGRATION
- Process user message with emotion detection
- Context-aware conversation management
- Adaptive difficulty based on performance
- Intelligent AI provider selection
- Real-time ability estimation
- Flow state optimization
"""

import logging
import time
import uuid
from typing import Optional, List
from datetime import datetime

from core.models import (
    AIResponse, EmotionState, LearningReadiness, 
    Message, MessageRole, ContextInfo, AbilityInfo
)
from core.ai_providers import ProviderManager
from core.context_manager import ContextManager
from core.adaptive_learning import AdaptiveLearningEngine, PerformanceMetrics as AdaptivePerformanceMetrics
from services.emotion.emotion_engine import EmotionEngine
from utils.errors import MasterXError
from utils.cost_tracker import cost_tracker
from utils.database import get_database

logger = logging.getLogger(__name__)


class MasterXEngine:
    """
    Main orchestrator for MasterX learning platform
    
    PHASE 3 COMPLETE - Full Intelligence Integration:
    - Emotion detection (18 emotions, BERT/RoBERTa)
    - Multi-AI provider routing (dynamic benchmarking)
    - Context management (conversation memory, semantic search)
    - Adaptive learning (IRT, cognitive load, flow state)
    - Performance tracking (ability estimation, velocity)
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize MasterX engine with all intelligence components (Phase 1 Optimized).
        
        Args:
            config: Configuration dictionary (from settings)
        """
        self.provider_manager = ProviderManager()
        
        # Phase 1: Initialize emotion engine with config for optimizations
        emotion_config = None
        if config and 'emotion_detection' in config:
            emotion_config = config['emotion_detection']
        self.emotion_engine = EmotionEngine(config=emotion_config)
        
        # Phase 3: Initialize context and adaptive learning components
        # Database will be set during server startup
        self.context_manager = None
        self.adaptive_engine = None
        self._db_initialized = False
        
        logger.info("✅ MasterXEngine initialized (Phase 1 Optimized + Phase 3: Full Intelligence)")
    
    def initialize_intelligence_layer(self, db):
        """
        Initialize Phase 3 intelligence components with database
        
        Args:
            db: MongoDB database instance
        """
        if not self._db_initialized:
            try:
                # Initialize context manager
                self.context_manager = ContextManager(
                    db=db,
                    max_context_tokens=8000,
                    short_term_memory_size=20
                )
                
                # Initialize adaptive learning engine
                self.adaptive_engine = AdaptiveLearningEngine(db=db)
                
                self._db_initialized = True
                logger.info("✅ Intelligence layer initialized (context + adaptive learning)")
            except Exception as e:
                logger.error(f"Failed to initialize intelligence layer: {e}")
                raise
    
    async def process_request(
        self,
        user_id: str,
        message: str,
        session_id: str,
        context: Optional[dict] = None,
        subject: str = "general"
    ) -> AIResponse:
        """
        Process user request with full Phase 3 intelligence
        
        PHASE 3 COMPLETE FLOW:
        1. Retrieve conversation context (semantic memory)
        2. Analyze emotion (18 emotions, learning readiness)
        3. Get ability estimate and recommend difficulty
        4. Detect category and select best AI provider
        5. Generate context-aware, difficulty-adapted response
        6. Store message with embeddings
        7. Update ability based on interaction
        
        Args:
            user_id: User identifier
            message: User message
            session_id: Session identifier
            context: Additional context (optional)
            subject: Learning subject/topic
        
        Returns:
            AIResponse with emotion, context, and difficulty information
        """
        
        start_time = time.time()
        
        try:
            # Ensure intelligence layer is initialized
            if not self._db_initialized:
                logger.warning("Intelligence layer not initialized, using basic mode")
                return await self._process_basic(user_id, message, session_id, context)
            
            # ====================================================================
            # PHASE 3 STEP 1: RETRIEVE CONVERSATION CONTEXT
            # ====================================================================
            logger.info(f"🧠 Retrieving conversation context for session {session_id}...")
            context_start = time.time()
            
            conversation_context = await self.context_manager.get_context(
                session_id=session_id,
                include_semantic=True,
                semantic_query=message
            )
            
            recent_messages = conversation_context.get('recent_messages', [])
            relevant_messages = conversation_context.get('relevant_messages', [])
            
            context_time_ms = (time.time() - context_start) * 1000
            logger.info(
                f"✅ Context retrieved: {len(recent_messages)} recent, "
                f"{len(relevant_messages)} relevant ({context_time_ms:.0f}ms)"
            )
            
            # ====================================================================
            # PHASE 3 STEP 2: ANALYZE EMOTION
            # ====================================================================
            logger.info(f"📊 Analyzing emotion for message: {message[:50]}...")
            emotion_start = time.time()
            
            emotion_result = await self.emotion_engine.analyze_emotion(
                user_id=user_id,
                text=message,
                context=context
            )
            
            emotion_time_ms = (time.time() - emotion_start) * 1000
            logger.info(f"✅ Emotion detected: {emotion_result.metrics.primary_emotion} ({emotion_time_ms:.0f}ms)")
            
            # Create EmotionState for response
            emotion_state = EmotionState(
                primary_emotion=emotion_result.metrics.primary_emotion,
                arousal=emotion_result.metrics.arousal,
                valence=emotion_result.metrics.valence,
                learning_readiness=LearningReadiness(emotion_result.metrics.learning_readiness)
            )
            
            # ====================================================================
            # PHASE 3 STEP 3: ADAPTIVE DIFFICULTY RECOMMENDATION
            # ====================================================================
            logger.info(f"🎯 Calculating optimal difficulty for {user_id}/{subject}...")
            difficulty_start = time.time()
            
            # Get current ability
            ability = await self.adaptive_engine.ability_estimator.get_ability(
                user_id=user_id,
                subject=subject
            )
            
            # Recommend difficulty based on ability and emotion
            difficulty_level = await self.adaptive_engine.recommend_difficulty(
                user_id=user_id,
                subject=subject,
                emotion_state=emotion_state,
                recent_performance=None  # Will be calculated from context
            )
            
            difficulty_time_ms = (time.time() - difficulty_start) * 1000
            logger.info(
                f"✅ Difficulty recommended: {difficulty_level.label} "
                f"({difficulty_level.value:.2f}, ability: {ability:.2f}) "
                f"({difficulty_time_ms:.0f}ms)"
            )
            
            # ====================================================================
            # PHASE 3 STEP 4: INTELLIGENT PROVIDER SELECTION
            # ====================================================================
            logger.info(f"🤖 Selecting best AI provider...")
            ai_start = time.time()
            
            # Detect category from message
            category = self.provider_manager.detect_category_from_message(
                message, 
                emotion_state
            )
            logger.info(f"📂 Detected category: {category}")
            
            # Select best provider for this category
            selected_provider = await self.provider_manager.select_best_provider_for_category(
                category,
                emotion_state
            )
            
            # ====================================================================
            # PHASE 3 STEP 5: GENERATE CONTEXT-AWARE RESPONSE
            # ====================================================================
            # Enhance prompt with emotion, context, and difficulty
            enhanced_prompt = self._enhance_prompt_phase3(
                message=message,
                emotion_result=emotion_result,
                recent_messages=recent_messages,
                relevant_messages=relevant_messages,
                difficulty_level=difficulty_level,
                ability=ability
            )
            
            response = await self.provider_manager.generate(
                prompt=enhanced_prompt,
                provider_name=selected_provider,
                max_tokens=1000
            )
            
            ai_time_ms = (time.time() - ai_start) * 1000
            logger.info(f"✅ AI response generated ({ai_time_ms:.0f}ms)")
            
            # ====================================================================
            # PHASE 3 STEP 6: STORE MESSAGES WITH EMBEDDINGS
            # ====================================================================
            logger.info(f"💾 Storing messages with embeddings...")
            storage_start = time.time()
            
            # Store user message
            user_message = Message(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                role=MessageRole.USER,
                content=message,
                timestamp=datetime.utcnow(),
                emotion_state=emotion_state
            )
            
            await self.context_manager.add_message(
                session_id=session_id,
                message=user_message,
                generate_embedding=True
            )
            
            # Store AI response message
            ai_message = Message(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content=response.content,
                timestamp=datetime.utcnow(),
                emotion_state=emotion_state,
                provider_used=response.provider,
                response_time_ms=response.response_time_ms,
                tokens_used=response.tokens_used,
                cost=response.cost
            )
            
            await self.context_manager.add_message(
                session_id=session_id,
                message=ai_message,
                generate_embedding=True
            )
            
            storage_time_ms = (time.time() - storage_start) * 1000
            logger.info(f"✅ Messages stored with embeddings ({storage_time_ms:.0f}ms)")
            
            # ====================================================================
            # PHASE 3 STEP 7: UPDATE ABILITY ESTIMATE
            # ====================================================================
            # Infer performance from interaction
            # (In full system, this would be based on correctness checks)
            # For now, use emotion and engagement as proxy
            interaction_success = self._infer_success_from_emotion(emotion_state)
            
            await self.adaptive_engine.ability_estimator.update_ability(
                user_id=user_id,
                subject=subject,
                item_difficulty=difficulty_level.value,
                result=interaction_success
            )
            
            # ====================================================================
            # FINALIZE RESPONSE WITH COMPREHENSIVE METADATA
            # ====================================================================
            response.emotion_state = emotion_state
            response.response_time_ms = (time.time() - start_time) * 1000
            
            # Track costs
            if response.tokens_used > 0:
                cost = await cost_tracker.track_request(
                    provider=response.provider,
                    model=response.model_name,
                    input_tokens=len(message.split()) * 2,
                    output_tokens=response.tokens_used,
                    user_id=user_id,
                    category=category
                )
                response.cost = cost
            
            # Add Phase 2 metadata
            response.category = category
            
            # Add Phase 3 metadata - Context Info
            response.context_info = ContextInfo(
                recent_messages_count=len(recent_messages),
                relevant_messages_count=len(relevant_messages),
                has_context=len(recent_messages) > 0 or len(relevant_messages) > 0,
                retrieval_time_ms=context_time_ms
            )
            
            # Add Phase 3 metadata - Ability Info
            response.ability_info = AbilityInfo(
                ability_level=ability,
                recommended_difficulty=difficulty_level.value,
                cognitive_load=emotion_result.metrics.arousal,  # Using arousal as proxy
                flow_state_score=None  # Could be calculated from emotion trajectory
            )
            
            # Mark that ability was updated
            response.ability_updated = True
            
            # Add Phase 4 metadata - Processing breakdown
            response.processing_breakdown = {
                "context_retrieval_ms": context_time_ms,
                "emotion_detection_ms": emotion_time_ms,
                "difficulty_calculation_ms": difficulty_time_ms,
                "ai_generation_ms": ai_time_ms,
                "storage_ms": storage_time_ms,
                "total_ms": (time.time() - start_time) * 1000
            }
            
            total_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"✅ Request processed in {total_time_ms:.0f}ms "
                f"(context: {context_time_ms:.0f}ms, "
                f"emotion: {emotion_time_ms:.0f}ms, "
                f"difficulty: {difficulty_time_ms:.0f}ms, "
                f"AI: {ai_time_ms:.0f}ms, "
                f"storage: {storage_time_ms:.0f}ms)"
            )
            
            return response
        
        except Exception as e:
            logger.error(f"❌ Error processing request: {e}", exc_info=True)
            raise MasterXError(
                f"Failed to process request: {str(e)}",
                details={
                    'user_id': user_id,
                    'session_id': session_id,
                    'error': str(e)
                }
            )
    
    async def _process_basic(
        self,
        user_id: str,
        message: str,
        session_id: str,
        context: Optional[dict] = None
    ) -> AIResponse:
        """
        Fallback to basic processing if intelligence layer not initialized
        (Phase 1 mode for backwards compatibility)
        """
        logger.warning("Using basic processing mode (Phase 1 fallback)")
        
        # Basic emotion detection
        emotion_result = await self.emotion_engine.analyze_emotion(
            user_id=user_id,
            text=message,
            context=context
        )
        
        emotion_state = EmotionState(
            primary_emotion=emotion_result.metrics.primary_emotion,
            arousal=emotion_result.metrics.arousal,
            valence=emotion_result.metrics.valence,
            learning_readiness=LearningReadiness(emotion_result.metrics.learning_readiness)
        )
        
        # Basic provider selection
        category = self.provider_manager.detect_category_from_message(message, emotion_state)
        selected_provider = await self.provider_manager.select_best_provider_for_category(
            category, emotion_state
        )
        
        # Basic prompt enhancement
        enhanced_prompt = self._enhance_prompt_with_emotion(message, emotion_result)
        
        # Generate response
        response = await self.provider_manager.generate(
            prompt=enhanced_prompt,
            provider_name=selected_provider,
            max_tokens=1000
        )
        
        response.emotion_state = emotion_state
        
        return response
    
    def _enhance_prompt_with_emotion(self, message: str, emotion_result) -> str:
        """
        Enhance prompt with emotion context (Phase 1 basic version)
        """
        emotion_guidance = self._get_emotion_guidance(emotion_result)
        
        enhanced_prompt = f"""{emotion_guidance}

User message: {message}

Provide a helpful, clear, and supportive response."""
        
        return enhanced_prompt
    
    def _enhance_prompt_phase3(
        self,
        message: str,
        emotion_result,
        recent_messages: List[Message],
        relevant_messages: List[Message],
        difficulty_level,
        ability: float
    ) -> str:
        """
        Phase 3: Advanced prompt enhancement with context and difficulty
        
        Args:
            message: Current user message
            emotion_result: Emotion analysis result
            recent_messages: Recent conversation history
            relevant_messages: Semantically relevant past messages
            difficulty_level: Recommended difficulty level
            ability: Current ability estimate
        
        Returns:
            Enhanced prompt with full context
        """
        
        # Build conversation history
        history_text = ""
        if recent_messages:
            history_text = "Recent conversation:\n"
            for msg in recent_messages[-5:]:  # Last 5 messages
                role_label = "User" if msg.role == MessageRole.USER else "Assistant"
                history_text += f"{role_label}: {msg.content[:100]}...\n"
        
        # Build relevant context
        relevant_text = ""
        if relevant_messages:
            relevant_text = "\nRelevant past context:\n"
            for msg in relevant_messages[:2]:  # Top 2 relevant
                relevant_text += f"- {msg.content[:80]}...\n"
        
        # Get emotion guidance
        emotion_guidance = self._get_emotion_guidance(emotion_result)
        
        # Build difficulty guidance
        difficulty_guidance = f"""
Learner ability level: {ability:.2f} (0.0=beginner, 1.0=expert)
Recommended difficulty: {difficulty_level.label} ({difficulty_level.value:.2f})
{difficulty_level.explanation}

Adapt your response to match this difficulty level."""
        
        # Combine everything
        enhanced_prompt = f"""You are an adaptive AI tutor. Consider the following:

EMOTIONAL STATE:
{emotion_guidance}

LEARNER ABILITY:
{difficulty_guidance}

{history_text}

{relevant_text}

CURRENT QUESTION:
{message}

Provide a response that:
1. Matches the learner's ability level
2. Addresses their emotional state appropriately
3. Builds on previous conversation context
4. Is clear, supportive, and educational"""
        
        return enhanced_prompt
    
    def _infer_success_from_emotion(self, emotion_state: EmotionState) -> bool:
        """
        Infer interaction success from emotional state
        
        Uses multiple signals:
        1. Primary emotion (positive vs negative)
        2. Valence (positive vs negative affect)
        3. Learning readiness (indicator of engagement)
        
        In full system, this would be based on correctness checks.
        
        Args:
            emotion_state: Current emotional state
        
        Returns:
            True if interaction seems successful
        """
        positive_emotions = [
            'joy', 'achievement', 'flow_state', 'engagement',
            'curiosity', 'confidence', 'satisfaction', 'excitement'
        ]
        
        negative_emotions = [
            'frustration', 'confusion', 'anxiety', 'boredom',
            'overwhelmed', 'disappointment'
        ]
        
        # Multi-signal approach
        emotion_positive = emotion_state.primary_emotion in positive_emotions
        emotion_negative = emotion_state.primary_emotion in negative_emotions
        
        # Valence: > 0.55 = positive, < 0.45 = negative
        valence_positive = emotion_state.valence > 0.55
        valence_negative = emotion_state.valence < 0.45
        
        # Readiness: high/moderate = engaged, low = struggling
        readiness_engaged = emotion_state.learning_readiness in [
            LearningReadiness.HIGH_READINESS,
            LearningReadiness.MODERATE_READINESS
        ]
        
        # Weighted decision (prioritize valence and readiness over emotion label)
        # This helps when emotion detection is inaccurate
        if emotion_negative or valence_negative:
            return False  # Clearly struggling
        elif emotion_positive and valence_positive and readiness_engaged:
            return True  # Clearly succeeding
        elif valence_positive or readiness_engaged:
            return True  # Probably succeeding (valence/readiness are more reliable)
        else:
            # Default: neutral/uncertain
            # Use valence as tiebreaker (0.5 = neutral)
            return emotion_state.valence >= 0.5
    
    def _get_emotion_guidance(self, emotion_result) -> str:
        """Get guidance for AI based on detected emotion"""
        
        emotion = emotion_result.metrics.primary_emotion
        readiness = emotion_result.metrics.learning_readiness
        
        # Emotion-specific guidance
        if emotion in ['frustration', 'anxiety', 'overwhelmed']:
            return """The learner is experiencing some frustration or anxiety. Please:
- Be extra patient and encouraging
- Break down concepts into smaller steps
- Validate their feelings
- Offer alternative explanations if needed"""
        
        elif emotion in ['confusion', 'uncertainty']:
            return """The learner seems confused. Please:
- Clarify concepts with clear examples
- Use analogies to explain difficult ideas
- Check for understanding
- Offer to explain differently"""
        
        elif emotion in ['joy', 'achievement', 'flow_state']:
            return """The learner is engaged and doing well! Please:
- Reinforce their progress
- Challenge them appropriately
- Maintain the momentum
- Celebrate their understanding"""
        
        elif emotion in ['boredom', 'disengagement']:
            return """The learner might be disengaged. Please:
- Make the content more interesting
- Use engaging examples
- Connect to real-world applications
- Increase the challenge slightly"""
        
        else:
            return """Provide a clear, helpful, and encouraging response."""
    
    def get_available_providers(self):
        """Get list of available AI providers"""
        return self.provider_manager.get_available_providers()
