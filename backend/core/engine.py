"""
MasterX Engine - Main Orchestrator
Following specifications from 5.DEVELOPMENT_HANDOFF_GUIDE.md

Phase 1: Simplified orchestrator
- Process user message
- Analyze emotion
- Generate AI response
- Return combined result
"""

import logging
import time
from typing import Optional
from core.models import AIResponse, EmotionState, LearningReadiness
from core.ai_providers import ProviderManager
from services.emotion.emotion_engine import EmotionEngine
from utils.errors import MasterXError
from utils.cost_tracker import cost_tracker

logger = logging.getLogger(__name__)


class MasterXEngine:
    """
    Main orchestrator for MasterX learning platform
    Phase 1: Simple emotion detection + AI response
    Phase 2: Add context management and adaptive learning
    """
    
    def __init__(self):
        self.provider_manager = ProviderManager()
        self.emotion_engine = EmotionEngine()
        logger.info("✅ MasterXEngine initialized")
    
    async def process_request(
        self,
        user_id: str,
        message: str,
        session_id: str,
        context: Optional[dict] = None
    ) -> AIResponse:
        """
        Process user request and generate AI response
        
        Phase 1 Flow:
        1. Analyze emotion (fast, < 100ms)
        2. Generate AI response
        3. Combine emotion + response
        4. Return result
        """
        
        start_time = time.time()
        
        try:
            # Phase 1: Analyze emotion
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
            
            # Phase 2: Generate AI response with smart routing
            logger.info(f"🤖 Generating AI response...")
            ai_start = time.time()
            
            # Enhance prompt with emotion context
            enhanced_prompt = self._enhance_prompt_with_emotion(message, emotion_result)
            
            # Try smart routing first (Phase 2), fall back to simple if not available
            if hasattr(self.provider_manager, '_smart_routing_enabled') and self.provider_manager._smart_routing_enabled:
                response = await self.provider_manager.generate_with_smart_routing(
                    message=enhanced_prompt,
                    emotion_state=emotion_state,
                    session_id=session_id,
                    max_tokens=1000
                )
            else:
                response = await self.provider_manager.generate(
                    prompt=enhanced_prompt,
                    max_tokens=1000
                )
            
            ai_time_ms = (time.time() - ai_start) * 1000
            logger.info(f"✅ AI response generated ({ai_time_ms:.0f}ms)")
            
            # Phase 3: Enhance response with emotion info
            response.emotion_state = emotion_state
            response.response_time_ms = (time.time() - start_time) * 1000
            
            # Track costs
            if response.tokens_used > 0:
                cost = await cost_tracker.track_request(
                    provider=response.provider,
                    model=response.model_name,
                    input_tokens=len(message.split()) * 2,  # Rough estimate
                    output_tokens=response.tokens_used,
                    user_id=user_id,
                    category="general"
                )
                response.cost = cost
            
            total_time_ms = (time.time() - start_time) * 1000
            logger.info(f"✅ Request processed in {total_time_ms:.0f}ms (emotion: {emotion_time_ms:.0f}ms, AI: {ai_time_ms:.0f}ms)")
            
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
    
    def _enhance_prompt_with_emotion(self, message: str, emotion_result) -> str:
        """
        Enhance prompt with emotion context
        Phase 1: Simple emotion-aware prompting
        Phase 2: Advanced context and difficulty adaptation
        """
        
        # Get emotion-specific guidance
        emotion_guidance = self._get_emotion_guidance(emotion_result)
        
        enhanced_prompt = f"""{emotion_guidance}

User message: {message}

Provide a helpful, clear, and supportive response."""
        
        return enhanced_prompt
    
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
