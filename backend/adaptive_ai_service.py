"""
Adaptive AI Service for MasterX - Personalized Learning Assistant

This service integrates with the personalization engine to provide
highly personalized and adaptive AI responses based on learning DNA,
mood analysis, and real-time context awareness.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
import os
from dataclasses import asdict

# Third-party imports
import groq
from groq import AsyncGroq

# Local imports
from database import db_service
from models import ChatSession, ChatMessage, MentorResponse
from personalization_engine import (
    personalization_engine, 
    LearningDNA, 
    AdaptiveContentParameters, 
    MoodBasedAdaptation,
    LearningStyle,
    EmotionalState,
    LearningPace
)

logger = logging.getLogger(__name__)

class AdaptiveAIService:
    """Adaptive AI service with personalization"""
    
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise ValueError("GROQ_API_KEY is required")
        
        self.client = AsyncGroq(api_key=self.groq_api_key)
        self.model = "deepseek-r1"  # Using DeepSeek R1 for reasoning capabilities
        
        # Personality templates for different moods and learning styles
        self._initialize_personality_templates()
        
        logger.info("Adaptive AI Service initialized successfully")
    
    def _initialize_personality_templates(self):
        """Initialize personality templates for different contexts"""
        
        self.base_personality = """You are MasterX, an advanced AI mentor with deep personalization capabilities. 
        You adapt your teaching style, content complexity, and interaction approach based on each learner's unique profile.
        
        Core Principles:
        - Personalize every interaction based on learning DNA
        - Adapt to emotional state and energy levels
        - Provide content at optimal complexity and pace
        - Use preferred learning modalities
        - Maintain encouraging and supportive tone
        - Foster deep understanding and critical thinking
        """
        
        self.mood_adaptations = {
            EmotionalState.EXCITED: {
                "tone": "enthusiastic and energetic",
                "approach": "leverage enthusiasm with challenging content",
                "pacing": "maintain high energy while ensuring comprehension"
            },
            EmotionalState.FRUSTRATED: {
                "tone": "patient and reassuring",
                "approach": "break down complex concepts into smaller steps",
                "pacing": "slow down and provide extra support"
            },
            EmotionalState.STRESSED: {
                "tone": "calm and supportive",
                "approach": "simplify content and reduce cognitive load",
                "pacing": "very gentle pacing with frequent check-ins"
            },
            EmotionalState.CONFIDENT: {
                "tone": "encouraging yet challenging",
                "approach": "introduce advanced concepts and applications",
                "pacing": "maintain momentum with progressive difficulty"
            },
            EmotionalState.CURIOUS: {
                "tone": "explorative and engaging",
                "approach": "provide deep explanations and related topics",
                "pacing": "follow curiosity while maintaining structure"
            },
            EmotionalState.OVERWHELMED: {
                "tone": "very supportive and patient",
                "approach": "focus on one concept at a time",
                "pacing": "extremely gentle with lots of encouragement"
            }
        }
        
        self.learning_style_adaptations = {
            LearningStyle.VISUAL: {
                "content_format": "use visual metaphors, diagrams, and structured layouts",
                "examples": "provide visual examples and imagery",
                "organization": "use clear headings, bullet points, and visual hierarchy"
            },
            LearningStyle.AUDITORY: {
                "content_format": "use conversational tone and audio-friendly explanations",
                "examples": "use verbal analogies and sound-based examples",
                "organization": "use rhythmic patterns and verbal cues"
            },
            LearningStyle.KINESTHETIC: {
                "content_format": "provide hands-on exercises and practical applications",
                "examples": "use real-world scenarios and action-based examples",
                "organization": "break into actionable steps and activities"
            },
            LearningStyle.READING_WRITING: {
                "content_format": "provide detailed written explanations",
                "examples": "use text-based examples and written exercises",
                "organization": "use structured text with clear sections"
            },
            LearningStyle.MULTIMODAL: {
                "content_format": "combine multiple formats for rich experience",
                "examples": "use varied examples across different modalities",
                "organization": "flexible structure adapting to topic needs"
            }
        }
    
    async def get_personalized_response(
        self,
        user_message: str,
        session: ChatSession,
        context: Dict[str, Any] = None,
        stream: bool = False
    ) -> MentorResponse:
        """Get personalized AI response based on user's learning DNA and current state"""
        
        try:
            # Get recent messages for context
            recent_messages = await db_service.get_recent_messages(session.id, limit=10)
            
            # Analyze learning DNA
            learning_dna = await personalization_engine.analyze_learning_dna(session.user_id)
            
            # Analyze mood and get adaptations
            mood_adaptation = await personalization_engine.analyze_mood_and_adapt(
                session.user_id, recent_messages, context
            )
            
            # Get adaptive content parameters
            content_params = await personalization_engine.get_adaptive_content_parameters(
                session.user_id, context
            )
            
            # Build personalized prompt
            personalized_prompt = await self._build_personalized_prompt(
                user_message, session, learning_dna, mood_adaptation, content_params, recent_messages
            )
            
            # Get AI response
            if stream:
                return await self._get_streaming_response(personalized_prompt, learning_dna, mood_adaptation)
            else:
                return await self._get_complete_response(personalized_prompt, learning_dna, mood_adaptation)
                
        except Exception as e:
            logger.error(f"Error getting personalized response: {str(e)}")
            return self._create_fallback_response()
    
    async def _build_personalized_prompt(
        self,
        user_message: str,
        session: ChatSession,
        learning_dna: LearningDNA,
        mood_adaptation: MoodBasedAdaptation,
        content_params: AdaptiveContentParameters,
        recent_messages: List[ChatMessage]
    ) -> str:
        """Build a personalized prompt based on all available data"""
        
        # Base context
        context_info = f"""
LEARNER PROFILE ANALYSIS:
=======================
User ID: {learning_dna.user_id}
Learning Style: {learning_dna.learning_style.value}
Cognitive Patterns: {', '.join([cp.value for cp in learning_dna.cognitive_patterns])}
Preferred Pace: {learning_dna.preferred_pace.value}
Motivation Style: {learning_dna.motivation_style.value}
Attention Span: {learning_dna.attention_span_minutes} minutes
Difficulty Preference: {learning_dna.difficulty_preference:.1f}/1.0
Interaction Level: {learning_dna.interaction_preference}
Feedback Preference: {learning_dna.feedback_preference}

CURRENT STATE ANALYSIS:
======================
Detected Mood: {mood_adaptation.detected_mood.value}
Energy Level: {mood_adaptation.energy_level:.1f}/1.0
Stress Level: {mood_adaptation.stress_level:.1f}/1.0
Recommended Pace: {mood_adaptation.recommended_pace.value}
Content Tone: {mood_adaptation.content_tone}
Interaction Style: {mood_adaptation.interaction_style}

ADVANCED METRICS:
================
Learning Velocity: {learning_dna.learning_velocity:.1f} concepts/hour
Curiosity Index: {learning_dna.curiosity_index:.1f}/1.0
Perseverance Score: {learning_dna.perseverance_score:.1f}/1.0
Metacognitive Awareness: {learning_dna.metacognitive_awareness:.1f}/1.0
Concept Retention Rate: {learning_dna.concept_retention_rate:.1f}/1.0

CONTENT ADAPTATION PARAMETERS:
=============================
Complexity Level: {content_params.complexity_level:.1f}/1.0
Explanation Depth: {content_params.explanation_depth}
Example Count: {content_params.example_count}
Visual Elements: {content_params.visual_elements}
Interactive Elements: {content_params.interactive_elements}
Reinforcement Frequency: {content_params.reinforcement_frequency:.1f}/1.0

MOTIVATION TRIGGERS:
==================
{', '.join(learning_dna.motivation_triggers) if learning_dna.motivation_triggers else 'Standard encouragement'}

LEARNING BLOCKERS TO AVOID:
===========================
{', '.join(learning_dna.learning_blockers) if learning_dna.learning_blockers else 'None identified'}

CURRENT SESSION CONTEXT:
=======================
Subject: {session.subject or 'General Learning'}
Current Topic: {session.current_topic or 'Not specified'}
Learning Objectives: {', '.join(session.learning_objectives) if session.learning_objectives else 'Not specified'}
"""
        
        # Add recent context
        if recent_messages:
            context_info += f"\nRECENT CONVERSATION:\n"
            for msg in recent_messages[-5:]:  # Last 5 messages
                role = "User" if msg.sender == "user" else "Assistant"
                context_info += f"{role}: {msg.message[:200]}{'...' if len(msg.message) > 200 else ''}\n"
        
        # Get personality adaptation
        mood_adaptation_text = self.mood_adaptations.get(mood_adaptation.detected_mood, {})
        style_adaptation_text = self.learning_style_adaptations.get(learning_dna.learning_style, {})
        
        # Build the complete prompt
        full_prompt = f"""{self.base_personality}

{context_info}

PERSONALIZATION INSTRUCTIONS:
============================
1. Mood Adaptation:
   - Tone: {mood_adaptation_text.get('tone', 'neutral and supportive')}
   - Approach: {mood_adaptation_text.get('approach', 'standard teaching approach')}
   - Pacing: {mood_adaptation_text.get('pacing', 'moderate pacing')}

2. Learning Style Adaptation:
   - Content Format: {style_adaptation_text.get('content_format', 'clear and structured')}
   - Examples: {style_adaptation_text.get('examples', 'relevant examples')}
   - Organization: {style_adaptation_text.get('organization', 'logical structure')}

3. Content Complexity:
   - Adjust explanation complexity to {content_params.complexity_level:.1f}/1.0 level
   - Provide {content_params.explanation_depth} explanations
   - Include {content_params.example_count} relevant examples
   - {'Include visual elements' if content_params.visual_elements else 'Focus on text-based content'}
   - {'Make it interactive' if content_params.interactive_elements else 'Provide clear exposition'}

4. Special Considerations:
   - Current energy level is {mood_adaptation.energy_level:.1f}/1.0 - adjust accordingly
   - Stress level is {mood_adaptation.stress_level:.1f}/1.0 - {'be extra supportive' if mood_adaptation.stress_level > 0.6 else 'maintain normal supportiveness'}
   - Attention span is {learning_dna.attention_span_minutes} minutes - keep responses appropriately sized
   - {'Suggest a break soon' if mood_adaptation.break_recommendation else 'Continue with learning flow'}

5. Response Requirements:
   - Use {mood_adaptation.content_tone} tone throughout
   - Apply {mood_adaptation.interaction_style} interaction style
   - Incorporate motivation triggers: {', '.join(learning_dna.motivation_triggers)}
   - Avoid learning blockers: {', '.join(learning_dna.learning_blockers)}

CURRENT USER MESSAGE:
====================
{user_message}

Please provide a highly personalized response that adapts to all the above factors. Focus on the user's specific learning needs, current emotional state, and optimal learning parameters. Make this interaction feel tailored specifically for this individual learner.
"""
        
        return full_prompt
    
    async def _get_complete_response(
        self,
        prompt: str,
        learning_dna: LearningDNA,
        mood_adaptation: MoodBasedAdaptation
    ) -> MentorResponse:
        """Get complete AI response"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Extract structured information from response
            formatted_response = self._format_personalized_response(content, learning_dna, mood_adaptation)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return self._create_fallback_response()
    
    async def _get_streaming_response(
        self,
        prompt: str,
        learning_dna: LearningDNA,
        mood_adaptation: MoodBasedAdaptation
    ) -> AsyncGenerator:
        """Get streaming AI response with adaptive pacing"""
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            # Apply adaptive pacing based on learning DNA
            pacing_delay = self._calculate_adaptive_pacing(learning_dna, mood_adaptation)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk
                    # Adaptive delay between chunks
                    if pacing_delay > 0:
                        await asyncio.sleep(pacing_delay)
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            # Yield error response
            error_response = self._create_fallback_response()
            yield error_response
    
    def _calculate_adaptive_pacing(self, learning_dna: LearningDNA, mood_adaptation: MoodBasedAdaptation) -> float:
        """Calculate adaptive pacing delay between response chunks"""
        
        base_delay = 0.05  # 50ms base delay
        
        # Adjust for learning pace preference
        if learning_dna.preferred_pace == LearningPace.SLOW_DEEP:
            base_delay *= 2.0
        elif learning_dna.preferred_pace == LearningPace.FAST_OVERVIEW:
            base_delay *= 0.5
        
        # Adjust for current mood and energy
        if mood_adaptation.detected_mood in [EmotionalState.STRESSED, EmotionalState.OVERWHELMED]:
            base_delay *= 1.5
        elif mood_adaptation.detected_mood == EmotionalState.EXCITED:
            base_delay *= 0.7
        
        # Adjust for energy level
        energy_factor = 1.0 - (mood_adaptation.energy_level - 0.5) * 0.5
        base_delay *= energy_factor
        
        return max(0.01, min(0.2, base_delay))  # Clamp between 10ms and 200ms
    
    def _format_personalized_response(
        self,
        content: str,
        learning_dna: LearningDNA,
        mood_adaptation: MoodBasedAdaptation
    ) -> MentorResponse:
        """Format the AI response with personalization metadata"""
        
        # Extract suggested actions (simplified)
        suggested_actions = []
        if "practice" in content.lower():
            suggested_actions.append("Try a practice exercise")
        if "example" in content.lower():
            suggested_actions.append("Review more examples")
        if "question" in content.lower():
            suggested_actions.append("Ask follow-up questions")
        
        # Extract concepts covered (simplified)
        concepts_covered = []
        # This would be more sophisticated in production
        
        # Determine next steps based on learning DNA
        next_steps = self._generate_personalized_next_steps(learning_dna, mood_adaptation)
        
        # Create comprehensive metadata
        metadata = {
            "personalization": {
                "learning_dna_applied": learning_dna.to_dict(),
                "mood_adaptation_applied": mood_adaptation.to_dict(),
                "personalization_confidence": learning_dna.confidence_score,
                "adaptation_reason": f"Adapted for {learning_dna.learning_style.value} learner in {mood_adaptation.detected_mood.value} mood"
            },
            "content_parameters": {
                "complexity_level": learning_dna.difficulty_preference,
                "explanation_depth": "detailed",  # This would be dynamic
                "learning_style_optimization": learning_dna.learning_style.value,
                "mood_optimization": mood_adaptation.detected_mood.value
            },
            "adaptive_features": {
                "pacing_adjusted": True,
                "tone_adapted": True,
                "complexity_optimized": True,
                "interaction_style_applied": True
            },
            "learning_analytics": {
                "predicted_engagement": self._predict_engagement(learning_dna, mood_adaptation),
                "expected_retention": learning_dna.concept_retention_rate,
                "difficulty_match": abs(learning_dna.difficulty_preference - 0.6) < 0.2
            }
        }
        
        return MentorResponse(
            response=content,
            response_type="personalized_explanation",
            suggested_actions=suggested_actions,
            concepts_covered=concepts_covered,
            next_steps=next_steps,
            metadata=metadata
        )
    
    def _generate_personalized_next_steps(self, learning_dna: LearningDNA, mood_adaptation: MoodBasedAdaptation) -> str:
        """Generate personalized next steps"""
        
        next_steps = []
        
        # Based on learning style
        if learning_dna.learning_style == LearningStyle.KINESTHETIC:
            next_steps.append("Try a hands-on exercise to reinforce this concept")
        elif learning_dna.learning_style == LearningStyle.VISUAL:
            next_steps.append("Create a visual diagram or mind map of this concept")
        
        # Based on mood
        if mood_adaptation.detected_mood == EmotionalState.CURIOUS:
            next_steps.append("Explore related topics that might interest you")
        elif mood_adaptation.detected_mood in [EmotionalState.STRESSED, EmotionalState.OVERWHELMED]:
            next_steps.append("Take a short break, then review the key points")
        
        # Based on energy level
        if mood_adaptation.energy_level > 0.8:
            next_steps.append("You seem energized - ready for the next challenging topic?")
        elif mood_adaptation.energy_level < 0.4:
            next_steps.append("Consider reviewing this material again when you're more refreshed")
        
        # Default next step
        if not next_steps:
            next_steps.append("Continue with the next concept when you're ready")
        
        return " | ".join(next_steps)
    
    def _predict_engagement(self, learning_dna: LearningDNA, mood_adaptation: MoodBasedAdaptation) -> float:
        """Predict user engagement based on personalization match"""
        
        base_engagement = 0.7
        
        # Boost for good mood-content match
        if mood_adaptation.detected_mood in [EmotionalState.CURIOUS, EmotionalState.EXCITED]:
            base_engagement += 0.2
        elif mood_adaptation.detected_mood in [EmotionalState.FRUSTRATED, EmotionalState.OVERWHELMED]:
            base_engagement -= 0.1
        
        # Boost for high personalization confidence
        base_engagement += learning_dna.confidence_score * 0.2
        
        # Boost for good energy level
        base_engagement += (mood_adaptation.energy_level - 0.5) * 0.2
        
        return max(0.1, min(1.0, base_engagement))
    
    def _create_fallback_response(self) -> MentorResponse:
        """Create fallback response when personalization fails"""
        
        return MentorResponse(
            response="I'm here to help you learn! Let me know what specific topic you'd like to explore, and I'll provide personalized guidance based on your learning style and current needs.",
            response_type="fallback",
            suggested_actions=["Ask a specific question", "Choose a topic to explore"],
            concepts_covered=[],
            next_steps="Please share what you'd like to learn about",
            metadata={
                "personalization": {
                    "status": "fallback",
                    "reason": "Unable to load personalization data"
                }
            }
        )

# Global adaptive AI service instance
adaptive_ai_service = AdaptiveAIService()