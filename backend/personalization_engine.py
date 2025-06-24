"""
Advanced Personalization Engine for MasterX AI Mentor System

This module implements sophisticated learning DNA profiling, adaptive content generation,
and mood-based learning adaptation for a truly personalized learning experience.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import uuid

# Local imports
from database import db_service
from models import User, ChatSession, ChatMessage

logger = logging.getLogger(__name__)

class LearningStyle(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"
    MULTIMODAL = "multimodal"

class CognitivePattern(Enum):
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    CONCRETE = "concrete"
    ABSTRACT = "abstract"
    ANALYTICAL = "analytical"
    HOLISTIC = "holistic"

class LearningPace(Enum):
    SLOW_DEEP = "slow_deep"
    MODERATE = "moderate"
    FAST_OVERVIEW = "fast_overview"
    ADAPTIVE = "adaptive"

class MotivationStyle(Enum):
    ACHIEVEMENT = "achievement"
    SOCIAL = "social"
    AUTONOMY = "autonomy"
    MASTERY = "mastery"
    PURPOSE = "purpose"

class EmotionalState(Enum):
    EXCITED = "excited"
    FOCUSED = "focused"
    CALM = "calm"
    STRESSED = "stressed"
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"
    CURIOUS = "curious"
    OVERWHELMED = "overwhelmed"

@dataclass
class LearningDNA:
    """Comprehensive learning DNA profile"""
    user_id: str
    learning_style: LearningStyle
    cognitive_patterns: List[CognitivePattern]
    preferred_pace: LearningPace
    motivation_style: MotivationStyle
    attention_span_minutes: int
    difficulty_preference: float  # 0.0 to 1.0
    interaction_preference: str  # "high", "medium", "low"
    feedback_preference: str  # "immediate", "periodic", "milestone"
    
    # Behavioral patterns
    session_frequency: float  # sessions per week
    optimal_session_length: int  # minutes
    peak_performance_hours: List[int]  # hours of day (0-23)
    concept_retention_rate: float  # 0.0 to 1.0
    knowledge_transfer_ability: float  # 0.0 to 1.0
    
    # Advanced metrics
    learning_velocity: float  # concepts per hour
    curiosity_index: float  # 0.0 to 1.0
    perseverance_score: float  # 0.0 to 1.0
    collaboration_preference: float  # 0.0 to 1.0
    metacognitive_awareness: float  # 0.0 to 1.0
    
    # Dynamic factors
    current_energy_level: float  # 0.0 to 1.0
    stress_indicators: List[str]
    motivation_triggers: List[str]
    learning_blockers: List[str]
    
    # Temporal data
    last_updated: datetime
    confidence_score: float  # 0.0 to 1.0 - how confident we are in this profile
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert enums to strings
        data['learning_style'] = self.learning_style.value
        data['cognitive_patterns'] = [cp.value for cp in self.cognitive_patterns]
        data['preferred_pace'] = self.preferred_pace.value
        data['motivation_style'] = self.motivation_style.value
        data['last_updated'] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningDNA':
        """Create from dictionary"""
        data['learning_style'] = LearningStyle(data['learning_style'])
        data['cognitive_patterns'] = [CognitivePattern(cp) for cp in data['cognitive_patterns']]
        data['preferred_pace'] = LearningPace(data['preferred_pace'])
        data['motivation_style'] = MotivationStyle(data['motivation_style'])
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

@dataclass
class AdaptiveContentParameters:
    """Parameters for adaptive content generation"""
    complexity_level: float  # 0.0 to 1.0
    explanation_depth: str  # "overview", "detailed", "comprehensive"
    example_count: int
    visual_elements: bool
    interactive_elements: bool
    pacing_delay: float  # seconds between concepts
    reinforcement_frequency: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MoodBasedAdaptation:
    """Mood-based learning adaptations"""
    detected_mood: EmotionalState
    confidence: float  # 0.0 to 1.0
    energy_level: float  # 0.0 to 1.0
    stress_level: float  # 0.0 to 1.0
    recommended_pace: LearningPace
    content_tone: str  # "encouraging", "neutral", "challenging"
    interaction_style: str  # "supportive", "standard", "challenging"
    break_recommendation: Optional[int]  # minutes until suggested break
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['detected_mood'] = self.detected_mood.value
        data['recommended_pace'] = self.recommended_pace.value
        return data

class PersonalizationEngine:
    """Advanced personalization engine for MasterX"""
    
    def __init__(self):
        self.learning_dna_cache = {}  # user_id -> LearningDNA
        self.session_analytics = defaultdict(list)  # user_id -> [session_data]
        self.interaction_patterns = defaultdict(deque)  # user_id -> deque of interactions
        self.mood_history = defaultdict(deque)  # user_id -> deque of mood data
        self.content_effectiveness = defaultdict(dict)  # user_id -> content_type -> effectiveness
        
        # Initialize pattern recognition models
        self._initialize_pattern_models()
    
    def _initialize_pattern_models(self):
        """Initialize pattern recognition models"""
        # Simple pattern recognition - in production, use ML models
        self.attention_patterns = {}
        self.learning_velocity_models = {}
        self.mood_detection_models = {}
        
        logger.info("Personalization engine initialized")
    
    async def analyze_learning_dna(self, user_id: str) -> LearningDNA:
        """Analyze user's learning DNA from historical data"""
        try:
            # Check cache first
            if user_id in self.learning_dna_cache:
                cached_dna = self.learning_dna_cache[user_id]
                if (datetime.now() - cached_dna.last_updated).total_seconds() < 3600:  # 1 hour cache
                    return cached_dna
            
            # Get user's historical data
            user = await db_service.get_user(user_id)
            if not user:
                return self._create_default_dna(user_id)
            
            sessions = await db_service.get_user_sessions(user_id, active_only=False)
            messages = []
            
            # Collect all messages from recent sessions
            for session in sessions[-20:]:  # Last 20 sessions
                session_messages = await db_service.get_session_messages(session.id, limit=100)
                messages.extend(session_messages)
            
            # Analyze patterns
            learning_dna = await self._analyze_interaction_patterns(user_id, sessions, messages)
            
            # Cache the result
            self.learning_dna_cache[user_id] = learning_dna
            
            # Store in database
            await self._store_learning_dna(learning_dna)
            
            return learning_dna
            
        except Exception as e:
            logger.error(f"Error analyzing learning DNA for user {user_id}: {str(e)}")
            return self._create_default_dna(user_id)
    
    async def _analyze_interaction_patterns(
        self, 
        user_id: str, 
        sessions: List[ChatSession], 
        messages: List[ChatMessage]
    ) -> LearningDNA:
        """Analyze user interaction patterns to build learning DNA"""
        
        # Analyze session patterns
        session_patterns = self._analyze_session_patterns(sessions)
        
        # Analyze message patterns
        message_patterns = self._analyze_message_patterns(messages)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(sessions, messages)
        
        # Analyze content preferences
        content_patterns = self._analyze_content_preferences(messages)
        
        # Analyze performance patterns
        performance_patterns = self._analyze_performance_patterns(sessions, messages)
        
        # Determine learning style
        learning_style = self._determine_learning_style(message_patterns, content_patterns)
        
        # Determine cognitive patterns
        cognitive_patterns = self._determine_cognitive_patterns(message_patterns, session_patterns)
        
        # Determine preferred pace
        preferred_pace = self._determine_preferred_pace(temporal_patterns, session_patterns)
        
        # Determine motivation style
        motivation_style = self._determine_motivation_style(session_patterns, message_patterns)
        
        # Calculate advanced metrics
        learning_velocity = performance_patterns.get('learning_velocity', 0.5)
        curiosity_index = message_patterns.get('question_ratio', 0.3)
        perseverance_score = session_patterns.get('completion_rate', 0.7)
        
        # Determine current energy and stress levels
        current_energy = temporal_patterns.get('recent_energy', 0.7)
        stress_indicators = self._detect_stress_indicators(messages[-10:])
        
        # Create learning DNA
        learning_dna = LearningDNA(
            user_id=user_id,
            learning_style=learning_style,
            cognitive_patterns=cognitive_patterns,
            preferred_pace=preferred_pace,
            motivation_style=motivation_style,
            attention_span_minutes=session_patterns.get('avg_session_length', 30),
            difficulty_preference=performance_patterns.get('difficulty_preference', 0.6),
            interaction_preference=message_patterns.get('interaction_level', 'medium'),
            feedback_preference=message_patterns.get('feedback_preference', 'immediate'),
            session_frequency=temporal_patterns.get('sessions_per_week', 3.0),
            optimal_session_length=session_patterns.get('optimal_length', 25),
            peak_performance_hours=temporal_patterns.get('peak_hours', [9, 14, 19]),
            concept_retention_rate=performance_patterns.get('retention_rate', 0.7),
            knowledge_transfer_ability=performance_patterns.get('transfer_ability', 0.6),
            learning_velocity=learning_velocity,
            curiosity_index=curiosity_index,
            perseverance_score=perseverance_score,
            collaboration_preference=session_patterns.get('collaboration_pref', 0.5),
            metacognitive_awareness=message_patterns.get('metacognitive_score', 0.6),
            current_energy_level=current_energy,
            stress_indicators=stress_indicators,
            motivation_triggers=self._identify_motivation_triggers(messages),
            learning_blockers=self._identify_learning_blockers(messages),
            last_updated=datetime.now(),
            confidence_score=self._calculate_confidence_score(len(sessions), len(messages))
        )
        
        return learning_dna
    
    def _analyze_session_patterns(self, sessions: List[ChatSession]) -> Dict[str, Any]:
        """Analyze session patterns"""
        if not sessions:
            return {}
        
        # Calculate session statistics
        session_lengths = []
        completion_rates = []
        
        for session in sessions:
            # Estimate session length from creation to last update
            if session.updated_at and session.created_at:
                length = (session.updated_at - session.created_at).total_seconds() / 60
                session_lengths.append(length)
                
                # Estimate completion rate (simplified)
                completion_rate = 1.0 if session.is_active else 0.8
                completion_rates.append(completion_rate)
        
        avg_length = sum(session_lengths) / len(session_lengths) if session_lengths else 30
        avg_completion = sum(completion_rates) / len(completion_rates) if completion_rates else 0.7
        
        return {
            'avg_session_length': avg_length,
            'optimal_length': min(45, max(15, avg_length)),
            'completion_rate': avg_completion,
            'total_sessions': len(sessions),
            'collaboration_pref': 0.5  # Default, would need group data
        }
    
    def _analyze_message_patterns(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Analyze message patterns"""
        if not messages:
            return {}
        
        user_messages = [m for m in messages if m.sender == 'user']
        mentor_messages = [m for m in messages if m.sender == 'mentor']
        
        # Analyze question patterns
        question_count = sum(1 for m in user_messages if '?' in m.message)
        question_ratio = question_count / len(user_messages) if user_messages else 0
        
        # Analyze message lengths
        avg_user_message_length = sum(len(m.message.split()) for m in user_messages) / len(user_messages) if user_messages else 0
        
        # Analyze interaction level
        interaction_level = 'high' if len(user_messages) > len(mentor_messages) * 0.8 else 'medium'
        
        # Analyze feedback preference (simplified)
        feedback_words = ['thanks', 'good', 'helpful', 'more', 'explain', 'clarify']
        feedback_count = sum(1 for m in user_messages if any(word in m.message.lower() for word in feedback_words))
        feedback_preference = 'immediate' if feedback_count > len(user_messages) * 0.3 else 'periodic'
        
        # Analyze metacognitive indicators
        metacognitive_words = ['understand', 'confused', 'think', 'realize', 'strategy', 'approach']
        metacognitive_count = sum(1 for m in user_messages if any(word in m.message.lower() for word in metacognitive_words))
        metacognitive_score = metacognitive_count / len(user_messages) if user_messages else 0
        
        return {
            'question_ratio': question_ratio,
            'avg_message_length': avg_user_message_length,
            'interaction_level': interaction_level,
            'feedback_preference': feedback_preference,
            'metacognitive_score': min(1.0, metacognitive_score),
            'total_messages': len(user_messages)
        }
    
    def _analyze_temporal_patterns(self, sessions: List[ChatSession], messages: List[ChatMessage]) -> Dict[str, Any]:
        """Analyze temporal patterns"""
        if not sessions:
            return {'sessions_per_week': 3.0, 'peak_hours': [9, 14, 19], 'recent_energy': 0.7}
        
        # Calculate sessions per week
        if len(sessions) > 1:
            days_span = (sessions[-1].created_at - sessions[0].created_at).days
            sessions_per_week = len(sessions) * 7 / max(1, days_span)
        else:
            sessions_per_week = 1.0
        
        # Analyze peak hours
        hours = [session.created_at.hour for session in sessions]
        peak_hours = sorted(set(hours), key=hours.count, reverse=True)[:3]
        
        # Recent energy level (based on recent session frequency)
        recent_sessions = [s for s in sessions if (datetime.now() - s.created_at).days <= 7]
        recent_energy = min(1.0, len(recent_sessions) / 7.0 * 2)
        
        return {
            'sessions_per_week': sessions_per_week,
            'peak_hours': peak_hours,
            'recent_energy': recent_energy
        }
    
    def _analyze_content_preferences(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Analyze content preferences"""
        if not messages:
            return {}
        
        # Look for preferences in message metadata and content
        visual_indicators = ['diagram', 'chart', 'image', 'visual', 'picture', 'graph']
        audio_indicators = ['explain', 'tell', 'describe', 'say']
        hands_on_indicators = ['practice', 'try', 'do', 'exercise', 'example']
        
        user_messages = [m for m in messages if m.sender == 'user']
        
        visual_count = sum(1 for m in user_messages if any(word in m.message.lower() for word in visual_indicators))
        audio_count = sum(1 for m in user_messages if any(word in m.message.lower() for word in audio_indicators))
        hands_on_count = sum(1 for m in user_messages if any(word in m.message.lower() for word in hands_on_indicators))
        
        return {
            'visual_preference': visual_count / len(user_messages) if user_messages else 0,
            'audio_preference': audio_count / len(user_messages) if user_messages else 0,
            'hands_on_preference': hands_on_count / len(user_messages) if user_messages else 0
        }
    
    def _analyze_performance_patterns(self, sessions: List[ChatSession], messages: List[ChatMessage]) -> Dict[str, Any]:
        """Analyze performance patterns"""
        # Simplified performance analysis
        # In production, this would use exercise results, assessment scores, etc.
        
        return {
            'learning_velocity': 0.6,  # concepts per hour
            'difficulty_preference': 0.6,  # 0.0 to 1.0
            'retention_rate': 0.7,  # 0.0 to 1.0
            'transfer_ability': 0.6  # 0.0 to 1.0
        }
    
    def _determine_learning_style(self, message_patterns: Dict, content_patterns: Dict) -> LearningStyle:
        """Determine primary learning style"""
        visual_score = content_patterns.get('visual_preference', 0)
        audio_score = content_patterns.get('audio_preference', 0)
        hands_on_score = content_patterns.get('hands_on_preference', 0)
        
        if visual_score > audio_score and visual_score > hands_on_score:
            return LearningStyle.VISUAL
        elif audio_score > hands_on_score:
            return LearningStyle.AUDITORY
        elif hands_on_score > 0.3:
            return LearningStyle.KINESTHETIC
        else:
            return LearningStyle.MULTIMODAL
    
    def _determine_cognitive_patterns(self, message_patterns: Dict, session_patterns: Dict) -> List[CognitivePattern]:
        """Determine cognitive patterns"""
        patterns = []
        
        # Analyze message complexity and structure
        if message_patterns.get('metacognitive_score', 0) > 0.6:
            patterns.append(CognitivePattern.ANALYTICAL)
        else:
            patterns.append(CognitivePattern.HOLISTIC)
        
        # Analyze session patterns
        if session_patterns.get('completion_rate', 0) > 0.8:
            patterns.append(CognitivePattern.SEQUENTIAL)
        else:
            patterns.append(CognitivePattern.RANDOM)
        
        return patterns
    
    def _determine_preferred_pace(self, temporal_patterns: Dict, session_patterns: Dict) -> LearningPace:
        """Determine preferred learning pace"""
        session_length = session_patterns.get('avg_session_length', 30)
        sessions_per_week = temporal_patterns.get('sessions_per_week', 3)
        
        if session_length > 45 and sessions_per_week < 3:
            return LearningPace.SLOW_DEEP
        elif session_length < 20 and sessions_per_week > 5:
            return LearningPace.FAST_OVERVIEW
        else:
            return LearningPace.MODERATE
    
    def _determine_motivation_style(self, session_patterns: Dict, message_patterns: Dict) -> MotivationStyle:
        """Determine motivation style"""
        completion_rate = session_patterns.get('completion_rate', 0.7)
        question_ratio = message_patterns.get('question_ratio', 0.3)
        
        if completion_rate > 0.9:
            return MotivationStyle.ACHIEVEMENT
        elif question_ratio > 0.5:
            return MotivationStyle.MASTERY
        else:
            return MotivationStyle.AUTONOMY
    
    def _detect_stress_indicators(self, recent_messages: List[ChatMessage]) -> List[str]:
        """Detect stress indicators in recent messages"""
        stress_words = ['confused', 'frustrated', 'difficult', 'hard', 'stuck', 'overwhelmed']
        indicators = []
        
        user_messages = [m for m in recent_messages if m.sender == 'user']
        
        for word in stress_words:
            if any(word in m.message.lower() for m in user_messages):
                indicators.append(word)
        
        return indicators
    
    def _identify_motivation_triggers(self, messages: List[ChatMessage]) -> List[str]:
        """Identify what motivates the user"""
        triggers = []
        
        # Analyze positive responses
        positive_words = ['great', 'awesome', 'helpful', 'clear', 'understood', 'makes sense']
        trigger_contexts = []
        
        for i, message in enumerate(messages):
            if message.sender == 'user' and any(word in message.message.lower() for word in positive_words):
                # Look at previous mentor message for context
                if i > 0 and messages[i-1].sender == 'mentor':
                    trigger_contexts.append(messages[i-1].message_type)
        
        # Determine common triggers
        if trigger_contexts.count('exercise') > 2:
            triggers.append('practical_exercises')
        if trigger_contexts.count('explanation') > 3:
            triggers.append('clear_explanations')
        
        return triggers or ['progress_feedback', 'achievement_recognition']
    
    def _identify_learning_blockers(self, messages: List[ChatMessage]) -> List[str]:
        """Identify learning blockers"""
        blockers = []
        
        user_messages = [m for m in messages if m.sender == 'user']
        
        # Look for common blocker patterns
        blocker_patterns = {
            'information_overload': ['too much', 'overwhelming', 'too fast'],
            'unclear_concepts': ['confused', 'dont understand', 'unclear'],
            'lack_of_examples': ['example', 'show me', 'demonstrate'],
            'pacing_issues': ['slow down', 'too fast', 'too slow']
        }
        
        for blocker, keywords in blocker_patterns.items():
            if any(any(keyword in m.message.lower() for keyword in keywords) for m in user_messages):
                blockers.append(blocker)
        
        return blockers
    
    def _calculate_confidence_score(self, session_count: int, message_count: int) -> float:
        """Calculate confidence in the learning DNA profile"""
        # More data = higher confidence
        data_score = min(1.0, (session_count * 0.1 + message_count * 0.01))
        return data_score
    
    def _create_default_dna(self, user_id: str) -> LearningDNA:
        """Create default learning DNA for new users"""
        return LearningDNA(
            user_id=user_id,
            learning_style=LearningStyle.MULTIMODAL,
            cognitive_patterns=[CognitivePattern.ANALYTICAL, CognitivePattern.SEQUENTIAL],
            preferred_pace=LearningPace.MODERATE,
            motivation_style=MotivationStyle.MASTERY,
            attention_span_minutes=25,
            difficulty_preference=0.6,
            interaction_preference='medium',
            feedback_preference='immediate',
            session_frequency=3.0,
            optimal_session_length=25,
            peak_performance_hours=[9, 14, 19],
            concept_retention_rate=0.7,
            knowledge_transfer_ability=0.6,
            learning_velocity=0.5,
            curiosity_index=0.6,
            perseverance_score=0.7,
            collaboration_preference=0.5,
            metacognitive_awareness=0.6,
            current_energy_level=0.7,
            stress_indicators=[],
            motivation_triggers=['progress_feedback'],
            learning_blockers=[],
            last_updated=datetime.now(),
            confidence_score=0.1
        )
    
    async def _store_learning_dna(self, learning_dna: LearningDNA):
        """Store learning DNA in database"""
        try:
            collection = db_service.db['learning_dna']
            await collection.replace_one(
                {'user_id': learning_dna.user_id},
                learning_dna.to_dict(),
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error storing learning DNA: {str(e)}")
    
    async def get_adaptive_content_parameters(self, user_id: str, context: Dict[str, Any] = None) -> AdaptiveContentParameters:
        """Get adaptive content parameters for user"""
        try:
            learning_dna = await self.analyze_learning_dna(user_id)
            
            # Determine complexity level
            complexity_level = learning_dna.difficulty_preference
            
            # Adjust based on current context
            if context:
                if context.get('struggling', False):
                    complexity_level = max(0.1, complexity_level - 0.2)
                elif context.get('excelling', False):
                    complexity_level = min(1.0, complexity_level + 0.2)
            
            # Determine explanation depth
            explanation_depth = "detailed"
            if learning_dna.preferred_pace == LearningPace.FAST_OVERVIEW:
                explanation_depth = "overview"
            elif learning_dna.preferred_pace == LearningPace.SLOW_DEEP:
                explanation_depth = "comprehensive"
            
            # Determine visual and interactive elements
            visual_elements = learning_dna.learning_style in [LearningStyle.VISUAL, LearningStyle.MULTIMODAL]
            interactive_elements = learning_dna.learning_style in [LearningStyle.KINESTHETIC, LearningStyle.MULTIMODAL]
            
            # Determine pacing
            pacing_delay = 0.5  # default
            if learning_dna.preferred_pace == LearningPace.SLOW_DEEP:
                pacing_delay = 1.0
            elif learning_dna.preferred_pace == LearningPace.FAST_OVERVIEW:
                pacing_delay = 0.2
            
            return AdaptiveContentParameters(
                complexity_level=complexity_level,
                explanation_depth=explanation_depth,
                example_count=3 if visual_elements else 2,
                visual_elements=visual_elements,
                interactive_elements=interactive_elements,
                pacing_delay=pacing_delay,
                reinforcement_frequency=0.3 if learning_dna.concept_retention_rate < 0.7 else 0.2
            )
            
        except Exception as e:
            logger.error(f"Error getting adaptive content parameters: {str(e)}")
            return AdaptiveContentParameters(
                complexity_level=0.6,
                explanation_depth="detailed",
                example_count=2,
                visual_elements=True,
                interactive_elements=True,
                pacing_delay=0.5,
                reinforcement_frequency=0.2
            )
    
    async def analyze_mood_and_adapt(self, user_id: str, recent_messages: List[ChatMessage], context: Dict[str, Any] = None) -> MoodBasedAdaptation:
        """Analyze user's mood and provide adaptation recommendations"""
        try:
            # Analyze recent messages for mood indicators
            mood_analysis = self._analyze_mood_from_messages(recent_messages)
            
            # Get learning DNA for context
            learning_dna = await self.analyze_learning_dna(user_id)
            
            # Detect energy and stress levels
            energy_level = self._detect_energy_level(recent_messages, learning_dna)
            stress_level = self._detect_stress_level(recent_messages, learning_dna)
            
            # Determine recommended adaptations
            recommended_pace = self._recommend_pace_for_mood(mood_analysis['mood'], energy_level, stress_level)
            content_tone = self._recommend_content_tone(mood_analysis['mood'], stress_level)
            interaction_style = self._recommend_interaction_style(mood_analysis['mood'], energy_level)
            
            # Determine if break is needed
            break_recommendation = None
            if stress_level > 0.7 or energy_level < 0.3:
                break_recommendation = 10  # 10 minute break
            
            return MoodBasedAdaptation(
                detected_mood=mood_analysis['mood'],
                confidence=mood_analysis['confidence'],
                energy_level=energy_level,
                stress_level=stress_level,
                recommended_pace=recommended_pace,
                content_tone=content_tone,
                interaction_style=interaction_style,
                break_recommendation=break_recommendation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing mood: {str(e)}")
            return MoodBasedAdaptation(
                detected_mood=EmotionalState.CALM,
                confidence=0.5,
                energy_level=0.7,
                stress_level=0.3,
                recommended_pace=LearningPace.MODERATE,
                content_tone="neutral",
                interaction_style="standard",
                break_recommendation=None
            )
    
    def _analyze_mood_from_messages(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Analyze mood from recent messages"""
        if not messages:
            return {'mood': EmotionalState.CALM, 'confidence': 0.5}
        
        user_messages = [m for m in messages if m.sender == 'user']
        
        # Define mood indicators
        mood_indicators = {
            EmotionalState.EXCITED: ['great', 'awesome', 'amazing', 'love', 'fantastic'],
            EmotionalState.FRUSTRATED: ['frustrated', 'annoying', 'difficult', 'hard', 'stuck'],
            EmotionalState.STRESSED: ['stressed', 'overwhelmed', 'too much', 'pressure', 'anxious'],
            EmotionalState.CONFIDENT: ['understand', 'got it', 'clear', 'easy', 'makes sense'],
            EmotionalState.CURIOUS: ['why', 'how', 'what if', 'interesting', 'more about'],
            EmotionalState.OVERWHELMED: ['too much', 'overwhelming', 'confused', 'lost', 'dont get it']
        }
        
        # Score each mood
        mood_scores = {}
        for mood, indicators in mood_indicators.items():
            score = sum(1 for m in user_messages if any(ind in m.message.lower() for ind in indicators))
            mood_scores[mood] = score / len(user_messages) if user_messages else 0
        
        # Determine dominant mood
        if not mood_scores or max(mood_scores.values()) == 0:
            return {'mood': EmotionalState.CALM, 'confidence': 0.5}
        
        dominant_mood = max(mood_scores, key=mood_scores.get)
        confidence = mood_scores[dominant_mood]
        
        return {'mood': dominant_mood, 'confidence': min(1.0, confidence * 3)}
    
    def _detect_energy_level(self, messages: List[ChatMessage], learning_dna: LearningDNA) -> float:
        """Detect user's energy level"""
        if not messages:
            return learning_dna.current_energy_level
        
        user_messages = [m for m in messages if m.sender == 'user']
        
        # High energy indicators
        high_energy_words = ['great', 'awesome', 'let\'s', 'more', 'next', 'excited']
        low_energy_words = ['tired', 'slow', 'later', 'enough', 'break', 'stop']
        
        high_energy_count = sum(1 for m in user_messages if any(word in m.message.lower() for word in high_energy_words))
        low_energy_count = sum(1 for m in user_messages if any(word in m.message.lower() for word in low_energy_words))
        
        if high_energy_count > low_energy_count:
            return min(1.0, 0.7 + (high_energy_count / len(user_messages)))
        elif low_energy_count > 0:
            return max(0.1, 0.7 - (low_energy_count / len(user_messages)))
        else:
            return learning_dna.current_energy_level
    
    def _detect_stress_level(self, messages: List[ChatMessage], learning_dna: LearningDNA) -> float:
        """Detect user's stress level"""
        if not messages:
            return 0.3  # default low stress
        
        user_messages = [m for m in messages if m.sender == 'user']
        
        stress_indicators = ['confused', 'frustrated', 'difficult', 'hard', 'stuck', 'overwhelmed', 'anxious']
        stress_count = sum(1 for m in user_messages if any(word in m.message.lower() for word in stress_indicators))
        
        return min(1.0, stress_count / len(user_messages) * 2) if user_messages else 0.3
    
    def _recommend_pace_for_mood(self, mood: EmotionalState, energy_level: float, stress_level: float) -> LearningPace:
        """Recommend learning pace based on mood and energy"""
        if mood in [EmotionalState.EXCITED, EmotionalState.CONFIDENT] and energy_level > 0.7:
            return LearningPace.FAST_OVERVIEW
        elif mood in [EmotionalState.STRESSED, EmotionalState.OVERWHELMED] or stress_level > 0.6:
            return LearningPace.SLOW_DEEP
        else:
            return LearningPace.MODERATE
    
    def _recommend_content_tone(self, mood: EmotionalState, stress_level: float) -> str:
        """Recommend content tone based on mood"""
        if mood in [EmotionalState.FRUSTRATED, EmotionalState.STRESSED] or stress_level > 0.6:
            return "encouraging"
        elif mood in [EmotionalState.CONFIDENT, EmotionalState.EXCITED]:
            return "challenging"
        else:
            return "neutral"
    
    def _recommend_interaction_style(self, mood: EmotionalState, energy_level: float) -> str:
        """Recommend interaction style based on mood and energy"""
        if mood in [EmotionalState.FRUSTRATED, EmotionalState.OVERWHELMED]:
            return "supportive"
        elif mood in [EmotionalState.CONFIDENT, EmotionalState.EXCITED] and energy_level > 0.7:
            return "challenging"
        else:
            return "standard"

# Global personalization engine instance
personalization_engine = PersonalizationEngine()