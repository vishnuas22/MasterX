"""
Emotion Core - Core components for emotion detection and learning readiness.

This module provides:
- Emotion categories and states
- Learning readiness indicators
- Emotional trajectory tracking
- PAD (Pleasure-Arousal-Dominance) dimensional model
- Intervention level definitions

Author: MasterX AI Team
Version: 1.0 (Enhanced from v9.0)
"""

import logging
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# EMOTION CONSTANTS
# ============================================================================

class EmotionConstants:
    """Constants for emotion detection system."""
    
    # Performance targets
    TARGET_ANALYSIS_TIME_MS = 15.0
    OPTIMAL_ANALYSIS_TIME_MS = 10.0
    
    # Accuracy thresholds (adaptive)
    MIN_CONFIDENCE_THRESHOLD = 0.70
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    
    # Processing limits
    MAX_CONCURRENT_ANALYSES = 1000
    EMOTION_HISTORY_LIMIT = 1000
    
    # Circuit breaker config
    FAILURE_THRESHOLD = 3
    RECOVERY_TIMEOUT = 30.0
    SUCCESS_THRESHOLD = 5


# ============================================================================
# EMOTION CATEGORIES
# ============================================================================

class EmotionCategory(Enum):
    """Core emotion categories for learning contexts."""
    
    # Basic emotions
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"
    
    # Learning-specific emotions
    FRUSTRATION = "frustration"
    SATISFACTION = "satisfaction"
    CURIOSITY = "curiosity"
    CONFIDENCE = "confidence"
    ANXIETY = "anxiety"
    EXCITEMENT = "excitement"
    CONFUSION = "confusion"
    ENGAGEMENT = "engagement"
    
    # Advanced learning states
    FLOW_STATE = "flow_state"
    COGNITIVE_OVERLOAD = "cognitive_overload"
    BREAKTHROUGH_MOMENT = "breakthrough_moment"
    MASTERY_JOY = "mastery_joy"


class InterventionLevel(Enum):
    """Intervention levels for emotional support."""
    
    NONE = "none"
    PREVENTIVE = "preventive"
    MILD = "mild"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"


class LearningReadiness(Enum):
    """Learning readiness states based on emotional factors."""
    
    OPTIMAL_READINESS = "optimal_readiness"
    HIGH_READINESS = "high_readiness"
    MODERATE_READINESS = "moderate_readiness"
    LOW_READINESS = "low_readiness"
    NOT_READY = "not_ready"


class EmotionalTrajectory(Enum):
    """Emotional trajectory over time."""
    
    IMPROVING = "improving"
    STABLE_POSITIVE = "stable_positive"
    STABLE_NEUTRAL = "stable_neutral"
    DECLINING = "declining"
    VOLATILE = "volatile"
    RECOVERING = "recovering"


# ============================================================================
# EMOTION METRICS
# ============================================================================

@dataclass
class EmotionMetrics:
    """
    Comprehensive emotion metrics for a user interaction.
    
    Includes categorical emotions, dimensional model (PAD), and learning indicators.
    """
    
    # Primary emotion data
    primary_emotion: str = EmotionCategory.NEUTRAL.value
    emotion_distribution: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    
    # PAD dimensional model
    arousal: float = 0.5  # 0 (calm) to 1 (excited)
    valence: float = 0.5  # 0 (negative) to 1 (positive)
    dominance: float = 0.5  # 0 (submissive) to 1 (dominant)
    
    # Learning indicators
    learning_readiness: str = LearningReadiness.MODERATE_READINESS.value
    cognitive_load: float = 0.5  # 0 (low) to 1 (high)
    engagement_level: float = 0.5  # 0 (disengaged) to 1 (highly engaged)
    
    # Intervention needs
    intervention_level: str = InterventionLevel.NONE.value
    intervention_confidence: float = 0.5
    
    # Trajectory
    emotional_trajectory: str = EmotionalTrajectory.STABLE_NEUTRAL.value
    
    # Metadata
    analysis_time_ms: float = 0.0
    model_version: str = "1.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'primary_emotion': self.primary_emotion,
            'emotion_distribution': self.emotion_distribution,
            'confidence': self.confidence,
            'arousal': self.arousal,
            'valence': self.valence,
            'dominance': self.dominance,
            'learning_readiness': self.learning_readiness,
            'cognitive_load': self.cognitive_load,
            'engagement_level': self.engagement_level,
            'intervention_level': self.intervention_level,
            'intervention_confidence': self.intervention_confidence,
            'emotional_trajectory': self.emotional_trajectory,
            'analysis_time_ms': self.analysis_time_ms,
            'model_version': self.model_version,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionMetrics':
        """Create metrics from dictionary."""
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            primary_emotion=data.get('primary_emotion', EmotionCategory.NEUTRAL.value),
            emotion_distribution=data.get('emotion_distribution', {}),
            confidence=data.get('confidence', 0.5),
            arousal=data.get('arousal', 0.5),
            valence=data.get('valence', 0.5),
            dominance=data.get('dominance', 0.5),
            learning_readiness=data.get('learning_readiness', LearningReadiness.MODERATE_READINESS.value),
            cognitive_load=data.get('cognitive_load', 0.5),
            engagement_level=data.get('engagement_level', 0.5),
            intervention_level=data.get('intervention_level', InterventionLevel.NONE.value),
            intervention_confidence=data.get('intervention_confidence', 0.5),
            emotional_trajectory=data.get('emotional_trajectory', EmotionalTrajectory.STABLE_NEUTRAL.value),
            analysis_time_ms=data.get('analysis_time_ms', 0.0),
            model_version=data.get('model_version', '1.0'),
            timestamp=timestamp or datetime.utcnow()
        )


# ============================================================================
# EMOTION RESULT
# ============================================================================

@dataclass
class EmotionResult:
    """
    Complete emotion analysis result with recommendations.
    """
    
    # Core metrics
    metrics: EmotionMetrics = field(default_factory=EmotionMetrics)
    
    # User context
    user_id: str = ""
    session_id: Optional[str] = None
    
    # Analysis details
    text_analyzed: str = ""
    context_factors: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    intervention_needed: bool = False
    intervention_type: Optional[str] = None
    intervention_suggestions: List[str] = field(default_factory=list)
    
    # Adaptive learning
    difficulty_adjustment: float = 0.0  # -1 (easier) to +1 (harder)
    pacing_adjustment: float = 0.0  # -1 (slower) to +1 (faster)
    support_level: str = "standard"  # minimal, standard, enhanced, intensive
    
    # Performance tracking
    prediction_quality: float = 0.5
    model_type: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'metrics': self.metrics.to_dict(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'text_analyzed': self.text_analyzed,
            'context_factors': self.context_factors,
            'intervention_needed': self.intervention_needed,
            'intervention_type': self.intervention_type,
            'intervention_suggestions': self.intervention_suggestions,
            'difficulty_adjustment': self.difficulty_adjustment,
            'pacing_adjustment': self.pacing_adjustment,
            'support_level': self.support_level,
            'prediction_quality': self.prediction_quality,
            'model_type': self.model_type
        }
    
    def is_struggling(self) -> bool:
        """Check if user is showing signs of struggle."""
        struggling_emotions = {
            EmotionCategory.FRUSTRATION.value,
            EmotionCategory.CONFUSION.value,
            EmotionCategory.ANXIETY.value,
            EmotionCategory.COGNITIVE_OVERLOAD.value
        }
        
        return (
            self.metrics.primary_emotion in struggling_emotions or
            self.metrics.cognitive_load > 0.7 or
            self.metrics.learning_readiness in [
                LearningReadiness.LOW_READINESS.value,
                LearningReadiness.NOT_READY.value
            ]
        )
    
    def is_thriving(self) -> bool:
        """Check if user is in optimal learning state."""
        thriving_emotions = {
            EmotionCategory.FLOW_STATE.value,
            EmotionCategory.ENGAGEMENT.value,
            EmotionCategory.CURIOSITY.value,
            EmotionCategory.EXCITEMENT.value
        }
        
        return (
            self.metrics.primary_emotion in thriving_emotions or
            self.metrics.learning_readiness == LearningReadiness.OPTIMAL_READINESS.value or
            self.metrics.engagement_level > 0.7
        )
    
    def requires_intervention(self) -> bool:
        """Check if intervention is recommended."""
        return (
            self.intervention_needed or
            self.metrics.intervention_level in [
                InterventionLevel.SIGNIFICANT.value,
                InterventionLevel.CRITICAL.value
            ]
        )


# ============================================================================
# BEHAVIORAL PATTERNS
# ============================================================================

@dataclass
class BehavioralPattern:
    """Tracks behavioral patterns for a user."""
    
    user_id: str
    total_interactions: int = 0
    emotional_history: List[str] = field(default_factory=list)
    
    # Engagement metrics
    avg_engagement: float = 0.5
    engagement_trend: str = "stable"
    
    # Learning patterns
    avg_cognitive_load: float = 0.5
    struggle_frequency: float = 0.0  # 0 to 1
    breakthrough_frequency: float = 0.0  # 0 to 1
    
    # Adaptive thresholds
    optimal_difficulty: float = 0.5
    optimal_pacing: float = 0.5
    preferred_support_level: str = "standard"
    
    # Timestamp
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_from_result(self, result: EmotionResult) -> None:
        """Update patterns based on new emotion result."""
        self.total_interactions += 1
        
        # Update emotional history (keep last 50)
        self.emotional_history.append(result.metrics.primary_emotion)
        if len(self.emotional_history) > 50:
            self.emotional_history = self.emotional_history[-50:]
        
        # Update engagement (exponential moving average)
        alpha = 0.1
        self.avg_engagement = (
            alpha * result.metrics.engagement_level +
            (1 - alpha) * self.avg_engagement
        )
        
        # Update cognitive load
        self.avg_cognitive_load = (
            alpha * result.metrics.cognitive_load +
            (1 - alpha) * self.avg_cognitive_load
        )
        
        # Update struggle frequency
        if result.is_struggling():
            self.struggle_frequency = (
                alpha * 1.0 + (1 - alpha) * self.struggle_frequency
            )
        else:
            self.struggle_frequency = (
                alpha * 0.0 + (1 - alpha) * self.struggle_frequency
            )
        
        # Update breakthrough frequency
        if result.metrics.primary_emotion == EmotionCategory.BREAKTHROUGH_MOMENT.value:
            self.breakthrough_frequency = (
                alpha * 1.0 + (1 - alpha) * self.breakthrough_frequency
            )
        else:
            self.breakthrough_frequency = (
                alpha * 0.0 + (1 - alpha) * self.breakthrough_frequency
            )
        
        self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'total_interactions': self.total_interactions,
            'emotional_history': self.emotional_history[-10:],  # Last 10 only
            'avg_engagement': self.avg_engagement,
            'engagement_trend': self.engagement_trend,
            'avg_cognitive_load': self.avg_cognitive_load,
            'struggle_frequency': self.struggle_frequency,
            'breakthrough_frequency': self.breakthrough_frequency,
            'optimal_difficulty': self.optimal_difficulty,
            'optimal_pacing': self.optimal_pacing,
            'preferred_support_level': self.preferred_support_level,
            'last_updated': self.last_updated.isoformat()
        }


__all__ = [
    'EmotionConstants',
    'EmotionCategory',
    'InterventionLevel',
    'LearningReadiness',
    'EmotionalTrajectory',
    'EmotionMetrics',
    'EmotionResult',
    'BehavioralPattern'
]