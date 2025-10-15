"""
Emotion Core - Core components for emotion detection and learning readiness.

This module provides:
- Emotion categories and states
- Learning readiness indicators
- Emotional trajectory tracking
- PAD (Pleasure-Arousal-Dominance) dimensional model
- Intervention level definitions

PRINCIPLES (AGENTS.md):
- Type-safe with Pydantic models
- Runtime validation for all data
- No hardcoded values
- Clean, professional naming

Author: MasterX AI Team
Version: 2.0 (Enhanced with Pydantic type safety)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

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

class EmotionMetrics(BaseModel):
    """
    Comprehensive emotion metrics for a user interaction.
    
    Includes categorical emotions, dimensional model (PAD), and learning indicators.
    Uses Pydantic for type safety and runtime validation (AGENTS.md compliant).
    """
    
    # Primary emotion data
    primary_emotion: str = Field(
        default=EmotionCategory.NEUTRAL.value,
        description="Primary detected emotion"
    )
    emotion_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Distribution of all detected emotions"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in emotion detection (0.0-1.0)"
    )
    
    # PAD dimensional model
    arousal: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Arousal level: 0 (calm) to 1 (excited)"
    )
    valence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Valence: 0 (negative) to 1 (positive)"
    )
    dominance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Dominance: 0 (submissive) to 1 (dominant)"
    )
    
    # Learning indicators
    learning_readiness: str = Field(
        default=LearningReadiness.MODERATE_READINESS.value,
        description="Current learning readiness state"
    )
    cognitive_load: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Cognitive load: 0 (low) to 1 (high)"
    )
    engagement_level: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Engagement: 0 (disengaged) to 1 (highly engaged)"
    )
    
    # Intervention needs
    intervention_level: str = Field(
        default=InterventionLevel.NONE.value,
        description="Required intervention level"
    )
    intervention_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in intervention recommendation"
    )
    
    # Trajectory
    emotional_trajectory: str = Field(
        default=EmotionalTrajectory.STABLE_NEUTRAL.value,
        description="Emotional trajectory over time"
    )
    
    # Metadata
    analysis_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for analysis in milliseconds"
    )
    model_version: str = Field(
        default="2.0",
        description="Model version used for analysis"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of analysis"
    )
    
    @field_validator('confidence', 'arousal', 'valence', 'dominance', 
                     'cognitive_load', 'engagement_level', 'intervention_confidence')
    @classmethod
    def validate_range(cls, v: float) -> float:
        """Validate that values are within 0.0-1.0 range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Value must be between 0.0 and 1.0, got {v}")
        return v
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        validate_assignment = True  # Validate on attribute assignment
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary.
        
        Uses Pydantic's model_dump for type-safe serialization.
        """
        return self.model_dump(mode='json')
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionMetrics':
        """
        Create metrics from dictionary with type validation.
        
        Uses Pydantic's model_validate for runtime type checking.
        """
        return cls.model_validate(data)


# ============================================================================
# EMOTION RESULT
# ============================================================================

class EmotionResult(BaseModel):
    """
    Complete emotion analysis result with recommendations.
    
    Uses Pydantic for type safety and runtime validation (AGENTS.md compliant).
    """
    
    # Core metrics
    metrics: EmotionMetrics = Field(
        default_factory=EmotionMetrics,
        description="Emotion metrics from analysis"
    )
    
    # User context
    user_id: str = Field(
        default="",
        description="User identifier"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier"
    )
    
    # Analysis details
    text_analyzed: str = Field(
        default="",
        description="Text that was analyzed"
    )
    context_factors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context factors"
    )
    
    # Recommendations
    intervention_needed: bool = Field(
        default=False,
        description="Whether intervention is needed"
    )
    intervention_type: Optional[str] = Field(
        default=None,
        description="Type of intervention recommended"
    )
    intervention_suggestions: List[str] = Field(
        default_factory=list,
        description="Specific intervention suggestions"
    )
    
    # Adaptive learning
    difficulty_adjustment: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Difficulty adjustment: -1 (easier) to +1 (harder)"
    )
    pacing_adjustment: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Pacing adjustment: -1 (slower) to +1 (faster)"
    )
    support_level: str = Field(
        default="standard",
        description="Support level: minimal, standard, enhanced, intensive"
    )
    
    # Performance tracking
    prediction_quality: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Quality of prediction"
    )
    model_type: str = Field(
        default="unknown",
        description="Type of model used"
    )
    
    @field_validator('difficulty_adjustment', 'pacing_adjustment')
    @classmethod
    def validate_adjustment_range(cls, v: float) -> float:
        """Validate adjustment values are within -1.0 to 1.0 range"""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"Adjustment must be between -1.0 and 1.0, got {v}")
        return v
    
    @field_validator('support_level')
    @classmethod
    def validate_support_level(cls, v: str) -> str:
        """Validate support level is one of allowed values"""
        allowed = ["minimal", "standard", "enhanced", "intensive"]
        if v not in allowed:
            raise ValueError(f"Support level must be one of {allowed}, got {v}")
        return v
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True  # Validate on attribute assignment
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Uses Pydantic's model_dump for type-safe serialization.
        """
        return self.model_dump(mode='json')
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionResult':
        """
        Create result from dictionary with type validation.
        
        Uses Pydantic's model_validate for runtime type checking.
        """
        return cls.model_validate(data)
    
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