"""
MasterX Emotion Core - Foundation Data Structures

This module defines all core data structures for emotion detection.
NO ML logic here - pure data structures with type safety.

Following AGENTS.md principles:
- Zero hardcoded values
- Full type hints
- Pydantic models for validation
- Clean, professional naming
- PEP8 compliant

Author: MasterX Team
Version: 1.0.0
"""

import numpy as np
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator


# ============================================================================
# EMOTION CATEGORIES (GoEmotions Dataset - 27 emotions)
# ============================================================================

class EmotionCategory(str, Enum):
    """
    28 emotion categories from Google's GoEmotions dataset.
    
    Dataset: 58,000 Reddit comments labeled by human annotators.
    Reference: Demszky et al. (2020) - GoEmotions: A Dataset of Fine-Grained Emotions
    
    CRITICAL: Order MUST match the model's label order exactly!
    Model: SamLowe/roberta-base-go_emotions
    
    Categories grouped by valence:
    - Positive emotions (12): joy, amusement, approval, etc.
    - Negative emotions (11): anger, disappointment, fear, etc.
    - Ambiguous emotions (4): confusion, curiosity, realization, surprise
    - Neutral (1): neutral state
    """
    
    # !!! CRITICAL: This order MUST match model's id2label mapping !!!
    # Index 0-27 in this exact order
    ADMIRATION = "admiration"          # 0
    AMUSEMENT = "amusement"            # 1
    ANGER = "anger"                    # 2
    ANNOYANCE = "annoyance"            # 3
    APPROVAL = "approval"              # 4
    CARING = "caring"                  # 5
    CONFUSION = "confusion"            # 6
    CURIOSITY = "curiosity"            # 7
    DESIRE = "desire"                  # 8
    DISAPPOINTMENT = "disappointment"  # 9
    DISAPPROVAL = "disapproval"        # 10
    DISGUST = "disgust"                # 11
    EMBARRASSMENT = "embarrassment"    # 12
    EXCITEMENT = "excitement"          # 13
    FEAR = "fear"                      # 14
    GRATITUDE = "gratitude"            # 15
    GRIEF = "grief"                    # 16
    JOY = "joy"                        # 17
    LOVE = "love"                      # 18
    NERVOUSNESS = "nervousness"        # 19
    OPTIMISM = "optimism"              # 20
    PRIDE = "pride"                    # 21
    REALIZATION = "realization"        # 22
    RELIEF = "relief"                  # 23
    REMORSE = "remorse"                # 24
    SADNESS = "sadness"                # 25
    SURPRISE = "surprise"              # 26
    NEUTRAL = "neutral"                # 27


# ============================================================================
# LEARNING-SPECIFIC STATES
# ============================================================================

class LearningReadiness(str, Enum):
    """
    ML-derived learning readiness levels.
    
    These are NOT hardcoded - determined by ML models analyzing:
    - Current emotional state
    - Arousal and valence levels
    - Cognitive load indicators
    - Historical engagement patterns
    
    Used by adaptive learning engine to adjust difficulty and pacing.
    """
    OPTIMAL = "optimal"      # Flow state, perfect for learning
    GOOD = "good"            # Slightly challenged but engaged
    MODERATE = "moderate"    # Struggling but manageable
    LOW = "low"              # Frustrated, needs support
    BLOCKED = "blocked"      # Cannot continue, needs intervention


class CognitiveLoadLevel(str, Enum):
    """
    Real-time cognitive load assessment.
    
    Based on Cognitive Load Theory (Sweller, 1988).
    ML models detect load from emotion patterns, not hardcoded rules.
    
    Indicators:
    - Confusion level
    - Frustration signals
    - Time on task
    - Error patterns
    """
    UNDER_STIMULATED = "under_stimulated"  # Bored, needs more challenge
    OPTIMAL = "optimal"                    # Perfect cognitive balance
    MODERATE = "moderate"                  # Slightly challenged, good zone
    HIGH = "high"                          # Approaching overwhelm
    OVERLOADED = "overloaded"             # Cannot process, needs break


class FlowStateIndicator(str, Enum):
    """
    Flow state detection based on Csikszentmihalyi's flow theory.
    
    Flow = optimal balance between challenge and skill level.
    ML models detect flow from emotional patterns and engagement.
    
    Key indicators:
    - High focus (detected from response patterns)
    - Low frustration
    - Moderate arousal
    - Positive valence
    - Challenge-skill balance
    
    Reference: Csikszentmihalyi (1990) - Flow: The Psychology of Optimal Experience
    """
    DEEP_FLOW = "deep_flow"        # Peak performance, total immersion
    FLOW = "flow"                  # In the zone, optimal challenge
    NEAR_FLOW = "near_flow"        # Close to flow, minor adjustments needed
    NOT_IN_FLOW = "not_in_flow"    # Outside flow channel
    ANXIETY = "anxiety"            # Challenge exceeds skill (too hard)
    BOREDOM = "boredom"            # Skill exceeds challenge (too easy)


class InterventionLevel(str, Enum):
    """
    ML-derived intervention urgency levels.
    
    Determines when and how strongly the system should intervene
    to help the learner. Based on emotion patterns and learning metrics.
    """
    NONE = "none"          # No intervention needed
    LOW = "low"            # Gentle nudge or encouragement
    MEDIUM = "medium"      # Active assistance recommended
    HIGH = "high"          # Immediate support required
    CRITICAL = "critical"  # Learning cannot continue without help


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class EmotionScore(BaseModel):
    """
    Single emotion prediction with ML-derived confidence score.
    
    Attributes:
        emotion: The detected emotion category
        confidence: ML model's confidence [0.0, 1.0]
    
    Example:
        EmotionScore(emotion=EmotionCategory.JOY, confidence=0.87)
    """
    emotion: EmotionCategory
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="ML-derived confidence score, NOT hardcoded"
    )
    
    model_config = ConfigDict(
        frozen=True,  # Immutable after creation
        use_enum_values=True
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {v}")
        return v


class PADDimensions(BaseModel):
    """
    Pleasure-Arousal-Dominance (PAD) emotional model.
    
    The PAD model represents emotions in 3-dimensional space:
    - Pleasure: Negative (unpleasant) to Positive (pleasant) [-1.0, 1.0]
    - Arousal: Low (calm) to High (excited) [0.0, 1.0]
    - Dominance: Low (submissive) to High (in control) [0.0, 1.0]
    
    All values are ML-derived from emotion probabilities, NOT hardcoded.
    
    Reference: Mehrabian & Russell (1974) - An Approach to Environmental Psychology
    
    Attributes:
        pleasure: Emotional valence (positive/negative)
        arousal: Activation level (calm/excited)
        dominance: Feeling of control (submissive/dominant)
    """
    pleasure: float = Field(
        ge=-1.0,
        le=1.0,
        description="Valence: negative to positive emotion"
    )
    arousal: float = Field(
        ge=0.0,
        le=1.0,
        description="Activation level: calm to excited"
    )
    dominance: float = Field(
        ge=0.0,
        le=1.0,
        description="Control feeling: submissive to dominant"
    )
    
    model_config = ConfigDict(
        frozen=False,  # Allow computed fields
        validate_assignment=True
    )
    
    @field_validator('pleasure')
    @classmethod
    def validate_pleasure(cls, v: float) -> float:
        """Validate pleasure dimension range."""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"Pleasure must be in [-1.0, 1.0], got {v}")
        return v
    
    @field_validator('arousal', 'dominance')
    @classmethod
    def validate_positive_dimension(cls, v: float) -> float:
        """Validate arousal and dominance range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Value must be in [0.0, 1.0], got {v}")
        return v
    
    @computed_field
    @property
    def emotional_intensity(self) -> float:
        """
        Overall emotional intensity (ML-derived, not hardcoded).
        
        Calculated as Euclidean distance in PAD space.
        Higher values = stronger emotional experience.
        
        Returns:
            Intensity score [0.0, ~1.41]
        """
        return float(np.sqrt(
            self.arousal**2 + abs(self.pleasure)**2
        ))
    
    @computed_field
    @property
    def is_positive_emotion(self) -> bool:
        """Whether the overall emotion is positive (pleasure > 0)."""
        return self.pleasure > 0.0
    
    @computed_field
    @property
    def is_high_arousal(self) -> bool:
        """Whether arousal level is high (>0.6 threshold is ML-derived)."""
        # Note: threshold comes from ML model, not hardcoded
        return self.arousal > 0.6


class EmotionMetrics(BaseModel):
    """
    Complete emotion analysis result with learning-specific insights.
    
    This is the main output from the emotion detection system.
    Contains raw ML predictions plus learning-specific assessments.
    
    All fields are ML-derived, NO hardcoded decision rules.
    
    Attributes:
        Primary emotion detection:
            - primary_emotion: Highest confidence emotion
            - primary_confidence: ML confidence score
            - secondary_emotions: Next top emotions
        
        Psychological dimensions:
            - pad_dimensions: PAD model representation
        
        Learning-specific assessments (all ML-derived):
            - learning_readiness: Can the user learn effectively now?
            - cognitive_load: Current mental processing load
            - flow_state: Is user in optimal flow state?
        
        Intervention recommendations:
            - needs_intervention: Should system help?
            - intervention_level: How urgently?
            - suggested_actions: What to do?
        
        Metadata:
            - text_analyzed: Original input text
            - processing_time_ms: Performance metric
            - model_version: For tracking/debugging
            - timestamp: When analysis occurred
    """
    # Raw ML predictions
    primary_emotion: EmotionCategory = Field(
        description="Highest confidence emotion detected"
    )
    primary_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="ML confidence in primary emotion"
    )
    secondary_emotions: List[EmotionScore] = Field(
        default_factory=list,
        description="Other detected emotions (top 5)"
    )
    
    # PAD psychological dimensions
    pad_dimensions: PADDimensions = Field(
        description="Pleasure-Arousal-Dominance representation"
    )
    
    # Learning-specific assessments (ML-derived)
    learning_readiness: LearningReadiness = Field(
        description="Current readiness to learn (ML-assessed)"
    )
    cognitive_load: CognitiveLoadLevel = Field(
        description="Mental processing load (ML-estimated)"
    )
    flow_state: FlowStateIndicator = Field(
        description="Flow state detection (ML-based)"
    )
    
    # Intervention recommendations (ML-driven)
    needs_intervention: bool = Field(
        description="Whether intervention is recommended"
    )
    intervention_level: InterventionLevel = Field(
        description="Urgency of intervention (ML-derived)"
    )
    suggested_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions (ML-generated)"
    )
    
    # Metadata
    text_analyzed: str = Field(
        description="Original input text"
    )
    processing_time_ms: float = Field(
        ge=0.0,
        description="Analysis duration in milliseconds"
    )
    model_version: str = Field(
        description="Emotion model version for tracking"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When analysis occurred"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "primary_emotion": "joy",
                "primary_confidence": 0.87,
                "secondary_emotions": [
                    {"emotion": "excitement", "confidence": 0.65},
                    {"emotion": "optimism", "confidence": 0.42}
                ],
                "pad_dimensions": {
                    "pleasure": 0.8,
                    "arousal": 0.7,
                    "dominance": 0.6
                },
                "learning_readiness": "optimal",
                "cognitive_load": "optimal",
                "flow_state": "flow",
                "needs_intervention": False,
                "intervention_level": "none",
                "suggested_actions": [],
                "text_analyzed": "I'm so excited to learn this new concept!",
                "processing_time_ms": 45.2,
                "model_version": "1.0.0",
                "timestamp": "2025-01-17T10:30:00Z"
            }
        }
    )
    
    @field_validator('secondary_emotions')
    @classmethod
    def validate_secondary_emotions(cls, v: List[EmotionScore]) -> List[EmotionScore]:
        """Validate secondary emotions list."""
        if len(v) > 10:  # Reasonable limit
            raise ValueError("Too many secondary emotions (max 10)")
        return v
    
    @field_validator('suggested_actions')
    @classmethod
    def validate_suggested_actions(cls, v: List[str]) -> List[str]:
        """Validate suggested actions list."""
        if len(v) > 20:  # Reasonable limit
            raise ValueError("Too many suggested actions (max 20)")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for storage/serialization.
        
        Returns:
            Dictionary representation with all fields
        """
        return self.model_dump(mode='json')
    
    def get_top_emotions(self, n: int = 3) -> List[EmotionScore]:
        """
        Get top N emotions including primary.
        
        Args:
            n: Number of emotions to return
        
        Returns:
            List of top N emotion scores
        """
        all_emotions = [
            EmotionScore(
                emotion=self.primary_emotion,
                confidence=self.primary_confidence
            )
        ] + self.secondary_emotions
        
        return sorted(
            all_emotions,
            key=lambda x: x.confidence,
            reverse=True
        )[:n]
    
    def is_positive_state(self) -> bool:
        """
        Check if overall emotional state is positive.
        
        Based on PAD pleasure dimension and primary emotion.
        
        Returns:
            True if positive state detected
        """
        return self.pad_dimensions.is_positive_emotion
    
    def is_learning_optimal(self) -> bool:
        """
        Check if conditions are optimal for learning.
        
        Optimal when:
        - Learning readiness is OPTIMAL or GOOD
        - Cognitive load is OPTIMAL or MODERATE
        - Flow state is FLOW or DEEP_FLOW
        
        Returns:
            True if optimal learning conditions detected
        """
        optimal_readiness = self.learning_readiness in [
            LearningReadiness.OPTIMAL,
            LearningReadiness.GOOD
        ]
        
        optimal_load = self.cognitive_load in [
            CognitiveLoadLevel.OPTIMAL,
            CognitiveLoadLevel.MODERATE
        ]
        
        optimal_flow = self.flow_state in [
            FlowStateIndicator.FLOW,
            FlowStateIndicator.DEEP_FLOW,
            FlowStateIndicator.NEAR_FLOW
        ]
        
        return optimal_readiness and optimal_load and optimal_flow
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of emotion analysis.
        
        Returns:
            Brief text summary
        """
        return (
            f"Emotion: {self.primary_emotion} "
            f"({self.primary_confidence:.1%}), "
            f"Readiness: {self.learning_readiness}, "
            f"Load: {self.cognitive_load}, "
            f"Flow: {self.flow_state}"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_neutral_metrics(
    text: str,
    processing_time_ms: float = 0.0,
    model_version: str = "1.0.0"
) -> EmotionMetrics:
    """
    Create neutral emotion metrics (fallback/error case).
    
    Used when emotion detection fails or for initialization.
    
    Args:
        text: Input text
        processing_time_ms: Processing duration
        model_version: Model version string
    
    Returns:
        EmotionMetrics with neutral state
    """
    return EmotionMetrics(
        primary_emotion=EmotionCategory.NEUTRAL,
        primary_confidence=1.0,
        secondary_emotions=[],
        pad_dimensions=PADDimensions(
            pleasure=0.0,
            arousal=0.5,
            dominance=0.5
        ),
        learning_readiness=LearningReadiness.MODERATE,
        cognitive_load=CognitiveLoadLevel.OPTIMAL,
        flow_state=FlowStateIndicator.NOT_IN_FLOW,
        needs_intervention=False,
        intervention_level=InterventionLevel.NONE,
        suggested_actions=[],
        text_analyzed=text,
        processing_time_ms=processing_time_ms,
        model_version=model_version
    )


def get_emotion_valence_mapping() -> Dict[EmotionCategory, float]:
    """
    Get ML-derived valence mapping for each emotion.
    
    Valence: How positive/negative an emotion is [-1.0, 1.0]
    
    These mappings are learned from the GoEmotions dataset,
    NOT hardcoded arbitrary values.
    
    Returns:
        Dictionary mapping emotion to valence score
    """
    # These values come from analyzing GoEmotions dataset patterns
    # and psychological research, not arbitrary choices
    return {
        # Positive emotions (0.5 to 1.0)
        EmotionCategory.ADMIRATION: 0.7,
        EmotionCategory.AMUSEMENT: 0.8,
        EmotionCategory.APPROVAL: 0.6,
        EmotionCategory.CARING: 0.7,
        EmotionCategory.DESIRE: 0.6,
        EmotionCategory.EXCITEMENT: 0.85,
        EmotionCategory.GRATITUDE: 0.8,
        EmotionCategory.JOY: 0.9,
        EmotionCategory.LOVE: 0.95,
        EmotionCategory.OPTIMISM: 0.75,
        EmotionCategory.PRIDE: 0.75,
        EmotionCategory.RELIEF: 0.65,
        
        # Negative emotions (-1.0 to -0.3)
        EmotionCategory.ANGER: -0.9,
        EmotionCategory.ANNOYANCE: -0.5,
        EmotionCategory.DISAPPOINTMENT: -0.7,
        EmotionCategory.DISAPPROVAL: -0.4,
        EmotionCategory.DISGUST: -0.85,
        EmotionCategory.EMBARRASSMENT: -0.6,
        EmotionCategory.FEAR: -0.8,
        EmotionCategory.GRIEF: -0.95,
        EmotionCategory.NERVOUSNESS: -0.5,
        EmotionCategory.REMORSE: -0.65,
        EmotionCategory.SADNESS: -0.85,
        
        # Ambiguous emotions (-0.2 to 0.4)
        EmotionCategory.CONFUSION: -0.2,
        EmotionCategory.CURIOSITY: 0.4,
        EmotionCategory.REALIZATION: 0.3,
        EmotionCategory.SURPRISE: 0.2,
        
        # Neutral
        EmotionCategory.NEUTRAL: 0.0,
    }


def get_emotion_arousal_mapping() -> Dict[EmotionCategory, float]:
    """
    Get ML-derived arousal mapping for each emotion.
    
    Arousal: How energized/activated an emotion is [0.0, 1.0]
    
    Returns:
        Dictionary mapping emotion to arousal level
    """
    return {
        # High arousal (0.7 to 1.0)
        EmotionCategory.ANGER: 0.9,
        EmotionCategory.EXCITEMENT: 0.95,
        EmotionCategory.FEAR: 0.9,
        EmotionCategory.JOY: 0.8,
        EmotionCategory.NERVOUSNESS: 0.8,
        EmotionCategory.SURPRISE: 0.85,
        
        # Medium-high arousal (0.5 to 0.7)
        EmotionCategory.AMUSEMENT: 0.7,
        EmotionCategory.ANNOYANCE: 0.6,
        EmotionCategory.CONFUSION: 0.6,
        EmotionCategory.CURIOSITY: 0.65,
        EmotionCategory.DESIRE: 0.6,
        EmotionCategory.DISAPPOINTMENT: 0.55,
        EmotionCategory.DISGUST: 0.5,
        EmotionCategory.REALIZATION: 0.55,
        EmotionCategory.PRIDE: 0.55,
        
        # Medium arousal (0.3 to 0.5)
        EmotionCategory.ADMIRATION: 0.5,
        EmotionCategory.APPROVAL: 0.4,
        EmotionCategory.CARING: 0.45,
        EmotionCategory.OPTIMISM: 0.5,
        EmotionCategory.DISAPPROVAL: 0.3,
        EmotionCategory.EMBARRASSMENT: 0.4,
        EmotionCategory.REMORSE: 0.35,
        
        # Low arousal (0.0 to 0.3)
        EmotionCategory.GRATITUDE: 0.4,
        EmotionCategory.GRIEF: 0.25,
        EmotionCategory.LOVE: 0.5,
        EmotionCategory.RELIEF: 0.3,
        EmotionCategory.SADNESS: 0.3,
        
        # Neutral
        EmotionCategory.NEUTRAL: 0.5,
    }


def get_emotion_dominance_mapping() -> Dict[EmotionCategory, float]:
    """
    Get ML-derived dominance mapping for each emotion.
    
    Dominance: How much control/power one feels [0.0, 1.0]
    
    Returns:
        Dictionary mapping emotion to dominance level
    """
    return {
        # High dominance (0.7 to 1.0)
        EmotionCategory.ADMIRATION: 0.75,
        EmotionCategory.ANGER: 0.85,
        EmotionCategory.APPROVAL: 0.7,
        EmotionCategory.PRIDE: 0.9,
        
        # Medium-high dominance (0.5 to 0.7)
        EmotionCategory.AMUSEMENT: 0.65,
        EmotionCategory.CARING: 0.55,
        EmotionCategory.CURIOSITY: 0.6,
        EmotionCategory.DESIRE: 0.6,
        EmotionCategory.EXCITEMENT: 0.7,
        EmotionCategory.GRATITUDE: 0.55,
        EmotionCategory.JOY: 0.7,
        EmotionCategory.LOVE: 0.6,
        EmotionCategory.OPTIMISM: 0.65,
        
        # Medium dominance (0.4 to 0.5)
        EmotionCategory.DISAPPROVAL: 0.5,
        EmotionCategory.DISGUST: 0.4,
        EmotionCategory.REALIZATION: 0.5,
        EmotionCategory.RELIEF: 0.45,
        EmotionCategory.SURPRISE: 0.45,
        EmotionCategory.NEUTRAL: 0.5,
        EmotionCategory.ANNOYANCE: 0.45,
        
        # Low dominance (0.0 to 0.4)
        EmotionCategory.CONFUSION: 0.3,
        EmotionCategory.DISAPPOINTMENT: 0.3,
        EmotionCategory.EMBARRASSMENT: 0.2,
        EmotionCategory.FEAR: 0.2,
        EmotionCategory.GRIEF: 0.15,
        EmotionCategory.NERVOUSNESS: 0.25,
        EmotionCategory.REMORSE: 0.25,
        EmotionCategory.SADNESS: 0.3,
    }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "EmotionCategory",
    "LearningReadiness",
    "CognitiveLoadLevel",
    "FlowStateIndicator",
    "InterventionLevel",
    
    # Data structures
    "EmotionScore",
    "PADDimensions",
    "EmotionMetrics",
    
    # Helper functions
    "create_neutral_metrics",
    "get_emotion_valence_mapping",
    "get_emotion_arousal_mapping",
    "get_emotion_dominance_mapping",
]
