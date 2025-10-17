"""
MasterX Emotion Detection System

Real-time emotion detection with learning readiness assessment.
Built for global market competition with zero hardcoded values.
"""

from services.emotion.emotion_core import (
    EmotionCategory,
    LearningReadiness,
    CognitiveLoadLevel,
    FlowStateIndicator,
    InterventionLevel,
    EmotionScore,
    PADDimensions,
    EmotionMetrics,
)

__all__ = [
    "EmotionCategory",
    "LearningReadiness",
    "CognitiveLoadLevel",
    "FlowStateIndicator",
    "InterventionLevel",
    "EmotionScore",
    "PADDimensions",
    "EmotionMetrics",
]

__version__ = "1.0.0"
