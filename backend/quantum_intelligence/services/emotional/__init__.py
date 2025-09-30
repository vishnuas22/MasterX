"""
Emotional AI and Wellbeing Services for Quantum Intelligence Engine.

Enterprise-grade emotion detection system:
- Transformer-based emotion recognition (BERT/RoBERTa)
- Adaptive behavioral pattern analysis
- Real-time learning state prediction
- ML-driven intervention systems
- Multimodal emotion fusion

Author: MasterX AI Team
Version: 1.0 (Enhanced from v9.0)
"""

# Import new enhanced components (clean naming)
try:
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
    NEW_CORE_AVAILABLE = True
except ImportError:
    NEW_CORE_AVAILABLE = False

try:
    from .emotion_transformer import (
        EmotionTransformer,
        EmotionClassifier,
        AdaptiveThresholdManager
    )
    NEW_TRANSFORMER_AVAILABLE = True
except ImportError:
    NEW_TRANSFORMER_AVAILABLE = False

# Import V9.0 legacy components (backward compatibility)
try:
    from .authentic_emotion_core_v9 import (
        AuthenticEmotionV9Constants,
        AuthenticEmotionCategoryV9,
        AuthenticInterventionLevelV9,
        AuthenticLearningReadinessV9,
        AuthenticEmotionalTrajectoryV9,
        AuthenticEmotionMetricsV9,
        AuthenticEmotionResultV9,
        AuthenticBehavioralAnalyzer,
        AuthenticPatternRecognitionEngine
    )
    LEGACY_V9_AVAILABLE = True
except ImportError:
    LEGACY_V9_AVAILABLE = False

try:
    from .authentic_transformer_v9 import AuthenticEmotionTransformerV9
    LEGACY_TRANSFORMER_AVAILABLE = True
except ImportError:
    LEGACY_TRANSFORMER_AVAILABLE = False

try:
    from .authentic_emotion_engine_v9 import (
        RevolutionaryAuthenticEmotionEngineV9,
        authentic_emotion_engine_v9
    )
    LEGACY_ENGINE_AVAILABLE = True
except ImportError:
    LEGACY_ENGINE_AVAILABLE = False

# Import V8.0 Legacy Components (for compatibility)
try:
    from .emotion_detection_v8 import (
        UltraEnterpriseEmotionDetectionEngine,
        UltraEnterpriseEmotionResult,
        EmotionCategoryV8,
        InterventionLevelV8,
        LearningReadinessV8,
        EmotionDetectionV8Constants
    )
    LEGACY_V8_AVAILABLE = True
except ImportError:
    LEGACY_V8_AVAILABLE = False

# Import additional services
try:
    from .stress_monitoring import (
        StressMonitoringSystem,
        StressLevelData,
        BurnoutPreventionEngine
    )
    STRESS_MONITORING_AVAILABLE = True
except ImportError:
    STRESS_MONITORING_AVAILABLE = False

try:
    from .motivation import (
        MotivationBoostEngine,
        MotivationAnalysis,
        PersonalizedMotivationSystem
    )
    MOTIVATION_AVAILABLE = True
except ImportError:
    MOTIVATION_AVAILABLE = False

try:
    from .wellbeing import (
        MentalWellbeingTracker,
        WellbeingMetrics,
        BreakRecommendationEngine
    )
    WELLBEING_AVAILABLE = True
except ImportError:
    WELLBEING_AVAILABLE = False

# Define exports
__all__ = []

# Add new enhanced components
if NEW_CORE_AVAILABLE:
    __all__.extend([
        "EmotionConstants",
        "EmotionCategory",
        "InterventionLevel",
        "LearningReadiness",
        "EmotionalTrajectory",
        "EmotionMetrics",
        "EmotionResult",
        "BehavioralPattern"
    ])

if NEW_TRANSFORMER_AVAILABLE:
    __all__.extend([
        "EmotionTransformer",
        "EmotionClassifier",
        "AdaptiveThresholdManager"
    ])

# Add V9.0 legacy components for backward compatibility
if LEGACY_V9_AVAILABLE:
    __all__.extend([
        "AuthenticEmotionV9Constants",
        "AuthenticEmotionCategoryV9",
        "AuthenticInterventionLevelV9",
        "AuthenticLearningReadinessV9", 
        "AuthenticEmotionalTrajectoryV9",
        "AuthenticEmotionMetricsV9",
        "AuthenticEmotionResultV9",
        "AuthenticBehavioralAnalyzer",
        "AuthenticPatternRecognitionEngine"
    ])

if LEGACY_TRANSFORMER_AVAILABLE:
    __all__.append("AuthenticEmotionTransformerV9")

if LEGACY_ENGINE_AVAILABLE:
    __all__.extend([
        "RevolutionaryAuthenticEmotionEngineV9",
        "authentic_emotion_engine_v9"
    ])

# Add V8.0 legacy components if available
if LEGACY_V8_AVAILABLE:
    __all__.extend([
        "UltraEnterpriseEmotionDetectionEngine",
        "UltraEnterpriseEmotionResult", 
        "EmotionCategoryV8",
        "InterventionLevelV8",
        "LearningReadinessV8",
        "EmotionDetectionV8Constants"
    ])

# Add additional services if available
if STRESS_MONITORING_AVAILABLE:
    __all__.extend([
        "StressMonitoringSystem",
        "StressLevelData",
        "BurnoutPreventionEngine"
    ])

if MOTIVATION_AVAILABLE:
    __all__.extend([
        "MotivationBoostEngine", 
        "MotivationAnalysis",
        "PersonalizedMotivationSystem"
    ])

if WELLBEING_AVAILABLE:
    __all__.extend([
        "MentalWellbeingTracker",
        "WellbeingMetrics", 
        "BreakRecommendationEngine"
    ])

# Module-level convenience functions
def get_authentic_emotion_engine():
    """Get the V9.0 authentic emotion detection engine"""
    return authentic_emotion_engine_v9

def get_legacy_emotion_engine():
    """Get the V8.0 legacy emotion detection engine (if available)"""
    if LEGACY_V8_AVAILABLE:
        try:
            from .emotion_detection_v8 import ultra_enterprise_emotion_engine
            return ultra_enterprise_emotion_engine
        except ImportError:
            return None
    return None

async def initialize_all_emotion_services():
    """Initialize all available emotion detection services"""
    results = {}
    
    # Initialize V9.0 authentic engine
    try:
        v9_success = await authentic_emotion_engine_v9.initialize()
        results['v9_authentic'] = v9_success
    except Exception as e:
        results['v9_authentic'] = False
        results['v9_error'] = str(e)
    
    # Initialize V8.0 legacy engine if available
    if LEGACY_V8_AVAILABLE:
        try:
            legacy_engine = get_legacy_emotion_engine()
            if legacy_engine:
                v8_success = await legacy_engine.initialize({})
                results['v8_legacy'] = v8_success
        except Exception as e:
            results['v8_legacy'] = False
            results['v8_error'] = str(e)
    
    return results

# Version information
__version__ = "9.0.0"
__author__ = "MasterX Quantum Intelligence Team"
__description__ = "Revolutionary Authentic Emotion Detection with NO hardcoded values"