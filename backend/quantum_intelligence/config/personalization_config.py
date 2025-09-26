# MAXIMUM PERSONALIZATION CONFIGURATION
# This configuration prioritizes personalization over speed

PERSONALIZATION_MODE = "maximum"
AI_OPTIMIZATION_DISABLED = True
FORCE_REAL_AI_PROVIDERS = True
ENABLE_EMOTION_DETECTION = True
ENABLE_DIFFICULTY_ADAPTATION = True
ENABLE_EMPATHY_ANALYSIS = True

# Performance targets (relaxed for better personalization)
TARGET_RESPONSE_TIME_MS = 3000  # 3 seconds for quality AI analysis
ULTRA_TARGET_MS = 1500  # 1.5 seconds for urgent requests
MAX_RESPONSE_TIME_MS = 5000  # 5 seconds maximum

# Real AI provider preferences
PREFERRED_PROVIDERS = ["groq", "emergent", "gemini"]
FALLBACK_TO_MOCK = False  # Never use mock responses
CACHE_REAL_RESPONSES = True  # Cache real AI responses for efficiency
