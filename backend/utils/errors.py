"""
Unified Error Handling System
Following specifications from 2.CRITICAL_INITIAL_SETUP.md Section 6
"""

from typing import Dict, Any


class MasterXError(Exception):
    """Base exception for all MasterX errors"""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ProviderError(MasterXError):
    """AI provider errors"""
    pass


class BenchmarkError(MasterXError):
    """Benchmarking errors"""
    pass


class EmotionDetectionError(MasterXError):
    """Emotion detection errors"""
    pass


class DatabaseError(MasterXError):
    """Database errors"""
    pass


class ValidationError(MasterXError):
    """Validation errors"""
    pass


class AuthenticationError(MasterXError):
    """Authentication errors"""
    pass


class RateLimitError(MasterXError):
    """Rate limit exceeded errors"""
    pass


class CostThresholdError(MasterXError):
    """Cost threshold exceeded errors"""
    pass
