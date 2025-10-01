"""
Database Models & Schemas for MasterX
Following specifications from 2.CRITICAL_INITIAL_SETUP.md Section 1

All models use UUID4 for IDs (NOT MongoDB ObjectId)
Full type hints, Pydantic V2 validation
"""

from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    PREMIUM = "premium"


class LearningStyle(str, Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class LearningReadiness(str, Enum):
    HIGH_READINESS = "high_readiness"
    MODERATE_READINESS = "moderate_readiness"
    LOW_READINESS = "low_readiness"
    NEEDS_BREAK = "needs_break"


class ProviderStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class CircuitBreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ============================================================================
# USER MODELS
# ============================================================================

class LearningPreferences(BaseModel):
    """User learning preferences"""
    preferred_subjects: List[str] = Field(default_factory=list)
    learning_style: LearningStyle = LearningStyle.VISUAL
    difficulty_preference: str = "adaptive"


class EmotionalProfile(BaseModel):
    """User emotional baseline profile"""
    baseline_engagement: float = Field(default=0.7, ge=0.0, le=1.0)
    frustration_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    celebration_responsiveness: float = Field(default=0.8, ge=0.0, le=1.0)


class UserProfile(BaseModel):
    """User profile model - matches MongoDB users collection"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    email: EmailStr
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    learning_preferences: LearningPreferences = Field(default_factory=LearningPreferences)
    emotional_profile: EmotionalProfile = Field(default_factory=EmotionalProfile)
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    total_sessions: int = 0
    last_active: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "john@example.com",
                "name": "John Doe",
                "subscription_tier": "free"
            }
        }


# ============================================================================
# SESSION MODELS
# ============================================================================

class LearningSession(BaseModel):
    """Learning session model - matches MongoDB sessions collection"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    user_id: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    current_topic: Optional[str] = None
    assigned_provider: Optional[str] = None
    total_messages: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time_ms: float = 0.0
    emotion_trajectory: List[str] = Field(default_factory=list)
    performance_score: float = 0.75
    status: SessionStatus = SessionStatus.ACTIVE

    class Config:
        populate_by_name = True


# ============================================================================
# MESSAGE MODELS
# ============================================================================

class EmotionState(BaseModel):
    """Emotion analysis state"""
    primary_emotion: str
    arousal: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=0.0, le=1.0)
    learning_readiness: LearningReadiness


class Message(BaseModel):
    """Message model - matches MongoDB messages collection"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    session_id: str
    user_id: str
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    emotion_state: Optional[EmotionState] = None
    provider_used: Optional[str] = None
    response_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    embedding: Optional[List[float]] = None
    quality_rating: Optional[int] = Field(None, ge=1, le=5)

    class Config:
        populate_by_name = True


# ============================================================================
# BENCHMARK MODELS
# ============================================================================

class BenchmarkTestResult(BaseModel):
    """Individual benchmark test result"""
    test_id: str
    quality: float = Field(ge=0.0, le=100.0)
    time_ms: float
    passed: bool


class BenchmarkResult(BaseModel):
    """Provider benchmark results - matches MongoDB benchmark_results collection"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    category: str
    provider: str
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    quality_score: float = Field(ge=0.0, le=100.0)
    speed_score: float = Field(ge=0.0, le=100.0)
    cost_score: float = Field(ge=0.0, le=100.0)
    final_score: float = Field(ge=0.0, le=100.0)
    avg_response_time_ms: float
    avg_cost: float
    tests_passed: int
    tests_total: int
    test_results: List[BenchmarkTestResult] = Field(default_factory=list)

    class Config:
        populate_by_name = True


# ============================================================================
# PROVIDER HEALTH MODELS
# ============================================================================

class ProviderHealth(BaseModel):
    """Provider health status - matches MongoDB provider_health collection"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    provider: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: ProviderStatus
    success_rate: float = Field(ge=0.0, le=1.0)
    avg_response_time_ms: float
    requests_last_hour: int
    errors_last_hour: int
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None

    class Config:
        populate_by_name = True


# ============================================================================
# USER PERFORMANCE MODELS
# ============================================================================

class UserPerformance(BaseModel):
    """User learning performance - matches MongoDB user_performance collection"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    user_id: str
    subject: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ability_level: float = Field(ge=0.0, le=1.0)
    difficulty_preference: float = Field(ge=0.0, le=1.0)
    learning_velocity: float
    mastery_topics: List[str] = Field(default_factory=list)
    struggling_topics: List[str] = Field(default_factory=list)
    total_practice_time_hours: float = 0.0
    improvement_rate: float = 0.0

    class Config:
        populate_by_name = True


# ============================================================================
# COST TRACKING MODELS
# ============================================================================

class CategoryBreakdown(BaseModel):
    """Cost breakdown by category"""
    coding: float = 0.0
    math: float = 0.0
    empathy: float = 0.0
    research: float = 0.0
    general: float = 0.0


class CostTracking(BaseModel):
    """API cost tracking - matches MongoDB cost_tracking collection"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    date: str  # YYYY-MM-DD format
    provider: str
    user_id: Optional[str] = None
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_cost_per_request: float = 0.0
    category_breakdown: CategoryBreakdown = Field(default_factory=CategoryBreakdown)

    class Config:
        populate_by_name = True


# ============================================================================
# AI RESPONSE MODELS
# ============================================================================

class AIResponse(BaseModel):
    """AI provider response"""
    content: str
    provider: str
    model_name: str
    tokens_used: int
    cost: float
    response_time_ms: float
    emotion_state: Optional[EmotionState] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# API REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat API request"""
    user_id: str
    session_id: Optional[str] = None
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat API response"""
    session_id: str
    message: str
    emotion_state: Optional[EmotionState] = None
    provider_used: str
    response_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# DATABASE INDEXES CONFIGURATION
# ============================================================================

INDEXES = {
    "users": [
        {"keys": [("email", 1)], "unique": True},
        {"keys": [("last_active", -1)]}
    ],
    "sessions": [
        {"keys": [("user_id", 1), ("started_at", -1)]},
        {"keys": [("status", 1), ("started_at", -1)]},
        {"keys": [("current_topic", 1)]}
    ],
    "messages": [
        {"keys": [("session_id", 1), ("timestamp", 1)]},
        {"keys": [("user_id", 1), ("timestamp", -1)]}
    ],
    "benchmark_results": [
        {"keys": [("category", 1), ("timestamp", -1)]},
        {"keys": [("provider", 1), ("category", 1), ("timestamp", -1)]},
        {"keys": [("timestamp", -1)]}
    ],
    "provider_health": [
        {"keys": [("provider", 1), ("timestamp", -1)]},
        {"keys": [("status", 1), ("timestamp", -1)]}
    ],
    "user_performance": [
        {"keys": [("user_id", 1), ("subject", 1), ("timestamp", -1)]},
        {"keys": [("timestamp", -1)]}
    ],
    "cost_tracking": [
        {"keys": [("date", -1), ("provider", 1)]},
        {"keys": [("user_id", 1), ("date", -1)]}
    ]
}
