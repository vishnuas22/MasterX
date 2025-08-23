"""
🚀 ENHANCED LLM-OPTIMIZED DATA MODELS
Revolutionary data structures optimized for maximum LLM performance

BREAKTHROUGH OPTIMIZATIONS:
- Context compression for efficient token usage
- LLM-friendly data serialization
- Performance-tracked conversation structures
- AI-provider optimized user profiles
- Real-time adaptation data structures

PERFORMANCE FEATURES:
- Token-efficient context serialization
- Relevance-weighted information structures
- Dynamic context adjustment capabilities
- Real-time performance feedback integration
- Predictive user modeling structures

Author: MasterX Enhancement Team
Version: 4.0 - LLM Performance Optimization
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json

# ============================================================================
# ENHANCED ENUMS FOR LLM OPTIMIZATION
# ============================================================================

class ContextRelevanceLevel(str, Enum):
    """Context relevance levels for LLM optimization"""
    CRITICAL = "critical"       # Must include in context
    HIGH = "high"              # Very important for response quality
    MODERATE = "moderate"      # Helpful but not essential
    LOW = "low"               # Background information
    MINIMAL = "minimal"       # Only if space available

class LLMTaskType(str, Enum):
    """Optimized task types for provider selection"""
    EMOTIONAL_SUPPORT = "emotional_support"      # Groq excels (89% empathy)
    COMPLEX_EXPLANATION = "complex_explanation"  # Gemini optimal
    PERSONALIZED_LEARNING = "personalized_learning" # Emergent specialized
    QUICK_RESPONSE = "quick_response"           # Speed-optimized
    CREATIVE_CONTENT = "creative_content"       # Creativity-focused
    ANALYTICAL_REASONING = "analytical_reasoning" # Logic-focused
    CODE_EXAMPLES = "code_examples"            # Technical content
    BEGINNER_CONCEPTS = "beginner_concepts"    # Simple explanations
    ADVANCED_CONCEPTS = "advanced_concepts"    # Complex topics

class AdaptationTrigger(str, Enum):
    """Real-time adaptation triggers"""
    CONFUSION_DETECTED = "confusion_detected"
    FRUSTRATION_RISING = "frustration_rising"
    ENGAGEMENT_DROPPING = "engagement_dropping"
    COMPREHENSION_POOR = "comprehension_poor"
    PACE_TOO_FAST = "pace_too_fast"
    PACE_TOO_SLOW = "pace_too_slow"
    EMOTIONAL_DISTRESS = "emotional_distress"
    MOTIVATION_LOW = "motivation_low"

# ============================================================================
# LLM-OPTIMIZED USER PROFILE
# ============================================================================

class LLMOptimizedLearningProfile(BaseModel):
    """LLM-optimized user profile for maximum context efficiency"""
    user_id: str
    profile_version: str = "4.0"
    
    # CORE IDENTITY (Always included in context)
    learning_identity: Dict[str, Any] = Field(default_factory=lambda: {
        "preferred_name": None,
        "learning_style": "adaptive",
        "difficulty_preference": "moderate",
        "interaction_pace": "moderate"
    })
    
    # CONTEXT-CRITICAL PREFERENCES (High relevance)
    critical_preferences: Dict[str, Any] = Field(default_factory=lambda: {
        "explanation_style": "balanced",      # visual, analytical, conversational
        "content_depth": "moderate",          # brief, moderate, detailed
        "feedback_frequency": "regular",      # minimal, regular, frequent
        "challenge_tolerance": 0.5,           # 0.0-1.0
        "encouragement_level": "moderate"     # minimal, moderate, high
    })
    
    # ADAPTIVE PARAMETERS (Real-time optimization)
    adaptive_parameters: Dict[str, float] = Field(default_factory=lambda: {
        "current_difficulty": 0.5,           # Current optimal difficulty
        "learning_velocity": 0.6,            # Rate of learning
        "engagement_level": 0.7,             # Current engagement
        "comprehension_rate": 0.6,           # Understanding rate
        "frustration_tolerance": 0.5,        # Frustration threshold
        "motivation_level": 0.7              # Current motivation
    })
    
    # CONTEXT OPTIMIZATION
    context_optimization: Dict[str, Any] = Field(default_factory=lambda: {
        "max_context_tokens": 8000,          # Token budget for context
        "context_priorities": {              # Priority weights for context types
            "learning_preferences": 0.9,
            "recent_conversation": 0.8,
            "emotional_state": 0.7,
            "learning_progress": 0.6,
            "background_knowledge": 0.4
        },
        "compression_level": "moderate",      # Context compression level
        "relevance_threshold": 0.3           # Minimum relevance to include
    })
    
    # PERFORMANCE TRACKING
    performance_metrics: Dict[str, float] = Field(default_factory=lambda: {
        "personalization_effectiveness": 0.5,
        "context_utilization_success": 0.5,
        "adaptation_response_rate": 0.5,
        "learning_acceleration": 1.0,
        "satisfaction_trend": 0.7
    })
    
    # PROVIDER OPTIMIZATION
    provider_preferences: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        "task_performance": {},              # Task -> Provider performance scores
        "response_quality": {},              # Provider -> Quality scores
        "user_satisfaction": {},             # Provider -> Satisfaction scores
        "adaptation_effectiveness": {}        # Provider -> Adaptation success
    })
    
    # REAL-TIME STATE
    current_state: Dict[str, Any] = Field(default_factory=lambda: {
        "emotional_state": "engaged",
        "energy_level": "moderate",
        "focus_level": "good",
        "learning_mode": "active",
        "session_context": {},
        "last_adaptation": None,
        "active_triggers": []
    })
    
    # TIMESTAMPS
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    last_interaction: datetime = Field(default_factory=datetime.utcnow)

    def get_context_optimized_profile(self, task_type: LLMTaskType, token_budget: int = 2000) -> Dict[str, Any]:
        """Get token-optimized profile for LLM context"""
        context = {
            "user_id": self.user_id,
            "learning_identity": self.learning_identity,
            "critical_preferences": self.critical_preferences,
            "current_parameters": self.adaptive_parameters,
            "current_state": self.current_state
        }
        
        # Add task-specific optimizations
        if task_type == LLMTaskType.EMOTIONAL_SUPPORT:
            context["emotional_focus"] = {
                "emotional_state": self.current_state.get("emotional_state"),
                "frustration_tolerance": self.adaptive_parameters.get("frustration_tolerance"),
                "encouragement_level": self.critical_preferences.get("encouragement_level")
            }
        elif task_type == LLMTaskType.PERSONALIZED_LEARNING:
            context["learning_focus"] = {
                "learning_velocity": self.adaptive_parameters.get("learning_velocity"),
                "difficulty_preference": self.learning_identity.get("difficulty_preference"),
                "comprehension_rate": self.adaptive_parameters.get("comprehension_rate")
            }
        
        return context

    def update_performance_feedback(self, provider: str, task_type: str, effectiveness: float):
        """Update provider performance based on user feedback"""
        if "task_performance" not in self.provider_preferences:
            self.provider_preferences["task_performance"] = {}
        
        task_key = f"{provider}_{task_type}"
        current_score = self.provider_preferences["task_performance"].get(task_key, 0.5)
        # Weighted average with recent feedback
        new_score = (current_score * 0.7) + (effectiveness * 0.3)
        self.provider_preferences["task_performance"][task_key] = new_score
        self.last_updated = datetime.utcnow()

# ============================================================================
# ENHANCED CONVERSATION MODELS
# ============================================================================

class LLMOptimizedMessage(BaseModel):
    """LLM-optimized message structure for efficient processing"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    
    # MESSAGE CONTENT
    content: str
    sender: Literal["user", "ai"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # LLM OPTIMIZATION DATA
    token_count: Optional[int] = None
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    context_importance: ContextRelevanceLevel = ContextRelevanceLevel.MODERATE
    
    # AI GENERATION METADATA
    ai_provider: Optional[str] = None
    ai_model: Optional[str] = None
    generation_time_ms: Optional[float] = None
    context_tokens_used: Optional[int] = None
    
    # PERFORMANCE TRACKING
    user_satisfaction: Optional[float] = Field(None, ge=0.0, le=1.0)
    learning_impact: float = Field(default=0.0, ge=-1.0, le=1.0)
    comprehension_boost: float = Field(default=0.0, ge=-1.0, le=1.0)
    engagement_change: float = Field(default=0.0, ge=-1.0, le=1.0)
    
    # ADAPTATION DATA
    adaptations_triggered: List[AdaptationTrigger] = Field(default_factory=list)
    context_effectiveness: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    def get_context_summary(self, max_tokens: int = 100) -> str:
        """Get token-efficient summary for context"""
        if self.token_count and self.token_count <= max_tokens:
            return self.content
        
        # Simple truncation for now - could use more sophisticated summarization
        words = self.content.split()
        if len(words) <= max_tokens // 4:  # Rough token estimation
            return self.content
        
        truncated = " ".join(words[:max_tokens // 4])
        return f"{truncated}..."

class LLMOptimizedConversation(BaseModel):
    """LLM-optimized conversation structure"""
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    
    # CONVERSATION METADATA
    title: Optional[str] = None
    primary_topic: Optional[str] = None
    learning_objectives: List[str] = Field(default_factory=list)
    
    # MESSAGE MANAGEMENT
    messages: List[LLMOptimizedMessage] = Field(default_factory=list)
    message_count: int = 0
    
    # CONTEXT OPTIMIZATION
    context_window_size: int = 10          # Number of recent messages to consider
    max_context_tokens: int = 6000         # Maximum tokens for conversation context
    context_compression_ratio: float = 0.7  # How much to compress older context
    
    # REAL-TIME ANALYTICS
    current_engagement: float = Field(default=0.7, ge=0.0, le=1.0)
    learning_progress: float = Field(default=0.0, ge=-1.0, le=1.0)
    comprehension_trend: List[float] = Field(default_factory=list)
    emotional_trend: List[str] = Field(default_factory=list)
    
    # ADAPTATION TRACKING
    active_adaptations: List[str] = Field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = Field(default_factory=list)
    adaptation_effectiveness: Dict[str, float] = Field(default_factory=dict)
    
    # PROVIDER PERFORMANCE
    provider_usage: Dict[str, int] = Field(default_factory=dict)
    provider_satisfaction: Dict[str, List[float]] = Field(default_factory=dict)
    
    # TIMESTAMPS
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    def add_message(self, message: LLMOptimizedMessage):
        """Add message and update conversation metrics"""
        self.messages.append(message)
        self.message_count = len(self.messages)
        self.last_activity = datetime.utcnow()
        
        # Update provider usage
        if message.ai_provider:
            self.provider_usage[message.ai_provider] = self.provider_usage.get(message.ai_provider, 0) + 1
        
        # Update trends
        if message.learning_impact != 0:
            self.learning_progress += message.learning_impact * 0.1
            self.learning_progress = max(-1.0, min(1.0, self.learning_progress))
        
        # Update comprehension trend
        if hasattr(message, 'comprehension_boost'):
            self.comprehension_trend.append(message.comprehension_boost)
            if len(self.comprehension_trend) > 20:
                self.comprehension_trend = self.comprehension_trend[-20:]

    def get_optimized_context(self, token_budget: int = 4000) -> Dict[str, Any]:
        """Get token-optimized conversation context"""
        # Get recent messages within token budget
        recent_messages = []
        token_count = 0
        
        for message in reversed(self.messages):
            message_tokens = message.token_count or len(message.content.split()) * 1.3
            if token_count + message_tokens > token_budget:
                break
            recent_messages.insert(0, message)
            token_count += message_tokens
        
        # Create context structure
        context = {
            "conversation_id": self.conversation_id,
            "primary_topic": self.primary_topic,
            "learning_objectives": self.learning_objectives,
            "recent_messages": [
                {
                    "sender": msg.sender,
                    "content": msg.get_context_summary(100),
                    "timestamp": msg.timestamp,
                    "relevance": msg.relevance_score
                }
                for msg in recent_messages
            ],
            "current_metrics": {
                "engagement": self.current_engagement,
                "learning_progress": self.learning_progress,
                "comprehension_trend": self.comprehension_trend[-5:] if self.comprehension_trend else [],
                "active_adaptations": self.active_adaptations
            }
        }
        
        return context

# ============================================================================
# ENHANCED CONTEXT INJECTION MODELS
# ============================================================================

class LLMContextInjection(BaseModel):
    """LLM-optimized context injection system"""
    injection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # CONTEXT COMPONENTS
    user_profile_context: Optional[str] = None
    conversation_context: Optional[str] = None
    learning_analytics_context: Optional[str] = None
    emotional_context: Optional[str] = None
    adaptation_instructions: Optional[str] = None
    
    # OPTIMIZATION PARAMETERS
    total_tokens: int = 0
    relevance_weights: Dict[str, float] = Field(default_factory=lambda: {
        "user_profile": 0.9,
        "conversation": 0.8,
        "emotional": 0.7,
        "analytics": 0.6,
        "adaptation": 0.8
    })
    
    # PERFORMANCE TRACKING
    effectiveness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    user_response_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    context_utilization: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # TIMESTAMPS
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_optimized_injection(self, token_budget: int = 2000) -> str:
        """Get token-optimized context injection"""
        components = []
        remaining_tokens = token_budget
        
        # Priority order based on weights
        context_items = [
            ("user_profile", self.user_profile_context, self.relevance_weights.get("user_profile", 0.5)),
            ("conversation", self.conversation_context, self.relevance_weights.get("conversation", 0.5)),
            ("adaptation", self.adaptation_instructions, self.relevance_weights.get("adaptation", 0.5)),
            ("emotional", self.emotional_context, self.relevance_weights.get("emotional", 0.5)),
            ("analytics", self.learning_analytics_context, self.relevance_weights.get("analytics", 0.5))
        ]
        
        # Sort by relevance weight
        context_items.sort(key=lambda x: x[2], reverse=True)
        
        for context_type, content, weight in context_items:
            if not content or remaining_tokens <= 0:
                continue
                
            # Estimate tokens (rough approximation)
            content_tokens = len(content.split()) * 1.3
            
            if content_tokens <= remaining_tokens:
                components.append(f"[{context_type.upper()}]: {content}")
                remaining_tokens -= content_tokens
            elif remaining_tokens > 50:  # Enough space for truncated version
                words = content.split()
                truncated_words = int(remaining_tokens / 1.3)
                truncated_content = " ".join(words[:truncated_words])
                components.append(f"[{context_type.upper()}]: {truncated_content}...")
                remaining_tokens = 0
        
        return "\n".join(components)

# ============================================================================
# REAL-TIME ADAPTATION MODELS
# ============================================================================

class RealTimeAdaptation(BaseModel):
    """Real-time adaptation tracking and optimization"""
    adaptation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    conversation_id: str
    
    # TRIGGER INFORMATION
    trigger_type: AdaptationTrigger
    trigger_confidence: float = Field(ge=0.0, le=1.0)
    trigger_indicators: List[str] = Field(default_factory=list)
    
    # ADAPTATION RESPONSE
    adaptation_type: str  # difficulty_adjustment, pace_change, style_change, etc.
    adaptation_parameters: Dict[str, Any] = Field(default_factory=dict)
    expected_improvement: float = Field(default=0.0, ge=-1.0, le=1.0)
    
    # IMPLEMENTATION
    context_modifications: Dict[str, str] = Field(default_factory=dict)
    provider_adjustments: Dict[str, Any] = Field(default_factory=dict)
    
    # PERFORMANCE TRACKING
    effectiveness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    user_response_improvement: Optional[float] = Field(None, ge=-1.0, le=1.0)
    learning_impact: Optional[float] = Field(None, ge=-1.0, le=1.0)
    
    # TIMESTAMPS
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    implemented_at: Optional[datetime] = None
    measured_at: Optional[datetime] = None

# ============================================================================
# PROVIDER PERFORMANCE OPTIMIZATION
# ============================================================================

class LLMProviderPerformance(BaseModel):
    """LLM provider performance optimization tracking"""
    performance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    provider_name: str
    model_name: str
    
    # TASK-SPECIFIC PERFORMANCE
    task_performance: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        "emotional_support": {"score": 0.0, "count": 0},
        "complex_explanation": {"score": 0.0, "count": 0},
        "personalized_learning": {"score": 0.0, "count": 0},
        "quick_response": {"score": 0.0, "count": 0},
        "creative_content": {"score": 0.0, "count": 0}
    })
    
    # REAL-TIME METRICS
    current_performance: Dict[str, float] = Field(default_factory=lambda: {
        "response_time_ms": 0.0,
        "token_efficiency": 0.0,
        "context_utilization": 0.0,
        "user_satisfaction": 0.0,
        "adaptation_effectiveness": 0.0
    })
    
    # OPTIMIZATION PARAMETERS
    optimal_contexts: Dict[str, Any] = Field(default_factory=dict)
    performance_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    
    # TIMESTAMPS
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def update_task_performance(self, task_type: str, score: float):
        """Update task-specific performance score"""
        if task_type not in self.task_performance:
            self.task_performance[task_type] = {"score": score, "count": 1}
        else:
            current = self.task_performance[task_type]
            new_count = current["count"] + 1
            new_score = ((current["score"] * current["count"]) + score) / new_count
            self.task_performance[task_type] = {"score": new_score, "count": new_count}
        
        self.last_updated = datetime.utcnow()
    
    def get_task_recommendation(self, task_type: str) -> float:
        """Get recommendation score for specific task type"""
        return self.task_performance.get(task_type, {}).get("score", 0.5)

# ============================================================================
# EXPORT ALL MODELS
# ============================================================================

__all__ = [
    # Enums
    "ContextRelevanceLevel", "LLMTaskType", "AdaptationTrigger",
    
    # User Profile Models
    "LLMOptimizedLearningProfile",
    
    # Conversation Models  
    "LLMOptimizedMessage", "LLMOptimizedConversation",
    
    # Context Models
    "LLMContextInjection",
    
    # Adaptation Models
    "RealTimeAdaptation",
    
    # Provider Models
    "LLMProviderPerformance"
]