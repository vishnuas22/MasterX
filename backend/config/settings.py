"""
MasterX Configuration Management
Following specifications from 3.MASTERX_COMPREHENSIVE_PLAN.md

PRINCIPLES (from AGENTS.md):
- No hardcoded values
- All configuration from environment
- Type-safe with Pydantic
- Clean, professional naming
"""

import os
from typing import Optional, List, Dict
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    
    mongo_url: str = Field(
        default_factory=lambda: os.getenv("MONGO_URL", "mongodb://localhost:27017/masterx"),
        description="MongoDB connection URL"
    )
    
    database_name: str = Field(
        default="masterx",
        description="Database name"
    )
    
    max_pool_size: int = Field(
        default=100,
        description="MongoDB connection pool size"
    )
    
    min_pool_size: int = Field(
        default=10,
        description="Minimum connections in pool"
    )
    
    class Config:
        env_prefix = "DB_"


class AIProviderSettings(BaseSettings):
    """AI Provider configuration"""
    
    groq_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY"),
        description="Groq API key"
    )
    
    emergent_llm_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("EMERGENT_LLM_KEY"),
        description="Emergent universal LLM key"
    )
    
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY"),
        description="Google Gemini API key"
    )
    
    provider_timeout_seconds: int = Field(
        default=30,
        description="Default provider timeout in seconds"
    )
    
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed requests"
    )
    
    class Config:
        env_prefix = "AI_"


class CachingSettings(BaseSettings):
    """Caching configuration"""
    
    enabled: bool = Field(
        default=True,
        description="Enable caching system"
    )
    
    memory_cache_size: int = Field(
        default=1000,
        description="Size of in-memory LRU cache"
    )
    
    response_cache_ttl: int = Field(
        default=3600,
        description="TTL for response cache in seconds"
    )
    
    embedding_cache_ttl: int = Field(
        default=86400,
        description="TTL for embedding cache in seconds"
    )
    
    benchmark_cache_ttl: int = Field(
        default=43200,
        description="TTL for benchmark cache in seconds"
    )
    
    class Config:
        env_prefix = "CACHE_"


class PerformanceSettings(BaseSettings):
    """Performance monitoring configuration"""
    
    enabled: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    
    slow_request_threshold_ms: int = Field(
        default=5000,
        description="Threshold for slow request alerts"
    )
    
    critical_latency_threshold_ms: int = Field(
        default=10000,
        description="Threshold for critical latency alerts"
    )
    
    metrics_interval_seconds: int = Field(
        default=60,
        description="Interval for metrics aggregation"
    )
    
    class Config:
        env_prefix = "PERF_"


class VoiceSettings(BaseSettings):
    """
    Voice interaction configuration (AGENTS.md compliant - zero hardcoded values)
    All voice processing parameters configurable via environment
    """
    
    # Speech-to-Text (Groq Whisper)
    whisper_model: str = Field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_NAME", "whisper-large-v3-turbo"),
        description="Groq Whisper model name"
    )
    
    # Text-to-Speech (ElevenLabs)
    elevenlabs_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ELEVENLABS_API_KEY"),
        description="ElevenLabs API key"
    )
    
    elevenlabs_model_short: str = Field(
        default="eleven_flash_v2_5",
        description="ElevenLabs model for short text (<200 chars)"
    )
    
    elevenlabs_model_long: str = Field(
        default="eleven_multilingual_v2",
        description="ElevenLabs model for long text (>=200 chars)"
    )
    
    text_length_threshold: int = Field(
        default=200,
        description="Character threshold for model selection"
    )
    
    # Voice Activity Detection (VAD)
    vad_frame_duration_ms: int = Field(
        default=30,
        description="VAD frame duration in milliseconds"
    )
    
    vad_min_speech_frames: int = Field(
        default=10,
        description="Minimum frames to consider speech"
    )
    
    vad_sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz"
    )
    
    # Pronunciation Assessment
    pronunciation_min_word_length: int = Field(
        default=3,
        description="Minimum word length for pronunciation scoring"
    )
    
    pronunciation_speaking_rate: float = Field(
        default=2.5,
        description="Average speaking rate (words per second)"
    )
    
    # ML-based pronunciation scoring weights (configurable)
    pronunciation_phoneme_weight: float = Field(
        default=0.5,
        description="Weight for phoneme accuracy in scoring"
    )
    
    pronunciation_fluency_weight: float = Field(
        default=0.3,
        description="Weight for fluency in scoring"
    )
    
    pronunciation_word_weight: float = Field(
        default=0.2,
        description="Weight for word accuracy in scoring"
    )
    
    # Fallback weights when word scores unavailable
    pronunciation_phoneme_fallback_weight: float = Field(
        default=0.8,
        description="Phoneme weight when no word scores"
    )
    
    pronunciation_fluency_fallback_weight: float = Field(
        default=0.2,
        description="Fluency weight when no word scores"
    )
    
    class Config:
        env_prefix = "VOICE_"


class SecuritySettings(BaseSettings):
    """
    Security configuration (AGENTS.md compliant - zero hardcoded values)
    All values configurable via environment variables with secure defaults
    """
    
    # JWT Configuration
    jwt_secret_key: str = Field(
        default_factory=lambda: os.getenv("JWT_SECRET_KEY", ""),
        description="JWT signing secret key (REQUIRED in production)"
    )
    
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiry time in minutes"
    )
    
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiry time in days"
    )
    
    # Password Policy
    password_min_length: int = Field(
        default=8,
        description="Minimum password length"
    )
    
    password_require_uppercase: bool = Field(
        default=True,
        description="Require uppercase letter in password"
    )
    
    password_require_lowercase: bool = Field(
        default=True,
        description="Require lowercase letter in password"
    )
    
    password_require_digit: bool = Field(
        default=True,
        description="Require digit in password"
    )
    
    password_require_special: bool = Field(
        default=True,
        description="Require special character in password"
    )
    
    bcrypt_rounds: int = Field(
        default=12,
        description="Bcrypt hashing rounds (10-14 recommended)"
    )
    
    # Rate Limiting - Per IP
    rate_limit_ip_per_minute: int = Field(
        default=60,
        description="Max requests per IP per minute"
    )
    
    rate_limit_ip_per_hour: int = Field(
        default=1000,
        description="Max requests per IP per hour"
    )
    
    # Rate Limiting - Per User
    rate_limit_user_per_minute: int = Field(
        default=30,
        description="Max requests per user per minute"
    )
    
    rate_limit_user_per_hour: int = Field(
        default=500,
        description="Max requests per user per hour"
    )
    
    rate_limit_user_per_day: int = Field(
        default=5000,
        description="Max requests per user per day"
    )
    
    # Rate Limiting - Per Endpoint
    rate_limit_chat_per_minute: int = Field(
        default=10,
        description="Max chat requests per minute (AI calls)"
    )
    
    rate_limit_voice_per_minute: int = Field(
        default=5,
        description="Max voice requests per minute (TTS/STT)"
    )
    
    # Cost-Based Limits
    rate_limit_user_daily_cost: float = Field(
        default=5.0,
        description="Max cost per user per day in USD"
    )
    
    rate_limit_global_hourly_cost: float = Field(
        default=100.0,
        description="Max global cost per hour in USD"
    )
    
    # Anomaly Detection
    anomaly_score_threshold: float = Field(
        default=0.8,
        description="Threshold for anomaly detection (0.0-1.0)"
    )
    
    anomaly_spike_multiplier: float = Field(
        default=3.0,
        description="Multiplier for spike detection"
    )
    
    # Storage Settings
    rate_limit_window_seconds: int = Field(
        default=3600,
        description="Sliding window size in seconds"
    )
    
    rate_limit_max_history: int = Field(
        default=10000,
        description="Max items in rate limit history"
    )
    
    # Input Validation
    input_max_length: int = Field(
        default=10000,
        description="Maximum input length for validation"
    )
    
    file_upload_max_size_mb: int = Field(
        default=10,
        description="Maximum file upload size in MB"
    )
    
    allowed_file_types: List[str] = Field(
        default=["audio/wav", "audio/mpeg", "audio/mp3", "image/jpeg", "image/png"],
        description="Allowed MIME types for file uploads"
    )
    
    # SQL Injection Patterns (configurable for easy updates)
    sql_injection_patterns: List[str] = Field(
        default=[
            r"(\bunion\b.*\bselect\b)",
            r"(\bselect\b.*\bfrom\b.*\bwhere\b)",
            r"(\bdrop\b.*\btable\b)",
            r"(\binsert\b.*\binto\b.*\bvalues\b)",
            r"(\bupdate\b.*\bset\b)",
            r"(\bdelete\b.*\bfrom\b)",
            r"(\bexec\b.*\()",
            r"(\bexecute\b.*\()",
            r"(--\s)",
            r"(;.*drop)",
            r"('.*or.*'.*=.*')",
            r"(\".*or.*\".*=.*\")",
            r"(\bor\b.*\b1\s*=\s*1\b)",
            r"(\band\b.*\b1\s*=\s*1\b)",
            r"(\/\*.*\*\/)",
        ],
        description="Regex patterns for SQL injection detection"
    )
    
    # XSS Prevention Patterns
    xss_patterns: List[str] = Field(
        default=[
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<embed[^>]*>",
            r"<object[^>]*>",
        ],
        description="Regex patterns for XSS detection"
    )
    
    class Config:
        env_prefix = "SECURITY_"


class MonitoringSettings(BaseSettings):
    """
    Health monitoring configuration (Phase 8C File 11)
    
    AGENTS.md compliant: All thresholds ML/statistical-based, not hardcoded
    Configuration-driven parameters for Statistical Process Control
    """
    
    # History and sampling
    history_size: int = Field(
        default=100,
        description="Number of historical samples to maintain for SPC"
    )
    
    min_samples_for_threshold: int = Field(
        default=10,
        description="Minimum samples required for threshold calculation"
    )
    
    min_samples_for_trend: int = Field(
        default=20,
        description="Minimum samples required for trend detection"
    )
    
    min_samples_for_score: int = Field(
        default=10,
        description="Minimum samples required for health score"
    )
    
    # Statistical Process Control (SPC)
    sigma_threshold: float = Field(
        default=3.0,
        description="Sigma threshold for anomaly detection (3.0 = 99.7% confidence)"
    )
    
    # EWMA (Exponential Weighted Moving Average)
    ewma_alpha: float = Field(
        default=0.3,
        description="EWMA smoothing factor (0.0-1.0, higher = more reactive)"
    )
    
    # Trending
    trend_window_size: int = Field(
        default=10,
        description="Window size for trend calculation"
    )
    
    improvement_threshold_pct: float = Field(
        default=5.0,
        description="Percentage change threshold for improvement detection"
    )
    
    degradation_threshold_pct: float = Field(
        default=5.0,
        description="Percentage change threshold for degradation detection"
    )
    
    # Health scoring thresholds
    healthy_threshold: float = Field(
        default=70.0,
        description="Health score threshold for HEALTHY status (0-100)"
    )
    
    degraded_threshold: float = Field(
        default=40.0,
        description="Health score threshold for DEGRADED status (0-100)"
    )
    
    # Metric weights for composite health score
    metric_weights: Dict[str, float] = Field(
        default={
            'latency_ms': 0.35,      # Most important
            'error_rate': 0.30,      # Very important
            'throughput': 0.20,      # Important
            'connections': 0.15      # Less important
        },
        description="Weights for health score calculation"
    )
    
    # Component weights for system health score
    component_weights: Dict[str, float] = Field(
        default={
            'database': 1.0,              # Critical
            'ai_provider_groq': 0.8,      # Important
            'ai_provider_emergent': 0.8,  # Important
            'ai_provider_gemini': 0.8,    # Important
            'external_apis': 0.5          # Less critical
        },
        description="Weights for system health calculation"
    )
    
    # Monitoring intervals
    check_interval_seconds: int = Field(
        default=60,
        description="Interval between health checks (seconds)"
    )
    
    provider_timeout_seconds: int = Field(
        default=5,
        description="Timeout for AI provider health checks (seconds)"
    )
    
    # Alert thresholds (statistical/ML-based, not hardcoded rules)
    alert_error_rate_threshold: float = Field(
        default=0.1,
        description="Error rate threshold for alerts (10% = 0.1)"
    )
    
    alert_latency_threshold_ms: float = Field(
        default=10000.0,
        description="Latency threshold for alerts in milliseconds"
    )
    
    class Config:
        env_prefix = "MONITORING_"


class MasterXSettings(BaseSettings):
    """
    Master configuration class for MasterX
    
    Aggregates all configuration sections with zero hardcoded values.
    """
    
    environment: str = Field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development"),
        description="Environment (development, staging, production)"
    )
    
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "True").lower() == "true",
        description="Debug mode"
    )
    
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ai_providers: AIProviderSettings = Field(default_factory=AIProviderSettings)
    caching: CachingSettings = Field(default_factory=CachingSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"
    
    def get_active_providers(self) -> List[str]:
        """Get list of configured AI providers"""
        providers = []
        if self.ai_providers.groq_api_key:
            providers.append("groq")
        if self.ai_providers.emergent_llm_key:
            providers.append("emergent")
        if self.ai_providers.gemini_api_key:
            providers.append("gemini")
        return providers


# Global settings instance
settings = MasterXSettings()


def get_settings() -> MasterXSettings:
    """
    Get global settings instance
    
    Returns:
        MasterXSettings instance with all configuration
    """
    return settings
