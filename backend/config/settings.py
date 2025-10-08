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
from typing import Optional, List
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
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    
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
