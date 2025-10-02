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
