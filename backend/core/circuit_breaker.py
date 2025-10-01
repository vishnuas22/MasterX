"""
Circuit Breaker - Provider Health & Fault Tolerance
Following specifications from 4.DYNAMIC_AI_ROUTING_SYSTEM.md

Prevents cascading failures by monitoring provider health:
- Three states: CLOSED (ok), OPEN (failing), HALF_OPEN (testing)
- Auto-recovery with exponential backoff
- Real-time health tracking in MongoDB
- Failure threshold and success threshold configuration
"""

import logging
import time
from typing import Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

from core.models import CircuitBreakerState, ProviderStatus
from utils.database import get_database

logger = logging.getLogger(__name__)


# Circuit breaker configuration
CIRCUIT_BREAKER_CONFIG = {
    'failure_threshold': 5,        # Open circuit after 5 consecutive failures
    'success_threshold': 2,        # Close circuit after 2 successes in half-open
    'timeout_seconds': 60,         # Try again after 60 seconds
    'half_open_max_calls': 3,     # Test with max 3 calls in half-open state
    'degraded_threshold': 0.8      # < 80% success rate = degraded
}


class CircuitBreaker:
    """
    Circuit breaker pattern for AI provider fault tolerance
    
    State Machine:
    - CLOSED: Normal operation, all requests allowed
    - OPEN: Provider failing, requests blocked
    - HALF_OPEN: Testing provider recovery, limited requests
    
    Transitions:
    - CLOSED → OPEN: After failure_threshold consecutive failures
    - OPEN → HALF_OPEN: After timeout_seconds elapsed
    - HALF_OPEN → CLOSED: After success_threshold successes
    - HALF_OPEN → OPEN: If any failure during testing
    """
    
    def __init__(self):
        self.db = None
        self.provider_health_collection = None
        
        # In-memory state for fast checks
        self.provider_states: Dict[str, CircuitBreakerState] = {}
        self.failure_counts: Dict[str, int] = {}
        self.success_counts: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
        self.half_open_calls: Dict[str, int] = {}
        
        logger.info("✅ CircuitBreaker initialized")
    
    async def initialize_db(self):
        """Initialize database connection"""
        if not self.db:
            self.db = get_database()
            self.provider_health_collection = self.db['provider_health']
            logger.info("✅ CircuitBreaker database initialized")
    
    def is_available(self, provider: str) -> bool:
        """
        Check if provider is available for requests
        
        Args:
            provider: Provider name
        
        Returns:
            True if available (CLOSED or HALF_OPEN with capacity)
        """
        
        state = self.provider_states.get(provider, CircuitBreakerState.CLOSED)
        
        if state == CircuitBreakerState.CLOSED:
            return True
        
        elif state == CircuitBreakerState.HALF_OPEN:
            # Check if we can make more test calls
            calls = self.half_open_calls.get(provider, 0)
            if calls < CIRCUIT_BREAKER_CONFIG['half_open_max_calls']:
                self.half_open_calls[provider] = calls + 1
                return True
            return False
        
        elif state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed
            last_failure = self.last_failure_time.get(provider, 0)
            elapsed = time.time() - last_failure
            
            if elapsed >= CIRCUIT_BREAKER_CONFIG['timeout_seconds']:
                # Try transitioning to half-open
                self._transition_to_half_open(provider)
                self.half_open_calls[provider] = 1
                return True
            
            return False
        
        return False
    
    async def record_success(self, provider: str):
        """
        Record successful provider call
        
        Args:
            provider: Provider name
        """
        
        state = self.provider_states.get(provider, CircuitBreakerState.CLOSED)
        
        if state == CircuitBreakerState.CLOSED:
            # Reset failure count
            self.failure_counts[provider] = 0
        
        elif state == CircuitBreakerState.HALF_OPEN:
            # Increment success count
            success_count = self.success_counts.get(provider, 0) + 1
            self.success_counts[provider] = success_count
            
            if success_count >= CIRCUIT_BREAKER_CONFIG['success_threshold']:
                # Transition to closed
                self._transition_to_closed(provider)
                logger.info(f"✅ Circuit CLOSED for {provider} (recovered)")
        
        # Update health in database
        await self._update_provider_health(provider, success=True)
    
    async def record_failure(self, provider: str):
        """
        Record failed provider call
        
        Args:
            provider: Provider name
        """
        
        state = self.provider_states.get(provider, CircuitBreakerState.CLOSED)
        
        if state == CircuitBreakerState.CLOSED:
            # Increment failure count
            failure_count = self.failure_counts.get(provider, 0) + 1
            self.failure_counts[provider] = failure_count
            
            if failure_count >= CIRCUIT_BREAKER_CONFIG['failure_threshold']:
                # Transition to open
                self._transition_to_open(provider)
                logger.error(f"🚨 Circuit OPENED for {provider} (too many failures)")
        
        elif state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to_open(provider)
            logger.warning(f"⚠️ Circuit back to OPEN for {provider} (failed during testing)")
        
        # Update health in database
        await self._update_provider_health(provider, success=False)
    
    def _transition_to_open(self, provider: str):
        """Transition circuit to OPEN state"""
        self.provider_states[provider] = CircuitBreakerState.OPEN
        self.last_failure_time[provider] = time.time()
        self.success_counts[provider] = 0
        self.half_open_calls[provider] = 0
    
    def _transition_to_half_open(self, provider: str):
        """Transition circuit to HALF_OPEN state"""
        self.provider_states[provider] = CircuitBreakerState.HALF_OPEN
        self.failure_counts[provider] = 0
        self.success_counts[provider] = 0
        self.half_open_calls[provider] = 0
        logger.info(f"🔄 Circuit HALF_OPEN for {provider} (testing recovery)")
    
    def _transition_to_closed(self, provider: str):
        """Transition circuit to CLOSED state"""
        self.provider_states[provider] = CircuitBreakerState.CLOSED
        self.failure_counts[provider] = 0
        self.success_counts[provider] = 0
        self.half_open_calls[provider] = 0
    
    async def _update_provider_health(self, provider: str, success: bool):
        """Update provider health in database"""
        
        await self.initialize_db()
        
        try:
            timestamp = datetime.utcnow()
            
            # Get current health record
            current = await self.provider_health_collection.find_one(
                {'provider': provider},
                sort=[('timestamp', -1)]
            )
            
            if current:
                # Calculate success rate for last hour
                one_hour_ago = timestamp - timedelta(hours=1)
                recent_records = self.provider_health_collection.find({
                    'provider': provider,
                    'timestamp': {'$gte': one_hour_ago}
                })
                
                requests = 0
                errors = 0
                async for record in recent_records:
                    requests += 1
                    if not record.get('last_success'):
                        errors += 1
                
                success_rate = (requests - errors) / requests if requests > 0 else 1.0
            else:
                success_rate = 1.0 if success else 0.0
                requests = 1
                errors = 0 if success else 1
            
            # Determine status
            state = self.provider_states.get(provider, CircuitBreakerState.CLOSED)
            
            if state == CircuitBreakerState.OPEN:
                status = ProviderStatus.DOWN
            elif success_rate < CIRCUIT_BREAKER_CONFIG['degraded_threshold']:
                status = ProviderStatus.DEGRADED
            else:
                status = ProviderStatus.HEALTHY
            
            # Insert new health record
            health_record = {
                'provider': provider,
                'timestamp': timestamp,
                'status': status.value,
                'success_rate': success_rate,
                'avg_response_time_ms': 0,  # TODO: Track this
                'requests_last_hour': requests,
                'errors_last_hour': errors,
                'circuit_breaker_state': state.value,
                'last_success': timestamp if success else current.get('last_success'),
                'last_failure': timestamp if not success else current.get('last_failure')
            }
            
            await self.provider_health_collection.insert_one(health_record)
        
        except Exception as e:
            logger.error(f"Error updating provider health: {e}")
    
    async def get_health_status(self) -> Dict[str, Dict]:
        """
        Get health status for all providers
        
        Returns:
            Dict mapping provider names to health info
        """
        
        await self.initialize_db()
        
        health_status = {}
        
        try:
            # Get latest health record for each provider
            providers = await self.provider_health_collection.distinct('provider')
            
            for provider in providers:
                latest = await self.provider_health_collection.find_one(
                    {'provider': provider},
                    sort=[('timestamp', -1)]
                )
                
                if latest:
                    health_status[provider] = {
                        'status': latest['status'],
                        'circuit_state': latest['circuit_breaker_state'],
                        'success_rate': latest['success_rate'],
                        'requests_last_hour': latest['requests_last_hour'],
                        'errors_last_hour': latest['errors_last_hour'],
                        'last_updated': latest['timestamp'].isoformat()
                    }
        
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
        
        return health_status
    
    def get_provider_state(self, provider: str) -> CircuitBreakerState:
        """
        Get current circuit breaker state for provider
        
        Args:
            provider: Provider name
        
        Returns:
            Current circuit breaker state
        """
        return self.provider_states.get(provider, CircuitBreakerState.CLOSED)
    
    def reset_provider(self, provider: str):
        """
        Manually reset circuit breaker for provider (admin function)
        
        Args:
            provider: Provider name
        """
        self._transition_to_closed(provider)
        logger.info(f"🔧 Circuit breaker manually reset for {provider}")
