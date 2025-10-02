"""
MasterX Performance Monitoring System
Following specifications from 3.MASTERX_COMPREHENSIVE_PLAN.md

PRINCIPLES (from AGENTS.md):
- No hardcoded values
- Real-time ML-driven performance tracking
- Clean, professional naming
- PEP8 compliant

Tracks and optimizes system performance in real-time.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    
    request_id: str
    endpoint: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    
    # Component timings
    emotion_time_ms: float = 0.0
    context_time_ms: float = 0.0
    difficulty_time_ms: float = 0.0
    ai_time_ms: float = 0.0
    storage_time_ms: float = 0.0
    
    # Resource usage
    tokens_used: int = 0
    cost: float = 0.0
    
    # Status
    success: bool = True
    error: Optional[str] = None
    
    def mark_complete(self):
        """Mark request as complete and calculate duration"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class PerformanceStats:
    """Aggregated performance statistics"""
    
    # Latency percentiles
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    
    # Counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Component timing averages
    avg_emotion_time_ms: float = 0.0
    avg_context_time_ms: float = 0.0
    avg_difficulty_time_ms: float = 0.0
    avg_ai_time_ms: float = 0.0
    avg_storage_time_ms: float = 0.0
    
    # Resource usage
    total_tokens: int = 0
    total_cost: float = 0.0
    
    # Time period
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: Optional[datetime] = None


class PerformanceTracker:
    """
    Real-time performance tracking
    
    Tracks request latencies, component timings, and resource usage.
    Calculates percentiles and detects performance issues.
    """
    
    def __init__(self):
        """Initialize performance tracker"""
        self.enabled = settings.performance.enabled
        
        if not self.enabled:
            logger.info("⚠️ Performance tracking disabled")
            return
        
        # Current requests being tracked
        self.active_requests: Dict[str, RequestMetrics] = {}
        
        # Completed requests (rolling window)
        self.completed_requests: deque = deque(maxlen=1000)
        
        # Per-endpoint metrics
        self.endpoint_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Thresholds
        self.slow_threshold_ms = settings.performance.slow_request_threshold_ms
        self.critical_threshold_ms = settings.performance.critical_latency_threshold_ms
        
        # Alerts
        self.slow_requests_count = 0
        self.critical_requests_count = 0
        
        logger.info(
            f"✅ Performance tracker initialized "
            f"(slow: {self.slow_threshold_ms}ms, critical: {self.critical_threshold_ms}ms)"
        )
    
    def start_request(self, request_id: str, endpoint: str) -> RequestMetrics:
        """
        Start tracking a request
        
        Args:
            request_id: Unique request identifier
            endpoint: API endpoint
        
        Returns:
            RequestMetrics object for this request
        """
        if not self.enabled:
            return None
        
        metrics = RequestMetrics(
            request_id=request_id,
            endpoint=endpoint,
            start_time=time.time()
        )
        
        self.active_requests[request_id] = metrics
        logger.debug(f"Started tracking request: {request_id}")
        
        return metrics
    
    def end_request(
        self,
        request_id: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        End tracking a request
        
        Args:
            request_id: Unique request identifier
            success: Whether request succeeded
            error: Error message if failed
        """
        if not self.enabled:
            return
        
        metrics = self.active_requests.get(request_id)
        if not metrics:
            logger.warning(f"Request {request_id} not found in active requests")
            return
        
        # Mark complete
        metrics.mark_complete()
        metrics.success = success
        metrics.error = error
        
        # Check thresholds
        if metrics.duration_ms > self.critical_threshold_ms:
            self.critical_requests_count += 1
            logger.error(
                f"🚨 CRITICAL LATENCY: {metrics.duration_ms:.0f}ms "
                f"(endpoint: {metrics.endpoint}, request: {request_id})"
            )
        elif metrics.duration_ms > self.slow_threshold_ms:
            self.slow_requests_count += 1
            logger.warning(
                f"⚠️ SLOW REQUEST: {metrics.duration_ms:.0f}ms "
                f"(endpoint: {metrics.endpoint}, request: {request_id})"
            )
        
        # Store in completed requests
        self.completed_requests.append(metrics)
        self.endpoint_metrics[metrics.endpoint].append(metrics)
        
        # Remove from active
        del self.active_requests[request_id]
        
        logger.debug(
            f"Completed request: {request_id} ({metrics.duration_ms:.0f}ms, "
            f"success: {success})"
        )
    
    def update_component_timing(
        self,
        request_id: str,
        component: str,
        duration_ms: float
    ) -> None:
        """
        Update component timing for a request
        
        Args:
            request_id: Unique request identifier
            component: Component name (emotion, context, ai, etc.)
            duration_ms: Duration in milliseconds
        """
        if not self.enabled:
            return
        
        metrics = self.active_requests.get(request_id)
        if not metrics:
            return
        
        # Update component timing
        if component == "emotion":
            metrics.emotion_time_ms = duration_ms
        elif component == "context":
            metrics.context_time_ms = duration_ms
        elif component == "difficulty":
            metrics.difficulty_time_ms = duration_ms
        elif component == "ai":
            metrics.ai_time_ms = duration_ms
        elif component == "storage":
            metrics.storage_time_ms = duration_ms
    
    def update_resource_usage(
        self,
        request_id: str,
        tokens: int = 0,
        cost: float = 0.0
    ) -> None:
        """
        Update resource usage for a request
        
        Args:
            request_id: Unique request identifier
            tokens: Tokens used
            cost: Cost incurred
        """
        if not self.enabled:
            return
        
        metrics = self.active_requests.get(request_id)
        if not metrics:
            return
        
        metrics.tokens_used += tokens
        metrics.cost += cost
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """
        Calculate percentile from list of values
        
        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)
        
        Returns:
            Percentile value
        """
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_stats(
        self,
        endpoint: Optional[str] = None,
        last_n: int = 100
    ) -> PerformanceStats:
        """
        Get performance statistics
        
        Args:
            endpoint: Specific endpoint (None for all)
            last_n: Number of recent requests to analyze
        
        Returns:
            PerformanceStats object
        """
        if not self.enabled:
            return PerformanceStats()
        
        # Select requests to analyze
        if endpoint:
            requests = list(self.endpoint_metrics.get(endpoint, []))[-last_n:]
        else:
            requests = list(self.completed_requests)[-last_n:]
        
        if not requests:
            return PerformanceStats()
        
        # Calculate statistics
        durations = [r.duration_ms for r in requests if r.duration_ms is not None]
        successful = [r for r in requests if r.success]
        failed = [r for r in requests if not r.success]
        
        stats = PerformanceStats(
            p50_ms=self._calculate_percentile(durations, 50),
            p95_ms=self._calculate_percentile(durations, 95),
            p99_ms=self._calculate_percentile(durations, 99),
            total_requests=len(requests),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_tokens=sum(r.tokens_used for r in requests),
            total_cost=sum(r.cost for r in requests)
        )
        
        # Calculate component averages
        if successful:
            stats.avg_emotion_time_ms = sum(r.emotion_time_ms for r in successful) / len(successful)
            stats.avg_context_time_ms = sum(r.context_time_ms for r in successful) / len(successful)
            stats.avg_difficulty_time_ms = sum(r.difficulty_time_ms for r in successful) / len(successful)
            stats.avg_ai_time_ms = sum(r.ai_time_ms for r in successful) / len(successful)
            stats.avg_storage_time_ms = sum(r.storage_time_ms for r in successful) / len(successful)
        
        return stats
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data
        
        Returns:
            Dictionary with dashboard metrics
        """
        if not self.enabled:
            return {"enabled": False}
        
        overall_stats = self.get_stats()
        
        # Per-endpoint stats
        endpoint_stats = {}
        for endpoint in self.endpoint_metrics.keys():
            endpoint_stats[endpoint] = self.get_stats(endpoint=endpoint).__dict__
        
        return {
            "enabled": True,
            "overall": overall_stats.__dict__,
            "by_endpoint": endpoint_stats,
            "active_requests": len(self.active_requests),
            "alerts": {
                "slow_requests": self.slow_requests_count,
                "critical_requests": self.critical_requests_count
            }
        }
    
    def reset_alerts(self) -> None:
        """Reset alert counters"""
        self.slow_requests_count = 0
        self.critical_requests_count = 0
        logger.info("Alert counters reset")


# Global performance tracker instance
_performance_tracker: Optional[PerformanceTracker] = None


def init_performance_tracker() -> PerformanceTracker:
    """
    Initialize global performance tracker
    
    Returns:
        PerformanceTracker instance
    """
    global _performance_tracker
    _performance_tracker = PerformanceTracker()
    return _performance_tracker


def get_performance_tracker() -> Optional[PerformanceTracker]:
    """
    Get global performance tracker instance
    
    Returns:
        PerformanceTracker instance or None if not initialized
    """
    return _performance_tracker
