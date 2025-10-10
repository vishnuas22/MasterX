"""
MasterX Graceful Shutdown Manager
Phase 8C File 13
Following specifications from PHASE_8C_FILES_13-15_CONTINUATION.md

PRINCIPLES (from AGENTS.md):
- Zero-downtime deployments
- Google SRE drain pattern
- No hardcoded timeouts (all configurable)
- Clean resource cleanup
- Type-safe with type hints
- Production-ready

Features:
- Request tracking (in-flight monitoring)
- Signal handlers (SIGTERM, SIGINT)
- Background task management
- Resource cleanup coordination
- Configurable timeouts
- Integration with health monitor

Date: October 10, 2025
"""

import asyncio
import logging
import signal
import os
from typing import Set, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ShutdownStatus:
    """Shutdown progress status"""
    is_shutting_down: bool = False
    start_time: Optional[datetime] = None
    phase: str = "idle"
    in_flight_count: int = 0
    background_tasks_count: int = 0
    completed: bool = False
    error: Optional[str] = None


# ============================================================================
# GRACEFUL SHUTDOWN MANAGER
# ============================================================================

class GracefulShutdown:
    """
    Graceful shutdown coordinator using Google SRE drain pattern
    
    Implements zero-downtime deployments by:
    1. Stop accepting new requests
    2. Drain in-flight requests
    3. Cancel background tasks
    4. Close connections
    5. Cleanup resources
    
    AGENTS.md compliant: Zero hardcoded timeouts, all configurable
    """
    
    def __init__(
        self,
        shutdown_timeout: Optional[float] = None,
        drain_timeout_ratio: float = 0.8,
        background_timeout_ratio: float = 0.2
    ):
        """
        Initialize graceful shutdown manager
        
        Args:
            shutdown_timeout: Maximum shutdown time in seconds (default from env)
            drain_timeout_ratio: Ratio of timeout for draining requests (0.0-1.0)
            background_timeout_ratio: Ratio of timeout for background tasks (0.0-1.0)
        """
        # Get timeout from environment or use default
        self.shutdown_timeout = shutdown_timeout or float(
            os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT", "30.0")
        )
        
        # Validate ratios
        if drain_timeout_ratio + background_timeout_ratio > 1.0:
            raise ValueError("Sum of timeout ratios must be <= 1.0")
        
        self.drain_timeout_ratio = drain_timeout_ratio
        self.background_timeout_ratio = background_timeout_ratio
        
        # State tracking
        self.is_shutting_down = False
        self.shutdown_start_time: Optional[datetime] = None
        self.in_flight_requests: Set[str] = set()
        self.background_tasks: Set[asyncio.Task] = set()
        self.shutdown_callbacks: List[Callable] = []
        
        # Synchronization
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()
        
        # Register signal handlers (SIGTERM, SIGINT)
        self._register_signal_handlers()
        
        logger.info(
            f"✅ Graceful shutdown initialized "
            f"(timeout: {self.shutdown_timeout}s, "
            f"drain: {drain_timeout_ratio * 100}%, "
            f"background: {background_timeout_ratio * 100}%)"
        )
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            logger.info("✅ Signal handlers registered (SIGTERM, SIGINT)")
        except Exception as e:
            logger.warning(f"⚠️ Could not register signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown")
        # Create task to run async shutdown
        asyncio.create_task(self.shutdown())
    
    # ========================================================================
    # REQUEST TRACKING
    # ========================================================================
    
    def track_request(self, request_id: str) -> bool:
        """
        Track in-flight request
        
        Args:
            request_id: Unique request identifier
            
        Returns:
            True if tracked, False if shutting down
        """
        if self.is_shutting_down:
            logger.debug(f"🛑 Rejecting new request {request_id} (shutting down)")
            return False
        
        self.in_flight_requests.add(request_id)
        logger.debug(f"📝 Tracking request {request_id} (total: {len(self.in_flight_requests)})")
        return True
    
    def complete_request(self, request_id: str):
        """
        Mark request as complete
        
        Args:
            request_id: Request identifier to remove
        """
        self.in_flight_requests.discard(request_id)
        logger.debug(
            f"✅ Request {request_id} completed "
            f"(remaining: {len(self.in_flight_requests)})"
        )
    
    def get_in_flight_count(self) -> int:
        """Get number of in-flight requests"""
        return len(self.in_flight_requests)
    
    # ========================================================================
    # BACKGROUND TASK MANAGEMENT
    # ========================================================================
    
    def register_background_task(self, task: asyncio.Task, name: Optional[str] = None):
        """
        Register background task for cleanup
        
        Args:
            task: Asyncio task to track
            name: Optional task name for logging
        """
        self.background_tasks.add(task)
        task_name = name or f"task-{id(task)}"
        logger.debug(f"📝 Registered background task: {task_name}")
        
        # Auto-remove when task completes
        task.add_done_callback(lambda t: self._remove_background_task(t, task_name))
    
    def _remove_background_task(self, task: asyncio.Task, name: str):
        """Remove completed background task"""
        self.background_tasks.discard(task)
        logger.debug(f"✅ Background task completed: {name}")
    
    def get_background_task_count(self) -> int:
        """Get number of active background tasks"""
        return len(self.background_tasks)
    
    # ========================================================================
    # CALLBACK REGISTRATION
    # ========================================================================
    
    def register_shutdown_callback(self, callback: Callable):
        """
        Register callback to run during shutdown
        
        Args:
            callback: Async callable to run during shutdown
        """
        self.shutdown_callbacks.append(callback)
        logger.debug(f"📝 Registered shutdown callback: {callback.__name__}")
    
    # ========================================================================
    # SHUTDOWN PHASES (Google SRE Drain Pattern)
    # ========================================================================
    
    async def _stop_accepting_requests(self):
        """Phase 1: Stop accepting new requests"""
        logger.info("📛 Phase 1: Stopping acceptance of new requests")
        
        async with self._lock:
            self.is_shutting_down = True
            self.shutdown_start_time = datetime.utcnow()
        
        # Give load balancer time to detect unhealthy state
        health_check_grace = float(os.getenv("GRACEFUL_SHUTDOWN_HEALTH_GRACE", "0.1"))
        await asyncio.sleep(health_check_grace)
        
        logger.info("✅ Phase 1 complete: No longer accepting requests")
    
    async def _drain_requests(self, timeout: float):
        """
        Phase 2: Wait for in-flight requests to complete
        
        Args:
            timeout: Maximum time to wait for draining
        """
        logger.info(
            f"⏳ Phase 2: Draining {len(self.in_flight_requests)} in-flight requests "
            f"(timeout: {timeout:.1f}s)"
        )
        
        start_time = asyncio.get_event_loop().time()
        check_interval = float(os.getenv("GRACEFUL_SHUTDOWN_CHECK_INTERVAL", "0.5"))
        
        while len(self.in_flight_requests) > 0:
            elapsed = asyncio.get_event_loop().time() - start_time
            
            if elapsed >= timeout:
                logger.warning(
                    f"⚠️ Phase 2 timeout: {len(self.in_flight_requests)} requests "
                    f"still in-flight after {timeout:.1f}s"
                )
                break
            
            remaining_time = timeout - elapsed
            logger.debug(
                f"⏳ Waiting for {len(self.in_flight_requests)} requests "
                f"(timeout in {remaining_time:.1f}s)"
            )
            
            await asyncio.sleep(check_interval)
        
        if len(self.in_flight_requests) == 0:
            logger.info("✅ Phase 2 complete: All requests drained")
        else:
            logger.warning(
                f"⚠️ Phase 2 forced completion: "
                f"{len(self.in_flight_requests)} requests abandoned"
            )
    
    async def _cancel_background_tasks(self, timeout: float):
        """
        Phase 3: Cancel background tasks with timeout
        
        Args:
            timeout: Maximum time to wait for task cancellation
        """
        task_count = len(self.background_tasks)
        logger.info(
            f"🔄 Phase 3: Canceling {task_count} background tasks "
            f"(timeout: {timeout:.1f}s)"
        )
        
        if task_count == 0:
            logger.info("✅ Phase 3 complete: No background tasks")
            return
        
        # Cancel all tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellation with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.background_tasks, return_exceptions=True),
                timeout=timeout
            )
            logger.info("✅ Phase 3 complete: All background tasks cancelled")
        except asyncio.TimeoutError:
            remaining = sum(1 for t in self.background_tasks if not t.done())
            logger.warning(
                f"⚠️ Phase 3 timeout: {remaining} tasks still running after {timeout:.1f}s"
            )
    
    async def _close_connections(self):
        """Phase 4: Close connections"""
        logger.info("🔌 Phase 4: Closing connections")
        
        # Run shutdown callbacks
        for callback in self.shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
                logger.debug(f"✅ Shutdown callback completed: {callback.__name__}")
            except Exception as e:
                logger.error(f"❌ Shutdown callback failed: {callback.__name__}: {e}")
        
        logger.info("✅ Phase 4 complete: Connections closed")
    
    async def _cleanup_resources(self):
        """Phase 5: Final cleanup"""
        logger.info("🧹 Phase 5: Cleaning up resources")
        
        # Clear tracking sets
        self.in_flight_requests.clear()
        self.background_tasks.clear()
        self.shutdown_callbacks.clear()
        
        # Set event
        self._shutdown_event.set()
        
        logger.info("✅ Phase 5 complete: Resources cleaned up")
    
    # ========================================================================
    # MAIN SHUTDOWN ORCHESTRATION
    # ========================================================================
    
    async def shutdown(self) -> bool:
        """
        Execute graceful shutdown following Google SRE drain pattern
        
        Returns:
            True if successful, False if already shutting down
        """
        # Check if already shutting down
        if self.is_shutting_down:
            logger.warning("⚠️ Shutdown already in progress")
            return False
        
        logger.info("🛑 ========== GRACEFUL SHUTDOWN INITIATED ==========")
        logger.info(f"⏱️  Timeout: {self.shutdown_timeout}s")
        logger.info(f"📊 In-flight requests: {len(self.in_flight_requests)}")
        logger.info(f"🔄 Background tasks: {len(self.background_tasks)}")
        
        try:
            # Calculate phase timeouts
            drain_timeout = self.shutdown_timeout * self.drain_timeout_ratio
            background_timeout = self.shutdown_timeout * self.background_timeout_ratio
            
            # Phase 1: Stop accepting new requests
            await self._stop_accepting_requests()
            
            # Phase 2: Drain in-flight requests
            await self._drain_requests(drain_timeout)
            
            # Phase 3: Cancel background tasks
            await self._cancel_background_tasks(background_timeout)
            
            # Phase 4: Close connections
            await self._close_connections()
            
            # Phase 5: Cleanup
            await self._cleanup_resources()
            
            elapsed = (datetime.utcnow() - self.shutdown_start_time).total_seconds()
            logger.info(f"✅ ========== GRACEFUL SHUTDOWN COMPLETE ({elapsed:.2f}s) ==========")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Graceful shutdown failed: {e}", exc_info=True)
            return False
    
    async def wait_for_shutdown(self):
        """Wait for shutdown to complete"""
        await self._shutdown_event.wait()
    
    def get_shutdown_status(self) -> ShutdownStatus:
        """
        Get current shutdown status
        
        Returns:
            ShutdownStatus with current state
        """
        return ShutdownStatus(
            is_shutting_down=self.is_shutting_down,
            start_time=self.shutdown_start_time,
            phase="shutdown" if self.is_shutting_down else "running",
            in_flight_count=len(self.in_flight_requests),
            background_tasks_count=len(self.background_tasks),
            completed=self._shutdown_event.is_set()
        )


# ============================================================================
# SINGLETON PATTERN
# ============================================================================

_graceful_shutdown_instance: Optional[GracefulShutdown] = None


def get_graceful_shutdown(
    shutdown_timeout: Optional[float] = None
) -> GracefulShutdown:
    """
    Get singleton instance of graceful shutdown manager
    
    Args:
        shutdown_timeout: Timeout in seconds (only used on first call)
        
    Returns:
        GracefulShutdown singleton instance
    """
    global _graceful_shutdown_instance
    
    if _graceful_shutdown_instance is None:
        _graceful_shutdown_instance = GracefulShutdown(shutdown_timeout)
    
    return _graceful_shutdown_instance


def reset_graceful_shutdown():
    """Reset singleton (for testing only)"""
    global _graceful_shutdown_instance
    _graceful_shutdown_instance = None
