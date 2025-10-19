"""
Brute Force Protection for Login Attempts
Zero hardcoded values - all configuration from environment
Follows AGENTS.md principles: clean naming, type safety, async patterns

Implements account lockout and rate limiting to prevent credential stuffing attacks.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from typing import Dict, List
import asyncio
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class BruteForceProtector:
    """
    Protect against brute force login attacks
    
    Features:
    - Track failed login attempts per identifier (username/email/IP)
    - Automatic account lockout after threshold breaches
    - Configurable attempt window and lockout duration
    - Automatic cleanup of old data
    - Zero hardcoded values (all from config)
    """
    
    def __init__(
        self,
        max_attempts: int = None,
        lockout_duration_minutes: int = None,
        attempt_window_minutes: int = None
    ):
        """
        Initialize brute force protector
        
        Args:
            max_attempts: Maximum failed attempts before lockout (from config)
            lockout_duration_minutes: How long to lock account (from config)
            attempt_window_minutes: Time window for counting attempts (from config)
        """
        # Get configuration from settings (zero hardcoded values)
        self.max_attempts = max_attempts or settings.security.max_login_attempts
        self.lockout_duration = timedelta(
            minutes=lockout_duration_minutes or settings.security.lockout_duration_minutes
        )
        self.attempt_window = timedelta(
            minutes=attempt_window_minutes or settings.security.attempt_window_minutes
        )
        
        # Track failed attempts: {identifier: [timestamp1, timestamp2, ...]}
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        
        # Track locked accounts: {identifier: lockout_until_timestamp}
        self.locked_accounts: Dict[str, datetime] = {}
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_old_data())
        
        logger.info(
            f"Brute force protector initialized: max_attempts={self.max_attempts}, "
            f"lockout={self.lockout_duration.total_seconds()/60:.0f}min, "
            f"window={self.attempt_window.total_seconds()/60:.0f}min"
        )
    
    async def _cleanup_old_data(self):
        """
        Periodically clean up old attempt data
        
        Runs in background to prevent memory leaks.
        Cleanup interval from config (zero hardcoded values).
        """
        cleanup_interval = settings.monitoring.check_interval_seconds * 5  # 5x monitoring interval
        
        while True:
            await asyncio.sleep(cleanup_interval)
            
            try:
                cutoff = datetime.now() - self.attempt_window
                
                # Clean old attempts
                for identifier in list(self.failed_attempts.keys()):
                    self.failed_attempts[identifier] = [
                        t for t in self.failed_attempts[identifier] if t > cutoff
                    ]
                    if not self.failed_attempts[identifier]:
                        del self.failed_attempts[identifier]
                
                # Clean expired lockouts
                now = datetime.now()
                for identifier in list(self.locked_accounts.keys()):
                    if now >= self.locked_accounts[identifier]:
                        del self.locked_accounts[identifier]
                        logger.info(f"Account lockout expired: {identifier}")
                
                logger.debug(
                    f"Cleanup completed: {len(self.failed_attempts)} active trackers, "
                    f"{len(self.locked_accounts)} locked accounts"
                )
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def check_and_record_attempt(
        self,
        identifier: str,
        success: bool
    ) -> None:
        """
        Check if attempt is allowed and record result
        
        Args:
            identifier: Username, email, or IP address
            success: True if login succeeded, False if failed
            
        Raises:
            HTTPException: If account is locked or too many attempts (429)
        """
        now = datetime.now()
        
        # Check if account is locked
        if identifier in self.locked_accounts:
            locked_until = self.locked_accounts[identifier]
            if now < locked_until:
                remaining = int((locked_until - now).total_seconds() / 60)
                logger.warning(
                    f"Blocked login attempt for locked account: {identifier}, "
                    f"remaining: {remaining}min"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Account locked due to too many failed login attempts. "
                        f"Try again in {remaining} minutes."
                    )
                )
            else:
                # Lock expired
                del self.locked_accounts[identifier]
                self.failed_attempts[identifier] = []
                logger.info(f"Lock expired for: {identifier}")
        
        if success:
            # Clear failed attempts on successful login
            if identifier in self.failed_attempts:
                logger.info(
                    f"Successful login, clearing {len(self.failed_attempts[identifier])} "
                    f"failed attempts for: {identifier}"
                )
                del self.failed_attempts[identifier]
            return
        
        # Record failed attempt
        cutoff = now - self.attempt_window
        self.failed_attempts[identifier] = [
            t for t in self.failed_attempts[identifier] if t > cutoff
        ]
        self.failed_attempts[identifier].append(now)
        
        attempt_count = len(self.failed_attempts[identifier])
        logger.warning(
            f"Failed login attempt {attempt_count}/{self.max_attempts} "
            f"for: {identifier}"
        )
        
        # Check if we should lock the account
        if attempt_count >= self.max_attempts:
            self.locked_accounts[identifier] = now + self.lockout_duration
            lockout_minutes = self.lockout_duration.total_seconds() / 60
            
            logger.error(
                f"Account locked due to {attempt_count} failed attempts: {identifier}, "
                f"duration: {lockout_minutes:.0f}min"
            )
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Too many failed login attempts. "
                    f"Account locked for {lockout_minutes:.0f} minutes."
                )
            )
    
    def get_remaining_attempts(self, identifier: str) -> int:
        """
        Get number of remaining login attempts before lockout
        
        Args:
            identifier: Username, email, or IP address
            
        Returns:
            Number of remaining attempts (0 if locked)
        """
        # Check if locked
        if identifier in self.locked_accounts:
            if datetime.now() < self.locked_accounts[identifier]:
                return 0
        
        # Count recent attempts
        cutoff = datetime.now() - self.attempt_window
        recent_attempts = [
            t for t in self.failed_attempts.get(identifier, []) if t > cutoff
        ]
        return max(0, self.max_attempts - len(recent_attempts))
    
    def get_lockout_remaining_time(self, identifier: str) -> int:
        """
        Get remaining lockout time in minutes
        
        Args:
            identifier: Username, email, or IP address
            
        Returns:
            Remaining minutes (0 if not locked)
        """
        if identifier in self.locked_accounts:
            locked_until = self.locked_accounts[identifier]
            if datetime.now() < locked_until:
                return int((locked_until - datetime.now()).total_seconds() / 60)
        return 0


# Global instance (singleton pattern for efficiency)
brute_force_protector = BruteForceProtector()
