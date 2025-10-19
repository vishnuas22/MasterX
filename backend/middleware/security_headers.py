"""
Security Headers Middleware
Adds essential security headers to all responses
Follows OWASP best practices
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses
    
    Headers added:
    - X-Content-Type-Options: nosniff (prevent MIME type sniffing)
    - X-Frame-Options: DENY (prevent clickjacking)
    - X-XSS-Protection: 1; mode=block (XSS protection for legacy browsers)
    - Strict-Transport-Security: enforce HTTPS (HSTS)
    - Content-Security-Policy: restrict resource loading
    - Referrer-Policy: control referrer information
    - Permissions-Policy: control browser features
    """
    
    def __init__(self, app, enable_hsts: bool = False):
        """
        Initialize security headers middleware
        
        Args:
            app: FastAPI application
            enable_hsts: Enable HSTS header (only for HTTPS deployments)
        """
        super().__init__(app)
        self.enable_hsts = enable_hsts
        
        logger.info(f"Security Headers Middleware initialized (HSTS: {enable_hsts})")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to response
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response with security headers added
        """
        # Process request
        response = await call_next(request)
        
        # Add security headers
        
        # 1. X-Content-Type-Options: Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # 2. X-Frame-Options: Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # 3. X-XSS-Protection: XSS filter for older browsers
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # 4. Strict-Transport-Security (HSTS): Enforce HTTPS
        # Only enable in production with HTTPS
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # 5. Content-Security-Policy: Restrict resource loading
        # Balanced policy: secure but allows common use cases
        csp_directives = [
            "default-src 'self'",  # Only load from same origin by default
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Allow inline scripts (needed for some frameworks)
            "style-src 'self' 'unsafe-inline'",  # Allow inline styles
            "img-src 'self' data: https:",  # Allow images from same origin, data URIs, and HTTPS
            "font-src 'self' data:",  # Allow fonts from same origin and data URIs
            "connect-src 'self'",  # API calls only to same origin
            "frame-ancestors 'none'",  # Don't allow framing (same as X-Frame-Options)
            "base-uri 'self'",  # Restrict base tag
            "form-action 'self'",  # Forms can only submit to same origin
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # 6. Referrer-Policy: Control referrer information leakage
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # 7. Permissions-Policy: Disable unnecessary browser features
        permissions = [
            "geolocation=()",  # Disable geolocation
            "microphone=()",   # Disable microphone
            "camera=()",       # Disable camera
            "payment=()",      # Disable payment API
            "usb=()",          # Disable USB access
            "magnetometer=()", # Disable magnetometer
            "gyroscope=()",    # Disable gyroscope
            "accelerometer=()",# Disable accelerometer
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions)
        
        # 8. Cache-Control: Prevent caching of sensitive data
        # Only for non-static resources
        if not request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
        
        return response
