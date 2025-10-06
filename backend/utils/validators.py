"""
Input Validation and Sanitization Module
Following OWASP security guidelines

Features:
- XSS prevention (HTML/JavaScript injection)
- SQL injection protection
- Path traversal prevention
- Email validation
- URL validation
- Content sanitization
- Length limits enforcement

PRINCIPLES (AGENTS.MD):
- No hardcoded limits (configurable)
- Industry-standard validation
- Clean, professional naming
- Comprehensive coverage
"""

import re
import os
import html
from typing import Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class ValidationConfig:
    """Validation configuration from environment"""
    
    # Length limits
    MAX_STRING_LENGTH: int = int(os.getenv("MAX_STRING_LENGTH", "10000"))
    MAX_MESSAGE_LENGTH: int = int(os.getenv("MAX_MESSAGE_LENGTH", "5000"))
    MAX_NAME_LENGTH: int = int(os.getenv("MAX_NAME_LENGTH", "100"))
    MAX_EMAIL_LENGTH: int = int(os.getenv("MAX_EMAIL_LENGTH", "255"))
    MAX_URL_LENGTH: int = int(os.getenv("MAX_URL_LENGTH", "2048"))
    
    # Content filtering
    ALLOW_HTML: bool = os.getenv("ALLOW_HTML", "false").lower() == "true"
    ALLOW_MARKDOWN: bool = os.getenv("ALLOW_MARKDOWN", "true").lower() == "true"
    
    # Security settings
    BLOCK_SUSPICIOUS_PATTERNS: bool = True


# ============================================================================
# VALIDATION PATTERNS
# ============================================================================

class ValidationPatterns:
    """Regex patterns for validation"""
    
    # XSS patterns (dangerous JavaScript/HTML)
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'onerror\s*=',
        r'onload\s*=',
        r'onclick\s*=',
        r'<iframe[^>]*>',
        r'<embed[^>]*>',
        r'<object[^>]*>',
    ]
    
    # SQL injection patterns
    SQL_PATTERNS = [
        r';\s*drop\s+table',
        r';\s*delete\s+from',
        r';\s*insert\s+into',
        r';\s*update\s+.*\s+set',
        r'union\s+select',
        r'or\s+1\s*=\s*1',
        r'and\s+1\s*=\s*1',
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL = [
        r'\.\./+',
        r'\.\.\\+',
        r'/etc/passwd',
        r'/etc/shadow',
        r'c:\\windows',
    ]
    
    # Email pattern (RFC 5322 simplified)
    EMAIL = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # URL pattern
    URL = r'^https?://[^\s/$.?#].[^\s]*$'


# ============================================================================
# SANITIZATION FUNCTIONS
# ============================================================================

class Sanitizer:
    """
    Content sanitization utilities
    
    Removes or escapes potentially dangerous content.
    """
    
    @staticmethod
    def sanitize_html(text: str, allow_basic_formatting: bool = False) -> str:
        """
        Sanitize HTML content
        
        Args:
            text: Input text that may contain HTML
            allow_basic_formatting: Allow <b>, <i>, <p> tags
            
        Returns:
            Sanitized text with HTML escaped or stripped
        """
        if not text:
            return ""
        
        if not allow_basic_formatting:
            # Escape all HTML
            return html.escape(text)
        
        # Remove XSS patterns
        for pattern in ValidationPatterns.XSS_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def sanitize_sql(text: str) -> str:
        """
        Sanitize SQL content
        
        Args:
            text: Input text that may contain SQL
            
        Returns:
            Sanitized text with SQL patterns removed
        """
        if not text:
            return ""
        
        for pattern in ValidationPatterns.SQL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("SQL injection attempt detected")
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def sanitize_path(path: str) -> str:
        """
        Sanitize file path
        
        Args:
            path: File path to sanitize
            
        Returns:
            Sanitized path without traversal attacks
        """
        if not path:
            return ""
        
        for pattern in ValidationPatterns.PATH_TRAVERSAL:
            if re.search(pattern, path, re.IGNORECASE):
                logger.warning("Path traversal attempt detected")
                raise ValueError("Invalid path: contains suspicious patterns")
        
        return os.path.normpath(path)
    
    @staticmethod
    def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
        """
        General text sanitization
        
        Args:
            text: Input text
            max_length: Maximum allowed length
            
        Returns:
            Sanitized and truncated text
        """
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Trim whitespace
        text = text.strip()
        
        # Truncate if needed
        if max_length and len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    @staticmethod
    def sanitize_message(message: str) -> str:
        """
        Sanitize user message for chat
        
        Args:
            message: User message
            
        Returns:
            Sanitized message safe for processing
        """
        if not message:
            return ""
        
        message = Sanitizer.sanitize_html(message, allow_basic_formatting=False)
        message = Sanitizer.sanitize_sql(message)
        message = Sanitizer.sanitize_text(message, ValidationConfig.MAX_MESSAGE_LENGTH)
        
        return message


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

class Validator:
    """
    Content validation utilities
    
    Checks if content meets security and format requirements.
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        if not email or len(email) > ValidationConfig.MAX_EMAIL_LENGTH:
            return False
        
        return bool(re.match(ValidationPatterns.EMAIL, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        if not url or len(url) > ValidationConfig.MAX_URL_LENGTH:
            return False
        
        if not re.match(ValidationPatterns.URL, url):
            return False
        
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def validate_length(text: str, min_length: int = 0, max_length: Optional[int] = None) -> bool:
        """Validate text length"""
        if not text:
            return min_length == 0
        
        length = len(text)
        return (length >= min_length) and (not max_length or length <= max_length)
    
    @staticmethod
    def validate_no_xss(text: str) -> bool:
        """Check for XSS attempts"""
        if not text:
            return True
        
        for pattern in ValidationPatterns.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("XSS attempt detected")
                return False
        
        return True
    
    @staticmethod
    def validate_no_sql_injection(text: str) -> bool:
        """Check for SQL injection attempts"""
        if not text:
            return True
        
        for pattern in ValidationPatterns.SQL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("SQL injection attempt detected")
                return False
        
        return True
    
    @staticmethod
    def validate_safe_text(text: str) -> bool:
        """Check if text contains only safe characters"""
        if not text:
            return True
        
        return Validator.validate_no_xss(text) and Validator.validate_no_sql_injection(text)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sanitize_input(text: str, input_type: str = "text") -> str:
    """Sanitize input based on type"""
    if input_type == "message":
        return Sanitizer.sanitize_message(text)
    elif input_type == "html":
        return Sanitizer.sanitize_html(text)
    elif input_type == "path":
        return Sanitizer.sanitize_path(text)
    else:
        return Sanitizer.sanitize_text(text)


def validate_input(text: str, input_type: str = "text") -> bool:
    """Validate input based on type"""
    if input_type == "email":
        return Validator.validate_email(text)
    elif input_type == "url":
        return Validator.validate_url(text)
    else:
        return Validator.validate_safe_text(text)


logger.info("✅ Validators module initialized")