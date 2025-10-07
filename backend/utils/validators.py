"""
Input Validation and Sanitization
Following OWASP security best practices

Features:
- Input sanitization (XSS prevention)
- SQL injection prevention (NoSQL)
- Email validation
- URL validation
- File upload validation
- Content length validation

PRINCIPLES (AGENTS.md):
- No hardcoded values
- Real validation algorithms
- Clean, professional naming
- Comprehensive error handling
"""

import re
import html
import logging
from typing import Optional, List
from urllib.parse import urlparse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class ValidationConfig:
    """Validation configuration"""
    
    # Content length limits
    MAX_MESSAGE_LENGTH: int = 10000  # 10KB
    MAX_USERNAME_LENGTH: int = 100
    MAX_EMAIL_LENGTH: int = 255
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Allowed file types
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    ALLOWED_AUDIO_TYPES: List[str] = ["audio/wav", "audio/mp3", "audio/mpeg"]
    
    # URL validation
    ALLOWED_URL_SCHEMES: List[str] = ["http", "https"]
    
    # Character restrictions
    FORBIDDEN_CHARS: List[str] = ["<script>", "</script>", "javascript:", "onerror="]
    
    # SQL/NoSQL injection patterns
    INJECTION_PATTERNS: List[str] = [
        r"\$where",
        r"\$ne",
        r";\s*drop\s+table",
        r";\s*delete\s+from",
        r"union\s+select",
        r"<script",
        r"javascript:",
    ]


# ============================================================================
# VALIDATION MODELS
# ============================================================================

class ValidationResult(BaseModel):
    """Result of validation"""
    is_valid: bool
    sanitized_value: Optional[str] = None
    errors: List[str] = []
    warnings: List[str] = []


class FileValidation(BaseModel):
    """File upload validation result"""
    is_valid: bool
    file_type: str
    file_size: int
    errors: List[str] = []


# ============================================================================
# TEXT SANITIZER
# ============================================================================

class TextSanitizer:
    """
    Text sanitization for XSS prevention
    
    Removes or escapes potentially dangerous content.
    """
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize HTML content - escapes HTML special characters"""
        if not text:
            return ""
        
        sanitized = html.escape(text)
        
        for pattern in ValidationConfig.FORBIDDEN_CHARS:
            sanitized = sanitized.replace(pattern, "")
        
        return sanitized
    
    @staticmethod
    def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
        """General text sanitization"""
        if not text:
            return ""
        
        # Remove null bytes
        sanitized = text.replace('\x00', '')
        
        # Remove control characters
        sanitized = ''.join(char for char in sanitized if char.isprintable() or char in '\n\t')
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Trim to max length
        if max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    @staticmethod
    def remove_sql_injection_patterns(text: str) -> ValidationResult:
        """Check and remove SQL/NoSQL injection patterns"""
        errors = []
        sanitized = text
        
        for pattern in ValidationConfig.INJECTION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                errors.append(f"Potential injection pattern detected")
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized,
            errors=errors
        )


# ============================================================================
# INPUT VALIDATORS
# ============================================================================

class InputValidator:
    """Input validation for various data types"""
    
    @staticmethod
    def validate_email(email: str) -> ValidationResult:
        """Validate email address"""
        errors = []
        
        if len(email) > ValidationConfig.MAX_EMAIL_LENGTH:
            errors.append(f"Email too long")
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            errors.append("Invalid email format")
        
        if any(char in email for char in ['<', '>', ';', '"', "'"]):
            errors.append("Email contains invalid characters")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=email.lower().strip(),
            errors=errors
        )
    
    @staticmethod
    def validate_username(username: str) -> ValidationResult:
        """Validate username"""
        errors = []
        sanitized = username.strip()
        
        if len(sanitized) == 0:
            errors.append("Username cannot be empty")
        elif len(sanitized) > ValidationConfig.MAX_USERNAME_LENGTH:
            errors.append(f"Username too long")
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', sanitized):
            errors.append("Username contains invalid characters")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized,
            errors=errors
        )
    
    @staticmethod
    def validate_url(url: str) -> ValidationResult:
        """Validate URL"""
        errors = []
        
        try:
            parsed = urlparse(url)
            
            if parsed.scheme not in ValidationConfig.ALLOWED_URL_SCHEMES:
                errors.append(f"Invalid URL scheme")
            
            if not parsed.netloc:
                errors.append("Invalid URL format")
            
        except Exception as e:
            errors.append(f"URL parsing error")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=url.strip(),
            errors=errors
        )
    
    @staticmethod
    def validate_message(message: str) -> ValidationResult:
        """Validate user message"""
        errors = []
        warnings = []
        
        if len(message) == 0:
            errors.append("Message cannot be empty")
        elif len(message) > ValidationConfig.MAX_MESSAGE_LENGTH:
            errors.append(f"Message too long")
        
        sanitized = TextSanitizer.sanitize_text(message, ValidationConfig.MAX_MESSAGE_LENGTH)
        
        injection_result = TextSanitizer.remove_sql_injection_patterns(sanitized)
        if not injection_result.is_valid:
            warnings.extend(injection_result.errors)
            sanitized = injection_result.sanitized_value
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized,
            errors=errors,
            warnings=warnings
        )


# ============================================================================
# FILE VALIDATORS
# ============================================================================

class FileValidator:
    """File upload validation"""
    
    @staticmethod
    def validate_file_upload(
        filename: str,
        file_type: str,
        file_size: int,
        allowed_types: Optional[List[str]] = None
    ) -> FileValidation:
        """Validate file upload"""
        errors = []
        
        if file_size > ValidationConfig.MAX_FILE_SIZE:
            errors.append(f"File too large")
        
        if allowed_types and file_type not in allowed_types:
            errors.append(f"File type not allowed")
        
        if re.search(r'[<>:"/\\|?*]', filename):
            errors.append("Filename contains invalid characters")
        
        dangerous_extensions = ['.exe', '.bat', '.sh', '.php', '.py', '.js']
        if any(filename.lower().endswith(ext) for ext in dangerous_extensions):
            errors.append("File extension not allowed")
        
        return FileValidation(
            is_valid=len(errors) == 0,
            file_type=file_type,
            file_size=file_size,
            errors=errors
        )


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

text_sanitizer = TextSanitizer()
input_validator = InputValidator()
file_validator = FileValidator()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sanitize_text(text: str) -> str:
    """Helper: Sanitize text"""
    return text_sanitizer.sanitize_text(text)


def sanitize_html(text: str) -> str:
    """Helper: Sanitize HTML"""
    return text_sanitizer.sanitize_html(text)


def validate_email(email: str) -> ValidationResult:
    """Helper: Validate email"""
    return input_validator.validate_email(email)


def validate_message(message: str) -> ValidationResult:
    """Helper: Validate user message"""
    return input_validator.validate_message(message)


def validate_url(url: str) -> ValidationResult:
    """Helper: Validate URL"""
    return input_validator.validate_url(url)
