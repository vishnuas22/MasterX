"""
Authentication Manager for MasterX Quantum Intelligence Platform

Comprehensive authentication and authorization system that integrates with
the API gateway and provides secure access to all quantum intelligence services.

🔐 AUTHENTICATION CAPABILITIES:
- JWT token-based authentication with refresh tokens
- Role-based access control (RBAC)
- API key authentication for service-to-service communication
- Integration with multiple LLM providers (Groq, Gemini)
- Session management and user profiling
- Rate limiting and security monitoring

Author: MasterX AI Team - API Integration Division
Version: 1.0 - Phase 12 Enhanced Backend APIs Integration
"""

import os
import jwt
import bcrypt
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

from .models import UserProfile, UserRole, LoginRequest, LoginResponse

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
JWT_SECRET = os.getenv('JWT_SECRET', 'masterx-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))

# LLM API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCmV-mlB7rag8GurIDj07ijRDhPuNwOiVA')

# ============================================================================
# AUTHENTICATION MODELS
# ============================================================================

class TokenData(BaseModel):
    """Token data model"""
    user_id: str
    email: str
    role: UserRole
    exp: datetime
    iat: datetime

class APIKeyData(BaseModel):
    """API key data model"""
    key_id: str
    service_name: str
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None

# ============================================================================
# AUTHENTICATION MANAGER
# ============================================================================

class AuthManager:
    """
    🔐 AUTHENTICATION MANAGER
    
    Comprehensive authentication and authorization system for the MasterX
    Quantum Intelligence Platform with JWT tokens, RBAC, and API key support.
    """
    
    def __init__(self):
        """Initialize the authentication manager"""
        
        # In-memory user store (replace with database in production)
        self.users_db = {
            "admin@masterx.ai": {
                "user_id": "admin_001",
                "email": "admin@masterx.ai",
                "name": "MasterX Admin",
                "password_hash": self._hash_password("admin123"),
                "role": UserRole.ADMIN,
                "created_at": datetime.now(),
                "last_login": None,
                "preferences": {}
            },
            "student@example.com": {
                "user_id": "student_001",
                "email": "student@example.com",
                "name": "Test Student",
                "password_hash": self._hash_password("student123"),
                "role": UserRole.STUDENT,
                "created_at": datetime.now(),
                "last_login": None,
                "preferences": {}
            },
            "teacher@example.com": {
                "user_id": "teacher_001",
                "email": "teacher@example.com",
                "name": "Test Teacher",
                "password_hash": self._hash_password("teacher123"),
                "role": UserRole.TEACHER,
                "created_at": datetime.now(),
                "last_login": None,
                "preferences": {}
            }
        }
        
        # API keys store (replace with database in production)
        self.api_keys_db = {
            "masterx_internal_001": {
                "key_id": "masterx_internal_001",
                "service_name": "internal_services",
                "permissions": ["*"],
                "created_at": datetime.now(),
                "expires_at": None
            }
        }
        
        # Active sessions store
        self.active_sessions = {}
        
        logger.info("🔐 Authentication Manager initialized")
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _generate_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """Generate a JWT token for a user"""
        
        now = datetime.utcnow()
        exp = now + timedelta(hours=JWT_EXPIRATION_HOURS)
        
        payload = {
            'user_id': user_data['user_id'],
            'email': user_data['email'],
            'role': user_data['role'].value if isinstance(user_data['role'], UserRole) else user_data['role'],
            'iat': now,
            'exp': exp
        }
        
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    def _generate_refresh_token(self, user_data: Dict[str, Any]) -> str:
        """Generate a refresh token for a user"""
        
        now = datetime.utcnow()
        exp = now + timedelta(days=30)  # Refresh tokens last 30 days
        
        payload = {
            'user_id': user_data['user_id'],
            'type': 'refresh',
            'iat': now,
            'exp': exp
        }
        
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    def _decode_jwt_token(self, token: str) -> TokenData:
        """Decode and validate a JWT token"""
        
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            return TokenData(
                user_id=payload['user_id'],
                email=payload['email'],
                role=UserRole(payload['role']),
                exp=datetime.fromtimestamp(payload['exp']),
                iat=datetime.fromtimestamp(payload['iat'])
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def authenticate_user(self, login_request: LoginRequest) -> LoginResponse:
        """Authenticate a user with email and password"""
        
        try:
            # Check if user exists
            if login_request.email not in self.users_db:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            user_data = self.users_db[login_request.email]
            
            # Verify password
            if not self._verify_password(login_request.password, user_data['password_hash']):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            
            # Update last login
            user_data['last_login'] = datetime.now()
            
            # Generate tokens
            access_token = self._generate_jwt_token(user_data)
            refresh_token = self._generate_refresh_token(user_data)
            
            # Store session
            session_id = f"session_{user_data['user_id']}_{int(datetime.now().timestamp())}"
            self.active_sessions[session_id] = {
                'user_id': user_data['user_id'],
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
            
            return LoginResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=JWT_EXPIRATION_HOURS * 3600,
                user_info={
                    'user_id': user_data['user_id'],
                    'email': user_data['email'],
                    'name': user_data['name'],
                    'role': user_data['role'].value if isinstance(user_data['role'], UserRole) else user_data['role']
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error"
            )
    
    async def get_current_user(self, token: str) -> UserProfile:
        """Get current user from JWT token"""
        
        try:
            # Decode token
            token_data = self._decode_jwt_token(token)
            
            # Get user data
            user_email = token_data.email
            if user_email not in self.users_db:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            user_data = self.users_db[user_email]
            
            return UserProfile(
                user_id=user_data['user_id'],
                email=user_data['email'],
                name=user_data['name'],
                role=user_data['role'],
                created_at=user_data['created_at'],
                last_login=user_data['last_login'],
                preferences=user_data['preferences']
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get current user error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User service error"
            )
    
    async def validate_api_key(self, api_key: str) -> APIKeyData:
        """Validate an API key"""
        
        try:
            if api_key not in self.api_keys_db:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )
            
            key_data = self.api_keys_db[api_key]
            
            # Check expiration
            if key_data['expires_at'] and datetime.now() > key_data['expires_at']:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key has expired"
                )
            
            return APIKeyData(**key_data)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API key validation error"
            )
    
    def check_permission(self, user: UserProfile, required_permission: str) -> bool:
        """Check if user has required permission"""
        
        # Admin has all permissions
        if user.role == UserRole.ADMIN:
            return True
        
        # Define role-based permissions
        role_permissions = {
            UserRole.STUDENT: [
                "chat:read", "chat:write",
                "learning:read", "learning:write",
                "progress:read",
                "content:read",
                "assessment:read", "assessment:write"
            ],
            UserRole.TEACHER: [
                "chat:read", "chat:write",
                "learning:read", "learning:write",
                "progress:read", "progress:write",
                "content:read", "content:write",
                "assessment:read", "assessment:write",
                "analytics:read",
                "personalization:read"
            ],
            UserRole.ADMIN: ["*"]  # All permissions
        }
        
        user_permissions = role_permissions.get(user.role, [])
        
        return "*" in user_permissions or required_permission in user_permissions
    
    def get_llm_config(self) -> Dict[str, str]:
        """Get LLM configuration with API keys"""
        
        return {
            "groq_api_key": GROQ_API_KEY,
            "gemini_api_key": GEMINI_API_KEY,
            "default_provider": "groq"  # Can be configured
        }

# ============================================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================================

# Global auth manager instance
auth_manager = AuthManager()

# Security scheme
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserProfile:
    """FastAPI dependency to get current authenticated user"""
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required"
        )
    
    return await auth_manager.get_current_user(credentials.credentials)

async def get_admin_user(current_user: UserProfile = Depends(get_current_user)) -> UserProfile:
    """FastAPI dependency to require admin user"""
    
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user

async def get_teacher_or_admin_user(current_user: UserProfile = Depends(get_current_user)) -> UserProfile:
    """FastAPI dependency to require teacher or admin user"""
    
    if current_user.role not in [UserRole.TEACHER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Teacher or admin access required"
        )
    
    return current_user

def require_permission(permission: str):
    """Decorator to require specific permission"""
    
    def permission_checker(current_user: UserProfile = Depends(get_current_user)) -> UserProfile:
        if not auth_manager.check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    
    return permission_checker
