"""
JWT Authentication Middleware
Zero hardcoded values - all configuration from environment
Follows AGENTS.md principles: clean naming, type safety, async patterns
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

# Security scheme for JWT bearer tokens
security = HTTPBearer(auto_error=False)


class AuthenticationError(Exception):
    """Custom authentication error"""
    pass


class AuthorizationError(Exception):
    """Custom authorization error"""
    pass


async def verify_jwt_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Verify JWT token and return payload
    
    Args:
        credentials: HTTP bearer credentials from request
        
    Returns:
        Token payload dictionary
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    
    try:
        # Decode JWT using settings from environment
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        # Verify expiration
        exp = payload.get("exp")
        if exp:
            exp_datetime = datetime.fromtimestamp(exp)
            if exp_datetime < datetime.utcnow():
                logger.warning(f"Expired token attempt: exp={exp_datetime}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"}
                )
        
        # Verify required fields
        if not payload.get("sub"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format: missing subject",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        logger.info(f"Token verified for user: {payload.get('sub')}")
        return payload
        
    except JWTError as e:
        logger.error(f"JWT validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user(payload: Dict[str, Any] = Depends(verify_jwt_token)) -> Dict[str, Any]:
    """
    Get current authenticated user from token payload
    
    Args:
        payload: JWT token payload from verify_jwt_token
        
    Returns:
        User information dictionary
    """
    return payload


async def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Require admin privileges for endpoint access
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        User information if admin
        
    Raises:
        HTTPException: If user is not admin (403)
    """
    # Check multiple possible admin indicators (flexible)
    is_admin = (
        current_user.get("is_admin", False) or
        current_user.get("admin", False) or
        current_user.get("role") == "admin" or
        current_user.get("roles", []) and "admin" in current_user.get("roles", [])
    )
    
    if not is_admin:
        user_id = current_user.get("sub", "unknown")
        logger.warning(f"Unauthorized admin access attempt by user: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this operation"
        )
    
    logger.info(f"Admin access granted to user: {current_user.get('sub')}")
    return current_user


async def require_user_or_admin(
    user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Allow access if user owns resource or is admin
    
    Args:
        user_id: ID of resource owner
        current_user: Current authenticated user
        
    Returns:
        User information if authorized
        
    Raises:
        HTTPException: If not authorized (403)
    """
    is_owner = current_user.get("sub") == user_id
    is_admin = current_user.get("is_admin", False) or current_user.get("role") == "admin"
    
    if not (is_owner or is_admin):
        logger.warning(
            f"Unauthorized resource access: user={current_user.get('sub')}, "
            f"resource_owner={user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this resource"
        )
    
    return current_user


async def optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Extract user from token if present, otherwise return None
    Used for endpoints that work with or without authentication
    
    Args:
        credentials: Optional HTTP bearer credentials
        
    Returns:
        User payload if token valid, None otherwise
    """
    if not credentials:
        return None
    
    try:
        return await verify_jwt_token(credentials)
    except HTTPException:
        return None
