"""
Authentication middleware for FastAPI.

This module provides API key authentication and authorization middleware.
"""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..models import ErrorResponse
from ..security import validate_api_key, security_auditor
from ..security.input_validation import SecurityValidationError

logger = logging.getLogger("marketing_project.middleware.auth")


class APIKeyAuth:
    """API Key authentication handler."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.valid_keys = self._load_valid_keys()
    
    def _load_valid_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load valid API keys from environment or configuration."""
        # In production, this would load from a secure store
        valid_keys = {}
        
        # Load from environment variables
        main_key = os.getenv("API_KEY")
        if main_key:
            valid_keys[main_key] = {
                "role": "admin",
                "permissions": ["read", "write", "delete", "admin"],
                "created_at": datetime.utcnow(),
                "expires_at": None
            }
        
        # Load additional keys from environment
        for i in range(1, 6):  # Support up to 5 additional keys
            key = os.getenv(f"API_KEY_{i}")
            if key:
                role = os.getenv(f"API_KEY_{i}_ROLE", "user")
                valid_keys[key] = {
                    "role": role,
                    "permissions": self._get_permissions_for_role(role),
                    "created_at": datetime.utcnow(),
                    "expires_at": None
                }
        
        return valid_keys
    
    def _get_permissions_for_role(self, role: str) -> list:
        """Get permissions for a given role."""
        role_permissions = {
            "admin": ["read", "write", "delete", "admin"],
            "user": ["read", "write"],
            "viewer": ["read"]
        }
        return role_permissions.get(role, ["read"])
    
    def validate_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return user info."""
        if not api_key:
            return None
        
        try:
            # Validate API key format and security
            validate_api_key(api_key)
        except SecurityValidationError as e:
            logger.warning(f"Invalid API key format: {e}")
            return None
        
        user_info = self.valid_keys.get(api_key)
        if not user_info:
            return None
        
        # Check if key has expired
        if user_info.get("expires_at") and user_info["expires_at"] < datetime.utcnow():
            return None
        
        return user_info
    
    def has_permission(self, api_key: str, required_permission: str) -> bool:
        """Check if API key has required permission."""
        user_info = self.validate_key(api_key)
        if not user_info:
            return False
        
        permissions = user_info.get("permissions", [])
        return required_permission in permissions or "admin" in permissions


# Global auth instance
auth_handler = APIKeyAuth(os.getenv("API_KEY", "default-key"))


class APIKeyHeaderAuth(APIKeyHeader):
    """API Key header authentication."""
    
    def __init__(self):
        super().__init__(name="X-API-Key", auto_error=False)
    
    async def __call__(self, request: Request) -> Optional[str]:
        """Extract API key from request header."""
        api_key = await super().__call__(request)
        if not api_key:
            # Try alternative header names
            api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
            if not api_key:
                api_key = request.headers.get("X-Api-Key")
        return api_key


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for FastAPI."""
    
    def __init__(self, app, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/ready",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware."""
        # Skip authentication for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Extract API key
        api_key = request.headers.get("X-API-Key") or request.headers.get("X-Api-Key")
        
        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=ErrorResponse(
                    success=False,
                    message="API key required",
                    error_code="MISSING_API_KEY",
                    error_details={"header": "X-API-Key"}
                ).dict()
            )
        
        # Validate API key
        user_info = auth_handler.validate_key(api_key)
        if not user_info:
            # Log authentication failure
            security_auditor.audit_logger.log_authentication_failure(
                source_ip=request.client.host if request.client else "unknown",
                api_key=api_key,
                endpoint=request.url.path,
                reason="Invalid API key",
                request_id=getattr(request.state, 'request_id', 'unknown')
            )
            
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=ErrorResponse(
                    success=False,
                    message="Invalid API key",
                    error_code="INVALID_API_KEY"
                ).dict()
            )
        
        # Log successful authentication
        security_auditor.audit_logger.log_authentication_success(
            source_ip=request.client.host if request.client else "unknown",
            user_id=user_info.get('role', 'unknown'),
            api_key=api_key,
            endpoint=request.url.path,
            request_id=getattr(request.state, 'request_id', 'unknown')
        )
        
        # Add user info to request state
        request.state.user = user_info
        request.state.api_key = api_key
        
        logger.info(f"Authenticated request from user with role: {user_info.get('role')}")
        
        return await call_next(request)


def require_permission(permission: str):
    """Decorator to require specific permission for endpoint."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get request from kwargs (FastAPI dependency injection)
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            api_key = getattr(request.state, 'api_key', None)
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not auth_handler.has_permission(api_key, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current user from request state."""
    return getattr(request.state, 'user', {})


def get_current_api_key(request: Request) -> str:
    """Get current API key from request state."""
    return getattr(request.state, 'api_key', '')
