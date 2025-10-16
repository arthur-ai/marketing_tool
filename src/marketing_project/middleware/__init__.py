"""
Middleware package for FastAPI.

This package contains all middleware components for the marketing project API.
"""

from .auth import AuthenticationMiddleware, APIKeyAuth, require_permission, get_current_user, get_current_api_key
from .cors import setup_cors
from .rate_limiting import RateLimitingMiddleware, RateLimiter
from .logging import LoggingMiddleware, RequestIDMiddleware
from .error_handling import ErrorHandlingMiddleware, create_error_response

__all__ = [
    "AuthenticationMiddleware",
    "APIKeyAuth", 
    "require_permission",
    "get_current_user",
    "get_current_api_key",
    "setup_cors",
    "RateLimitingMiddleware",
    "RateLimiter",
    "LoggingMiddleware",
    "RequestIDMiddleware",
    "ErrorHandlingMiddleware",
    "create_error_response"
]
