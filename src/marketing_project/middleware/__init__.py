"""
Middleware package for FastAPI.

This package contains essential middleware components for the marketing project API.
"""

from .cors import setup_cors
from .error_handling import ErrorHandlingMiddleware, create_error_response
from .keycloak_auth import get_current_user, is_public_path
from .logging import LoggingMiddleware, RequestIDMiddleware
from .rbac import require_all_roles, require_any_role, require_roles
from .trusted_host import TrustedHostMiddlewareWithHealthBypass

__all__ = [
    "setup_cors",
    "LoggingMiddleware",
    "RequestIDMiddleware",
    "ErrorHandlingMiddleware",
    "create_error_response",
    "TrustedHostMiddlewareWithHealthBypass",
    "get_current_user",
    "is_public_path",
    "require_roles",
    "require_any_role",
    "require_all_roles",
]
