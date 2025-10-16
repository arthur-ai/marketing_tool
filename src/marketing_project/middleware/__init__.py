"""
Middleware package for FastAPI.

This package contains essential middleware components for the marketing project API.
"""

from .cors import setup_cors
from .logging import LoggingMiddleware, RequestIDMiddleware
from .error_handling import ErrorHandlingMiddleware, create_error_response

__all__ = [
    "setup_cors",
    "LoggingMiddleware",
    "RequestIDMiddleware",
    "ErrorHandlingMiddleware",
    "create_error_response"
]
