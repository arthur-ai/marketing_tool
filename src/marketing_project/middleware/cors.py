"""
CORS middleware for FastAPI.

This module provides CORS (Cross-Origin Resource Sharing) configuration.
"""

import logging
import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("marketing_project.middleware.cors")


def setup_cors(
    app: FastAPI,
    allowed_origins: Optional[List[str]] = None,
    allowed_methods: Optional[List[str]] = None,
    allowed_headers: Optional[List[str]] = None,
    allow_credentials: Optional[bool] = None,
    max_age: int = 600,
) -> None:
    """
    Setup CORS middleware for FastAPI application.

    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins (default: reads from CORS_ORIGINS env var or uses defaults)
        allowed_methods: List of allowed HTTP methods
        allowed_headers: List of allowed headers
        allow_credentials: Whether to allow credentials (default: reads from CORS_ALLOW_CREDENTIALS env var or True)
        max_age: Maximum age for preflight requests
    """

    # Check environment variables first to determine if we should use regex
    cors_origins_env = os.getenv("CORS_ORIGINS")
    use_localhost_regex = os.getenv("CORS_ALLOW_LOCALHOST_REGEX", "true").lower() in (
        "true",
        "1",
        "yes",
    )

    # Read from environment variables if not provided
    if allowed_origins is None:
        if cors_origins_env and cors_origins_env.strip():
            # Parse comma-separated origins from environment variable
            allowed_origins = [
                origin.strip()
                for origin in cors_origins_env.split(",")
                if origin.strip()
            ]
        else:
            # No explicit CORS_ORIGINS set - will use regex if enabled
            allowed_origins = None

    # Determine if we should use regex pattern (when CORS_ORIGINS is not set and regex is enabled)
    # Use regex if: regex is enabled AND no explicit origins (from env or parameter)
    should_use_regex = use_localhost_regex and (
        allowed_origins is None
        or (isinstance(allowed_origins, list) and len(allowed_origins) == 0)
    )

    # If not using regex and no origins set, use default list
    if not should_use_regex and (
        allowed_origins is None
        or (isinstance(allowed_origins, list) and len(allowed_origins) == 0)
    ):
        # Default configuration for development
        allowed_origins = [
            "http://localhost:3000",  # Next.js/React dev server (default)
            "http://localhost:3001",  # Alternative React dev server
            "http://localhost:8080",  # Vue dev server
            "http://localhost:4200",  # Angular dev server
            "http://localhost:5173",  # Vite dev server
            "http://localhost:5174",  # Vite dev server (alternative)
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:4200",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
        ]

    if allow_credentials is None:
        allow_credentials_env = os.getenv("CORS_ALLOW_CREDENTIALS", "true")
        allow_credentials = allow_credentials_env.lower() in ("true", "1", "yes")

    if allowed_methods is None:
        allowed_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]

    if allowed_headers is None:
        allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Api-Key",
            "X-Requested-With",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers",
        ]

    # Use regex for localhost if conditions are met
    if should_use_regex:
        # Use regex to match any localhost or 127.0.0.1 with any port
        # This is safe for development as it only matches localhost
        regex_pattern = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=regex_pattern,
            allow_credentials=allow_credentials,
            allow_methods=allowed_methods,
            allow_headers=allowed_headers,
            max_age=max_age,
        )
        logger.info(
            f"CORS middleware configured with localhost regex pattern: {regex_pattern} "
            "(matches any localhost/127.0.0.1 port). "
            "Set CORS_ORIGINS environment variable to use explicit origins instead."
        )
    else:
        # Use explicit origins list (required when CORS_ORIGINS is set or regex is disabled)
        if not allowed_origins:
            # Fallback: if somehow we have no origins, use localhost:3000 as minimum
            allowed_origins = ["http://localhost:3000"]
            logger.warning(
                "No CORS origins configured, using default: http://localhost:3000"
            )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=allow_credentials,
            allow_methods=allowed_methods,
            allow_headers=allowed_headers,
            max_age=max_age,
        )
        logger.info(f"CORS middleware configured with origins: {allowed_origins}")
