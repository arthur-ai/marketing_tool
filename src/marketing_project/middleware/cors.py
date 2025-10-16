"""
CORS middleware for FastAPI.

This module provides CORS (Cross-Origin Resource Sharing) configuration.
"""

import logging
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("marketing_project.middleware.cors")


def setup_cors(app: FastAPI, 
               allowed_origins: Optional[List[str]] = None,
               allowed_methods: Optional[List[str]] = None,
               allowed_headers: Optional[List[str]] = None,
               allow_credentials: bool = True,
               max_age: int = 600) -> None:
    """
    Setup CORS middleware for FastAPI application.
    
    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins (default: ["*"] for development)
        allowed_methods: List of allowed HTTP methods
        allowed_headers: List of allowed headers
        allow_credentials: Whether to allow credentials
        max_age: Maximum age for preflight requests
    """
    
    # Default configuration
    if allowed_origins is None:
        allowed_origins = [
            "http://localhost:3000",  # React dev server
            "http://localhost:8080",  # Vue dev server
            "http://localhost:4200",  # Angular dev server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:4200",
        ]
    
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
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
        max_age=max_age,
    )
    
    logger.info(f"CORS middleware configured with origins: {allowed_origins}")
