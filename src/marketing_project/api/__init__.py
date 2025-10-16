"""
API package for Marketing Project.

This package contains all API endpoints organized by functionality.
"""

from fastapi import APIRouter

# Import all endpoint modules
from . import core, content, health, performance, security, system

# Create main API router
api_router = APIRouter(prefix="/api/v1", tags=["Marketing API"])

# Include all sub-routers
api_router.include_router(core.router, tags=["Core"])
api_router.include_router(content.router, tags=["Content Sources"])
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(performance.router, tags=["Performance"])
api_router.include_router(security.router, tags=["Security"])
api_router.include_router(system.router, tags=["System"])

__all__ = ["api_router"]
