"""
API package for Marketing Project.

This package contains all API endpoints organized by functionality.
"""

from fastapi import APIRouter

# Import all endpoint modules
from . import core, content, health, system, upload

# Create main API router
api_router = APIRouter(prefix="/api/v1", tags=["Marketing API"])

# Include all sub-routers
api_router.include_router(core.router, tags=["Core"])
api_router.include_router(content.router, tags=["Content Sources"])
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(system.router, tags=["System"])
api_router.include_router(upload.router, tags=["File Upload"])

__all__ = ["api_router"]
