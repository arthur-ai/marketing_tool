"""
API package for Marketing Project.

This package contains all API endpoints organized by functionality.
All routes are centralized in routes.py for easy management and visibility.
"""

from marketing_project.api.routes import register_routes

# Create and register all routes
api_router = register_routes()

__all__ = ["api_router"]
