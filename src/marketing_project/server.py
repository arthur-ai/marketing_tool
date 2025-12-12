"""
Marketing Project FastAPI server module.

This module defines the FastAPI application for the marketing project MCP server. It loads configuration, initializes shared services, and exposes comprehensive API endpoints for content processing.

Endpoints:
    DETERMINISTIC PROCESSORS (Direct Processing - Faster, Predictable):
    POST /api/v1/process/blog: Process blog posts through deterministic workflow
    POST /api/v1/process/release-notes: Process release notes through deterministic workflow
    POST /api/v1/process/transcript: Process transcripts through deterministic workflow

    ORCHESTRATED ROUTES (Auto-Routing - Intelligent, Flexible):
    POST /api/v1/analyze: Analyze content for marketing pipeline processing
    POST /api/v1/pipeline: Run the complete marketing pipeline on content (auto-routes to processors)

    CONTENT SOURCES:
    GET /api/v1/content-sources: List all configured content sources
    GET /api/v1/content-sources/{source_name}/status: Get status of a specific content source
    POST /api/v1/content-sources/{source_name}/fetch: Fetch content from a specific source

    HEALTH & SYSTEM:
    GET /api/v1/health: Health check endpoint for Kubernetes probes
    GET /api/v1/ready: Readiness check endpoint for Kubernetes probes

Usage:
    Run this module directly to start a local development server with Uvicorn.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

# Import using the same path as routes.py to ensure same module instance
from marketing_project.api import content

from . import __version__

# Import API endpoints
from .api import api_router

# Load config from centralized settings (MUST be before API imports to avoid circular dependency)
# This also loads .env file via config/settings.py
from .config.settings import PIPELINE_SPEC, PROMPTS_DIR, TEMPLATE_VERSION

# Import middleware
from .middleware.cors import setup_cors
from .middleware.error_handling import ErrorHandlingMiddleware
from .middleware.logging import LoggingMiddleware, RequestIDMiddleware
from .middleware.trusted_host import TrustedHostMiddlewareWithHealthBypass
from .scheduler import Scheduler

# Initialize logger
logger = logging.getLogger("marketing_project.server")

# Instantiate shared services
scheduler = Scheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    This works correctly with uvicorn's --reload mode.
    """
    # Startup
    logger.info("=" * 80)
    logger.info("FASTAPI APPLICATION STARTUP")
    logger.info("=" * 80)
    logger.info("Marketing Project API server starting up...")
    logger.info(f"API version: {__version__}")
    logger.info(f"Template version: {TEMPLATE_VERSION}")
    logger.info(f"Prompts directory: {PROMPTS_DIR}")

    # Initialize main database connection
    try:
        # Import models to register them with SQLAlchemy Base
        from marketing_project.models import db_models  # noqa: F401
        from marketing_project.services.database import get_database_manager

        db_manager = get_database_manager()
        if await db_manager.initialize():
            # Create tables if they don't exist
            await db_manager.create_tables()
            logger.info("✓ Database connection initialized and tables created")
        else:
            logger.warning(
                "⚠ Database not configured (DATABASE_URL or POSTGRES_URL not set). Configuration persistence will be disabled."
            )
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize database connection: {e}")

    # Initialize content sources from configuration
    logger.info("Calling initialize_content_sources()...")
    await content.initialize_content_sources()

    # Initialize scanned documents database
    try:
        from marketing_project.services.scanned_document_db import (
            get_scanned_document_db,
        )

        db = get_scanned_document_db()
        logger.info("✓ Scanned documents database initialized")
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize scanned documents database: {e}")

    # Initialize telemetry
    try:
        from marketing_project.services.telemetry import setup_tracing

        if setup_tracing():
            logger.info("✓ Telemetry initialized successfully")
        else:
            logger.info(
                "⚠ Telemetry not configured (missing ARTHUR_API_KEY or ARTHUR_TASK_ID)"
            )
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize telemetry: {e}")

    logger.info("=" * 80)
    logger.info("Startup completed successfully")
    logger.info("=" * 80)

    yield  # Application runs here

    # Shutdown
    logger.info("Marketing Project API server shutting down...")

    # Cleanup database connections
    try:
        from marketing_project.services.database import get_database_manager

        db_manager = get_database_manager()
        if db_manager.is_initialized:
            await db_manager.cleanup()
            logger.info("Database connections cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up database connections: {e}")

    # Cleanup Redis connections
    try:
        from marketing_project.services.redis_manager import get_redis_manager

        redis_manager = get_redis_manager()
        await redis_manager.cleanup()
        logger.info("Redis connections cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up Redis connections: {e}")

    # Cleanup telemetry
    try:
        from marketing_project.services.telemetry import cleanup_tracing

        cleanup_tracing()
        logger.info("Telemetry cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up telemetry: {e}")

    # Cleanup other services
    try:
        from marketing_project.services.job_manager import get_job_manager

        job_manager = get_job_manager()
        await job_manager.cleanup()
    except Exception as e:
        logger.warning(f"Error cleaning up job manager: {e}")

    try:
        from marketing_project.services.approval_manager import get_approval_manager

        approval_manager = await get_approval_manager()
        await approval_manager.cleanup()
    except Exception as e:
        logger.warning(f"Error cleaning up approval manager: {e}")

    try:
        from marketing_project.services.design_kit_manager import get_design_kit_manager

        design_kit_manager = await get_design_kit_manager()
        await design_kit_manager.cleanup()
    except Exception as e:
        logger.warning(f"Error cleaning up design kit manager: {e}")

    try:
        from marketing_project.services.internal_docs_manager import (
            get_internal_docs_manager,
        )

        internal_docs_manager = await get_internal_docs_manager()
        await internal_docs_manager.cleanup()
    except Exception as e:
        logger.warning(f"Error cleaning up internal docs manager: {e}")

    logger.info("Shutdown completed")


# Create FastAPI app with lifespan context manager
app = FastAPI(
    title="Marketing Project API",
    description="Comprehensive API for marketing content processing and analysis",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    servers=[
        {"url": "http://localhost:8000", "description": "Development server"},
        {
            "url": "https://api.marketing-project.com",
            "description": "Production server",
        },
    ],
)


# Configure OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Add middleware in correct order (last added is first executed)
# 1. Request ID middleware (innermost)
app.add_middleware(RequestIDMiddleware)

# 2. Logging middleware
app.add_middleware(LoggingMiddleware, log_requests=True, log_responses=True)

# 3. Error handling middleware
app.add_middleware(
    ErrorHandlingMiddleware, debug=os.getenv("DEBUG", "false").lower() == "true"
)

# 4. CORS middleware
# Read CORS configuration from environment variables
cors_origins_env = os.getenv("CORS_ORIGINS")
cors_origins = None
if cors_origins_env:
    cors_origins = [
        origin.strip() for origin in cors_origins_env.split(",") if origin.strip()
    ]

cors_allow_credentials_env = os.getenv("CORS_ALLOW_CREDENTIALS", "true")
cors_allow_credentials = cors_allow_credentials_env.lower() in ("true", "1", "yes")

setup_cors(app, allowed_origins=cors_origins, allow_credentials=cors_allow_credentials)

# 5. Trusted host middleware (outermost) - with health check bypass
app.add_middleware(
    TrustedHostMiddlewareWithHealthBypass,
    allowed_hosts=["localhost", "127.0.0.1", "*.arthur.ai"],
)

# Include API router
app.include_router(api_router)


if __name__ == "__main__":
    # local dev server
    uvicorn.run("marketing_project.server:app", host="0.0.0.0", port=8000, reload=True)
