"""
Marketing Project FastAPI server module.

This module defines the FastAPI application for the marketing project MCP server. It loads configuration, initializes shared services, and exposes comprehensive API endpoints for content processing.

Endpoints:
    POST /api/v1/analyze: Analyze content for marketing pipeline processing
    POST /api/v1/pipeline: Run the complete marketing pipeline on content
    GET /api/v1/content-sources: List all configured content sources
    GET /api/v1/content-sources/{source_name}/status: Get status of a specific content source
    POST /api/v1/content-sources/{source_name}/fetch: Fetch content from a specific source
    GET /api/v1/health: Health check endpoint for Kubernetes probes
    GET /api/v1/ready: Readiness check endpoint for Kubernetes probes

Usage:
    Run this module directly to start a local development server with Uvicorn.
"""

import logging
import os
from typing import Optional

import uvicorn
import yaml

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi

# Import middleware
from .middleware.auth import AuthenticationMiddleware
from .middleware.cors import setup_cors
from .middleware.rate_limiting import RateLimitingMiddleware
from .middleware.logging import LoggingMiddleware, RequestIDMiddleware
from .middleware.error_handling import ErrorHandlingMiddleware
from .middleware.performance import PerformanceMonitoringMiddleware

# Import API endpoints
from .api import api_router

from .runner import run_marketing_project_pipeline
from .scheduler import Scheduler

# Initialize logger
logger = logging.getLogger("marketing_project.server")

# Load config once
BASE = os.path.dirname(__file__)
SPEC_PATH = os.path.abspath(os.path.join(BASE, "..", "..", "config", "pipeline.yml"))
with open(SPEC_PATH) as f:
    PIPELINE_SPEC = yaml.safe_load(f)
TEMPLATE_VERSION = os.getenv("TEMPLATE_VERSION", "v1")
PROMPTS_DIR = os.path.abspath(os.path.join(BASE, "prompts", TEMPLATE_VERSION))

# Instantiate shared services
scheduler = Scheduler()

# Create FastAPI app with comprehensive configuration
app = FastAPI(
    title="Marketing Project API",
    description="Comprehensive API for marketing content processing and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    servers=[
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.marketing-project.com", "description": "Production server"}
    ]
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
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication"
        }
    }
    
    # Add security requirements
    openapi_schema["security"] = [{"ApiKeyAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Add middleware in correct order (last added is first executed)
# 1. Request ID middleware (innermost)
app.add_middleware(RequestIDMiddleware)

# 2. Logging middleware
app.add_middleware(LoggingMiddleware, log_requests=True, log_responses=True)

# 3. Error handling middleware
app.add_middleware(ErrorHandlingMiddleware, debug=os.getenv("DEBUG", "false").lower() == "true")

# 4. Rate limiting middleware
rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "100"))
rate_limit_burst = int(os.getenv("RATE_LIMIT_BURST_LIMIT", "20"))
app.add_middleware(
    RateLimitingMiddleware,
    requests_per_minute=rate_limit_requests,
    burst_limit=rate_limit_burst,
    per_ip=True,
    per_user=True
)

# 5. Authentication middleware
app.add_middleware(AuthenticationMiddleware)

# 6. Performance monitoring middleware
app.add_middleware(PerformanceMonitoringMiddleware, monitor_health_endpoints=False)

# 7. CORS middleware
setup_cors(app)

# 8. Trusted host middleware (outermost)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.marketing-project.com"]
)

# Include API router
app.include_router(api_router)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Marketing Project API server starting up...")
    logger.info(f"API version: 1.0.0")
    logger.info(f"Template version: {TEMPLATE_VERSION}")
    logger.info(f"Prompts directory: {PROMPTS_DIR}")
    logger.info(f"Rate limit: {rate_limit_requests} requests/minute")
    logger.info("Startup completed successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Marketing Project API server shutting down...")
    # Add any cleanup logic here
    logger.info("Shutdown completed")


if __name__ == "__main__":
    # local dev server
    uvicorn.run("marketing_project.server:app", host="0.0.0.0", port=8000, reload=True)
