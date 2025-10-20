"""
Health and readiness check API endpoints.
"""

import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from marketing_project.config.settings import PIPELINE_SPEC, PROMPTS_DIR

logger = logging.getLogger("marketing_project.api.health")

# Create router
router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint for Kubernetes probes.

    Returns 200 OK if the service is healthy.
    """
    try:

        health_status = {
            "status": "healthy",
            "service": "marketing-project",
            "version": "1.0.0",
            "checks": {
                "config_loaded": PIPELINE_SPEC is not None,
                "prompts_dir_exists": os.path.exists(PROMPTS_DIR),
            },
        }

        # Check if any critical checks failed
        if not all(health_status["checks"].values()):
            health_status["status"] = "unhealthy"
            return JSONResponse(status_code=503, content=health_status)

        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes probes.

    Returns 200 OK if the service is ready to accept traffic.
    """
    try:

        ready_status = {
            "status": "ready",
            "service": "marketing-project",
            "checks": {
                "config_loaded": PIPELINE_SPEC is not None,
                "prompts_dir_exists": os.path.exists(PROMPTS_DIR),
            },
        }

        if not all(ready_status["checks"].values()):
            ready_status["status"] = "not_ready"
            return JSONResponse(status_code=503, content=ready_status)

        return ready_status
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "error": str(e)}
        )
