"""
Health and readiness check API endpoints.
"""

import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from marketing_project import __version__
from marketing_project.config.settings import PIPELINE_SPEC, PROMPTS_DIR
from marketing_project.services.redis_manager import get_redis_manager

logger = logging.getLogger("marketing_project.api.health")

# Create router
router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint for Kubernetes probes.

    Returns 200 OK if the service is healthy.
    Includes Redis health status and metrics.
    """
    try:
        # Get Redis health status
        redis_manager = get_redis_manager()
        redis_health = redis_manager.get_health_status()

        health_status = {
            "status": "healthy",
            "service": "marketing-project",
            "version": __version__,
            "checks": {
                "config_loaded": PIPELINE_SPEC is not None,
                "prompts_dir_exists": os.path.exists(PROMPTS_DIR),
                "redis_healthy": redis_health["healthy"],
            },
            "redis": {
                "healthy": redis_health["healthy"],
                "circuit_breaker_state": redis_health["circuit_breaker_state"],
                "pool_info": redis_health.get("pool_info", {}),
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
    Includes Redis readiness check.
    """
    try:
        # Check Redis readiness
        redis_manager = get_redis_manager()
        redis_health = redis_manager.get_health_status()
        redis_ready = (
            redis_health["healthy"] and redis_health["circuit_breaker_state"] != "open"
        )

        ready_status = {
            "status": "ready",
            "service": "marketing-project",
            "checks": {
                "config_loaded": PIPELINE_SPEC is not None,
                "prompts_dir_exists": os.path.exists(PROMPTS_DIR),
                "redis_ready": redis_ready,
            },
            "redis": {
                "ready": redis_ready,
                "circuit_breaker_state": redis_health["circuit_breaker_state"],
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
