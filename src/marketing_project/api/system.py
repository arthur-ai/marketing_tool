"""
System and database management API endpoints.
"""

import logging
import os
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from marketing_project.middleware.auth import get_current_user

logger = logging.getLogger("marketing_project.api.system")

# Create router
router = APIRouter()


@router.get("/cache/stats")
async def get_cache_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get cache statistics and performance metrics.
    
    Returns cache performance metrics including:
    - Hit/miss rates
    - Cache size
    - Eviction statistics
    """
    try:
        from marketing_project.performance.optimization import get_cache_manager
        
        cache_manager = get_cache_manager()
        stats = await cache_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve cache statistics: {str(e)}"
        )


@router.post("/cache/clear")
async def clear_cache(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Clear all cache entries.
    
    Requires admin permissions.
    """
    try:
        # Check if user has admin permissions
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=403,
                detail="Admin permissions required to clear cache"
            )
        
        from marketing_project.performance.optimization import get_cache_manager
        
        cache_manager = get_cache_manager()
        await cache_manager.clear()
        
        return {"message": "Cache cleared successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/database/status")
async def get_database_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get database connection status and health.
    
    Returns status for all configured databases:
    - PostgreSQL
    - MongoDB
    - Redis
    """
    try:
        from marketing_project.services.database_source import (
            SQLContentSource, MongoDBContentSource, RedisContentSource
        )
        from marketing_project.core.content_sources import DatabaseSourceConfig
        
        status = {}
        
        # Check PostgreSQL
        try:
            postgres_config = DatabaseSourceConfig(
                name="postgres_check",
                source_type="database",
                connection_string=os.getenv("POSTGRES_URL", "postgresql://localhost:5432/test")
            )
            postgres_source = SQLContentSource(postgres_config)
            postgres_healthy = await postgres_source.health_check()
            status["postgresql"] = {"status": "healthy" if postgres_healthy else "unhealthy"}
        except Exception as e:
            status["postgresql"] = {"status": "error", "error": str(e)}
        
        # Check MongoDB
        try:
            mongodb_config = DatabaseSourceConfig(
                name="mongodb_check",
                source_type="database",
                connection_string=os.getenv("MONGODB_URL", "mongodb://localhost:27017/test")
            )
            mongodb_source = MongoDBContentSource(mongodb_config)
            mongodb_healthy = await mongodb_source.health_check()
            status["mongodb"] = {"status": "healthy" if mongodb_healthy else "unhealthy"}
        except Exception as e:
            status["mongodb"] = {"status": "error", "error": str(e)}
        
        # Check Redis
        try:
            redis_config = DatabaseSourceConfig(
                name="redis_check",
                source_type="database",
                connection_string=os.getenv("REDIS_URL", "redis://localhost:6379")
            )
            redis_source = RedisContentSource(redis_config)
            redis_healthy = await redis_source.health_check()
            status["redis"] = {"status": "healthy" if redis_healthy else "unhealthy"}
        except Exception as e:
            status["redis"] = {"status": "error", "error": str(e)}
        
        return status
    except Exception as e:
        logger.error(f"Failed to get database status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve database status: {str(e)}"
        )


@router.get("/system/info")
async def get_system_info(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get system information and configuration.
    
    Returns system information including:
    - Version information
    - Configuration status
    - Environment details
    - Feature flags
    """
    try:
        import platform
        import sys
        from marketing_project.server import PIPELINE_SPEC, PROMPTS_DIR
        
        info = {
            "service": "marketing-project",
            "version": "1.0.0",
            "python_version": sys.version,
            "platform": platform.platform(),
            "environment": {
                "debug": os.getenv("DEBUG", "false").lower() == "true",
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
                "template_version": os.getenv("TEMPLATE_VERSION", "v1"),
            },
            "configuration": {
                "pipeline_loaded": PIPELINE_SPEC is not None,
                "prompts_dir_exists": os.path.exists(PROMPTS_DIR),
                "prompts_dir": PROMPTS_DIR,
            },
            "features": {
                "security_audit": os.getenv("SECURITY_AUDIT_ENABLED", "true").lower() == "true",
                "performance_monitoring": os.getenv("PERFORMANCE_MONITORING_ENABLED", "true").lower() == "true",
                "caching": os.getenv("PERFORMANCE_CACHE_ENABLED", "true").lower() == "true",
                "rate_limiting": os.getenv("SECURITY_RATE_LIMIT_ENABLED", "true").lower() == "true",
            }
        }
        
        return info
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system information: {str(e)}"
        )
