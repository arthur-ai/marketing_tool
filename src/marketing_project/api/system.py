"""
System and database management API endpoints.
"""

import logging
import os

from fastapi import APIRouter, HTTPException

from marketing_project.config.settings import PIPELINE_SPEC, PROMPTS_DIR

logger = logging.getLogger("marketing_project.api.system")

# Create router
router = APIRouter()


@router.get("/system/info")
async def get_system_info():
    """
    Get system information and configuration.

    Returns system information including:
    - Version information
    - Configuration status
    - Environment details
    """
    try:
        import platform
        import sys

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
        }

        return info
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve system information: {str(e)}"
        )
