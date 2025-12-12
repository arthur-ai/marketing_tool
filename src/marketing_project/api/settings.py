"""
API endpoints for pipeline settings management.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.models.user_context import UserContext
from marketing_project.services.pipeline_settings_manager import (
    PipelineSettings,
    get_pipeline_settings_manager,
)

logger = logging.getLogger("marketing_project.api.settings")

router = APIRouter(prefix="/v1/settings", tags=["settings"])


class PipelineSettingsRequest(BaseModel):
    """Request to save pipeline settings."""

    pipeline_config: Dict[str, Any] = Field(
        default_factory=dict, description="Pipeline configuration"
    )
    optional_steps: list[str] = Field(
        default_factory=list, description="List of optional step names"
    )
    retry_strategy: Optional[Dict[str, Any]] = Field(
        None, description="Retry strategy configuration"
    )


class PipelineSettingsResponse(BaseModel):
    """Response with pipeline settings."""

    pipeline_config: Dict[str, Any] = Field(..., description="Pipeline configuration")
    optional_steps: list[str] = Field(..., description="List of optional step names")
    retry_strategy: Optional[Dict[str, Any]] = Field(
        None, description="Retry strategy configuration"
    )


@router.get("/pipeline", response_model=PipelineSettingsResponse)
async def get_pipeline_settings(user: UserContext = Depends(get_current_user)):
    """
    Get current pipeline settings.

    Returns:
        Current pipeline settings from database or cache
    """
    try:
        manager = get_pipeline_settings_manager()
        settings = await manager.load_settings()

        if not settings:
            # Return default settings if none found
            return PipelineSettingsResponse(
                pipeline_config={
                    "default_model": "gpt-5.1",
                    "default_temperature": 0.7,
                    "default_max_retries": 2,
                    "step_configs": {},
                    "seo_keywords_engine_config": {"default_engine": "llm"},
                },
                optional_steps=["suggested_links", "design_kit"],
                retry_strategy=None,
            )

        return PipelineSettingsResponse(
            pipeline_config=settings.pipeline_config,
            optional_steps=settings.optional_steps,
            retry_strategy=settings.retry_strategy,
        )
    except Exception as e:
        logger.error(f"Failed to get pipeline settings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get pipeline settings: {str(e)}"
        )


@router.post("/pipeline", response_model=PipelineSettingsResponse)
async def save_pipeline_settings(
    request: PipelineSettingsRequest, user: UserContext = Depends(get_current_user)
):
    """
    Save pipeline settings.

    Args:
        request: Pipeline settings to save

    Returns:
        Saved pipeline settings
    """
    try:
        manager = get_pipeline_settings_manager()
        settings = PipelineSettings(
            pipeline_config=request.pipeline_config,
            optional_steps=request.optional_steps,
            retry_strategy=request.retry_strategy,
        )

        await manager.save_settings(settings)

        logger.info("Pipeline settings saved successfully")

        return PipelineSettingsResponse(
            pipeline_config=settings.pipeline_config,
            optional_steps=settings.optional_steps,
            retry_strategy=settings.retry_strategy,
        )
    except Exception as e:
        logger.error(f"Failed to save pipeline settings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save pipeline settings: {str(e)}"
        )
