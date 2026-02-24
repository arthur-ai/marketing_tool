"""
Brand Kit API Endpoints.

Endpoints for managing brand kit configuration.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from ..middleware.keycloak_auth import get_current_user
from ..middleware.rbac import require_roles
from ..models.brand_kit_config import BrandKitConfig
from ..models.user_context import UserContext
from ..services.brand_kit_manager import get_brand_kit_manager
from ..services.job_manager import get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/config")
async def get_brand_kit_config(
    refresh: bool = Query(
        False,
        description="If true, submit a background job to regenerate config using AI",
    ),
    user: UserContext = Depends(get_current_user),
):
    """
    Get the currently active brand kit configuration.

    Args:
        refresh: If true, submit a background job to regenerate the config using AI.
                 Returns immediately with job_id. Use GET /api/v1/jobs/{job_id}/status to check progress.

    Returns:
        BrandKitConfig: The active configuration (or current config if refresh is requested)
        If refresh=true, returns dict with config and job_id
    """
    try:
        manager = await get_brand_kit_manager()

        # If refresh is requested, submit background job
        if refresh:
            logger.info("Refresh requested: submitting brand kit refresh job...")

            # Get job manager
            job_manager = get_job_manager()

            # Create job
            job = await job_manager.create_job(
                job_type="brand_kit_refresh",
                content_id="brand_kit",
                metadata={"operation": "refresh", "use_internal_docs": True},
                user_id=user.user_id,
                user_context=user,
            )

            # Submit job to ARQ for background processing with extended timeout
            # Brand kit generation can take longer due to multiple LLM calls
            arq_job_id = await job_manager.submit_to_arq(
                job.id,
                "refresh_brand_kit_job",  # ARQ function name
                True,  # use_internal_docs
                job.id,  # job_id
                _timeout=1800,  # 30 minutes timeout for brand kit jobs
            )

            logger.info(
                f"Brand kit refresh job {job.id} submitted to ARQ (arq_id: {arq_job_id})"
            )

            # Return current config immediately (job will update it in background)
            config = await manager.get_active_config()
            if not config:
                raise HTTPException(
                    status_code=404, detail="No active brand kit configuration found"
                )

            # Return config with job_id so client can track progress
            return {
                "job_id": job.id,
                "status_url": f"/api/v1/jobs/{job.id}/status",
                "config": config.model_dump(mode="json"),
            }

        # Otherwise, return existing config
        config = await manager.get_active_config()

        if not config:
            raise HTTPException(
                status_code=404, detail="No active brand kit configuration found"
            )

        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting brand kit config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get brand kit configuration: {str(e)}"
        )


@router.get("/config/{version}", response_model=BrandKitConfig)
async def get_brand_kit_config_by_version(
    version: str, user: UserContext = Depends(get_current_user)
):
    """
    Get brand kit configuration by version.

    Args:
        version: Version identifier

    Returns:
        BrandKitConfig: The configuration for the specified version
    """
    try:
        manager = await get_brand_kit_manager()
        config = await manager.get_config_by_version(version)

        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"Brand kit configuration version {version} not found",
            )

        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting brand kit config version {version}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get brand kit configuration version: {str(e)}",
        )


@router.get("/config/{content_type}/type", response_model=Dict[str, Any])
async def get_brand_kit_config_by_content_type(
    content_type: str, user: UserContext = Depends(get_current_user)
):
    """
    Get content-type-specific brand kit configuration.

    Args:
        content_type: Content type (blog_post, press_release, case_study)

    Returns:
        Dict[str, Any]: Merged configuration for the content type
    """
    try:
        manager = await get_brand_kit_manager()
        config = await manager.get_config_by_content_type(content_type)

        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"No active brand kit configuration found for content type {content_type}",
            )

        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting brand kit config for content type {content_type}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get brand kit configuration for content type: {str(e)}",
        )


@router.post("/config", response_model=BrandKitConfig)
async def create_or_update_brand_kit_config(
    request: dict = Body(..., description="Request body with config and set_active"),
    user: UserContext = Depends(require_roles(["admin"])),
):
    """
    Create or update brand kit configuration.

    This endpoint allows manual creation of brand kit configuration from scratch.

    Args:
        request: Dict with 'config' (BrandKitConfig) and 'set_active' (bool)

    Returns:
        BrandKitConfig: The saved configuration
    """
    try:
        config_dict = request.get("config", {})
        set_active = request.get("set_active", True)
        config = BrandKitConfig(**config_dict)

        manager = await get_brand_kit_manager()
        success = await manager.save_config(config, set_active=set_active)

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to save brand kit configuration"
            )

        # Return the saved config
        saved_config = await manager.get_config_by_version(config.version)
        return saved_config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving brand kit config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save brand kit configuration: {str(e)}"
        )


@router.post("/generate")
async def generate_brand_kit_config(
    request: dict = Body(..., description="Request body with use_internal_docs"),
    user: UserContext = Depends(require_roles(["admin"])),
):
    """
    Generate new brand kit configuration using AI/LLM (runs as background job).

    This endpoint submits a background job to generate a comprehensive brand kit configuration
    with all fields populated based on best practices and common patterns.
    It enriches with internal_docs_config if available.

    Args:
        request: Dict with 'use_internal_docs' (bool) - whether to use internal docs config

    Returns:
        Job submission response with job_id for tracking
    """
    try:
        use_internal_docs = request.get("use_internal_docs", True)

        logger.info("Submitting brand kit generation job...")

        # Get job manager
        job_manager = get_job_manager()

        # Create job
        job = await job_manager.create_job(
            job_type="brand_kit_generate",
            content_id="brand_kit",
            metadata={"operation": "generate", "use_internal_docs": use_internal_docs},
            user_id=user.user_id,
            user_context=user,
        )

        # Submit job to ARQ for background processing with extended timeout
        # Brand kit generation can take longer due to multiple LLM calls
        arq_job_id = await job_manager.submit_to_arq(
            job.id,
            "refresh_brand_kit_job",  # Same worker function as refresh
            use_internal_docs,
            job.id,  # job_id
            _timeout=1800,  # 30 minutes timeout for brand kit jobs
        )

        logger.info(
            f"Brand kit generation job {job.id} submitted to ARQ (arq_id: {arq_job_id})"
        )

        # Return current config immediately (job will update it in background)
        manager = await get_brand_kit_manager()
        config = await manager.get_active_config()

        # Return job info along with current config
        return {
            "job_id": job.id,
            "status_url": f"/api/v1/jobs/{job.id}/status",
            "config": config.model_dump(mode="json") if config else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting brand kit generation job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit brand kit generation job: {str(e)}",
        )


@router.get("/versions", response_model=List[str])
async def list_brand_kit_versions(
    user: UserContext = Depends(get_current_user),
):
    """
    List all available brand kit configuration versions.

    Returns:
        List[str]: List of version identifiers
    """
    try:
        manager = await get_brand_kit_manager()
        versions = await manager.list_versions()
        return versions
    except Exception as e:
        logger.error(f"Error listing brand kit versions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list brand kit versions: {str(e)}"
        )


@router.post("/activate/{version}")
async def activate_brand_kit_version(
    version: str, user: UserContext = Depends(require_roles(["admin"]))
):
    """
    Activate a specific brand kit configuration version.

    Args:
        version: Version identifier to activate

    Returns:
        dict: Success message
    """
    try:
        manager = await get_brand_kit_manager()
        success = await manager.activate_version(version)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to activate version {version}. Version may not exist.",
            )

        return {"message": f"Activated brand kit configuration version {version}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating brand kit version {version}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to activate brand kit version: {str(e)}"
        )
