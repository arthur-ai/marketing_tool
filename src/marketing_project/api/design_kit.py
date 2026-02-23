"""
Design Kit API Endpoints.

Endpoints for managing design kit configuration.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from ..middleware.keycloak_auth import get_current_user
from ..middleware.rbac import require_roles
from ..models.design_kit_config import DesignKitConfig
from ..models.user_context import UserContext
from ..services.design_kit_manager import get_design_kit_manager
from ..services.job_manager import get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/config")
async def get_design_kit_config(
    refresh: bool = Query(
        False,
        description="If true, submit a background job to regenerate config using AI",
    ),
    user: UserContext = Depends(get_current_user),
):
    """
    Get the currently active design kit configuration.

    Args:
        refresh: If true, submit a background job to regenerate the config using AI.
                 Returns immediately with job_id. Use GET /api/v1/jobs/{job_id}/status to check progress.

    Returns:
        DesignKitConfig: The active configuration (or current config if refresh is requested)
        If refresh=true, returns dict with config and job_id
    """
    try:
        manager = await get_design_kit_manager()

        # If refresh is requested, submit background job
        if refresh:
            logger.info("Refresh requested: submitting design kit refresh job...")

            # Get job manager
            job_manager = get_job_manager()

            # Create job
            job = await job_manager.create_job(
                job_type="design_kit_refresh",
                content_id="design_kit",
                metadata={"operation": "refresh", "use_internal_docs": True},
                user_id=user.user_id,
                user_context=user,
            )

            # Submit job to ARQ for background processing with extended timeout
            # Design kit generation can take longer due to multiple LLM calls
            arq_job_id = await job_manager.submit_to_arq(
                job.id,
                "refresh_design_kit_job",  # ARQ function name
                True,  # use_internal_docs
                job.id,  # job_id
                _timeout=1800,  # 30 minutes timeout for design kit jobs
            )

            logger.info(
                f"Design kit refresh job {job.id} submitted to ARQ (arq_id: {arq_job_id})"
            )

            # Return current config immediately (job will update it in background)
            config = await manager.get_active_config()
            if not config:
                raise HTTPException(
                    status_code=404, detail="No active design kit configuration found"
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
                status_code=404, detail="No active design kit configuration found"
            )

        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting design kit config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get design kit configuration: {str(e)}"
        )


@router.get("/config/{version}", response_model=DesignKitConfig)
async def get_design_kit_config_by_version(
    version: str, user: UserContext = Depends(get_current_user)
):
    """
    Get design kit configuration by version.

    Args:
        version: Version identifier

    Returns:
        DesignKitConfig: The configuration for the specified version
    """
    try:
        manager = await get_design_kit_manager()
        config = await manager.get_config_by_version(version)

        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"Design kit configuration version {version} not found",
            )

        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting design kit config version {version}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get design kit configuration version: {str(e)}",
        )


@router.get("/config/{content_type}/type", response_model=Dict[str, Any])
async def get_design_kit_config_by_content_type(
    content_type: str, user: UserContext = Depends(get_current_user)
):
    """
    Get content-type-specific design kit configuration.

    Args:
        content_type: Content type (blog_post, press_release, case_study)

    Returns:
        Dict[str, Any]: Merged configuration for the content type
    """
    try:
        manager = await get_design_kit_manager()
        config = await manager.get_config_by_content_type(content_type)

        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"No active design kit configuration found for content type {content_type}",
            )

        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting design kit config for content type {content_type}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get design kit configuration for content type: {str(e)}",
        )


@router.post("/config", response_model=DesignKitConfig)
async def create_or_update_design_kit_config(
    request: dict = Body(..., description="Request body with config and set_active"),
    user: UserContext = Depends(require_roles(["admin"])),
):
    """
    Create or update design kit configuration.

    This endpoint allows manual creation of design kit configuration from scratch.

    Args:
        request: Dict with 'config' (DesignKitConfig) and 'set_active' (bool)

    Returns:
        DesignKitConfig: The saved configuration
    """
    try:
        config_dict = request.get("config", {})
        set_active = request.get("set_active", True)
        config = DesignKitConfig(**config_dict)

        manager = await get_design_kit_manager()
        success = await manager.save_config(config, set_active=set_active)

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to save design kit configuration"
            )

        # Return the saved config
        saved_config = await manager.get_config_by_version(config.version)
        return saved_config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving design kit config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save design kit configuration: {str(e)}"
        )


@router.post("/generate")
async def generate_design_kit_config(
    request: dict = Body(..., description="Request body with use_internal_docs"),
    user: UserContext = Depends(require_roles(["admin"])),
):
    """
    Generate new design kit configuration using AI/LLM (runs as background job).

    This endpoint submits a background job to generate a comprehensive design kit configuration
    with all fields populated based on best practices and common patterns.
    It enriches with internal_docs_config if available.

    Args:
        request: Dict with 'use_internal_docs' (bool) - whether to use internal docs config

    Returns:
        Job submission response with job_id for tracking
    """
    try:
        use_internal_docs = request.get("use_internal_docs", True)

        logger.info("Submitting design kit generation job...")

        # Get job manager
        job_manager = get_job_manager()

        # Create job
        job = await job_manager.create_job(
            job_type="design_kit_generate",
            content_id="design_kit",
            metadata={"operation": "generate", "use_internal_docs": use_internal_docs},
            user_id=user.user_id,
            user_context=user,
        )

        # Submit job to ARQ for background processing with extended timeout
        # Design kit generation can take longer due to multiple LLM calls
        arq_job_id = await job_manager.submit_to_arq(
            job.id,
            "refresh_design_kit_job",  # Same worker function as refresh
            use_internal_docs,
            job.id,  # job_id
            _timeout=1800,  # 30 minutes timeout for design kit jobs
        )

        logger.info(
            f"Design kit generation job {job.id} submitted to ARQ (arq_id: {arq_job_id})"
        )

        # Return current config immediately (job will update it in background)
        manager = await get_design_kit_manager()
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
        logger.error(f"Error submitting design kit generation job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit design kit generation job: {str(e)}",
        )


@router.get("/versions", response_model=List[str])
async def list_design_kit_versions(
    user: UserContext = Depends(get_current_user),
):
    """
    List all available design kit configuration versions.

    Returns:
        List[str]: List of version identifiers
    """
    try:
        manager = await get_design_kit_manager()
        versions = await manager.list_versions()
        return versions
    except Exception as e:
        logger.error(f"Error listing design kit versions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list design kit versions: {str(e)}"
        )


@router.post("/activate/{version}")
async def activate_design_kit_version(
    version: str, user: UserContext = Depends(require_roles(["admin"]))
):
    """
    Activate a specific design kit configuration version.

    Args:
        version: Version identifier to activate

    Returns:
        dict: Success message
    """
    try:
        manager = await get_design_kit_manager()
        success = await manager.activate_version(version)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to activate version {version}. Version may not exist.",
            )

        return {"message": f"Activated design kit configuration version {version}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating design kit version {version}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to activate design kit version: {str(e)}"
        )
