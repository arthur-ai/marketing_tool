"""
API endpoints for batch processing multiple content items.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.models.processor_models import BlogProcessorRequest
from marketing_project.models.user_context import UserContext
from marketing_project.processors import (  # Exported for test compatibility
    process_blog_post,
)
from marketing_project.services.job_manager import get_job_manager

logger = logging.getLogger("marketing_project.api.batch")

router = APIRouter(prefix="/v1/batch", tags=["batch"])


class BatchBlogRequest(BaseModel):
    """Request to process multiple blog posts in batch."""

    content_items: List[BlogProcessorRequest] = Field(
        ...,
        description="List of blog post processing requests",
        min_items=1,
        max_items=20,
    )
    campaign_id: Optional[str] = Field(
        None, description="Optional campaign ID for grouping batch results"
    )


class BatchJobResponse(BaseModel):
    """Response for batch job submission."""

    success: bool = True
    message: str
    campaign_id: Optional[str] = None
    job_ids: List[str] = Field(..., description="List of job IDs created")
    total_items: int = Field(..., description="Total number of items in batch")


@router.post("/blog", response_model=BatchJobResponse)
async def process_batch_blog(
    request: BatchBlogRequest, user: UserContext = Depends(get_current_user)
):
    """
    Process multiple blog posts in a single batch.

    Creates individual jobs for each content item and groups them by campaign_id if provided.

    Args:
        request: Batch request with list of blog post requests

    Returns:
        Batch job response with list of job IDs
    """
    try:
        logger.info(
            f"Batch blog processing request for {len(request.content_items)} items"
        )

        job_manager = get_job_manager()
        job_ids = []

        # Process each content item
        for idx, item_request in enumerate(request.content_items):
            try:
                # Create job for this item
                job_metadata = {
                    "content_type": "blog_post",
                    "output_content_type": item_request.output_content_type
                    or "blog_post",
                    "batch_index": idx,
                    "batch_total": len(request.content_items),
                }

                # Add campaign ID if provided
                if request.campaign_id:
                    job_metadata["campaign_id"] = request.campaign_id

                # Add social media parameters if applicable
                if item_request.output_content_type == "social_media_post":
                    platforms = item_request.social_media_platforms
                    if not platforms and item_request.social_media_platform:
                        platforms = [item_request.social_media_platform]

                    if platforms:
                        if len(platforms) == 1:
                            job_metadata["social_media_platform"] = platforms[0]
                        else:
                            job_metadata["social_media_platforms"] = platforms

                    if item_request.email_type:
                        job_metadata["email_type"] = item_request.email_type
                    if item_request.variations_count:
                        job_metadata["variations_count"] = item_request.variations_count
                    if item_request.pipeline_config:
                        job_metadata["pipeline_config"] = (
                            item_request.pipeline_config.model_dump()
                            if hasattr(item_request.pipeline_config, "model_dump")
                            else item_request.pipeline_config
                        )

                # Create job
                job = await job_manager.create_job(
                    job_type="blog_post",
                    content_id=item_request.content.id,
                    metadata=job_metadata,
                    user_id=user.user_id,
                    user_context=user,
                )

                # Convert to JSON and submit to ARQ
                content_json = item_request.content.model_dump_json()

                # Determine ARQ function
                if item_request.output_content_type == "social_media_post":
                    platforms = item_request.social_media_platforms or (
                        [item_request.social_media_platform]
                        if item_request.social_media_platform
                        else []
                    )
                    if len(platforms) > 1:
                        arq_function_name = "process_multi_platform_social_media_job"
                    else:
                        arq_function_name = "process_social_media_job"
                else:
                    arq_function_name = "process_blog_job"

                # Submit to ARQ
                arq_job_id = await job_manager.submit_to_arq(
                    job.id,
                    arq_function_name,
                    content_json,
                    job.id,
                    metadata=job_metadata,
                )

                job_ids.append(job.id)
                logger.info(
                    f"Batch item {idx + 1}/{len(request.content_items)}: Created job {job.id}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to process batch item {idx + 1}: {e}",
                    exc_info=True,
                )
                # Continue with other items even if one fails
                continue

        if not job_ids:
            raise HTTPException(
                status_code=500, detail="Failed to create any jobs in batch"
            )

        return BatchJobResponse(
            message=f"Batch processing started for {len(job_ids)} items",
            campaign_id=request.campaign_id,
            job_ids=job_ids,
            total_items=len(request.content_items),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process batch blog posts: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process batch: {str(e)}"
        )


@router.get("/campaign/{campaign_id}/jobs")
async def get_campaign_jobs(
    campaign_id: str, user: UserContext = Depends(get_current_user)
):
    """
    Get all jobs for a specific campaign.

    Args:
        campaign_id: Campaign ID

    Returns:
        List of job IDs and their statuses
    """
    try:
        job_manager = get_job_manager()
        # Get all jobs and filter by campaign_id
        # Note: This is a simplified implementation - in production, you'd want
        # a more efficient query mechanism
        all_jobs = await job_manager.list_jobs(limit=1000, user_id=user.user_id)
        campaign_jobs = [
            job for job in all_jobs if job.metadata.get("campaign_id") == campaign_id
        ]

        return {
            "campaign_id": campaign_id,
            "total_jobs": len(campaign_jobs),
            "jobs": [
                {
                    "job_id": job.id,
                    "content_id": job.content_id,
                    "status": job.status,
                    "progress": job.progress,
                }
                for job in campaign_jobs
            ],
        }
    except Exception as e:
        logger.error(f"Failed to get campaign jobs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get campaign jobs: {str(e)}"
        )
