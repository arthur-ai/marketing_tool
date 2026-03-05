"""
Competitor Research API Endpoints.

Provides endpoints to submit and retrieve competitor content analysis jobs.
These endpoints allow users to analyze why competitors' blogs and social media
posts are performing well and get actionable strategic insights.
"""

import json
import logging
from typing import List, Optional

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, Depends, HTTPException, Query

from ..middleware.keycloak_auth import get_current_user
from ..models.competitor_models import (
    CompetitorResearchListItem,
    CompetitorResearchRequest,
    CompetitorResearchResult,
)
from ..models.user_context import UserContext
from ..services.competitor_research_service import get_competitor_research_service

logger = logging.getLogger(__name__)

router = APIRouter()


async def _enqueue_competitor_research_job(
    job_id: str, request: CompetitorResearchRequest
) -> None:
    """Enqueue the competitor research job in ARQ."""
    import os

    redis_settings = RedisSettings(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        database=int(os.getenv("REDIS_DATABASE", "0")),
        password=os.getenv("REDIS_PASSWORD"),
    )
    pool = await create_pool(redis_settings)
    try:
        await pool.enqueue_job(
            "process_competitor_research_job",
            job_id=job_id,
            request_json=request.model_dump_json(),
        )
    finally:
        await pool.aclose()


@router.post("", response_model=dict, status_code=202)
async def submit_competitor_research(
    request: CompetitorResearchRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    Submit a competitor research job.

    Analyzes competitor blogs and/or social media posts to identify:
    - Why their content performs well (structure, SEO, engagement tactics)
    - Content gaps and opportunities you can exploit
    - Actionable strategic recommendations

    You can provide content in two ways:
    1. **competitor_urls**: List of URLs to analyze (LLM reasons about the URL context)
    2. **competitor_content**: Raw content dicts with `title`, `content`, `url`, `platform`, `content_type`

    The job is processed asynchronously. Use GET /competitor-research/{job_id} to poll for results.
    """
    if not request.competitor_urls and not request.competitor_content:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'competitor_urls' or 'competitor_content' must be provided.",
        )

    service = get_competitor_research_service()

    try:
        job_id = await service.create_research_job(
            request=request,
            user_id=user.user_id if user else None,
        )

        # Enqueue background job
        try:
            await _enqueue_competitor_research_job(job_id, request)
        except Exception as e:
            logger.error(
                f"[COMPETITOR] Failed to enqueue job {job_id}: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=503,
                detail="Failed to enqueue competitor research job. Please try again.",
            )

        logger.info(
            f"[COMPETITOR] Submitted research job {job_id} for user {user.user_id if user else 'anonymous'}"
        )

        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Competitor research job submitted. Poll GET /competitor-research/{job_id} for results.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[COMPETITOR] Failed to create research job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to create research job: {str(e)}"
        )


@router.get("", response_model=List[CompetitorResearchListItem])
async def list_competitor_research_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: UserContext = Depends(get_current_user),
):
    """
    List competitor research jobs for the current user.

    Returns a lightweight list of jobs ordered by creation date (newest first).
    """
    service = get_competitor_research_service()

    try:
        # Admin users see all jobs; regular users see only their own
        user_filter = None
        if user and not getattr(user, "is_admin", False):
            user_filter = user.user_id

        items = await service.list_research_jobs(
            user_id=user_filter,
            limit=limit,
            offset=offset,
        )
        return items

    except Exception as e:
        logger.error(f"[COMPETITOR] Failed to list research jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to list research jobs: {str(e)}"
        )


@router.get("/{job_id}", response_model=dict)
async def get_competitor_research_result(
    job_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    Get the result of a competitor research job.

    Returns the full analysis including:
    - Individual content analyses with strength/weakness factors
    - Cross-competitor strategic summary
    - Actionable insights and quick wins

    While status is 'pending' or 'processing', result_data will be null.
    Poll until status is 'completed' or 'failed'.
    """
    service = get_competitor_research_service()

    try:
        record = await service.get_research_result(job_id)

        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Competitor research job '{job_id}' not found.",
            )

        return record

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[COMPETITOR] Failed to get research result for {job_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve research result: {str(e)}"
        )


@router.get("/{job_id}/crawled-content", response_model=List[dict])
async def get_crawled_url_content(
    job_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    Get the raw crawled page content for each URL in a competitor research job.

    Returns a list of records, one per URL, each containing:
    - url: The original URL
    - title: Page title
    - full_content: Full extracted page text
    - meta_description: Meta description tag
    - word_count: Number of words in the content
    - fetched_at: When the page was fetched
    """
    service = get_competitor_research_service()

    try:
        # Verify the job exists
        record = await service.get_research_result(job_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Competitor research job '{job_id}' not found.",
            )

        items = await service.get_crawled_url_content(job_id)
        return items

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[COMPETITOR] Failed to get crawled content for {job_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve crawled content: {str(e)}"
        )
