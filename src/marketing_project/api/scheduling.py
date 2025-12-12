"""
API endpoints for post scheduling.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.models.user_context import UserContext
from marketing_project.services.scheduling_service import get_scheduling_service

logger = logging.getLogger("marketing_project.api.scheduling")

router = APIRouter(prefix="/v1/schedule", tags=["scheduling"])


class SchedulePostRequest(BaseModel):
    """Request to schedule a post."""

    job_id: str = Field(..., description="Job ID of the post to schedule")
    content: str = Field(..., description="Post content to schedule")
    platform: str = Field(..., description="Platform: linkedin, hackernews, or email")
    scheduled_time: str = Field(
        ..., description="ISO format datetime when to publish the post"
    )
    subject_line: Optional[str] = Field(
        None, description="Email subject line if applicable"
    )
    hashtags: Optional[List[str]] = Field(None, description="Hashtags to include")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SchedulePostResponse(BaseModel):
    """Response after scheduling a post."""

    schedule_id: str = Field(..., description="Unique schedule ID")
    status: str = Field(..., description="Scheduling status")
    scheduled_time: str = Field(..., description="Scheduled time in ISO format")
    message: str = Field(..., description="Status message")


class ScheduledPostResponse(BaseModel):
    """Response with scheduled post details."""

    schedule_id: str
    job_id: str
    content: str
    platform: str
    scheduled_time: str
    status: str
    created_at: str
    metadata: Dict[str, Any]


@router.post("/post", response_model=SchedulePostResponse)
async def schedule_post(
    request: SchedulePostRequest, user: UserContext = Depends(get_current_user)
):
    """
    Schedule a social media post for publishing.

    Args:
        request: Schedule request with post details and time

    Returns:
        Scheduling result
    """
    try:
        scheduled_time = datetime.fromisoformat(
            request.scheduled_time.replace("Z", "+00:00")
        )

        scheduling_service = get_scheduling_service()
        result = await scheduling_service.schedule_post(
            job_id=request.job_id,
            content=request.content,
            platform=request.platform,
            scheduled_time=scheduled_time,
            metadata={
                "subject_line": request.subject_line,
                "hashtags": request.hashtags,
                **(request.metadata or {}),
            },
        )

        return SchedulePostResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to schedule post: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to schedule post: {str(e)}"
        )


@router.get("/posts", response_model=List[ScheduledPostResponse])
async def list_scheduled_posts(
    status: Optional[str] = Query(None, description="Filter by status"),
    platform: Optional[str] = Query(None, description="Platform filter"),
    limit: int = Query(
        50, ge=1, le=100, description="Maximum number of posts to return"
    ),
    user: UserContext = Depends(get_current_user),
):
    """
    List all scheduled posts.

    Args:
        platform: Optional platform filter
        limit: Maximum number of posts to return

    Returns:
        List of scheduled posts
    """
    try:
        scheduling_service = get_scheduling_service()
        posts = await scheduling_service.list_scheduled_posts(
            platform=platform, limit=limit
        )
        return [ScheduledPostResponse(**post) for post in posts]
    except Exception as e:
        logger.error(f"Failed to list scheduled posts: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list scheduled posts: {str(e)}"
        )


@router.delete("/posts/{schedule_id}")
async def cancel_scheduled_post(
    schedule_id: str, user: UserContext = Depends(get_current_user)
):
    """
    Cancel a scheduled post.

    Args:
        schedule_id: Schedule ID to cancel

    Returns:
        Cancellation result
    """
    try:
        scheduling_service = get_scheduling_service()
        result = await scheduling_service.cancel_scheduled_post(schedule_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to cancel scheduled post: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel scheduled post: {str(e)}"
        )


@router.get("/posts/{schedule_id}", response_model=ScheduledPostResponse)
async def get_scheduled_post(
    schedule_id: str, user: UserContext = Depends(get_current_user)
):
    """
    Get details of a scheduled post.

    Args:
        schedule_id: Schedule ID

    Returns:
        Scheduled post details
    """
    try:
        scheduling_service = get_scheduling_service()
        post = await scheduling_service.get_scheduled_post(schedule_id)
        if not post:
            raise HTTPException(status_code=404, detail="Scheduled post not found")
        return ScheduledPostResponse(**post)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scheduled post: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get scheduled post: {str(e)}"
        )
