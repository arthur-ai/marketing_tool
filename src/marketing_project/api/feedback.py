"""
API endpoints for feedback collection.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.models.user_context import UserContext
from marketing_project.services.feedback_loop import get_feedback_service

logger = logging.getLogger("marketing_project.api.feedback")

router = APIRouter(prefix="/v1/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Request to submit feedback."""

    job_id: str = Field(..., description="Job ID of the post")
    feedback_type: str = Field(
        ..., description="Type: approval, rejection, rating, etc."
    )
    rating: Optional[int] = Field(None, ge=1, le=5, description="Optional rating (1-5)")
    comments: Optional[str] = Field(None, description="Optional feedback comments")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""

    feedback_id: str = Field(..., description="Feedback record ID")
    message: str = Field(..., description="Confirmation message")


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest, user: UserContext = Depends(get_current_user)
):
    """
    Submit feedback on a generated post.

    Args:
        request: Feedback request

    Returns:
        Feedback response
    """
    try:
        feedback_service = get_feedback_service()
        feedback = await feedback_service.record_feedback(
            job_id=request.job_id,
            feedback_type=request.feedback_type,
            rating=request.rating,
            comments=request.comments,
            metadata=request.metadata,
        )

        return FeedbackResponse(
            feedback_id=feedback["feedback_id"],
            message="Feedback recorded successfully",
        )
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to record feedback: {str(e)}"
        )


@router.get("/stats")
async def get_feedback_stats(
    days: int = Query(30, ge=1, le=90, description="Number of days to look back"),
    platform: Optional[str] = Query(
        None, description="Platform filter: linkedin, hackernews, or email"
    ),
    user: UserContext = Depends(get_current_user),
):
    """
    Get feedback statistics.

    Returns:
        Feedback statistics including approval rates, ratings, etc.
    """
    try:
        feedback_service = get_feedback_service()
        stats = await feedback_service.get_feedback_stats(days=days, platform=platform)
        return stats
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get feedback stats: {str(e)}"
        )
