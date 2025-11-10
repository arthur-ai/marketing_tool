"""
Analytics API Endpoints.

Provides endpoints for retrieving system analytics and statistics.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from marketing_project.models.analytics_models import (
    AnalyticsResponse,
    ContentStats,
    DashboardStats,
    PipelineStats,
    RecentActivity,
    TrendData,
)
from marketing_project.services.analytics_service import get_analytics_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_analytics():
    """
    Get dashboard statistics.

    Returns overall system statistics including:
    - Total content count
    - Job counts by status
    - Success rates
    - Change indicators

    Results are cached for 60 seconds for performance.
    """
    try:
        analytics_service = get_analytics_service()
        stats = await analytics_service.get_dashboard_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get dashboard analytics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve dashboard analytics: {str(e)}"
        )


@router.get("/pipeline", response_model=PipelineStats)
async def get_pipeline_analytics():
    """
    Get pipeline-specific statistics.

    Returns metrics about pipeline runs including:
    - Total runs
    - Runs by status (completed, in-progress, failed, queued)
    - Average duration
    - Success rate

    Results are cached for 60 seconds for performance.
    """
    try:
        analytics_service = get_analytics_service()
        stats = await analytics_service.get_pipeline_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get pipeline analytics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve pipeline analytics: {str(e)}"
        )


@router.get("/content", response_model=ContentStats)
async def get_content_analytics():
    """
    Get content statistics.

    Returns information about content sources including:
    - Total content items
    - Breakdown by source
    - Number of active sources

    Results are cached for 60 seconds for performance.
    """
    try:
        analytics_service = get_analytics_service()
        stats = await analytics_service.get_content_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get content analytics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve content analytics: {str(e)}"
        )


@router.get("/recent-activity", response_model=RecentActivity)
async def get_recent_activity(
    days: int = Query(7, ge=1, le=30, description="Number of days to look back")
):
    """
    Get recent activity/jobs.

    Returns a list of recent jobs within the specified time window.
    Default is last 7 days, maximum is 30 days.

    Results are cached for 60 seconds for performance.
    """
    try:
        analytics_service = get_analytics_service()
        activity = await analytics_service.get_recent_activity(days=days)
        return activity
    except Exception as e:
        logger.error(f"Failed to get recent activity: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve recent activity: {str(e)}"
        )


@router.get("/trends", response_model=TrendData)
async def get_trends(
    days: int = Query(7, ge=1, le=90, description="Number of days to include in trend")
):
    """
    Get trend data for charts.

    Returns daily aggregated statistics for the specified period.
    Useful for creating time-series charts and visualizations.

    Default is 7 days, maximum is 90 days.
    Results are cached for 60 seconds for performance.
    """
    try:
        analytics_service = get_analytics_service()
        trends = await analytics_service.get_trends(days=days)
        return trends
    except Exception as e:
        logger.error(f"Failed to get trend data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve trend data: {str(e)}"
        )
