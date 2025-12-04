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
    CostMetrics,
    DashboardStats,
    PipelineStats,
    QualityTrends,
    RecentActivity,
    TrendData,
    UnifiedMonitoringMetrics,
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


@router.get("/social-media/posts")
async def get_social_media_performance(
    days: int = Query(30, ge=1, le=90, description="Number of days to look back"),
    platform: Optional[str] = Query(
        None, description="Platform filter: linkedin, hackernews, or email"
    ),
):
    """
    Get social media post performance metrics.

    Returns metrics including:
    - Total posts generated
    - Success rate
    - Average quality scores
    - Platform breakdown

    Results are cached for 60 seconds for performance.
    """
    try:
        analytics_service = get_analytics_service()
        performance = await analytics_service.get_social_media_performance(
            days=days, platform=platform
        )
        return performance
    except Exception as e:
        logger.error(f"Failed to get social media performance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve social media performance: {str(e)}",
        )


@router.get("/social-media/trends")
async def get_social_media_trends(
    days: int = Query(7, ge=1, le=30, description="Number of days to look back"),
    platform: Optional[str] = Query(
        None, description="Platform filter: linkedin, hackernews, or email"
    ),
):
    """
    Get social media performance trends over time.

    Returns time-series data showing:
    - Daily post counts
    - Success rates over time
    - Quality score trends

    Results are cached for 60 seconds for performance.
    """
    try:
        analytics_service = get_analytics_service()
        trends = await analytics_service.get_social_media_trends(
            days=days, platform=platform
        )
        return trends
    except Exception as e:
        logger.error(f"Failed to get social media trends: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve social media trends: {str(e)}",
        )


@router.get("/monitoring/unified", response_model=UnifiedMonitoringMetrics)
async def get_unified_monitoring_metrics(
    days: int = Query(30, ge=1, le=90, description="Number of days to look back"),
):
    """
    Get unified monitoring dashboard metrics.

    Returns comprehensive metrics including:
    - Dashboard statistics
    - Pipeline statistics
    - Cost tracking metrics
    - Quality trend analysis
    - Social media performance

    Results are cached for 60 seconds for performance.
    """
    try:
        analytics_service = get_analytics_service()
        metrics = await analytics_service.get_unified_monitoring_metrics(days=days)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get unified monitoring metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve unified monitoring metrics: {str(e)}",
        )


@router.get("/monitoring/cost", response_model=CostMetrics)
async def get_cost_metrics(
    days: int = Query(30, ge=1, le=90, description="Number of days to look back"),
):
    """
    Get cost tracking metrics.

    Returns:
    - Total cost in USD
    - Average cost per job
    - Cost breakdown by model
    - Cost breakdown by pipeline step
    - Token usage statistics

    Results are cached for 60 seconds for performance.
    """
    try:
        analytics_service = get_analytics_service()
        metrics = await analytics_service.get_cost_metrics(days=days)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get cost metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve cost metrics: {str(e)}"
        )


@router.get("/monitoring/quality", response_model=QualityTrends)
async def get_quality_trends(
    days: int = Query(30, ge=1, le=90, description="Number of days to look back"),
):
    """
    Get quality trend analysis.

    Returns:
    - Average quality score
    - Quality breakdown by platform
    - Quality trends over time
    - Quality improvement rate

    Results are cached for 60 seconds for performance.
    """
    try:
        analytics_service = get_analytics_service()
        trends = await analytics_service.get_quality_trends(days=days)
        return trends
    except Exception as e:
        logger.error(f"Failed to get quality trends: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve quality trends: {str(e)}"
        )
