"""
Analytics Models.

Pydantic models for analytics and statistics responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DashboardStats(BaseModel):
    """Dashboard statistics."""

    total_content: int = Field(..., description="Total number of content items")
    total_jobs: int = Field(..., description="Total number of jobs")
    jobs_completed: int = Field(..., description="Number of completed jobs")
    jobs_processing: int = Field(..., description="Number of jobs currently processing")
    jobs_failed: int = Field(..., description="Number of failed jobs")
    jobs_queued: int = Field(..., description="Number of queued jobs")
    success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Job success rate (0-1)"
    )

    # Change indicators (calculated from recent trends)
    content_change_percent: Optional[float] = Field(
        None, description="Percentage change in content"
    )
    jobs_change_percent: Optional[float] = Field(
        None, description="Percentage change in jobs"
    )
    success_rate_change_percent: Optional[float] = Field(
        None, description="Percentage change in success rate"
    )


class PipelineStats(BaseModel):
    """Pipeline-specific statistics."""

    total_runs: int = Field(..., description="Total number of pipeline runs")
    completed: int = Field(..., description="Number of completed runs")
    in_progress: int = Field(..., description="Number of runs in progress")
    failed: int = Field(..., description="Number of failed runs")
    queued: int = Field(..., description="Number of queued runs")

    # Performance metrics
    avg_duration_seconds: Optional[float] = Field(
        None, description="Average pipeline duration in seconds"
    )
    success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Pipeline success rate (0-1)"
    )

    # Change indicators
    completed_change_percent: Optional[float] = Field(
        None, description="Percentage change in completed"
    )
    total_change_percent: Optional[float] = Field(
        None, description="Percentage change in total runs"
    )


class ContentSourceStats(BaseModel):
    """Statistics for a single content source."""

    source_name: str = Field(..., description="Name of the content source")
    total_items: int = Field(..., description="Total items from this source")
    active: bool = Field(..., description="Whether the source is active")


class ContentStats(BaseModel):
    """Content statistics."""

    total_content: int = Field(
        ..., description="Total number of content items across all sources"
    )
    by_source: List[ContentSourceStats] = Field(
        default_factory=list, description="Breakdown by source"
    )
    active_sources: int = Field(..., description="Number of active content sources")


class RecentActivityItem(BaseModel):
    """A single recent activity/job item."""

    job_id: str = Field(..., description="Job ID")
    job_type: str = Field(..., description="Job type (blog, release_notes, etc.)")
    title: Optional[str] = Field(None, description="Job title or content title")
    status: str = Field(..., description="Job status")
    progress: int = Field(..., ge=0, le=100, description="Job progress percentage")
    created_at: datetime = Field(..., description="When the job was created")
    completed_at: Optional[datetime] = Field(None, description="When the job completed")
    error: Optional[str] = Field(None, description="Error message if failed")


class RecentActivity(BaseModel):
    """Recent activity/jobs list."""

    activities: List[RecentActivityItem] = Field(
        ..., description="List of recent activities"
    )
    total: int = Field(..., description="Total number of activities")


class TrendDataPoint(BaseModel):
    """A single data point in a trend chart."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    total_jobs: int = Field(default=0, description="Total jobs on this date")
    completed: int = Field(default=0, description="Completed jobs on this date")
    failed: int = Field(default=0, description="Failed jobs on this date")
    success_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Success rate on this date"
    )


class TrendData(BaseModel):
    """Trend data for charts."""

    data_points: List[TrendDataPoint] = Field(..., description="List of data points")
    start_date: str = Field(..., description="Start date of trend data")
    end_date: str = Field(..., description="End date of trend data")
    days: int = Field(..., description="Number of days in the trend")


class AnalyticsResponse(BaseModel):
    """Generic analytics response wrapper."""

    success: bool = True
    message: str = "Analytics data retrieved successfully"
    data: Dict = Field(..., description="Analytics data payload")


class CostMetrics(BaseModel):
    """Cost tracking metrics."""

    total_cost_usd: float = Field(..., description="Total cost in USD")
    avg_cost_per_job: float = Field(..., description="Average cost per job")
    cost_by_model: Dict[str, float] = Field(
        default_factory=dict, description="Cost breakdown by model"
    )
    cost_by_step: Dict[str, float] = Field(
        default_factory=dict, description="Cost breakdown by pipeline step"
    )
    token_usage: Dict[str, int] = Field(
        default_factory=dict, description="Token usage statistics"
    )


class QualityTrends(BaseModel):
    """Quality trend analysis."""

    avg_quality_score: float = Field(
        ..., ge=0.0, le=100.0, description="Average quality score"
    )
    quality_by_platform: Dict[str, float] = Field(
        default_factory=dict, description="Average quality by platform"
    )
    quality_trend_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Time-series quality data"
    )
    improvement_rate: Optional[float] = Field(
        None, description="Quality improvement rate over time"
    )


class UnifiedMonitoringMetrics(BaseModel):
    """Unified monitoring dashboard metrics."""

    dashboard_stats: DashboardStats = Field(..., description="Dashboard statistics")
    pipeline_stats: PipelineStats = Field(..., description="Pipeline statistics")
    cost_metrics: Optional[CostMetrics] = Field(
        None, description="Cost tracking metrics"
    )
    quality_trends: Optional[QualityTrends] = Field(
        None, description="Quality trend analysis"
    )
    social_media_performance: Optional[Dict[str, Any]] = Field(
        None, description="Social media performance metrics"
    )
