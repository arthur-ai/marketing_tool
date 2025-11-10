"""
Analytics Service.

Provides analytics and statistics computation with Redis caching.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import redis.asyncio as redis

from marketing_project.api.content import get_content_manager
from marketing_project.models.analytics_models import (
    ContentSourceStats,
    ContentStats,
    DashboardStats,
    PipelineStats,
    RecentActivity,
    RecentActivityItem,
    TrendData,
    TrendDataPoint,
)
from marketing_project.services.job_manager import JobStatus, get_job_manager
from marketing_project.services.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)

# Cache TTL in seconds
CACHE_TTL = 60  # 1 minute


class AnalyticsService:
    """
    Service for computing and caching analytics statistics.

    Uses Redis for caching to improve performance and reduce computation overhead.
    Uses centralized RedisManager for connection pooling and resilience.
    """

    def __init__(self):
        self._redis_manager = get_redis_manager()

    async def get_redis(self) -> redis.Redis:
        """Get Redis client from RedisManager."""
        return await self._redis_manager.get_redis()

    async def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached data from Redis."""
        try:
            cache_key = f"analytics:{key}"

            async def get_operation(redis_client: redis.Redis):
                return await redis_client.get(cache_key)

            cached = await self._redis_manager.execute(get_operation)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(
                f"Failed to get cached analytics for key '{key}': {e}", exc_info=False
            )
        return None

    async def _set_cached(self, key: str, data: Dict):
        """Set cached data in Redis with TTL."""
        try:
            cache_key = f"analytics:{key}"
            cache_data = json.dumps(data, default=str)

            async def setex_operation(redis_client: redis.Redis):
                return await redis_client.setex(cache_key, CACHE_TTL, cache_data)

            await self._redis_manager.execute(setex_operation)
        except Exception as e:
            logger.warning(
                f"Failed to cache analytics for key '{key}': {e}", exc_info=False
            )

    async def get_dashboard_stats(self) -> DashboardStats:
        """
        Get dashboard statistics.

        Returns overall system statistics including job counts, success rate, etc.
        """
        # Try cache first
        cached = await self._get_cached("dashboard")
        if cached:
            return DashboardStats(**cached)

        # Compute stats
        job_manager = get_job_manager()
        content_manager = get_content_manager()

        # Get all jobs (no limit for accurate stats)
        all_jobs = await job_manager.list_jobs(limit=1000)

        # Count by status
        total_jobs = len(all_jobs)
        jobs_completed = sum(1 for j in all_jobs if j.status == JobStatus.COMPLETED)
        jobs_processing = sum(1 for j in all_jobs if j.status == JobStatus.PROCESSING)
        jobs_failed = sum(1 for j in all_jobs if j.status == JobStatus.FAILED)
        jobs_queued = sum(1 for j in all_jobs if j.status == JobStatus.QUEUED)

        # Calculate success rate
        finished_jobs = jobs_completed + jobs_failed
        success_rate = jobs_completed / finished_jobs if finished_jobs > 0 else 0.0

        # Get total content count
        sources = await content_manager.list_sources()
        total_content = sum(
            s.get("item_count", 0) for s in sources if s.get("active", False)
        )

        stats = DashboardStats(
            total_content=total_content,
            total_jobs=total_jobs,
            jobs_completed=jobs_completed,
            jobs_processing=jobs_processing,
            jobs_failed=jobs_failed,
            jobs_queued=jobs_queued,
            success_rate=success_rate,
            # Change percentages could be calculated from historical data
            # For now, we'll leave them as None
            content_change_percent=None,
            jobs_change_percent=None,
            success_rate_change_percent=None,
        )

        # Cache the result
        await self._set_cached("dashboard", stats.model_dump())

        return stats

    async def get_pipeline_stats(self) -> PipelineStats:
        """
        Get pipeline-specific statistics.

        Returns statistics about pipeline runs and performance.
        """
        # Try cache first
        cached = await self._get_cached("pipeline")
        if cached:
            return PipelineStats(**cached)

        # Compute stats
        job_manager = get_job_manager()
        all_jobs = await job_manager.list_jobs(limit=1000)

        # Filter for pipeline jobs (job types that are actual pipeline runs)
        pipeline_jobs = [
            j
            for j in all_jobs
            if j.type in ["blog", "release_notes", "transcript", "pipeline"]
        ]

        total_runs = len(pipeline_jobs)
        completed = sum(1 for j in pipeline_jobs if j.status == JobStatus.COMPLETED)
        in_progress = sum(1 for j in pipeline_jobs if j.status == JobStatus.PROCESSING)
        failed = sum(1 for j in pipeline_jobs if j.status == JobStatus.FAILED)
        queued = sum(1 for j in pipeline_jobs if j.status == JobStatus.QUEUED)

        # Calculate success rate
        finished = completed + failed
        success_rate = completed / finished if finished > 0 else 0.0

        # Calculate average duration for completed jobs
        completed_jobs_with_duration = [
            j
            for j in pipeline_jobs
            if j.status == JobStatus.COMPLETED and j.completed_at and j.started_at
        ]

        avg_duration = None
        if completed_jobs_with_duration:
            durations = [
                (j.completed_at - j.started_at).total_seconds()
                for j in completed_jobs_with_duration
            ]
            avg_duration = sum(durations) / len(durations)

        stats = PipelineStats(
            total_runs=total_runs,
            completed=completed,
            in_progress=in_progress,
            failed=failed,
            queued=queued,
            avg_duration_seconds=avg_duration,
            success_rate=success_rate,
            completed_change_percent=None,
            total_change_percent=None,
        )

        # Cache the result
        await self._set_cached("pipeline", stats.model_dump())

        return stats

    async def get_content_stats(self) -> ContentStats:
        """
        Get content statistics.

        Returns statistics about content sources and items.
        """
        # Try cache first
        cached = await self._get_cached("content")
        if cached:
            return ContentStats(**cached)

        # Compute stats
        content_manager = get_content_manager()
        sources = await content_manager.list_sources()

        by_source = []
        total_content = 0
        active_sources = 0

        for source in sources:
            is_active = source.get("active", False)
            item_count = source.get("item_count", 0)

            if is_active:
                active_sources += 1
                total_content += item_count

            by_source.append(
                ContentSourceStats(
                    source_name=source.get("name", "unknown"),
                    total_items=item_count,
                    active=is_active,
                )
            )

        stats = ContentStats(
            total_content=total_content,
            by_source=by_source,
            active_sources=active_sources,
        )

        # Cache the result
        await self._set_cached("content", stats.model_dump())

        return stats

    async def get_recent_activity(self, days: int = 7) -> RecentActivity:
        """
        Get recent activity/jobs.

        Args:
            days: Number of days to look back (default: 7)

        Returns:
            Recent activity items within the specified time window
        """
        # Try cache first
        cache_key = f"recent_activity_{days}"
        cached = await self._get_cached(cache_key)
        if cached:
            return RecentActivity(**cached)

        # Compute recent activity
        job_manager = get_job_manager()
        all_jobs = await job_manager.list_jobs(limit=1000)

        # Filter jobs from the last N days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_jobs = [j for j in all_jobs if j.created_at >= cutoff_date]

        # Sort by created_at descending (most recent first)
        recent_jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Convert to RecentActivityItem
        activities = []
        for job in recent_jobs[:50]:  # Limit to 50 most recent
            # Try to extract a title from metadata
            title = None
            if job.metadata:
                title = job.metadata.get("title") or job.metadata.get("content_title")

            activities.append(
                RecentActivityItem(
                    job_id=job.id,
                    job_type=job.type,
                    title=title,
                    status=job.status.value,
                    progress=job.progress,
                    created_at=job.created_at,
                    completed_at=job.completed_at,
                    error=job.error,
                )
            )

        result = RecentActivity(
            activities=activities,
            total=len(activities),
        )

        # Cache the result
        await self._set_cached(cache_key, result.model_dump())

        return result

    async def get_trends(self, days: int = 7) -> TrendData:
        """
        Get trend data for charts.

        Args:
            days: Number of days to include in trend (default: 7)

        Returns:
            Daily aggregated statistics for the specified period
        """
        # Try cache first
        cache_key = f"trends_{days}"
        cached = await self._get_cached(cache_key)
        if cached:
            return TrendData(**cached)

        # Compute trends
        job_manager = get_job_manager()
        all_jobs = await job_manager.list_jobs(limit=1000)

        # Create date range
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days - 1)

        # Initialize data points for each day
        data_points_dict: Dict[str, TrendDataPoint] = {}
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            data_points_dict[date_str] = TrendDataPoint(
                date=date_str,
                total_jobs=0,
                completed=0,
                failed=0,
                success_rate=0.0,
            )
            current_date += timedelta(days=1)

        # Aggregate jobs by date
        for job in all_jobs:
            job_date = job.created_at.date()
            date_str = job_date.strftime("%Y-%m-%d")

            if date_str in data_points_dict:
                data_points_dict[date_str].total_jobs += 1

                if job.status == JobStatus.COMPLETED:
                    data_points_dict[date_str].completed += 1
                elif job.status == JobStatus.FAILED:
                    data_points_dict[date_str].failed += 1

        # Calculate success rates
        for point in data_points_dict.values():
            finished = point.completed + point.failed
            if finished > 0:
                point.success_rate = point.completed / finished

        # Convert to sorted list
        data_points = [
            data_points_dict[date_str] for date_str in sorted(data_points_dict.keys())
        ]

        result = TrendData(
            data_points=data_points,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            days=days,
        )

        # Cache the result
        await self._set_cached(cache_key, result.model_dump())

        return result

    async def cleanup(self):
        """Cleanup Redis connections."""
        # RedisManager cleanup is handled globally
        # This method is here for consistency with other managers
        pass


# Singleton instance
_analytics_service: Optional[AnalyticsService] = None


def get_analytics_service() -> AnalyticsService:
    """Get the singleton analytics service instance."""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = AnalyticsService()
    return _analytics_service
