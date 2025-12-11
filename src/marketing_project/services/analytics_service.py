"""
Analytics Service.

Provides analytics and statistics computation with Redis caching.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import redis.asyncio as redis

from marketing_project.api.content import get_content_manager
from marketing_project.models.analytics_models import (
    ContentSourceStats,
    ContentStats,
    CostMetrics,
    DashboardStats,
    PipelineStats,
    QualityTrends,
    RecentActivity,
    RecentActivityItem,
    TrendData,
    TrendDataPoint,
    UnifiedMonitoringMetrics,
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
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
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
        end_date = datetime.now(timezone.utc).date()
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

    async def get_social_media_performance(
        self, days: int = 30, platform: Optional[str] = None
    ) -> Dict:
        """
        Get social media post performance metrics.

        Args:
            days: Number of days to look back
            platform: Optional platform filter (linkedin, hackernews, email)

        Returns:
            Dictionary with performance metrics
        """
        cache_key = f"social_media_performance:{days}:{platform or 'all'}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        job_manager = get_job_manager()
        all_jobs = await job_manager.list_jobs(limit=1000)

        # Filter for social media jobs
        social_media_jobs = [
            j
            for j in all_jobs
            if j.metadata.get("output_content_type") == "social_media_post"
        ]

        # Filter by platform if specified
        if platform:
            social_media_jobs = [
                j
                for j in social_media_jobs
                if j.metadata.get("social_media_platform") == platform
                or platform in j.metadata.get("social_media_platforms", [])
            ]

        # Filter by date range
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_jobs = [j for j in social_media_jobs if j.created_at >= cutoff_date]

        # Calculate metrics
        total_posts = len(recent_jobs)
        completed_posts = sum(1 for j in recent_jobs if j.status == JobStatus.COMPLETED)
        failed_posts = sum(1 for j in recent_jobs if j.status == JobStatus.FAILED)

        # Get quality scores from job results
        quality_scores = []
        for job in recent_jobs:
            if job.status == JobStatus.COMPLETED:
                try:
                    result = await job_manager.get_job_result(job.job_id)
                    if result and "metadata" in result:
                        platform_scores = result["metadata"].get(
                            "platform_quality_scores", {}
                        )
                        if platform_scores:
                            quality_scores.append(platform_scores)
                except Exception:
                    pass

        # Calculate average quality scores
        avg_scores = {}
        if quality_scores:
            for score_dict in quality_scores:
                for key, value in score_dict.items():
                    if isinstance(value, (int, float)):
                        if key not in avg_scores:
                            avg_scores[key] = []
                        avg_scores[key].append(value)

            avg_scores = {
                key: sum(values) / len(values) for key, values in avg_scores.items()
            }

        # Calculate success rate
        finished_posts = completed_posts + failed_posts
        success_rate = completed_posts / finished_posts if finished_posts > 0 else 0.0

        # Group by platform
        platform_breakdown = {}
        for job in recent_jobs:
            platforms = (
                [job.metadata.get("social_media_platform")]
                if job.metadata.get("social_media_platform")
                else job.metadata.get("social_media_platforms", [])
            )
            for p in platforms:
                if p not in platform_breakdown:
                    platform_breakdown[p] = {
                        "total": 0,
                        "completed": 0,
                        "failed": 0,
                    }
                platform_breakdown[p]["total"] += 1
                if job.status == JobStatus.COMPLETED:
                    platform_breakdown[p]["completed"] += 1
                elif job.status == JobStatus.FAILED:
                    platform_breakdown[p]["failed"] += 1

        result = {
            "total_posts": total_posts,
            "completed_posts": completed_posts,
            "failed_posts": failed_posts,
            "success_rate": success_rate,
            "average_quality_scores": avg_scores,
            "platform_breakdown": platform_breakdown,
            "days": days,
            "platform": platform,
        }

        await self._set_cached(cache_key, result)
        return result

    async def get_social_media_trends(
        self, days: int = 7, platform: Optional[str] = None
    ) -> Dict:
        """
        Get social media performance trends over time.

        Args:
            days: Number of days to look back
            platform: Optional platform filter

        Returns:
            Dictionary with trend data points
        """
        cache_key = f"social_media_trends:{days}:{platform or 'all'}"
        cached = await self._get_cached(cache_key)
        if cached:
            return cached

        job_manager = get_job_manager()
        all_jobs = await job_manager.list_jobs(limit=1000)

        # Filter for social media jobs
        social_media_jobs = [
            j
            for j in all_jobs
            if j.metadata.get("output_content_type") == "social_media_post"
        ]

        if platform:
            social_media_jobs = [
                j
                for j in social_media_jobs
                if j.metadata.get("social_media_platform") == platform
                or platform in j.metadata.get("social_media_platforms", [])
            ]

        # Group by date
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        daily_stats = {}
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            daily_stats[date_str] = {
                "date": date_str,
                "total": 0,
                "completed": 0,
                "failed": 0,
                "quality_scores": [],
            }
            current_date += timedelta(days=1)

        # Populate daily stats
        for job in social_media_jobs:
            if job.created_at >= start_date:
                date_str = job.created_at.strftime("%Y-%m-%d")
                if date_str in daily_stats:
                    daily_stats[date_str]["total"] += 1
                    if job.status == JobStatus.COMPLETED:
                        daily_stats[date_str]["completed"] += 1
                    elif job.status == JobStatus.FAILED:
                        daily_stats[date_str]["failed"] += 1

                    # Try to get quality score
                    if job.status == JobStatus.COMPLETED:
                        try:
                            result = await job_manager.get_job_result(job.job_id)
                            if result and "metadata" in result:
                                scores = result["metadata"].get(
                                    "platform_quality_scores", {}
                                )
                                if scores:
                                    daily_stats[date_str]["quality_scores"].append(
                                        scores
                                    )
                        except Exception:
                            pass

        # Calculate averages
        trend_points = []
        for date_str in sorted(daily_stats.keys()):
            stats = daily_stats[date_str]
            avg_quality = 0.0
            if stats["quality_scores"]:
                all_scores = []
                for score_dict in stats["quality_scores"]:
                    for value in score_dict.values():
                        if isinstance(value, (int, float)):
                            all_scores.append(value)
                if all_scores:
                    avg_quality = sum(all_scores) / len(all_scores)

            trend_points.append(
                {
                    "date": date_str,
                    "total": stats["total"],
                    "completed": stats["completed"],
                    "failed": stats["failed"],
                    "success_rate": (
                        stats["completed"] / (stats["completed"] + stats["failed"])
                        if (stats["completed"] + stats["failed"]) > 0
                        else 0.0
                    ),
                    "average_quality": avg_quality,
                }
            )

        result = {
            "trend_points": trend_points,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "days": days,
            "platform": platform,
        }

        await self._set_cached(cache_key, result)
        return result

    async def get_cost_metrics(self, days: int = 30) -> CostMetrics:
        """
        Get cost tracking metrics.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            CostMetrics with cost breakdown and token usage
        """
        # Try cache first
        cache_key = f"cost_metrics_{days}"
        cached = await self._get_cached(cache_key)
        if cached:
            return CostMetrics(**cached)

        # Compute cost metrics
        job_manager = get_job_manager()
        all_jobs = await job_manager.list_jobs(limit=1000)

        # Filter jobs from the last N days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_jobs = [j for j in all_jobs if j.created_at >= cutoff_date]

        # Calculate costs from job metadata
        total_cost = 0.0
        cost_by_model = {}
        cost_by_step = {}
        token_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

        for job in recent_jobs:
            if job.metadata:
                # Extract cost information from metadata
                job_cost = job.metadata.get("cost", 0.0)
                if isinstance(job_cost, (int, float)):
                    total_cost += float(job_cost)

                # Cost by model
                model = job.metadata.get("model", "unknown")
                if model not in cost_by_model:
                    cost_by_model[model] = 0.0
                cost_by_model[model] += (
                    float(job_cost) if isinstance(job_cost, (int, float)) else 0.0
                )

                # Cost by step
                step_name = job.metadata.get("step_name") or job.metadata.get(
                    "pipeline_step"
                )
                if step_name:
                    if step_name not in cost_by_step:
                        cost_by_step[step_name] = 0.0
                    cost_by_step[step_name] += (
                        float(job_cost) if isinstance(job_cost, (int, float)) else 0.0
                    )

                # Token usage
                tokens = job.metadata.get("tokens_used", 0)
                if isinstance(tokens, int):
                    token_usage["total_tokens"] += tokens

                prompt_tokens = job.metadata.get("prompt_tokens", 0)
                if isinstance(prompt_tokens, int):
                    token_usage["prompt_tokens"] += prompt_tokens

                completion_tokens = job.metadata.get("completion_tokens", 0)
                if isinstance(completion_tokens, int):
                    token_usage["completion_tokens"] += completion_tokens

        # Calculate average cost per job
        avg_cost_per_job = total_cost / len(recent_jobs) if recent_jobs else 0.0

        metrics = CostMetrics(
            total_cost_usd=total_cost,
            avg_cost_per_job=avg_cost_per_job,
            cost_by_model=cost_by_model,
            cost_by_step=cost_by_step,
            token_usage=token_usage,
        )

        # Cache the result
        await self._set_cached(cache_key, metrics.model_dump())

        return metrics

    async def get_quality_trends(self, days: int = 30) -> QualityTrends:
        """
        Get quality trend analysis.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            QualityTrends with quality scores and trends
        """
        # Try cache first
        cache_key = f"quality_trends_{days}"
        cached = await self._get_cached(cache_key)
        if cached:
            return QualityTrends(**cached)

        # Compute quality trends
        job_manager = get_job_manager()
        all_jobs = await job_manager.list_jobs(limit=1000)

        # Filter jobs from the last N days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_jobs = [
            j
            for j in all_jobs
            if j.created_at >= cutoff_date and j.status == JobStatus.COMPLETED
        ]

        # Collect quality scores
        quality_scores = []
        quality_by_platform = {}
        quality_trend_data = []

        for job in recent_jobs:
            try:
                result = await job_manager.get_job_result(job.job_id)
                if result:
                    # Extract quality scores from result
                    quality_score = None
                    if "metadata" in result:
                        quality_score = result["metadata"].get("quality_score")
                        if quality_score is None:
                            # Try to get from step results
                            step_results = result.get("step_results", {})
                            for step_result in step_results.values():
                                if isinstance(step_result, dict):
                                    quality_score = step_result.get(
                                        "confidence_score"
                                    ) or step_result.get("quality_score")
                                    if quality_score is not None:
                                        break

                    if quality_score is not None and isinstance(
                        quality_score, (int, float)
                    ):
                        quality_scores.append(float(quality_score))

                    # Quality by platform
                    platform = (
                        job.metadata.get("social_media_platform")
                        if job.metadata
                        else None
                    )
                    if platform:
                        platform_scores = result.get("metadata", {}).get(
                            "platform_quality_scores", {}
                        )
                        if platform in platform_scores:
                            if platform not in quality_by_platform:
                                quality_by_platform[platform] = []
                            quality_by_platform[platform].append(
                                float(platform_scores[platform])
                            )
            except Exception:
                pass

        # Calculate averages
        avg_quality_score = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )

        # Average by platform
        quality_by_platform_avg = {
            platform: sum(scores) / len(scores)
            for platform, scores in quality_by_platform.items()
            if scores
        }

        # Calculate improvement rate (simple: compare first half vs second half)
        improvement_rate = None
        if len(quality_scores) > 10:
            mid_point = len(quality_scores) // 2
            first_half_avg = sum(quality_scores[:mid_point]) / mid_point
            second_half_avg = sum(quality_scores[mid_point:]) / (
                len(quality_scores) - mid_point
            )
            if first_half_avg > 0:
                improvement_rate = (
                    (second_half_avg - first_half_avg) / first_half_avg
                ) * 100

        # Create trend data (daily averages)
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days - 1)
        daily_quality = {}
        current_date = start_date
        while current_date <= end_date:
            daily_quality[current_date.strftime("%Y-%m-%d")] = []
            current_date += timedelta(days=1)

        # Group quality scores by date
        for job in recent_jobs:
            try:
                result = await job_manager.get_job_result(job.job_id)
                if result:
                    quality_score = None
                    if "metadata" in result:
                        quality_score = result["metadata"].get("quality_score")
                    if quality_score is None:
                        step_results = result.get("step_results", {})
                        for step_result in step_results.values():
                            if isinstance(step_result, dict):
                                quality_score = step_result.get(
                                    "confidence_score"
                                ) or step_result.get("quality_score")
                                if quality_score is not None:
                                    break

                    if quality_score is not None and isinstance(
                        quality_score, (int, float)
                    ):
                        date_str = job.created_at.strftime("%Y-%m-%d")
                        if date_str in daily_quality:
                            daily_quality[date_str].append(float(quality_score))
            except Exception:
                pass

        # Calculate daily averages
        for date_str, scores in daily_quality.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            quality_trend_data.append(
                {
                    "date": date_str,
                    "average_quality": avg_score,
                    "sample_count": len(scores),
                }
            )

        trends = QualityTrends(
            avg_quality_score=avg_quality_score,
            quality_by_platform=quality_by_platform_avg,
            quality_trend_data=quality_trend_data,
            improvement_rate=improvement_rate,
        )

        # Cache the result
        await self._set_cached(cache_key, trends.model_dump())

        return trends

    async def get_unified_metrics(self, days: int = 30) -> UnifiedMonitoringMetrics:
        """
        Get unified monitoring dashboard metrics.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            UnifiedMonitoringMetrics with all metrics combined
        """
        # Try cache first
        cache_key = f"unified_metrics_{days}"
        cached = await self._get_cached(cache_key)
        if cached:
            return UnifiedMonitoringMetrics(**cached)

        # Get all metrics
        dashboard_stats = await self.get_dashboard_stats()
        pipeline_stats = await self.get_pipeline_stats()
        cost_metrics = await self.get_cost_metrics(days=days)
        quality_trends = await self.get_quality_trends(days=days)
        social_media_performance = await self.get_social_media_performance(days=days)

        metrics = UnifiedMonitoringMetrics(
            dashboard_stats=dashboard_stats,
            pipeline_stats=pipeline_stats,
            cost_metrics=cost_metrics,
            quality_trends=quality_trends,
            social_media_performance=social_media_performance,
        )

        # Cache the result
        await self._set_cached(cache_key, metrics.model_dump())

        return metrics

    # Alias for backward compatibility
    async def get_unified_monitoring_metrics(
        self, days: int = 30
    ) -> UnifiedMonitoringMetrics:
        """Alias for get_unified_metrics for backward compatibility."""
        return await self.get_unified_metrics(days=days)


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
