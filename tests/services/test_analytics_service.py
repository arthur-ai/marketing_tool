"""
Tests for Analytics Service.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.analytics_models import (
    ContentStats,
    DashboardStats,
    PipelineStats,
    RecentActivity,
    TrendData,
)
from marketing_project.services.analytics_service import AnalyticsService


@pytest.fixture
def analytics_service():
    """Create AnalyticsService instance."""
    return AnalyticsService()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock(return_value=True)
    return redis_mock


@pytest.mark.asyncio
class TestAnalyticsService:
    """Test suite for AnalyticsService."""

    async def test_get_dashboard_stats_success(self, analytics_service):
        """Test successful dashboard stats retrieval."""
        with (
            patch(
                "marketing_project.services.analytics_service.get_job_manager"
            ) as mock_job_manager,
            patch(
                "marketing_project.services.analytics_service.get_content_manager"
            ) as mock_content_manager,
            patch.object(analytics_service, "_set_cached") as mock_cache,
        ):
            # Mock job manager
            mock_job_manager_instance = AsyncMock()
            mock_job_manager.return_value = mock_job_manager_instance
            mock_job_manager_instance.list_jobs = AsyncMock(return_value=[])

            # Mock content manager
            mock_content_manager_instance = AsyncMock()
            mock_content_manager.return_value = mock_content_manager_instance
            mock_content_manager_instance.list_sources = AsyncMock(return_value=[])

            stats = await analytics_service.get_dashboard_stats()
            assert isinstance(stats, DashboardStats)
            assert stats.total_jobs == 0

    async def test_get_dashboard_stats_cached(self, analytics_service):
        """Test dashboard stats retrieval from cache."""
        cached_data = {
            "total_content": 100,
            "total_jobs": 50,
            "jobs_by_status": {"completed": 40},
            "success_rate": 80.0,
        }

        with patch.object(analytics_service, "_get_cached") as mock_get_cache:
            mock_get_cache.return_value = cached_data

            stats = await analytics_service.get_dashboard_stats()
            assert stats.total_content == 100
            assert stats.total_jobs == 50

    async def test_get_pipeline_stats_success(self, analytics_service):
        """Test successful pipeline stats retrieval."""
        with (
            patch(
                "marketing_project.services.analytics_service.get_job_manager"
            ) as mock_job_manager,
            patch.object(analytics_service, "_set_cached") as mock_cache,
        ):
            mock_job_manager_instance = AsyncMock()
            mock_job_manager.return_value = mock_job_manager_instance
            mock_job_manager_instance.list_jobs = AsyncMock(return_value=[])

            stats = await analytics_service.get_pipeline_stats()
            assert isinstance(stats, PipelineStats)

    async def test_get_content_stats_success(self, analytics_service):
        """Test successful content stats retrieval."""
        with (
            patch(
                "marketing_project.services.analytics_service.get_content_manager"
            ) as mock_content_manager,
            patch.object(analytics_service, "_set_cached") as mock_cache,
        ):
            mock_content_manager_instance = AsyncMock()
            mock_content_manager.return_value = mock_content_manager_instance
            mock_content_manager_instance.list_sources = AsyncMock(return_value=[])

            stats = await analytics_service.get_content_stats()
            assert isinstance(stats, ContentStats)

    async def test_get_recent_activity_success(self, analytics_service):
        """Test successful recent activity retrieval."""
        with (
            patch(
                "marketing_project.services.analytics_service.get_job_manager"
            ) as mock_job_manager,
            patch.object(analytics_service, "_set_cached") as mock_cache,
        ):
            mock_job_manager_instance = AsyncMock()
            mock_job_manager.return_value = mock_job_manager_instance
            mock_job_manager_instance.list_jobs = AsyncMock(return_value=[])

            activity = await analytics_service.get_recent_activity(days=7)
            assert isinstance(activity, RecentActivity)
            assert activity.days == 7

    async def test_get_trends_success(self, analytics_service):
        """Test successful trends retrieval."""
        with (
            patch(
                "marketing_project.services.analytics_service.get_job_manager"
            ) as mock_job_manager,
            patch.object(analytics_service, "_set_cached") as mock_cache,
        ):
            mock_job_manager_instance = AsyncMock()
            mock_job_manager.return_value = mock_job_manager_instance
            mock_job_manager_instance.list_jobs = AsyncMock(return_value=[])

            trends = await analytics_service.get_trends(days=7)
            assert isinstance(trends, TrendData)
            assert trends.days == 7

    async def test_cache_operations(self, analytics_service):
        """Test cache get and set operations."""
        test_data = {"key": "value"}

        with patch.object(analytics_service, "_redis_manager") as mock_redis_manager:
            mock_redis_client = AsyncMock()
            mock_redis_manager.get_redis = AsyncMock(return_value=mock_redis_client)
            mock_redis_manager.execute = AsyncMock(return_value=None)

            # Test get cached (cache miss)
            result = await analytics_service._get_cached("test_key")
            assert result is None

            # Test set cached
            await analytics_service._set_cached("test_key", test_data)
            mock_redis_manager.execute.assert_called()
