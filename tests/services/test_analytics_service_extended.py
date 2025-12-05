"""
Extended tests for analytics service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.analytics_service import (
    AnalyticsService,
    get_analytics_service,
)


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    with patch(
        "marketing_project.services.analytics_service.get_redis_manager"
    ) as mock:
        manager = MagicMock()
        manager.get_redis = AsyncMock(return_value=MagicMock())
        manager.execute = AsyncMock(return_value=None)
        mock.return_value = manager
        yield manager


@pytest.fixture
def analytics_service(mock_redis_manager):
    """Create an AnalyticsService instance."""
    return AnalyticsService()


@pytest.mark.asyncio
async def test_get_dashboard_stats(analytics_service, mock_redis_manager):
    """Test getting dashboard statistics."""
    with patch(
        "marketing_project.services.analytics_service.get_job_manager"
    ) as mock_job_mgr:
        with patch(
            "marketing_project.services.analytics_service.get_content_manager"
        ) as mock_content_mgr:
            mock_job_manager = MagicMock()
            mock_job_manager.list_jobs = AsyncMock(return_value=[])
            mock_job_mgr.return_value = mock_job_manager

            mock_content_manager.return_value = MagicMock()

            stats = await analytics_service.get_dashboard_stats()

            assert stats is not None
            assert isinstance(stats, dict) or hasattr(stats, "total_jobs")


@pytest.mark.asyncio
async def test_get_pipeline_stats(analytics_service, mock_redis_manager):
    """Test getting pipeline statistics."""
    with patch(
        "marketing_project.services.analytics_service.get_job_manager"
    ) as mock_job_mgr:
        mock_job_manager = MagicMock()
        mock_job_manager.list_jobs = AsyncMock(return_value=[])
        mock_job_mgr.return_value = mock_job_manager

        stats = await analytics_service.get_pipeline_stats()

        assert stats is not None
        assert isinstance(stats, dict) or hasattr(stats, "total_runs")


@pytest.mark.asyncio
async def test_get_content_stats(analytics_service, mock_redis_manager):
    """Test getting content statistics."""
    with patch(
        "marketing_project.services.analytics_service.get_content_manager"
    ) as mock_content_mgr:
        mock_manager = MagicMock()
        mock_manager.list_sources = AsyncMock(return_value=[])
        mock_content_mgr.return_value = mock_manager

        stats = await analytics_service.get_content_stats()

        assert stats is not None
        assert isinstance(stats, dict) or hasattr(stats, "total_items")


@pytest.mark.asyncio
async def test_get_cost_metrics(analytics_service, mock_redis_manager):
    """Test getting cost metrics."""
    with patch(
        "marketing_project.services.analytics_service.get_job_manager"
    ) as mock_job_mgr:
        mock_job_manager = MagicMock()
        mock_job_manager.list_jobs = AsyncMock(return_value=[])
        mock_job_mgr.return_value = mock_job_manager

        metrics = await analytics_service.get_cost_metrics()

        assert metrics is not None
        assert isinstance(metrics, dict) or hasattr(metrics, "total_cost")


@pytest.mark.asyncio
async def test_get_quality_trends(analytics_service, mock_redis_manager):
    """Test getting quality trends."""
    trends = await analytics_service.get_quality_trends()

    assert trends is not None
    assert isinstance(trends, dict) or hasattr(trends, "trends")


@pytest.mark.asyncio
async def test_get_recent_activity(analytics_service, mock_redis_manager):
    """Test getting recent activity."""
    with patch(
        "marketing_project.services.analytics_service.get_job_manager"
    ) as mock_job_mgr:
        mock_job_manager = MagicMock()
        mock_job_manager.list_jobs = AsyncMock(return_value=[])
        mock_job_mgr.return_value = mock_job_manager

        activity = await analytics_service.get_recent_activity()

        assert activity is not None
        assert isinstance(activity, dict) or hasattr(activity, "activities")


@pytest.mark.asyncio
async def test_get_unified_metrics(analytics_service, mock_redis_manager):
    """Test getting unified metrics."""
    metrics = await analytics_service.get_unified_metrics()

    assert metrics is not None
    assert isinstance(metrics, dict)


def test_get_analytics_service_singleton():
    """Test that get_analytics_service returns a singleton."""
    service1 = get_analytics_service()
    service2 = get_analytics_service()

    assert service1 is service2
