"""
Extended tests for analytics API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.analytics import router
from marketing_project.server import app

# Add router to app
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_analytics_service():
    """Mock analytics service."""
    from marketing_project.models.analytics_models import (
        ContentSourceStats,
        ContentStats,
        CostMetrics,
        DashboardStats,
        PipelineStats,
        QualityTrends,
        RecentActivity,
        UnifiedMonitoringMetrics,
    )

    with patch("marketing_project.api.analytics.get_analytics_service") as mock:
        service = MagicMock()
        service.get_dashboard_stats = AsyncMock(
            return_value=DashboardStats(
                total_content=100,
                total_jobs=50,
                jobs_completed=40,
                jobs_processing=5,
                jobs_failed=5,
                jobs_queued=0,
                success_rate=0.8,
            )
        )
        service.get_pipeline_stats = AsyncMock(
            return_value=PipelineStats(
                total_runs=50,
                completed=40,
                in_progress=5,
                failed=5,
                queued=0,
                avg_duration_seconds=30.5,
                success_rate=0.8,
            )
        )
        service.get_content_stats = AsyncMock(
            return_value=ContentStats(
                total_content=100,
                by_source=[
                    ContentSourceStats(source_name="blog", total_items=50, active=True),
                    ContentSourceStats(
                        source_name="transcript", total_items=30, active=True
                    ),
                    ContentSourceStats(
                        source_name="release_notes", total_items=20, active=True
                    ),
                ],
                active_sources=3,
            )
        )
        service.get_cost_metrics = AsyncMock(
            return_value=CostMetrics(
                total_cost_usd=100.0,
                avg_cost_per_job=2.0,
                cost_by_step={"seo_keywords": 20.0},
                cost_by_model={},
                token_usage={},
            )
        )
        service.get_quality_trends = AsyncMock(
            return_value=QualityTrends(
                avg_quality_score=85.0,
                quality_by_platform={},
                quality_trend_data=[],
                improvement_rate=None,
            )
        )
        service.get_recent_activity = AsyncMock(
            return_value=RecentActivity(
                activities=[],
                total=0,
            )
        )
        service.get_unified_metrics = AsyncMock(
            return_value=UnifiedMonitoringMetrics(
                dashboard_stats=DashboardStats(
                    total_content=100,
                    total_jobs=50,
                    jobs_completed=40,
                    jobs_processing=5,
                    jobs_failed=5,
                    jobs_queued=0,
                    success_rate=0.8,
                ),
                pipeline_stats=PipelineStats(
                    total_runs=50,
                    completed=40,
                    in_progress=5,
                    failed=5,
                    queued=0,
                    success_rate=0.8,
                ),
                cost_metrics=CostMetrics(
                    total_cost_usd=100.0,
                    avg_cost_per_job=2.0,
                    cost_by_step={"seo_keywords": 20.0},
                    cost_by_model={},
                    token_usage={},
                ),
                quality_trends=QualityTrends(
                    avg_quality_score=85.0,
                    quality_by_platform={},
                    quality_trend_data=[],
                    improvement_rate=None,
                ),
                social_media_performance=None,
            )
        )
        mock.return_value = service
        yield service


@pytest.mark.asyncio
async def test_get_dashboard_analytics(mock_analytics_service):
    """Test /dashboard endpoint."""
    response = client.get("/dashboard")

    assert response.status_code == 200
    data = response.json()
    assert "total_content" in data or "total_jobs" in data


@pytest.mark.asyncio
async def test_get_pipeline_analytics(mock_analytics_service):
    """Test /pipeline endpoint."""
    response = client.get("/pipeline")

    assert response.status_code == 200
    data = response.json()
    assert "total_runs" in data or "completed" in data


@pytest.mark.asyncio
async def test_get_content_analytics(mock_analytics_service):
    """Test /content endpoint."""
    response = client.get("/content")

    assert response.status_code == 200
    data = response.json()
    assert "total_content" in data or "by_source" in data


@pytest.mark.asyncio
async def test_get_cost_metrics(mock_analytics_service):
    """Test /cost endpoint."""
    response = client.get("/cost")

    assert response.status_code == 200
    data = response.json()
    assert "total_cost_usd" in data or "cost_by_step" in data


@pytest.mark.asyncio
async def test_get_quality_trends(mock_analytics_service):
    """Test /quality/trends endpoint."""
    response = client.get("/quality/trends")

    assert response.status_code == 200
    data = response.json()
    assert "quality_trend_data" in data or "avg_quality_score" in data


@pytest.mark.asyncio
async def test_get_recent_activity(mock_analytics_service):
    """Test /activity/recent endpoint."""
    response = client.get("/activity/recent")

    assert response.status_code == 200
    data = response.json()
    assert "activities" in data or "total" in data


@pytest.mark.asyncio
async def test_get_unified_metrics(mock_analytics_service):
    """Test /metrics/unified endpoint."""
    response = client.get("/metrics/unified")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
