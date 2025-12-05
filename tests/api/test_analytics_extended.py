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
    with patch("marketing_project.api.analytics.get_analytics_service") as mock:
        service = MagicMock()
        service.get_dashboard_stats = AsyncMock(
            return_value={
                "total_content": 100,
                "total_jobs": 50,
                "jobs_by_status": {"completed": 40, "failed": 5, "in_progress": 5},
                "success_rate": 0.8,
            }
        )
        service.get_pipeline_stats = AsyncMock(
            return_value={
                "total_runs": 50,
                "runs_by_status": {"completed": 40},
                "average_duration": 30.5,
                "success_rate": 0.8,
            }
        )
        service.get_content_stats = AsyncMock(
            return_value={
                "total_items": 100,
                "by_source": {"blog": 50, "transcript": 30, "release_notes": 20},
                "active_sources": 3,
            }
        )
        service.get_cost_metrics = AsyncMock(
            return_value={
                "total_cost": 100.0,
                "cost_by_step": {"seo_keywords": 20.0},
                "average_cost_per_job": 2.0,
            }
        )
        service.get_quality_trends = AsyncMock(
            return_value={
                "trends": [],
                "average_quality": 0.85,
            }
        )
        service.get_recent_activity = AsyncMock(
            return_value={
                "activities": [],
                "total": 0,
            }
        )
        service.get_unified_metrics = AsyncMock(
            return_value={
                "metrics": {},
            }
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
    assert "total_runs" in data or "runs_by_status" in data


@pytest.mark.asyncio
async def test_get_content_analytics(mock_analytics_service):
    """Test /content endpoint."""
    response = client.get("/content")

    assert response.status_code == 200
    data = response.json()
    assert "total_items" in data or "by_source" in data


@pytest.mark.asyncio
async def test_get_cost_metrics(mock_analytics_service):
    """Test /cost endpoint."""
    response = client.get("/cost")

    assert response.status_code == 200
    data = response.json()
    assert "total_cost" in data or "cost_by_step" in data


@pytest.mark.asyncio
async def test_get_quality_trends(mock_analytics_service):
    """Test /quality/trends endpoint."""
    response = client.get("/quality/trends")

    assert response.status_code == 200
    data = response.json()
    assert "trends" in data or "average_quality" in data


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
