"""
Tests for Analytics API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.analytics import router
from marketing_project.models.analytics_models import (
    ContentStats,
    DashboardStats,
    PipelineStats,
    RecentActivity,
    TrendData,
)


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.mark.asyncio
class TestAnalyticsAPI:
    """Test suite for Analytics API endpoints."""

    async def test_get_dashboard_analytics_success(self, client):
        """Test successful dashboard analytics retrieval."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_stats = DashboardStats(
                total_content=100,
                total_jobs=50,
                jobs_by_status={"completed": 40, "failed": 5, "processing": 5},
                success_rate=80.0,
                change_indicators={"jobs": "+10%", "success_rate": "+5%"},
            )
            mock_service_instance.get_dashboard_stats = AsyncMock(
                return_value=mock_stats
            )

            response = await client.get("/api/v1/analytics/dashboard")
            assert response.status_code == 200
            data = response.json()
            assert data["total_content"] == 100
            assert data["total_jobs"] == 50
            assert data["success_rate"] == 80.0

    async def test_get_dashboard_analytics_error(self, client):
        """Test dashboard analytics error handling."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.get_dashboard_stats = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = await client.get("/api/v1/analytics/dashboard")
            assert response.status_code == 500
            assert "Failed to retrieve dashboard analytics" in response.json()["detail"]

    async def test_get_pipeline_analytics_success(self, client):
        """Test successful pipeline analytics retrieval."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_stats = PipelineStats(
                total_runs=100,
                runs_by_status={
                    "completed": 80,
                    "in_progress": 10,
                    "failed": 5,
                    "queued": 5,
                },
                average_duration=120.5,
                success_rate=80.0,
            )
            mock_service_instance.get_pipeline_stats = AsyncMock(
                return_value=mock_stats
            )

            response = await client.get("/api/v1/analytics/pipeline")
            assert response.status_code == 200
            data = response.json()
            assert data["total_runs"] == 100
            assert data["average_duration"] == 120.5

    async def test_get_pipeline_analytics_error(self, client):
        """Test pipeline analytics error handling."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.get_pipeline_stats = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = await client.get("/api/v1/analytics/pipeline")
            assert response.status_code == 500

    async def test_get_content_analytics_success(self, client):
        """Test successful content analytics retrieval."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_stats = ContentStats(
                total_items=200,
                items_by_source={"file": 100, "api": 50, "database": 50},
                active_sources=3,
            )
            mock_service_instance.get_content_stats = AsyncMock(return_value=mock_stats)

            response = await client.get("/api/v1/analytics/content")
            assert response.status_code == 200
            data = response.json()
            assert data["total_items"] == 200
            assert data["active_sources"] == 3

    async def test_get_content_analytics_error(self, client):
        """Test content analytics error handling."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.get_content_stats = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = await client.get("/api/v1/analytics/content")
            assert response.status_code == 500

    async def test_get_recent_activity_success(self, client):
        """Test successful recent activity retrieval."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_activity = RecentActivity(
                items=[],
                total_items=0,
                days=7,
            )
            mock_service_instance.get_recent_activity = AsyncMock(
                return_value=mock_activity
            )

            response = await client.get("/api/v1/analytics/recent-activity?days=7")
            assert response.status_code == 200
            data = response.json()
            assert data["days"] == 7

    async def test_get_recent_activity_with_custom_days(self, client):
        """Test recent activity with custom days parameter."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_activity = RecentActivity(
                items=[],
                total_items=0,
                days=14,
            )
            mock_service_instance.get_recent_activity = AsyncMock(
                return_value=mock_activity
            )

            response = await client.get("/api/v1/analytics/recent-activity?days=14")
            assert response.status_code == 200
            data = response.json()
            assert data["days"] == 14

    async def test_get_recent_activity_error(self, client):
        """Test recent activity error handling."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.get_recent_activity = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = await client.get("/api/v1/analytics/recent-activity")
            assert response.status_code == 500

    async def test_get_trends_success(self, client):
        """Test successful trends retrieval."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_trends = TrendData(
                days=7,
                data_points=[],
            )
            mock_service_instance.get_trends = AsyncMock(return_value=mock_trends)

            response = await client.get("/api/v1/analytics/trends?days=7")
            assert response.status_code == 200
            data = response.json()
            assert data["days"] == 7

    async def test_get_trends_with_custom_days(self, client):
        """Test trends with custom days parameter."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_trends = TrendData(
                days=30,
                data_points=[],
            )
            mock_service_instance.get_trends = AsyncMock(return_value=mock_trends)

            response = await client.get("/api/v1/analytics/trends?days=30")
            assert response.status_code == 200
            data = response.json()
            assert data["days"] == 30

    async def test_get_trends_error(self, client):
        """Test trends error handling."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.get_trends = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = await client.get("/api/v1/analytics/trends")
            assert response.status_code == 500
