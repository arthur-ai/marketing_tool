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
    app.include_router(router, prefix="/api/v1/analytics")
    return TestClient(app)


class TestAnalyticsAPI:
    """Test suite for Analytics API endpoints."""

    def test_get_dashboard_analytics_success(self, client):
        """Test successful dashboard analytics retrieval."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_stats = DashboardStats(
                total_content=100,
                total_jobs=50,
                jobs_completed=40,
                jobs_processing=5,
                jobs_failed=5,
                jobs_queued=0,
                success_rate=0.8,  # 80% as decimal (0-1)
            )
            mock_service_instance.get_dashboard_stats = AsyncMock(
                return_value=mock_stats
            )

            response = client.get("/api/v1/analytics/dashboard")
            assert response.status_code == 200
            data = response.json()
            assert data["total_content"] == 100
            assert data["total_jobs"] == 50
            assert data["success_rate"] == 0.8

    def test_get_dashboard_analytics_error(self, client):
        """Test dashboard analytics error handling."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.get_dashboard_stats = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = client.get("/api/v1/analytics/dashboard")
            assert response.status_code == 500
            assert "Failed to retrieve dashboard analytics" in response.json()["detail"]

    def test_get_pipeline_analytics_success(self, client):
        """Test successful pipeline analytics retrieval."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_stats = PipelineStats(
                total_runs=100,
                completed=80,
                in_progress=10,
                failed=5,
                queued=5,
                avg_duration_seconds=120.5,
                success_rate=0.8,  # 80% as decimal (0-1)
            )
            mock_service_instance.get_pipeline_stats = AsyncMock(
                return_value=mock_stats
            )

            response = client.get("/api/v1/analytics/pipeline")
            assert response.status_code == 200
            data = response.json()
            assert data["total_runs"] == 100
            assert data["avg_duration_seconds"] == 120.5

    def test_get_pipeline_analytics_error(self, client):
        """Test pipeline analytics error handling."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.get_pipeline_stats = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = client.get("/api/v1/analytics/pipeline")
            assert response.status_code == 500

    def test_get_content_analytics_success(self, client):
        """Test successful content analytics retrieval."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            from marketing_project.models.analytics_models import ContentSourceStats

            mock_stats = ContentStats(
                total_content=200,
                by_source=[
                    ContentSourceStats(
                        source_name="file", total_items=100, active=True
                    ),
                    ContentSourceStats(source_name="api", total_items=50, active=True),
                    ContentSourceStats(
                        source_name="database", total_items=50, active=True
                    ),
                ],
                active_sources=3,
            )
            mock_service_instance.get_content_stats = AsyncMock(return_value=mock_stats)

            response = client.get("/api/v1/analytics/content")
            assert response.status_code == 200
            data = response.json()
            assert data["total_content"] == 200
            assert data["active_sources"] == 3

    def test_get_content_analytics_error(self, client):
        """Test content analytics error handling."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.get_content_stats = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = client.get("/api/v1/analytics/content")
            assert response.status_code == 500

    def test_get_recent_activity_success(self, client):
        """Test successful recent activity retrieval."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_activity = RecentActivity(
                activities=[],
                total=0,
            )
            mock_service_instance.get_recent_activity = AsyncMock(
                return_value=mock_activity
            )

            response = client.get("/api/v1/analytics/recent-activity?days=7")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0

    def test_get_recent_activity_with_custom_days(self, client):
        """Test recent activity with custom days parameter."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_activity = RecentActivity(
                activities=[],
                total=0,
            )
            mock_service_instance.get_recent_activity = AsyncMock(
                return_value=mock_activity
            )

            response = client.get("/api/v1/analytics/recent-activity?days=14")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0

    def test_get_recent_activity_error(self, client):
        """Test recent activity error handling."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.get_recent_activity = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = client.get("/api/v1/analytics/recent-activity")
            assert response.status_code == 500

    def test_get_trends_success(self, client):
        """Test successful trends retrieval."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_trends = TrendData(
                data_points=[],
                start_date="2024-01-01",
                end_date="2024-01-07",
                days=7,
            )
            mock_service_instance.get_trends = AsyncMock(return_value=mock_trends)

            response = client.get("/api/v1/analytics/trends?days=7")
            assert response.status_code == 200
            data = response.json()
            assert data["days"] == 7

    def test_get_trends_with_custom_days(self, client):
        """Test trends with custom days parameter."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance

            mock_trends = TrendData(
                data_points=[],
                start_date="2024-01-01",
                end_date="2024-01-30",
                days=30,
            )
            mock_service_instance.get_trends = AsyncMock(return_value=mock_trends)

            response = client.get("/api/v1/analytics/trends?days=30")
            assert response.status_code == 200
            data = response.json()
            assert data["days"] == 30

    def test_get_trends_error(self, client):
        """Test trends error handling."""
        with patch(
            "marketing_project.api.analytics.get_analytics_service"
        ) as mock_service:
            mock_service_instance = AsyncMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.get_trends = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = client.get("/api/v1/analytics/trends")
            assert response.status_code == 500
