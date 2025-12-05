"""
Tests for feedback API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.feedback import router
from marketing_project.server import app

# Add router to app
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_feedback_service():
    """Mock feedback service."""
    with patch("marketing_project.api.feedback.get_feedback_service") as mock:
        service = MagicMock()
        service.record_feedback = AsyncMock(return_value={"feedback_id": "feedback-1"})
        service.get_feedback_stats = AsyncMock(
            return_value={
                "total_feedback": 10,
                "approval_rate": 0.8,
                "average_rating": 4.5,
                "platform_stats": {},
            }
        )
        mock.return_value = service
        yield service


@pytest.mark.asyncio
async def test_submit_feedback(mock_feedback_service):
    """Test submitting feedback."""
    request_data = {
        "job_id": "test-job-1",
        "feedback_type": "approval",
        "rating": 5,
        "comments": "Great post!",
    }

    response = client.post("/v1/feedback", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["feedback_id"] == "feedback-1"
    assert "message" in data


@pytest.mark.asyncio
async def test_submit_feedback_with_metadata(mock_feedback_service):
    """Test submitting feedback with metadata."""
    request_data = {
        "job_id": "test-job-1",
        "feedback_type": "rating",
        "rating": 4,
        "metadata": {"source": "user", "context": "dashboard"},
    }

    response = client.post("/v1/feedback", json=request_data)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_submit_feedback_invalid_rating(mock_feedback_service):
    """Test submitting feedback with invalid rating."""
    request_data = {
        "job_id": "test-job-1",
        "feedback_type": "rating",
        "rating": 10,  # Max is 5
    }

    response = client.post("/v1/feedback", json=request_data)

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_get_feedback_stats(mock_feedback_service):
    """Test getting feedback statistics."""
    response = client.get("/v1/feedback/stats")

    assert response.status_code == 200
    data = response.json()
    assert "total_feedback" in data
    assert "approval_rate" in data


@pytest.mark.asyncio
async def test_get_feedback_stats_with_filters(mock_feedback_service):
    """Test getting feedback statistics with filters."""
    response = client.get("/v1/feedback/stats?days=7&platform=linkedin")

    assert response.status_code == 200
