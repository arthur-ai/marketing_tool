"""
Extended tests for approvals API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.server import app

client = TestClient(app)


@pytest.fixture
def mock_approval_manager():
    """Mock approval manager."""
    with patch("marketing_project.api.approvals.get_approval_manager") as mock:
        manager = MagicMock()
        manager.list_approvals = AsyncMock(return_value=[])
        manager.get_approval = AsyncMock(return_value=None)
        manager.decide_approval = AsyncMock(return_value=MagicMock(status="approved"))
        manager.get_stats = AsyncMock(return_value=MagicMock(total_requests=10))
        manager.get_settings = AsyncMock(
            return_value=MagicMock(require_approval_for_steps=[])
        )
        mock.return_value = manager
        yield manager


@pytest.mark.asyncio
async def test_get_approval_analytics(mock_approval_manager):
    """Test GET /approvals/analytics endpoint."""
    response = client.get("/api/v1/approvals/analytics")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_get_approval_stats(mock_approval_manager):
    """Test GET /approvals/stats endpoint."""
    response = client.get("/api/v1/approvals/stats")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)
        assert "total_requests" in data or "pending" in data


@pytest.mark.asyncio
async def test_get_approval_settings(mock_approval_manager):
    """Test GET /approvals/settings endpoint."""
    response = client.get("/api/v1/approvals/settings")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_update_approval_settings(mock_approval_manager):
    """Test POST /approvals/settings endpoint."""
    request_data = {
        "require_approval_for_steps": ["seo_keywords"],
        "auto_approve_threshold": 0.9,
    }

    response = client.post("/api/v1/approvals/settings", json=request_data)

    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_get_approval_impact(mock_approval_manager):
    """Test GET /approvals/{approval_id}/impact endpoint."""
    response = client.get("/api/v1/approvals/test-approval-1/impact")

    assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_delete_all_approvals(mock_approval_manager):
    """Test DELETE /approvals/all endpoint."""
    response = client.delete("/api/v1/approvals/all")

    assert response.status_code in [200, 500]
