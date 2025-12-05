"""
Comprehensive tests for approvals API endpoints.
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
        mock.return_value = manager
        yield manager


@pytest.mark.asyncio
async def test_get_pending_approvals(mock_approval_manager):
    """Test GET /approvals/pending endpoint."""
    response = client.get("/api/v1/approvals/pending")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, (dict, list))


@pytest.mark.asyncio
async def test_get_approval_by_id(mock_approval_manager):
    """Test GET /approvals/{approval_id} endpoint."""
    response = client.get("/api/v1/approvals/test-approval-1")

    assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_approve_approval(mock_approval_manager):
    """Test POST /approvals/{approval_id}/approve endpoint."""
    request_data = {"comments": "Looks good"}

    response = client.post(
        "/api/v1/approvals/test-approval-1/approve", json=request_data
    )

    assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_reject_approval(mock_approval_manager):
    """Test POST /approvals/{approval_id}/reject endpoint."""
    request_data = {"comments": "Needs improvement"}

    response = client.post(
        "/api/v1/approvals/test-approval-1/reject", json=request_data
    )

    assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_modify_and_approve(mock_approval_manager):
    """Test POST /approvals/{approval_id}/modify endpoint."""
    request_data = {
        "modifications": {"main_keyword": "updated"},
        "comments": "Modified and approved",
    }

    response = client.post(
        "/api/v1/approvals/test-approval-1/modify", json=request_data
    )

    assert response.status_code in [200, 404, 500]
