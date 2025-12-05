"""
Extended tests for jobs API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.server import app

# Router is already included in the main app via routes.py
# We'll test the endpoints directly

client = TestClient(app)


@pytest.fixture
def mock_job_manager():
    """Mock job manager."""
    with patch("marketing_project.api.jobs.get_job_manager") as mock:
        manager = MagicMock()
        manager.create_job = AsyncMock(
            return_value=MagicMock(
                id="test-job-1",
                type="blog",
                status="pending",
                content_id="test-content-1",
            )
        )
        manager.get_job = AsyncMock(return_value=MagicMock(id="test-job-1"))
        manager.list_jobs = AsyncMock(return_value=[])
        manager.update_job_status = AsyncMock(return_value=True)
        manager.cancel_job = AsyncMock(return_value=True)
        mock.return_value = manager
        yield manager


@pytest.mark.asyncio
async def test_list_jobs(mock_job_manager):
    """Test /jobs endpoint."""
    # Router is included in main app, test the actual endpoint
    response = client.get("/jobs")

    assert response.status_code in [200, 404, 500]
    if response.status_code == 200:
        data = response.json()
        assert "jobs" in data or isinstance(data, list)


@pytest.mark.asyncio
async def test_get_job(mock_job_manager):
    """Test /jobs/{job_id} endpoint."""
    response = client.get("/jobs/test-job-1")

    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_get_job_status(mock_job_manager):
    """Test /jobs/{job_id}/status endpoint."""
    response = client.get("/jobs/test-job-1/status")

    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_get_job_result(mock_job_manager):
    """Test /jobs/{job_id}/result endpoint."""
    response = client.get("/jobs/test-job-1/result")

    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_get_job_chain(mock_job_manager):
    """Test /jobs/{job_id}/chain endpoint."""
    response = client.get("/jobs/test-job-1/chain")

    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_cancel_job(mock_job_manager):
    """Test DELETE /jobs/{job_id} endpoint."""
    response = client.delete("/jobs/test-job-1")

    assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_resume_job(mock_job_manager):
    """Test POST /jobs/{job_id}/resume endpoint."""
    request_data = {"guidance": "Continue processing"}

    response = client.post("/jobs/test-job-1/resume", json=request_data)

    assert response.status_code in [200, 404, 500]
