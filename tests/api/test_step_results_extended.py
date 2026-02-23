"""
Extended tests for step results API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.step_results import router
from marketing_project.server import app

# Add router to app
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_job_manager_ownership():
    """Mock get_job_manager so verify_job_ownership finds the job."""
    mock_job = MagicMock()
    mock_job.user_id = "test-user-123"
    mock_mgr = MagicMock()
    mock_mgr.get_job = AsyncMock(return_value=mock_job)
    with patch(
        "marketing_project.api.step_results.get_job_manager",
        return_value=mock_mgr,
    ):
        yield


@pytest.fixture
def mock_step_result_manager(mock_job_manager_ownership):
    """Mock step result manager."""
    with patch("marketing_project.api.step_results.get_step_result_manager") as mock:
        manager = MagicMock()
        manager.list_all_jobs = AsyncMock(return_value=[])
        manager.get_job_results = AsyncMock(
            return_value={
                "job_id": "test-job-1",
                "metadata": {},
                "steps": [],
                "total_steps": 0,
            }
        )
        manager.get_step_result = AsyncMock(return_value={"result": {}, "metadata": {}})
        mock.return_value = manager
        yield manager


@pytest.mark.asyncio
async def test_list_jobs(mock_step_result_manager):
    """Test /results/jobs endpoint."""
    response = client.get("/results/jobs")

    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_list_jobs_with_limit(mock_step_result_manager):
    """Test /results/jobs with limit parameter."""
    response = client.get("/results/jobs?limit=10")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_job_results(mock_step_result_manager):
    """Test /results/jobs/{job_id} endpoint."""
    response = client.get("/results/jobs/test-job-1")

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "steps" in data


@pytest.mark.asyncio
async def test_get_job_results_not_found(mock_step_result_manager):
    """Test /results/jobs/{job_id} with non-existent job."""
    mock_step_result_manager.get_job_results = AsyncMock(
        side_effect=FileNotFoundError("Job not found")
    )

    response = client.get("/results/jobs/non-existent")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_step_result_file(mock_step_result_manager):
    """Test /results/jobs/{job_id}/steps/{step_filename} endpoint."""
    response = client.get("/results/jobs/test-job-1/steps/01_seo_keywords.json")

    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_download_step_result(mock_step_result_manager):
    """Test /results/jobs/{job_id}/steps/{step_filename}/download endpoint."""
    response = client.get(
        "/results/jobs/test-job-1/steps/01_seo_keywords.json/download"
    )

    assert response.status_code in [200, 404]
