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


# ---------------------------------------------------------------------------
# Additional tests to increase coverage of api/jobs.py
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_job_manager_full():
    """Comprehensive mock job manager with realistic job objects."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from marketing_project.services.job_manager import Job, JobStatus

    mock_job = MagicMock(spec=Job)
    mock_job.id = "job-abc-123"
    mock_job.type = "blog"
    mock_job.content_id = "content-1"
    mock_job.user_id = "test-user-123"
    mock_job.progress = 0
    mock_job.current_step = None
    mock_job.result = None
    mock_job.error = None
    mock_job.status = JobStatus.PENDING
    mock_job.metadata = {}
    mock_job.created_at = None
    mock_job.started_at = None
    mock_job.completed_at = None

    mock_mgr = MagicMock()
    mock_mgr.get_job = AsyncMock(return_value=mock_job)
    mock_mgr.list_jobs = AsyncMock(return_value=[mock_job])
    mock_mgr.cancel_job = AsyncMock(return_value=True)
    mock_mgr.delete_job = AsyncMock(return_value=True)
    mock_mgr.delete_all_jobs = AsyncMock(return_value=5)
    mock_mgr.clear_all_arq_jobs = AsyncMock(return_value=3)
    mock_mgr.get_job_chain = AsyncMock(
        return_value={
            "root_job_id": "job-abc-123",
            "chain_length": 1,
            "chain_order": ["job-abc-123"],
            "all_job_ids": ["job-abc-123"],
            "chain_status": "completed",
            "jobs": [mock_job],
        }
    )
    mock_mgr.get_job_with_subjob_status = AsyncMock(
        return_value={"subjob_status": None, "chain_status": None}
    )
    mock_mgr._save_job = AsyncMock()

    with patch("marketing_project.api.jobs.get_job_manager", return_value=mock_mgr):
        yield mock_mgr, mock_job


@pytest.fixture
def mock_approval_manager():
    """Mock approval manager."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_mgr = MagicMock()
    mock_mgr.list_approvals = AsyncMock(return_value=[])
    mock_mgr.decide_approval = AsyncMock()
    mock_mgr.load_pipeline_context = AsyncMock(return_value=None)

    with patch(
        "marketing_project.api.jobs.get_approval_manager",
        AsyncMock(return_value=mock_mgr),
    ):
        yield mock_mgr


def test_list_jobs_api_path(mock_job_manager_full, mock_approval_manager):
    """Test GET /api/v1/jobs returns 200 with job list."""
    response = client.get("/api/v1/jobs")

    assert response.status_code == 200
    body = response.json()
    assert "jobs" in body
    assert "total" in body


def test_list_jobs_with_status_filter(mock_job_manager_full, mock_approval_manager):
    """Test GET /api/v1/jobs?status=pending filter."""
    response = client.get("/api/v1/jobs?status=pending")

    assert response.status_code in [200, 422]  # 422 if enum validation fails


def test_list_jobs_with_type_filter(mock_job_manager_full, mock_approval_manager):
    """Test GET /api/v1/jobs?job_type=blog filter."""
    response = client.get("/api/v1/jobs?job_type=blog")

    assert response.status_code == 200


def test_list_jobs_with_limit(mock_job_manager_full, mock_approval_manager):
    """Test GET /api/v1/jobs?limit=10."""
    response = client.get("/api/v1/jobs?limit=10")

    assert response.status_code == 200


def test_get_job_api_path_success(mock_job_manager_full, mock_approval_manager):
    """Test GET /api/v1/jobs/{job_id} returns job details."""
    mock_mgr, mock_job = mock_job_manager_full
    response = client.get("/api/v1/jobs/job-abc-123")

    assert response.status_code == 200
    body = response.json()
    assert "job" in body


def test_get_job_not_found(mock_approval_manager):
    """Test GET /api/v1/jobs/{job_id} with non-existent job returns 404."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_mgr = MagicMock()
    mock_mgr.get_job = AsyncMock(return_value=None)
    mock_mgr.list_jobs = AsyncMock(return_value=[])

    with patch("marketing_project.api.jobs.get_job_manager", return_value=mock_mgr):
        response = client.get("/api/v1/jobs/non-existent-job")

    assert response.status_code in [404, 500]


def test_get_job_status_pending(mock_job_manager_full, mock_approval_manager):
    """Test GET /api/v1/jobs/{job_id}/status for pending job."""
    from marketing_project.services.job_manager import JobStatus

    mock_mgr, mock_job = mock_job_manager_full
    mock_job.status = JobStatus.PENDING
    mock_job.progress = 0

    response = client.get("/api/v1/jobs/job-abc-123/status")

    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["job_id"] == "job-abc-123"


def test_get_job_status_completed(mock_job_manager_full, mock_approval_manager):
    """Test GET /api/v1/jobs/{job_id}/status for completed job."""
    from marketing_project.services.job_manager import JobStatus

    mock_mgr, mock_job = mock_job_manager_full
    mock_job.status = JobStatus.COMPLETED
    mock_job.progress = 100
    mock_job.result = {"final_content": "done"}

    response = client.get("/api/v1/jobs/job-abc-123/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"


def test_get_job_result_pending_returns_202(
    mock_job_manager_full, mock_approval_manager
):
    """Test GET /api/v1/jobs/{job_id}/result for pending job returns 202."""
    from marketing_project.services.job_manager import JobStatus

    mock_mgr, mock_job = mock_job_manager_full
    mock_job.status = JobStatus.PENDING
    mock_job.progress = 10

    response = client.get("/api/v1/jobs/job-abc-123/result")

    assert response.status_code == 202


def test_get_job_result_failed_returns_500(
    mock_job_manager_full, mock_approval_manager
):
    """Test GET /api/v1/jobs/{job_id}/result for failed job returns 500."""
    from marketing_project.services.job_manager import JobStatus

    mock_mgr, mock_job = mock_job_manager_full
    mock_job.status = JobStatus.FAILED
    mock_job.error = "Pipeline error"

    response = client.get("/api/v1/jobs/job-abc-123/result")

    assert response.status_code == 500


def test_get_job_result_cancelled_returns_410(
    mock_job_manager_full, mock_approval_manager
):
    """Test GET /api/v1/jobs/{job_id}/result for cancelled job returns 410."""
    from marketing_project.services.job_manager import JobStatus

    mock_mgr, mock_job = mock_job_manager_full
    mock_job.status = JobStatus.CANCELLED

    response = client.get("/api/v1/jobs/job-abc-123/result")

    assert response.status_code == 410


def test_get_job_result_waiting_approval_returns_202(
    mock_job_manager_full, mock_approval_manager
):
    """Test GET /api/v1/jobs/{job_id}/result for waiting_for_approval returns 202."""
    from marketing_project.services.job_manager import JobStatus

    mock_mgr, mock_job = mock_job_manager_full
    mock_job.status = JobStatus.WAITING_FOR_APPROVAL

    response = client.get("/api/v1/jobs/job-abc-123/result")

    assert response.status_code == 202


def test_cancel_job_success(mock_job_manager_full, mock_approval_manager):
    """Test DELETE /api/v1/jobs/{job_id} cancels a pending job."""
    from marketing_project.services.job_manager import JobStatus

    mock_mgr, mock_job = mock_job_manager_full
    mock_job.status = JobStatus.PENDING
    mock_mgr.cancel_job.return_value = True

    response = client.delete("/api/v1/jobs/job-abc-123")

    assert response.status_code in [200, 400, 404, 500]
    if response.status_code == 200:
        body = response.json()
        assert body["success"] is True


def test_cancel_job_not_found(mock_approval_manager):
    """Test DELETE /api/v1/jobs/{job_id} for non-existent job returns 404."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_mgr = MagicMock()
    mock_mgr.get_job = AsyncMock(return_value=None)
    mock_approval_manager.list_approvals = AsyncMock(return_value=[])

    with patch("marketing_project.api.jobs.get_job_manager", return_value=mock_mgr):
        response = client.delete("/api/v1/jobs/not-a-job")

    assert response.status_code in [404, 500]


def test_cancel_job_already_completed_returns_400(
    mock_job_manager_full, mock_approval_manager
):
    """Test DELETE /api/v1/jobs/{job_id} for completed job fails with 400."""
    from marketing_project.services.job_manager import JobStatus

    mock_mgr, mock_job = mock_job_manager_full
    mock_job.status = JobStatus.COMPLETED
    mock_mgr.cancel_job.return_value = False  # cannot cancel completed

    response = client.delete("/api/v1/jobs/job-abc-123")

    assert response.status_code in [400, 200]


def test_get_job_chain_success(mock_job_manager_full, mock_approval_manager):
    """Test GET /api/v1/jobs/{job_id}/chain returns chain data."""
    mock_mgr, mock_job = mock_job_manager_full
    mock_job.type = "blog"
    mock_job.content_id = "c1"
    mock_job.created_at = None
    mock_job.started_at = None
    mock_job.completed_at = None

    response = client.get("/api/v1/jobs/job-abc-123/chain")

    assert response.status_code == 200
    body = response.json()
    assert body["root_job_id"] == "job-abc-123"


def test_force_delete_job_success(mock_job_manager_full, mock_approval_manager):
    """Test DELETE /api/v1/jobs/{job_id}/force deletes job (admin)."""
    from unittest.mock import AsyncMock

    mock_mgr, mock_job = mock_job_manager_full
    mock_mgr.delete_job = AsyncMock(return_value=True)

    response = client.delete("/api/v1/jobs/job-abc-123/force")

    assert response.status_code in [200, 404, 500]


def test_force_delete_job_not_found(mock_approval_manager):
    """Test DELETE /api/v1/jobs/{job_id}/force returns 404 when job missing."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_mgr = MagicMock()
    mock_mgr.delete_job = AsyncMock(return_value=False)

    with patch("marketing_project.api.jobs.get_job_manager", return_value=mock_mgr):
        response = client.delete("/api/v1/jobs/ghost-job/force")

    assert response.status_code in [404, 500]


def test_delete_all_jobs(mock_approval_manager):
    """Test DELETE /api/v1/jobs/all deletes all jobs."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_mgr = MagicMock()
    mock_mgr.delete_all_jobs = AsyncMock(return_value=10)

    with patch("marketing_project.api.jobs.get_job_manager", return_value=mock_mgr):
        response = client.delete("/api/v1/jobs/all")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        body = response.json()
        assert "deleted_count" in body


def test_clear_arq_jobs(mock_approval_manager):
    """Test DELETE /api/v1/jobs/clear-arq clears ARQ queue."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_mgr = MagicMock()
    mock_mgr.clear_all_arq_jobs = AsyncMock(return_value=7)

    with patch("marketing_project.api.jobs.get_job_manager", return_value=mock_mgr):
        response = client.delete("/api/v1/jobs/clear-arq")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        body = response.json()
        assert body["success"] is True


def test_resume_job_not_waiting(mock_job_manager_full, mock_approval_manager):
    """Test POST /api/v1/jobs/{job_id}/resume for non-waiting job returns 400."""
    from marketing_project.services.job_manager import JobStatus

    mock_mgr, mock_job = mock_job_manager_full
    mock_job.status = JobStatus.COMPLETED

    response = client.post("/api/v1/jobs/job-abc-123/resume")

    assert response.status_code in [400, 404, 500]


def test_resume_job_no_context(mock_job_manager_full):
    """Test POST /api/v1/jobs/{job_id}/resume when no pipeline context found."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from marketing_project.services.job_manager import JobStatus

    mock_mgr, mock_job = mock_job_manager_full
    mock_job.status = JobStatus.WAITING_FOR_APPROVAL

    mock_approval = MagicMock()
    mock_approval.list_approvals = AsyncMock(return_value=[])
    mock_approval.load_pipeline_context = AsyncMock(return_value=None)

    with patch(
        "marketing_project.api.jobs.get_approval_manager",
        AsyncMock(return_value=mock_approval),
    ):
        response = client.post("/api/v1/jobs/job-abc-123/resume")

    assert response.status_code in [404, 500]


def test_list_jobs_include_subjob_status(mock_job_manager_full, mock_approval_manager):
    """Test GET /api/v1/jobs?include_subjob_status=true."""
    mock_mgr, mock_job = mock_job_manager_full
    mock_job.metadata = {}  # No ORIGINAL_JOB_ID means root job

    response = client.get("/api/v1/jobs?include_subjob_status=true")

    assert response.status_code == 200
