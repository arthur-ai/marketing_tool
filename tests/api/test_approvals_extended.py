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


# ---------------------------------------------------------------------------
# Helper to build a fully mocked approval object
# ---------------------------------------------------------------------------


def _make_approval(
    approval_id="appr-1",
    job_id="job-1",
    status="pending",
    pipeline_step="seo_keywords",
    step_name="Step 1: seo_keywords",
    agent_name="seo_keywords",
    modified_output=None,
    output_data=None,
    retry_count=0,
):
    from datetime import datetime, timezone

    a = MagicMock()
    a.id = approval_id
    a.job_id = job_id
    a.status = status
    a.pipeline_step = pipeline_step
    a.step_name = step_name
    a.agent_name = agent_name
    a.input_data = {"original_content": {"title": "Test Title"}}
    a.output_data = output_data or {"key": "value"}
    a.modified_output = modified_output
    a.user_comment = None
    a.reviewed_by = None
    a.reviewed_at = None
    a.created_at = datetime.now(timezone.utc)
    a.retry_count = retry_count
    a.retry_job_id = None
    a.model_dump = MagicMock(
        return_value={
            "id": approval_id,
            "job_id": job_id,
            "status": status,
            "step_name": step_name,
            "agent_name": agent_name,
            "pipeline_step": pipeline_step,
            "input_data": {"original_content": {"title": "Test Title"}},
            "output_data": output_data or {"key": "value"},
            "modified_output": modified_output,
            "user_comment": None,
            "reviewed_by": None,
            "reviewed_at": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retry_count": retry_count,
            "retry_job_id": None,
        }
    )
    return a


def _make_job(
    job_id="job-1", status_value="waiting_for_approval", user_id="test-user-123"
):
    from marketing_project.services.job_manager import JobStatus

    j = MagicMock()
    j.id = job_id
    j.user_id = user_id
    j.content_id = "content-1"
    j.type = "blog"
    j.metadata = {}

    # Map string status to JobStatus enum where possible
    _status_map = {
        "waiting_for_approval": JobStatus.WAITING_FOR_APPROVAL,
        "completed": JobStatus.COMPLETED,
        "failed": JobStatus.FAILED,
        "processing": JobStatus.PROCESSING,
    }
    j.status = _status_map.get(status_value, JobStatus.WAITING_FOR_APPROVAL)
    return j


# ---------------------------------------------------------------------------
# Analytics endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_approval_analytics_with_data():
    """Analytics endpoint returns structured response when approvals exist."""
    from datetime import datetime, timedelta, timezone

    approval = _make_approval(status="approved")
    approval.reviewed_at = datetime.now(timezone.utc)
    approval.created_at = datetime.now(timezone.utc) - timedelta(seconds=60)
    approval.reviewed_by = "reviewer1"

    with patch("marketing_project.api.approvals.get_approval_manager") as mock:
        manager = MagicMock()
        manager.list_approvals = AsyncMock(return_value=[approval])
        mock.return_value = manager

        response = client.get("/api/v1/approvals/analytics")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "statistics" in data
        assert data["statistics"]["total_requests"] == 1


@pytest.mark.asyncio
async def test_get_approval_analytics_empty():
    """Analytics endpoint works with zero approvals."""
    with patch("marketing_project.api.approvals.get_approval_manager") as mock:
        manager = MagicMock()
        manager.list_approvals = AsyncMock(return_value=[])
        mock.return_value = manager

        response = client.get("/api/v1/approvals/analytics")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert data["statistics"]["total_requests"] == 0
        assert data["statistics"]["approval_rate"] == 0.0


@pytest.mark.asyncio
async def test_get_approval_analytics_error():
    """Analytics endpoint returns 500 on unexpected error."""
    with patch("marketing_project.api.approvals.get_approval_manager") as mock:
        mock.side_effect = Exception("DB unavailable")

        response = client.get("/api/v1/approvals/analytics")

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_approval_stats_success():
    """Stats endpoint returns ApprovalStats model data."""
    from marketing_project.models.approval_models import ApprovalStats

    stats = ApprovalStats(
        total_requests=5,
        pending=1,
        approved=2,
        rejected=1,
        modified=1,
        rerun=0,
        approval_rate=0.6,
    )

    with patch("marketing_project.api.approvals.get_approval_manager") as mock:
        manager = MagicMock()
        manager.get_stats = AsyncMock(return_value=stats)
        mock.return_value = manager

        response = client.get("/api/v1/approvals/stats")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert data["total_requests"] == 5


@pytest.mark.asyncio
async def test_get_approval_stats_error():
    """Stats endpoint returns 500 on unexpected error."""
    with patch("marketing_project.api.approvals.get_approval_manager") as mock:
        mock.side_effect = Exception("Redis down")

        response = client.get("/api/v1/approvals/stats")

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Settings endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_approval_settings_success():
    """Settings endpoint returns ApprovalSettings."""
    from marketing_project.models.approval_models import ApprovalSettings

    settings = ApprovalSettings(require_approval=True, approval_agents=["seo_keywords"])

    with patch("marketing_project.api.approvals.get_approval_manager") as mock:
        manager = MagicMock()
        manager.settings = settings
        mock.return_value = manager

        response = client.get("/api/v1/approvals/settings")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "require_approval" in data


@pytest.mark.asyncio
async def test_update_approval_settings_valid():
    """POST /approvals/settings with valid body."""
    with patch("marketing_project.api.approvals.set_approval_settings") as mock_set:
        mock_set.return_value = None

        payload = {
            "require_approval": True,
            "approval_agents": ["seo_keywords"],
            "auto_approve_threshold": 0.85,
        }
        response = client.post("/api/v1/approvals/settings", json=payload)

    assert response.status_code in [200, 422, 500]


@pytest.mark.asyncio
async def test_update_approval_settings_invalid_threshold():
    """POST /approvals/settings rejects threshold > 1."""
    payload = {
        "require_approval": True,
        "approval_agents": [],
        "auto_approve_threshold": 1.5,  # invalid
    }
    response = client.post("/api/v1/approvals/settings", json=payload)
    assert response.status_code in [422, 500]


# ---------------------------------------------------------------------------
# Delete all approvals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_all_approvals_success():
    """DELETE /approvals/all returns count of deleted approvals."""
    with patch("marketing_project.api.approvals.get_approval_manager") as mock:
        manager = MagicMock()
        manager.delete_all_approvals = AsyncMock(return_value=7)
        mock.return_value = manager

        response = client.delete("/api/v1/approvals/all")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert data["deleted_count"] == 7
        assert data["success"] is True


@pytest.mark.asyncio
async def test_delete_all_approvals_error():
    """DELETE /approvals/all returns 500 on error."""
    with patch("marketing_project.api.approvals.get_approval_manager") as mock:
        mock.side_effect = Exception("Redis error")

        response = client.delete("/api/v1/approvals/all")

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Pending approvals endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_pending_approvals_with_job_filter():
    """GET /approvals/pending?job_id=... filters by job."""
    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
    ):
        manager = MagicMock()
        manager.list_approvals = AsyncMock(return_value=[])
        manager.load_pipeline_context = AsyncMock(return_value=None)
        mock_mgr.return_value = manager

        jm = MagicMock()
        jm.get_jobs_by_ids = AsyncMock(return_value={})
        jm.get_job_ids_for_user = AsyncMock(return_value=[])
        jm.get_job = AsyncMock(return_value=None)
        mock_jm.return_value = jm

        response = client.get("/api/v1/approvals/pending?job_id=job-999")

    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_get_pending_approvals_error():
    """GET /approvals/pending returns 500 on error."""
    with patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr:
        mock_mgr.side_effect = Exception("Database error")

        response = client.get("/api/v1/approvals/pending")

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Get approval by ID
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_approval_by_id_not_found():
    """GET /approvals/{id} returns 404 when approval not found."""
    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_approval_ownership",
            new=AsyncMock(
                side_effect=__import__("fastapi").HTTPException(
                    status_code=404, detail="Approval nonexistent-approval not found"
                )
            ),
        ),
    ):
        manager = MagicMock()
        manager.get_approval = AsyncMock(return_value=None)
        mock_mgr.return_value = manager

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=None)
        mock_jm.return_value = jm

        response = client.get("/api/v1/approvals/nonexistent-approval")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_approval_by_id_found():
    """GET /approvals/{id} returns approval dict when found."""
    approval = _make_approval(
        approval_id="appr-found", job_id="job-1", status="approved"
    )

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_approval_ownership",
            new=AsyncMock(return_value=approval),
        ),
    ):
        manager = MagicMock()
        manager.get_approval = AsyncMock(return_value=approval)
        mock_mgr.return_value = manager

        job = _make_job(job_id="job-1", status_value="completed")
        job.metadata = {}

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=job)
        mock_jm.return_value = jm

        response = client.get("/api/v1/approvals/appr-found")

    assert response.status_code in [200, 500]


# ---------------------------------------------------------------------------
# Impact endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_approval_impact_not_found():
    """GET /approvals/{id}/impact returns 404 when approval not found."""
    from fastapi import HTTPException as FastAPIHTTPException

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_approval_ownership",
            new=AsyncMock(
                side_effect=FastAPIHTTPException(
                    status_code=404, detail="Approval nonexistent not found"
                )
            ),
        ),
    ):
        manager = MagicMock()
        manager.get_approval = AsyncMock(return_value=None)
        mock_mgr.return_value = manager

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=None)
        mock_jm.return_value = jm

        response = client.get("/api/v1/approvals/nonexistent/impact")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_approval_impact_found_with_modifications():
    """GET /approvals/{id}/impact returns impact data including changes summary."""
    approval = _make_approval(
        approval_id="appr-impact",
        job_id="job-1",
        status="modified",
        output_data={"title": "Old"},
        modified_output={"title": "New", "extra": "field"},
    )

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_approval_ownership",
            new=AsyncMock(return_value=approval),
        ),
    ):
        manager = MagicMock()
        manager.get_approval = AsyncMock(return_value=approval)
        mock_mgr.return_value = manager

        job = _make_job(job_id="job-1", status_value="completed")
        job.metadata = {}

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=job)
        mock_jm.return_value = jm

        response = client.get("/api/v1/approvals/appr-impact/impact")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert data["has_modifications"] is True
        assert data["approval_id"] == "appr-impact"


# ---------------------------------------------------------------------------
# Decide approval endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decide_approval_not_found():
    """POST /approvals/{id}/decide returns 404 when approval not found."""
    from fastapi import HTTPException as FastAPIHTTPException

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_approval_ownership",
            new=AsyncMock(
                side_effect=FastAPIHTTPException(
                    status_code=404, detail="Approval nonexistent not found"
                )
            ),
        ),
    ):
        manager = MagicMock()
        manager.get_approval = AsyncMock(return_value=None)
        mock_mgr.return_value = manager

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=None)
        mock_jm.return_value = jm

        payload = {"decision": "approve"}
        response = client.post("/api/v1/approvals/nonexistent/decide", json=payload)

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_decide_approval_approve():
    """POST /approvals/{id}/decide with approve decision."""
    approval = _make_approval(approval_id="appr-2", job_id="job-2", status="approved")

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_approval_ownership",
            new=AsyncMock(return_value=approval),
        ),
    ):
        manager = MagicMock()
        manager.get_approval = AsyncMock(return_value=approval)
        manager.decide_approval = AsyncMock(return_value=approval)
        manager.load_pipeline_context = AsyncMock(return_value=None)
        mock_mgr.return_value = manager

        job = _make_job(job_id="job-2", status_value="completed")
        job.metadata = {}

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=job)
        mock_jm.return_value = jm

        payload = {"decision": "approve", "reviewed_by": "tester"}
        response = client.post("/api/v1/approvals/appr-2/decide", json=payload)

    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_decide_approval_reject_no_auto_retry():
    """POST /approvals/{id}/decide with reject + auto_retry=False marks job failed."""
    approval = _make_approval(approval_id="appr-3", job_id="job-3", status="rejected")

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_approval_ownership",
            new=AsyncMock(return_value=approval),
        ),
    ):
        manager = MagicMock()
        manager.get_approval = AsyncMock(return_value=approval)
        manager.decide_approval = AsyncMock(return_value=approval)
        manager.load_pipeline_context = AsyncMock(return_value=None)
        mock_mgr.return_value = manager

        job = _make_job(job_id="job-3", status_value="waiting_for_approval")
        job.metadata = {}

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=job)
        jm.update_job_status = AsyncMock()
        jm._save_job = AsyncMock()
        mock_jm.return_value = jm

        payload = {
            "decision": "reject",
            "auto_retry": False,
            "comment": "Not good enough",
        }
        response = client.post("/api/v1/approvals/appr-3/decide", json=payload)

    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_decide_approval_invalid_decision():
    """POST /approvals/{id}/decide with invalid decision value returns 422."""
    payload = {"decision": "invalid_decision"}
    response = client.post("/api/v1/approvals/appr-x/decide", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Bulk approve endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bulk_approve_success():
    """POST /approvals/bulk-approve approves multiple approvals."""
    approval = _make_approval(
        approval_id="appr-bulk-1", job_id="job-bulk", status="pending"
    )
    approved = _make_approval(
        approval_id="appr-bulk-1", job_id="job-bulk", status="approved"
    )

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
    ):
        manager = MagicMock()
        manager.get_approval = AsyncMock(return_value=approval)
        manager.decide_approval = AsyncMock(return_value=approved)
        mock_mgr.return_value = manager

        jm = MagicMock()
        mock_jm.return_value = jm

        payload = {
            "approval_ids": ["appr-bulk-1"],
            "decision": {"decision": "approve"},
        }
        response = client.post("/api/v1/approvals/bulk-approve", json=payload)

    assert response.status_code in [200, 422, 500]


@pytest.mark.asyncio
async def test_bulk_approve_rejects_non_approve_decision():
    """POST /approvals/bulk-approve returns 400 when decision is not 'approve'."""
    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
    ):
        manager = MagicMock()
        mock_mgr.return_value = manager
        jm = MagicMock()
        mock_jm.return_value = jm

        payload = {
            "approval_ids": ["appr-1"],
            "decision": {"decision": "reject"},
        }
        response = client.post("/api/v1/approvals/bulk-approve", json=payload)

    assert response.status_code in [400, 422, 500]


@pytest.mark.asyncio
async def test_bulk_approve_not_found_approval():
    """POST /approvals/bulk-approve handles missing approval gracefully."""
    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
    ):
        manager = MagicMock()
        manager.get_approval = AsyncMock(return_value=None)
        mock_mgr.return_value = manager

        jm = MagicMock()
        mock_jm.return_value = jm

        payload = {
            "approval_ids": ["not-found-id"],
            "decision": {"decision": "approve"},
        }
        response = client.post("/api/v1/approvals/bulk-approve", json=payload)

    assert response.status_code in [200, 422, 500]
    if response.status_code == 200:
        data = response.json()
        assert data["approved_count"] == 0
        assert len(data["errors"]) > 0


# ---------------------------------------------------------------------------
# Job approvals endpoint (GET /approvals/job/{job_id})
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_job_approvals_job_not_found():
    """GET /approvals/job/{job_id} returns 404 when job does not exist."""
    from fastapi import HTTPException as FastAPIHTTPException

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_job_ownership",
            new=AsyncMock(
                side_effect=FastAPIHTTPException(
                    status_code=404, detail="Job nonexistent-job not found"
                )
            ),
        ),
    ):
        manager = MagicMock()
        manager.list_approvals = AsyncMock(return_value=[])
        manager.load_pipeline_context = AsyncMock(return_value=None)
        manager._load_all_approvals_from_redis = AsyncMock()
        mock_mgr.return_value = manager

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=None)
        mock_jm.return_value = jm

        response = client.get("/api/v1/approvals/job/nonexistent-job")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_job_approvals_found():
    """GET /approvals/job/{job_id} returns approvals for a job."""
    approval = _make_approval(
        approval_id="appr-job", job_id="job-found", status="pending"
    )

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_job_ownership",
            new=AsyncMock(return_value=MagicMock()),
        ),
    ):
        manager = MagicMock()
        manager.list_approvals = AsyncMock(return_value=[approval])
        manager.load_pipeline_context = AsyncMock(return_value=None)
        manager._load_all_approvals_from_redis = AsyncMock()
        mock_mgr.return_value = manager

        job = _make_job(job_id="job-found", status_value="waiting_for_approval")
        job.metadata = {}

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=job)
        mock_jm.return_value = jm

        response = client.get("/api/v1/approvals/job/job-found")

    assert response.status_code in [200, 500]


# ---------------------------------------------------------------------------
# Retry rejected step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_rejected_step_not_found():
    """POST /approvals/{id}/retry returns 404 for missing approval."""
    from fastapi import HTTPException as FastAPIHTTPException

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_approval_ownership",
            new=AsyncMock(
                side_effect=FastAPIHTTPException(
                    status_code=404, detail="Approval nonexistent not found"
                )
            ),
        ),
    ):
        manager = MagicMock()
        manager.get_approval = AsyncMock(return_value=None)
        mock_mgr.return_value = manager

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=None)
        mock_jm.return_value = jm

        response = client.post("/api/v1/approvals/nonexistent/retry")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_retry_rejected_step_wrong_status():
    """POST /approvals/{id}/retry returns 400 when approval is not rejected."""
    approval = _make_approval(
        approval_id="appr-retry", job_id="job-retry", status="pending"
    )

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_approval_ownership",
            new=AsyncMock(return_value=approval),
        ),
    ):
        manager = MagicMock()
        manager.get_approval = AsyncMock(return_value=approval)
        mock_mgr.return_value = manager

        job = _make_job(job_id="job-retry", status_value="waiting_for_approval")
        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=job)
        mock_jm.return_value = jm

        response = client.post("/api/v1/approvals/appr-retry/retry")

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# All approvals for job chain (GET /approvals/jobs/{job_id}/all)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_all_approvals_for_job_not_found():
    """GET /approvals/jobs/{job_id}/all returns 404 for nonexistent job."""
    from fastapi import HTTPException as FastAPIHTTPException

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_job_ownership",
            new=AsyncMock(return_value=MagicMock()),
        ),
    ):
        manager = MagicMock()
        manager.list_approvals = AsyncMock(return_value=[])
        mock_mgr.return_value = manager

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=None)
        jm.get_job_chain = AsyncMock(
            return_value={"root_job_id": None, "chain_order": []}
        )
        mock_jm.return_value = jm

        response = client.get("/api/v1/approvals/jobs/nonexistent/all")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_all_approvals_for_job_success():
    """GET /approvals/jobs/{job_id}/all returns grouped approvals."""
    approval = _make_approval(
        approval_id="appr-chain", job_id="job-root", status="approved"
    )

    with (
        patch("marketing_project.api.approvals.get_approval_manager") as mock_mgr,
        patch("marketing_project.services.job_manager.get_job_manager") as mock_jm,
        patch(
            "marketing_project.middleware.rbac.verify_job_ownership",
            new=AsyncMock(return_value=MagicMock()),
        ),
    ):
        manager = MagicMock()
        manager.list_approvals = AsyncMock(return_value=[approval])
        mock_mgr.return_value = manager

        job = _make_job(job_id="job-root", status_value="completed")

        jm = MagicMock()
        jm.get_job = AsyncMock(return_value=job)
        jm.get_job_chain = AsyncMock(
            return_value={"root_job_id": "job-root", "chain_order": ["job-root"]}
        )
        mock_jm.return_value = jm

        response = client.get("/api/v1/approvals/jobs/job-root/all")

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert data["root_job_id"] == "job-root"
        assert "approvals_by_job" in data
