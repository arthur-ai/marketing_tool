"""
Tests for Jobs API endpoints.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.jobs import router
from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.services.job_manager import Job, JobStatus
from tests.utils.keycloak_test_helpers import create_user_context


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/jobs")
    mock_user = create_user_context(roles=["admin"])
    app.dependency_overrides[get_current_user] = lambda: mock_user
    return TestClient(app)


@pytest.fixture
def sample_job():
    """Sample job for testing."""
    return Job(
        id="test-job-123",
        type="blog_post",
        content_id="content-123",
        status=JobStatus.COMPLETED,
        created_at=datetime.utcnow(),
        progress=100,
        result={"output": "test result"},
    )


class TestJobsAPI:
    """Test suite for Jobs API endpoints."""

    def test_list_jobs_success(self, client, sample_job):
        """Test successful job listing."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.list_jobs = AsyncMock(return_value=[sample_job])

            response = client.get("/api/v1/jobs")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["jobs"]) == 1
            assert data["total"] == 1

    def test_list_jobs_with_filters(self, client, sample_job):
        """Test job listing with filters."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.list_jobs = AsyncMock(return_value=[sample_job])

            response = client.get("/api/v1/jobs?status=completed&limit=10")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_list_jobs_error(self, client):
        """Test job listing error handling."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.list_jobs = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = client.get("/api/v1/jobs")
            assert response.status_code == 500

    def test_get_job_success(self, client, sample_job):
        """Test successful job retrieval."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job = AsyncMock(return_value=sample_job)

            response = client.get("/api/v1/jobs/test-job-123")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["job"]["id"] == "test-job-123"

    def test_get_job_not_found(self, client):
        """Test job retrieval when job not found."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job = AsyncMock(return_value=None)

            response = client.get("/api/v1/jobs/non-existent")
            assert response.status_code == 404

    def test_get_job_status_success(self, client, sample_job):
        """Test successful job status retrieval."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job = AsyncMock(return_value=sample_job)

            response = client.get("/api/v1/jobs/test-job-123/status")
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "test-job-123"
            assert data["status"] == "completed"
            assert data["progress"] == 100

    def test_get_job_status_not_found(self, client):
        """Test job status when job not found."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job = AsyncMock(return_value=None)

            response = client.get("/api/v1/jobs/non-existent/status")
            assert response.status_code == 404

    def test_get_job_result_success(self, client, sample_job):
        """Test successful job result retrieval."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job = AsyncMock(return_value=sample_job)

            response = client.get("/api/v1/jobs/test-job-123/result")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["job_id"] == "test-job-123"

    def test_get_job_result_pending(self, client):
        """Test job result when job is still pending."""
        pending_job = Job(
            id="pending-job",
            type="blog_post",
            content_id="content-123",
            status=JobStatus.PROCESSING,
            created_at=datetime.utcnow(),
            progress=50,
        )

        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job = AsyncMock(return_value=pending_job)

            response = client.get("/api/v1/jobs/pending-job/result")
            assert response.status_code == 202

    def test_get_job_result_failed(self, client):
        """Test job result when job failed."""
        failed_job = Job(
            id="failed-job",
            type="blog_post",
            content_id="content-123",
            status=JobStatus.FAILED,
            created_at=datetime.utcnow(),
            error="Test error",
        )

        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job = AsyncMock(return_value=failed_job)

            response = client.get("/api/v1/jobs/failed-job/result")
            assert response.status_code == 500

    def test_cancel_job_success(self, client):
        """Test successful job cancellation."""
        processing_job = Job(
            id="processing-job",
            type="blog_post",
            content_id="content-123",
            status=JobStatus.PROCESSING,
            created_at=datetime.utcnow(),
        )

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_job_manager,
            patch(
                "marketing_project.api.jobs.get_approval_manager"
            ) as mock_approval_manager,
        ):
            mock_job_manager_instance = AsyncMock()
            mock_job_manager.return_value = mock_job_manager_instance
            mock_job_manager_instance.get_job = AsyncMock(return_value=processing_job)
            mock_job_manager_instance.cancel_job = AsyncMock(return_value=True)

            mock_approval_manager_instance = AsyncMock()
            mock_approval_manager.return_value = mock_approval_manager_instance
            mock_approval_manager_instance.list_approvals = AsyncMock(return_value=[])

            response = client.delete("/api/v1/jobs/processing-job")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_cancel_job_not_found(self, client):
        """Test job cancellation when job not found."""
        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_job_manager,
            patch(
                "marketing_project.api.jobs.get_approval_manager"
            ) as mock_approval_manager,
        ):
            mock_job_manager_instance = AsyncMock()
            mock_job_manager.return_value = mock_job_manager_instance
            mock_job_manager_instance.get_job = AsyncMock(return_value=None)

            mock_approval_manager_instance = AsyncMock()
            mock_approval_manager.return_value = mock_approval_manager_instance
            mock_approval_manager_instance.list_approvals = AsyncMock(return_value=[])

            response = client.delete("/api/v1/jobs/non-existent")
            assert response.status_code == 404

    def test_cancel_job_already_completed(self, client, sample_job):
        """Test job cancellation when job already completed."""
        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_job_manager,
            patch(
                "marketing_project.api.jobs.get_approval_manager"
            ) as mock_approval_manager,
        ):
            mock_job_manager_instance = AsyncMock()
            mock_job_manager.return_value = mock_job_manager_instance
            mock_job_manager_instance.get_job = AsyncMock(return_value=sample_job)
            mock_job_manager_instance.cancel_job = AsyncMock(return_value=False)

            mock_approval_manager_instance = AsyncMock()
            mock_approval_manager.return_value = mock_approval_manager_instance
            mock_approval_manager_instance.list_approvals = AsyncMock(return_value=[])

            response = client.delete("/api/v1/jobs/test-job-123")
            assert response.status_code == 400

    def test_get_job_chain_success(self, client, sample_job):
        """Test successful job chain retrieval."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job_chain = AsyncMock(
                return_value={
                    "root_job_id": "test-job-123",
                    "chain_length": 1,
                    "chain_order": ["test-job-123"],
                    "all_job_ids": ["test-job-123"],
                    "chain_status": "completed",
                    "jobs": [sample_job],
                }
            )

            response = client.get("/api/v1/jobs/test-job-123/chain")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["root_job_id"] == "test-job-123"

    def test_get_job_chain_not_found(self, client):
        """Test job chain when job not found."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job_chain = AsyncMock(
                return_value={"root_job_id": None}
            )

            response = client.get("/api/v1/jobs/non-existent/chain")
            assert response.status_code == 404

    def test_delete_all_jobs_success(self, client):
        """Test successful deletion of all jobs."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.delete_all_jobs = AsyncMock(return_value=5)

            response = client.delete("/api/v1/jobs/all")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["deleted_count"] == 5

    def test_clear_arq_jobs_success(self, client):
        """Test successful clearing of ARQ jobs."""
        with patch("marketing_project.api.jobs.get_job_manager") as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.clear_all_arq_jobs = AsyncMock(return_value=3)

            response = client.delete("/api/v1/jobs/clear-arq")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["deleted_count"] == 3
