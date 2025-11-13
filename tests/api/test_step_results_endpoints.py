"""
Tests for Step Results API endpoints.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.step_results import router


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


@pytest.fixture
def sample_step_info():
    """Sample step info for testing."""
    return {
        "filename": "01_seo_keywords.json",
        "step_number": 1,
        "step_name": "SEO Keywords",
        "timestamp": "2024-01-01T00:00:00Z",
        "has_result": True,
        "file_size": 1024,
        "job_id": "test-job-123",
        "root_job_id": "test-job-123",
        "status": "completed",
    }


@pytest.mark.asyncio
class TestStepResultsAPI:
    """Test suite for Step Results API endpoints."""

    async def test_list_jobs_success(self, client):
        """Test successful job listing with results."""
        with patch(
            "marketing_project.api.step_results.get_step_result_manager"
        ) as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.list_all_jobs = AsyncMock(
                return_value=[
                    {
                        "job_id": "test-job-123",
                        "content_type": "blog_post",
                        "step_count": 5,
                    }
                ]
            )

            response = client.get("/api/v1/results/jobs")
            assert response.status_code == 200
            data = response.json()
            assert len(data["jobs"]) == 1
            assert data["total"] == 1

    async def test_list_jobs_with_limit(self, client):
        """Test job listing with limit parameter."""
        with patch(
            "marketing_project.api.step_results.get_step_result_manager"
        ) as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.list_all_jobs = AsyncMock(return_value=[])

            response = client.get("/api/v1/results/jobs?limit=10")
            assert response.status_code == 200

    async def test_list_jobs_error(self, client):
        """Test job listing error handling."""
        with patch(
            "marketing_project.api.step_results.get_step_result_manager"
        ) as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.list_all_jobs = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = client.get("/api/v1/results/jobs")
            assert response.status_code == 500

    async def test_get_job_results_success(self, client, sample_step_info):
        """Test successful job results retrieval."""
        with patch(
            "marketing_project.api.step_results.get_step_result_manager"
        ) as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job_results = AsyncMock(
                return_value={
                    "job_id": "test-job-123",
                    "metadata": {},
                    "steps": [sample_step_info],
                    "total_steps": 1,
                }
            )

            response = client.get("/api/v1/results/jobs/test-job-123")
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == "test-job-123"
            assert len(data["steps"]) == 1

    async def test_get_job_results_with_filters(self, client, sample_step_info):
        """Test job results with filters."""
        with patch(
            "marketing_project.api.step_results.get_step_result_manager"
        ) as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job_results = AsyncMock(
                return_value={
                    "job_id": "test-job-123",
                    "metadata": {},
                    "steps": [sample_step_info],
                    "total_steps": 1,
                }
            )

            response = client.get(
                "/api/v1/results/jobs/test-job-123?filter_by_status=completed"
            )
            assert response.status_code == 200

    async def test_get_job_results_not_found(self, client):
        """Test job results when not found."""
        with patch(
            "marketing_project.api.step_results.get_step_result_manager"
        ) as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_job_results = AsyncMock(
                side_effect=FileNotFoundError("Job not found")
            )

            response = client.get("/api/v1/results/jobs/non-existent")
            assert response.status_code == 404

    async def test_get_step_result_success(self, client):
        """Test successful step result retrieval."""
        with patch(
            "marketing_project.api.step_results.get_step_result_manager"
        ) as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_step_result = AsyncMock(
                return_value={"keywords": ["test", "example"]}
            )

            response = client.get(
                "/api/v1/results/jobs/test-job-123/steps/01_seo_keywords.json"
            )
            assert response.status_code == 200
            data = response.json()
            assert "keywords" in data

    async def test_get_step_result_not_found(self, client):
        """Test step result when not found."""
        with patch(
            "marketing_project.api.step_results.get_step_result_manager"
        ) as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_step_result = AsyncMock(
                side_effect=FileNotFoundError("Step not found")
            )

            response = client.get(
                "/api/v1/results/jobs/test-job-123/steps/non-existent.json"
            )
            assert response.status_code == 404

    async def test_download_step_result_success(self, client):
        """Test successful step result download."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_data = {"keywords": ["test", "example"]}
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            with patch(
                "marketing_project.api.step_results.get_step_result_manager"
            ) as mock_manager:
                mock_manager_instance = AsyncMock()
                mock_manager.return_value = mock_manager_instance
                mock_manager_instance.get_step_file_path = AsyncMock(
                    return_value=temp_path
                )

                response = client.get(
                    "/api/v1/results/jobs/test-job-123/steps/01_seo_keywords.json/download"
                )
                assert response.status_code == 200
                assert response.headers["content-type"] == "application/json"
        finally:
            temp_path.unlink(missing_ok=True)

    async def test_download_step_result_not_found(self, client):
        """Test step result download when not found."""
        with patch(
            "marketing_project.api.step_results.get_step_result_manager"
        ) as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.get_step_file_path = AsyncMock(
                side_effect=FileNotFoundError("Step not found")
            )

            response = client.get(
                "/api/v1/results/jobs/test-job-123/steps/non-existent.json/download"
            )
            assert response.status_code == 404

    async def test_delete_job_results_success(self, client):
        """Test successful deletion of job results."""
        with patch(
            "marketing_project.api.step_results.get_step_result_manager"
        ) as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.cleanup_job = AsyncMock(return_value=True)

            response = client.delete("/api/v1/results/jobs/test-job-123")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    async def test_delete_job_results_not_found(self, client):
        """Test deletion when job results not found."""
        with patch(
            "marketing_project.api.step_results.get_step_result_manager"
        ) as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.cleanup_job = AsyncMock(return_value=False)

            response = client.delete("/api/v1/results/jobs/non-existent")
            assert response.status_code == 404

    async def test_get_job_timeline_success(self, client):
        """Test successful job timeline retrieval."""
        with (
            patch(
                "marketing_project.api.step_results.get_step_result_manager"
            ) as mock_step_manager,
            patch(
                "marketing_project.services.job_manager.get_job_manager"
            ) as mock_job_manager,
            patch(
                "marketing_project.services.approval_manager.get_approval_manager"
            ) as mock_approval_manager,
        ):
            mock_job_manager_instance = AsyncMock()
            mock_job_manager.return_value = mock_job_manager_instance
            mock_job_manager_instance.get_job_chain = AsyncMock(
                return_value={
                    "root_job_id": "test-job-123",
                    "chain_length": 1,
                    "chain_order": ["test-job-123"],
                    "all_job_ids": ["test-job-123"],
                }
            )

            from datetime import datetime

            from marketing_project.services.job_manager import Job, JobStatus

            mock_job = Job(
                id="test-job-123",
                type="blog_post",
                content_id="content-123",
                status=JobStatus.COMPLETED,
                created_at=datetime.utcnow(),
            )
            mock_job_manager_instance.get_job = AsyncMock(return_value=mock_job)

            mock_step_manager_instance = AsyncMock()
            mock_step_manager.return_value = mock_step_manager_instance
            mock_step_manager_instance.aggregate_steps_from_jobs = AsyncMock(
                return_value=[]
            )

            mock_approval_manager_instance = AsyncMock()
            mock_approval_manager.return_value = mock_approval_manager_instance
            mock_approval_manager_instance.list_approvals = AsyncMock(return_value=[])

            response = client.get("/api/v1/results/jobs/test-job-123/timeline")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "events" in data
