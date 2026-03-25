"""
Unit tests for GET /api/v1/jobs/{job_id}/quality
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marketing_project.api.jobs import router
from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.services.job_manager import Job, JobStatus
from tests.utils.keycloak_test_helpers import create_user_context

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user():
    return create_user_context(roles=["user"])


@pytest.fixture
def client(mock_user):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/jobs")
    app.dependency_overrides[get_current_user] = lambda: mock_user
    return TestClient(app)


def _make_job(status=JobStatus.COMPLETED, metadata=None):
    return Job(
        id="job-q-1",
        type="blog_post",
        content_id="c-1",
        status=status,
        created_at=datetime.utcnow(),
        progress=100 if status == JobStatus.COMPLETED else 50,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJobQualityEndpoint:
    """Tests for GET /api/v1/jobs/{job_id}/quality"""

    # ------------------------------------------------------------------
    # 409 when job not completed
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "status",
        [JobStatus.PROCESSING, JobStatus.PENDING, JobStatus.FAILED, JobStatus.QUEUED],
    )
    def test_409_when_job_not_completed(self, client, status):
        """Endpoint returns 409 for any non-COMPLETED job status."""
        job = _make_job(status=status)

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_mgr,
            patch("marketing_project.api.jobs.verify_job_ownership", return_value=job),
        ):
            mock_mgr_instance = AsyncMock()
            mock_mgr.return_value = mock_mgr_instance

            response = client.get("/api/v1/jobs/job-q-1/quality")

        assert response.status_code == 409
        assert "completed" in response.json()["detail"].lower()

    # ------------------------------------------------------------------
    # 404 when step result missing
    # ------------------------------------------------------------------

    def test_404_when_formatting_result_missing(self, client):
        """Returns 404 when content_formatting step result is not found."""
        job = _make_job()

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_mgr,
            patch("marketing_project.api.jobs.verify_job_ownership", return_value=job),
            patch(
                "marketing_project.services.step_result_manager.get_step_result_manager"
            ) as mock_srm,
        ):
            mock_mgr.return_value = AsyncMock()
            mock_srm_instance = AsyncMock()
            mock_srm_instance.get_step_result_by_name = AsyncMock(return_value=None)
            mock_srm.return_value = mock_srm_instance

            response = client.get("/api/v1/jobs/job-q-1/quality")

        assert response.status_code == 404

    def test_404_when_formatted_text_is_empty(self, client):
        """Returns 404 when the step result contains no text content."""
        job = _make_job()
        formatting_result = {"result": {"formatted_markdown": "", "formatted_html": ""}}

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_mgr,
            patch("marketing_project.api.jobs.verify_job_ownership", return_value=job),
            patch(
                "marketing_project.services.step_result_manager.get_step_result_manager"
            ) as mock_srm,
        ):
            mock_mgr.return_value = AsyncMock()
            mock_srm_instance = AsyncMock()
            mock_srm_instance.get_step_result_by_name = AsyncMock(
                return_value=formatting_result
            )
            mock_srm.return_value = mock_srm_instance

            response = client.get("/api/v1/jobs/job-q-1/quality")

        assert response.status_code == 404

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_returns_quality_metrics_for_completed_job(self, client):
        """Returns correct metrics structure for a completed job."""
        sample_text = (
            "The quick brown fox jumps over the lazy dog. " * 20
        )  # enough words
        job = _make_job(metadata={"input_content": {"topic": "quick brown fox"}})
        formatting_result = {"result": {"formatted_markdown": sample_text}}

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_mgr,
            patch("marketing_project.api.jobs.verify_job_ownership", return_value=job),
            patch(
                "marketing_project.services.step_result_manager.get_step_result_manager"
            ) as mock_srm,
        ):
            mock_mgr.return_value = AsyncMock()
            mock_srm_instance = AsyncMock()
            mock_srm_instance.get_step_result_by_name = AsyncMock(
                return_value=formatting_result
            )
            mock_srm.return_value = mock_srm_instance

            response = client.get("/api/v1/jobs/job-q-1/quality")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["job_id"] == "job-q-1"
        assert "flesch_kincaid_grade" in data
        assert "keyword_match_pct" in data
        assert "word_count" in data
        assert "warnings" in data
        assert isinstance(data["warnings"], list)

    def test_uses_formatted_html_fallback(self, client):
        """Falls back to formatted_html when formatted_markdown is absent."""
        sample_text = "Sample content. " * 20
        job = _make_job()
        formatting_result = {"result": {"formatted_html": sample_text}}

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_mgr,
            patch("marketing_project.api.jobs.verify_job_ownership", return_value=job),
            patch(
                "marketing_project.services.step_result_manager.get_step_result_manager"
            ) as mock_srm,
        ):
            mock_mgr.return_value = AsyncMock()
            mock_srm_instance = AsyncMock()
            mock_srm_instance.get_step_result_by_name = AsyncMock(
                return_value=formatting_result
            )
            mock_srm.return_value = mock_srm_instance

            response = client.get("/api/v1/jobs/job-q-1/quality")

        assert response.status_code == 200
        assert response.json()["word_count"] > 0

    # ------------------------------------------------------------------
    # Warning generation
    # ------------------------------------------------------------------

    def test_warning_when_word_count_below_100(self, client):
        """Emits a warning when output is very short."""
        short_text = "Short output."  # well below 100 words
        job = _make_job()
        formatting_result = {"result": {"formatted_markdown": short_text}}

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_mgr,
            patch("marketing_project.api.jobs.verify_job_ownership", return_value=job),
            patch(
                "marketing_project.services.step_result_manager.get_step_result_manager"
            ) as mock_srm,
        ):
            mock_mgr.return_value = AsyncMock()
            mock_srm_instance = AsyncMock()
            mock_srm_instance.get_step_result_by_name = AsyncMock(
                return_value=formatting_result
            )
            mock_srm.return_value = mock_srm_instance

            response = client.get("/api/v1/jobs/job-q-1/quality")

        assert response.status_code == 200
        warnings = response.json()["warnings"]
        assert any("100 words" in w for w in warnings)

    def test_warning_when_keyword_match_below_50_pct(self, client):
        """Emits a warning when fewer than half the topic keywords appear."""
        # Text contains none of the topic words
        text = "Completely unrelated content about weather patterns. " * 10
        job = _make_job(
            metadata={
                "input_content": {"topic": "machine learning artificial intelligence"}
            }
        )
        formatting_result = {"result": {"formatted_markdown": text}}

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_mgr,
            patch("marketing_project.api.jobs.verify_job_ownership", return_value=job),
            patch(
                "marketing_project.services.step_result_manager.get_step_result_manager"
            ) as mock_srm,
        ):
            mock_mgr.return_value = AsyncMock()
            mock_srm_instance = AsyncMock()
            mock_srm_instance.get_step_result_by_name = AsyncMock(
                return_value=formatting_result
            )
            mock_srm.return_value = mock_srm_instance

            response = client.get("/api/v1/jobs/job-q-1/quality")

        assert response.status_code == 200
        warnings = response.json()["warnings"]
        assert any("keyword" in w.lower() for w in warnings)

    def test_no_keyword_warning_without_topic(self, client):
        """No keyword warning when the job has no topic metadata."""
        text = "Some output content. " * 10
        job = _make_job(metadata={})  # no input_content
        formatting_result = {"result": {"formatted_markdown": text}}

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_mgr,
            patch("marketing_project.api.jobs.verify_job_ownership", return_value=job),
            patch(
                "marketing_project.services.step_result_manager.get_step_result_manager"
            ) as mock_srm,
        ):
            mock_mgr.return_value = AsyncMock()
            mock_srm_instance = AsyncMock()
            mock_srm_instance.get_step_result_by_name = AsyncMock(
                return_value=formatting_result
            )
            mock_srm.return_value = mock_srm_instance

            response = client.get("/api/v1/jobs/job-q-1/quality")

        assert response.status_code == 200
        warnings = response.json()["warnings"]
        assert not any("keyword" in w.lower() for w in warnings)

    def test_keyword_match_pct_zero_when_no_topic(self, client):
        """keyword_match_pct is 0.0 when there is no topic."""
        text = "Some content. " * 10
        job = _make_job(metadata={})
        formatting_result = {"result": {"formatted_markdown": text}}

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_mgr,
            patch("marketing_project.api.jobs.verify_job_ownership", return_value=job),
            patch(
                "marketing_project.services.step_result_manager.get_step_result_manager"
            ) as mock_srm,
        ):
            mock_mgr.return_value = AsyncMock()
            mock_srm_instance = AsyncMock()
            mock_srm_instance.get_step_result_by_name = AsyncMock(
                return_value=formatting_result
            )
            mock_srm.return_value = mock_srm_instance

            response = client.get("/api/v1/jobs/job-q-1/quality")

        assert response.status_code == 200
        assert response.json()["keyword_match_pct"] == 0.0

    # ------------------------------------------------------------------
    # Result data structure variants
    # ------------------------------------------------------------------

    def test_accepts_flat_result_dict(self, client):
        """Works when step result has no nested 'result' key (flat dict)."""
        text = "Content for testing. " * 10
        job = _make_job()
        # No nested "result" key — flat dict
        formatting_result = {"formatted_markdown": text}

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_mgr,
            patch("marketing_project.api.jobs.verify_job_ownership", return_value=job),
            patch(
                "marketing_project.services.step_result_manager.get_step_result_manager"
            ) as mock_srm,
        ):
            mock_mgr.return_value = AsyncMock()
            mock_srm_instance = AsyncMock()
            mock_srm_instance.get_step_result_by_name = AsyncMock(
                return_value=formatting_result
            )
            mock_srm.return_value = mock_srm_instance

            response = client.get("/api/v1/jobs/job-q-1/quality")

        assert response.status_code == 200
        assert response.json()["word_count"] > 0

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_500_on_unexpected_step_manager_error(self, client):
        """Returns 500 when step_result_manager raises an unexpected error."""
        job = _make_job()

        with (
            patch("marketing_project.api.jobs.get_job_manager") as mock_mgr,
            patch("marketing_project.api.jobs.verify_job_ownership", return_value=job),
            patch(
                "marketing_project.services.step_result_manager.get_step_result_manager"
            ) as mock_srm,
        ):
            mock_mgr.return_value = AsyncMock()
            mock_srm_instance = AsyncMock()
            mock_srm_instance.get_step_result_by_name = AsyncMock(
                side_effect=RuntimeError("DB error")
            )
            mock_srm.return_value = mock_srm_instance

            response = client.get("/api/v1/jobs/job-q-1/quality")

        assert response.status_code == 500
