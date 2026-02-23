"""
Tests for direct processor API endpoints.

These endpoints now return job IDs immediately for async processing.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.processors import router
from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.models import (
    BlogPostContext,
    BlogProcessorRequest,
    ReleaseNotesContext,
    ReleaseNotesProcessorRequest,
    TranscriptContext,
    TranscriptProcessorRequest,
)
from marketing_project.services.job_manager import Job, JobStatus
from tests.utils.keycloak_test_helpers import create_user_context


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    mock_user = create_user_context(roles=["admin"])
    app.dependency_overrides[get_current_user] = lambda: mock_user
    return TestClient(app)


@pytest.fixture
def sample_blog_content():
    """Sample blog content for testing."""
    return BlogPostContext(
        id="blog-123",
        title="Getting Started with FastAPI",
        content="FastAPI is a modern Python web framework...",
        author="Jane Smith",
        category="tutorial",
        tags=["python", "fastapi", "web"],
    )


@pytest.fixture
def sample_release_notes():
    """Sample release notes for testing."""
    return ReleaseNotesContext(
        id="v1.2.0",
        title="Version 1.2.0 Release Notes",
        version="1.2.0",
        release_date="2025-10-22",
        changes=["Added authentication", "Fixed memory leak"],
        features=["Authentication system"],
        bug_fixes=["Memory leak fix"],
    )


@pytest.fixture
def sample_transcript():
    """Sample transcript for testing."""
    return TranscriptContext(
        id="interview-789",
        title="Product Manager Interview",
        content="Interviewer: Tell me about your experience...",
        speakers=["Interviewer", "Candidate"],
        duration=1800,
    )


class TestBlogProcessorEndpoint:
    """Test the /process/blog endpoint."""

    @patch("marketing_project.api.processors.get_job_manager")
    @pytest.mark.asyncio
    async def test_process_blog_submits_job(
        self, mock_get_job_manager, client, sample_blog_content
    ):
        """Test that blog post processing submits a job and returns job ID."""
        # Setup mock job manager
        mock_manager = AsyncMock()
        job_id = str(uuid4())
        mock_job = Job(
            id=job_id,
            type="blog_post",
            status=JobStatus.QUEUED,
            content_id=sample_blog_content.id,
        )
        mock_manager.create_job = AsyncMock(return_value=mock_job)
        mock_manager.submit_to_arq = AsyncMock(return_value="arq-job-id-123")
        mock_get_job_manager.return_value = mock_manager

        request = BlogProcessorRequest(content=sample_blog_content)
        response = client.post("/process/blog", json=request.model_dump(mode="json"))

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["job_id"] == job_id
        assert data["content_id"] == sample_blog_content.id
        assert "status_url" in data
        assert f"/api/v1/jobs/{job_id}/status" in data["status_url"]

        # Verify job was created and submitted
        mock_manager.create_job.assert_called_once()
        mock_manager.submit_to_arq.assert_called_once()

    @patch("marketing_project.api.processors.get_job_manager")
    @pytest.mark.asyncio
    async def test_process_blog_with_output_content_type(
        self, mock_get_job_manager, client, sample_blog_content
    ):
        """Test blog post processing with custom output_content_type."""
        mock_manager = AsyncMock()
        job_id = str(uuid4())
        mock_job = Job(
            id=job_id,
            type="blog_post",
            status=JobStatus.QUEUED,
            content_id=sample_blog_content.id,
        )
        mock_manager.create_job = AsyncMock(return_value=mock_job)
        mock_manager.submit_to_arq = AsyncMock(return_value="arq-job-id-123")
        mock_get_job_manager.return_value = mock_manager

        request = BlogProcessorRequest(
            content=sample_blog_content, output_content_type="press_release"
        )
        response = client.post("/process/blog", json=request.model_dump(mode="json"))

        assert response.status_code == 200
        # Verify output_content_type was passed to job creation
        call_args = mock_manager.create_job.call_args
        assert call_args[1]["metadata"]["output_content_type"] == "press_release"

    @patch("marketing_project.api.processors.get_job_manager")
    @pytest.mark.asyncio
    async def test_process_blog_job_creation_failure(
        self, mock_get_job_manager, client, sample_blog_content
    ):
        """Test blog post processing when job creation fails."""
        mock_manager = AsyncMock()
        mock_manager.create_job = AsyncMock(
            side_effect=Exception("Redis connection failed")
        )
        mock_get_job_manager.return_value = mock_manager

        request = BlogProcessorRequest(content=sample_blog_content)
        response = client.post("/process/blog", json=request.model_dump(mode="json"))

        assert response.status_code == 500
        data = response.json()
        assert "Failed to submit job" in data["detail"]


class TestReleaseNotesProcessorEndpoint:
    """Test the /process/release-notes endpoint."""

    @patch("marketing_project.api.processors.get_job_manager")
    @pytest.mark.asyncio
    async def test_process_release_notes_submits_job(
        self, mock_get_job_manager, client, sample_release_notes
    ):
        """Test that release notes processing submits a job and returns job ID."""
        mock_manager = AsyncMock()
        job_id = str(uuid4())
        mock_job = Job(
            id=job_id,
            type="release_notes",
            status=JobStatus.QUEUED,
            content_id=sample_release_notes.id,
        )
        mock_manager.create_job = AsyncMock(return_value=mock_job)
        mock_manager.submit_to_arq = AsyncMock(return_value="arq-job-id-123")
        mock_get_job_manager.return_value = mock_manager

        request = ReleaseNotesProcessorRequest(content=sample_release_notes)
        response = client.post(
            "/process/release-notes", json=request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["job_id"] == job_id
        assert data["content_id"] == sample_release_notes.id
        assert "status_url" in data

        mock_manager.create_job.assert_called_once()
        mock_manager.submit_to_arq.assert_called_once()


class TestTranscriptProcessorEndpoint:
    """Test the /process/transcript endpoint."""

    @patch("marketing_project.api.processors.get_job_manager")
    @pytest.mark.asyncio
    async def test_process_transcript_submits_job(
        self, mock_get_job_manager, client, sample_transcript
    ):
        """Test that transcript processing submits a job and returns job ID."""
        mock_manager = AsyncMock()
        job_id = str(uuid4())
        mock_job = Job(
            id=job_id,
            type="transcript",
            status=JobStatus.QUEUED,
            content_id=sample_transcript.id,
        )
        mock_manager.create_job = AsyncMock(return_value=mock_job)
        mock_manager.submit_to_arq = AsyncMock(return_value="arq-job-id-123")
        mock_get_job_manager.return_value = mock_manager

        request = TranscriptProcessorRequest(content=sample_transcript)
        response = client.post(
            "/process/transcript", json=request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["job_id"] == job_id
        assert data["content_id"] == sample_transcript.id
        assert "status_url" in data

        mock_manager.create_job.assert_called_once()
        mock_manager.submit_to_arq.assert_called_once()
