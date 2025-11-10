"""
Tests for direct processor API endpoints.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.processors import router
from marketing_project.models import (
    BlogPostContext,
    BlogProcessorRequest,
    ReleaseNotesContext,
    ReleaseNotesProcessorRequest,
    TranscriptContext,
    TranscriptProcessorRequest,
)


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
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

    @patch("marketing_project.api.processors.process_blog_post")
    @pytest.mark.asyncio
    async def test_process_blog_success(
        self, mock_process, client, sample_blog_content
    ):
        """Test successful blog post processing."""
        # Setup mock processor response
        mock_process.return_value = json.dumps(
            {
                "status": "success",
                "content_type": "blog_post",
                "blog_type": "tutorial",
                "metadata": {
                    "author": "Jane Smith",
                    "category": "tutorial",
                    "tags": ["python", "fastapi", "web"],
                    "word_count": 150,
                },
                "pipeline_result": {
                    "seo_keywords": ["fastapi", "python", "web framework"],
                    "marketing_brief": "Comprehensive tutorial guide...",
                    "formatted_content": "# Getting Started with FastAPI\n...",
                },
                "validation": "passed",
                "processing_steps_completed": [
                    "type_analysis",
                    "structure_validation",
                    "metadata_extraction",
                    "pipeline_execution",
                ],
                "message": "Blog post processed successfully through 8-step pipeline",
            }
        )

        request = BlogProcessorRequest(content=sample_blog_content)
        response = client.post("/process/blog", json=request.model_dump(mode="json"))

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content_id"] == "blog-123"
        assert data["content_type"] == "blog_post"
        assert data["blog_type"] == "tutorial"
        assert "metadata" in data
        assert "pipeline_result" in data
        assert data["validation"] == "passed"

        # Verify processor was called
        mock_process.assert_called_once()

    @patch("marketing_project.api.processors.process_blog_post")
    @pytest.mark.asyncio
    async def test_process_blog_validation_error(
        self, mock_process, client, sample_blog_content
    ):
        """Test blog post processing with validation error."""
        # Setup mock processor to return validation error
        mock_process.return_value = json.dumps(
            {
                "status": "error",
                "error": "validation_failed",
                "message": "Blog post validation failed: Missing required field 'title'",
            }
        )

        request = BlogProcessorRequest(content=sample_blog_content)
        response = client.post("/process/blog", json=request.model_dump(mode="json"))

        assert response.status_code == 400
        data = response.json()
        assert "validation_failed" in data["detail"]

    @patch("marketing_project.api.processors.process_blog_post")
    @pytest.mark.asyncio
    async def test_process_blog_exception(
        self, mock_process, client, sample_blog_content
    ):
        """Test blog post processing with exception."""
        # Setup mock processor to raise exception
        mock_process.side_effect = Exception("Unexpected error")

        request = BlogProcessorRequest(content=sample_blog_content)
        response = client.post("/process/blog", json=request.model_dump(mode="json"))

        assert response.status_code == 500
        data = response.json()
        assert "Blog processing failed" in data["detail"]


class TestReleaseNotesProcessorEndpoint:
    """Test the /process/release-notes endpoint."""

    @patch("marketing_project.api.processors.process_release_notes")
    @pytest.mark.asyncio
    async def test_process_release_notes_success(
        self, mock_process, client, sample_release_notes
    ):
        """Test successful release notes processing."""
        # Setup mock processor response
        mock_process.return_value = json.dumps(
            {
                "status": "success",
                "content_type": "release_notes",
                "release_type": "minor",
                "metadata": {
                    "version": "1.2.0",
                    "release_date": "2025-10-22",
                    "changes": ["Added authentication", "Fixed memory leak"],
                    "features": ["Authentication system"],
                    "bug_fixes": ["Memory leak fix"],
                },
                "pipeline_result": {
                    "seo_keywords": ["release", "version", "authentication"],
                    "marketing_brief": "Major update with new authentication...",
                    "formatted_content": "# Version 1.2.0\n...",
                },
                "validation": "passed",
                "processing_steps_completed": [
                    "type_analysis",
                    "structure_validation",
                    "metadata_extraction",
                    "pipeline_execution",
                ],
                "message": "Release notes processed successfully",
            }
        )

        request = ReleaseNotesProcessorRequest(content=sample_release_notes)
        response = client.post(
            "/process/release-notes", json=request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content_id"] == "v1.2.0"
        assert data["content_type"] == "release_notes"
        assert data["release_type"] == "minor"
        assert "metadata" in data
        assert data["metadata"]["version"] == "1.2.0"

        # Verify processor was called
        mock_process.assert_called_once()

    @patch("marketing_project.api.processors.process_release_notes")
    @pytest.mark.asyncio
    async def test_process_release_notes_missing_version(
        self, mock_process, client, sample_release_notes
    ):
        """Test release notes processing with missing version."""
        # Setup mock processor to return validation error
        mock_process.return_value = json.dumps(
            {
                "status": "error",
                "error": "validation_failed",
                "message": "Release notes validation failed. Release notes REQUIRE version field.",
            }
        )

        request = ReleaseNotesProcessorRequest(content=sample_release_notes)
        response = client.post(
            "/process/release-notes", json=request.model_dump(mode="json")
        )

        assert response.status_code == 400
        data = response.json()
        assert "validation_failed" in data["detail"]


class TestTranscriptProcessorEndpoint:
    """Test the /process/transcript endpoint."""

    @patch("marketing_project.api.processors.process_transcript")
    @pytest.mark.asyncio
    async def test_process_transcript_success(
        self, mock_process, client, sample_transcript
    ):
        """Test successful transcript processing."""
        # Setup mock processor response
        mock_process.return_value = json.dumps(
            {
                "status": "success",
                "content_type": "transcript",
                "transcript_type": "interview",
                "metadata": {
                    "speakers": ["Interviewer", "Candidate"],
                    "duration": 1800,
                    "transcript_type": "interview",
                },
                "pipeline_result": {
                    "seo_keywords": ["interview", "product manager", "experience"],
                    "marketing_brief": "Professional interview transcript...",
                    "formatted_content": "# Interview Transcript\n...",
                },
                "validation": "passed",
                "processing_steps_completed": [
                    "type_analysis",
                    "structure_validation",
                    "metadata_extraction",
                    "pipeline_execution",
                ],
                "message": "Transcript processed successfully",
            }
        )

        request = TranscriptProcessorRequest(content=sample_transcript)
        response = client.post(
            "/process/transcript", json=request.model_dump(mode="json")
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content_id"] == "interview-789"
        assert data["content_type"] == "transcript"
        assert data["transcript_type"] == "interview"
        assert "metadata" in data
        assert len(data["metadata"]["speakers"]) == 2

        # Verify processor was called
        mock_process.assert_called_once()

    @patch("marketing_project.api.processors.process_transcript")
    @pytest.mark.asyncio
    async def test_process_transcript_invalid_json(
        self, mock_process, client, sample_transcript
    ):
        """Test transcript processing with invalid JSON response."""
        # Setup mock processor to return invalid JSON
        mock_process.return_value = "Not valid JSON"

        request = TranscriptProcessorRequest(content=sample_transcript)
        response = client.post(
            "/process/transcript", json=request.model_dump(mode="json")
        )

        assert response.status_code == 500
        data = response.json()
        assert "Processor returned invalid JSON" in data["detail"]
