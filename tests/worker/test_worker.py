"""
Tests for ARQ worker functions.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from marketing_project.worker import (
    process_blog_job,
    process_release_notes_job,
    process_transcript_job,
)


@pytest.fixture
def sample_blog_content_json():
    """Sample blog content as JSON string."""
    return json.dumps(
        {
            "id": "test-blog-1",
            "title": "Test Blog",
            "content": "Test content",
            "snippet": "Test snippet",
        }
    )


@pytest.fixture
def sample_release_notes_json():
    """Sample release notes as JSON string."""
    return json.dumps(
        {
            "id": "test-release-1",
            "title": "Version 1.0.0",
            "content": "Test content",
            "snippet": "Test snippet",
            "version": "1.0.0",
        }
    )


@pytest.fixture
def sample_transcript_json():
    """Sample transcript as JSON string."""
    return json.dumps(
        {
            "id": "test-transcript-1",
            "title": "Test Transcript",
            "content": "Speaker 1: Hello",
            "snippet": "Test snippet",
            "speakers": ["Speaker 1"],
            "duration": "10:00",
        }
    )


class TestProcessBlogJob:
    """Test process_blog_job function."""

    @pytest.mark.asyncio
    async def test_process_blog_job_success(self, sample_blog_content_json):
        """Test successful blog job processing."""
        job_id = str(uuid4())
        mock_ctx = MagicMock()

        mock_result = {
            "status": "success",
            "content_type": "blog_post",
            "pipeline_result": {"test": "result"},
        }

        with patch(
            "marketing_project.worker.process_blog_post", new_callable=AsyncMock
        ) as mock_process:
            mock_process.return_value = json.dumps(mock_result)

            with patch(
                "marketing_project.worker.get_job_manager"
            ) as mock_get_job_manager:
                mock_job_manager = AsyncMock()
                mock_job = MagicMock()
                mock_job.metadata = {}
                mock_job_manager.get_job = AsyncMock(return_value=mock_job)
                mock_job_manager.update_job_progress = AsyncMock()
                mock_job_manager._save_job_to_redis = AsyncMock()
                mock_get_job_manager.return_value = mock_job_manager

                result = await process_blog_job(
                    mock_ctx, sample_blog_content_json, job_id
                )

                assert result["status"] == "success"
                mock_process.assert_called_once()
                mock_job_manager.update_job_progress.assert_called()

    @pytest.mark.asyncio
    async def test_process_blog_job_error(self, sample_blog_content_json):
        """Test blog job processing with error."""
        job_id = str(uuid4())
        mock_ctx = MagicMock()

        mock_result = {
            "status": "error",
            "message": "Processing failed",
        }

        with patch(
            "marketing_project.worker.process_blog_post", new_callable=AsyncMock
        ) as mock_process:
            mock_process.return_value = json.dumps(mock_result)

            with patch(
                "marketing_project.worker.get_job_manager"
            ) as mock_get_job_manager:
                mock_job_manager = AsyncMock()
                mock_job_manager.update_job_progress = AsyncMock()
                mock_get_job_manager.return_value = mock_job_manager

                with pytest.raises(Exception, match="Processing failed"):
                    await process_blog_job(mock_ctx, sample_blog_content_json, job_id)


class TestProcessReleaseNotesJob:
    """Test process_release_notes_job function."""

    @pytest.mark.asyncio
    async def test_process_release_notes_job_success(self, sample_release_notes_json):
        """Test successful release notes job processing."""
        job_id = str(uuid4())
        mock_ctx = MagicMock()

        mock_result = {
            "status": "success",
            "content_type": "release_notes",
            "pipeline_result": {"test": "result"},
        }

        with patch(
            "marketing_project.worker.process_release_notes", new_callable=AsyncMock
        ) as mock_process:
            mock_process.return_value = json.dumps(mock_result)

            with patch(
                "marketing_project.worker.get_job_manager"
            ) as mock_get_job_manager:
                mock_job_manager = AsyncMock()
                mock_job = MagicMock()
                mock_job.metadata = {}
                mock_job_manager.get_job = AsyncMock(return_value=mock_job)
                mock_job_manager.update_job_progress = AsyncMock()
                mock_job_manager._save_job_to_redis = AsyncMock()
                mock_get_job_manager.return_value = mock_job_manager

                result = await process_release_notes_job(
                    mock_ctx, sample_release_notes_json, job_id
                )

                assert result["status"] == "success"
                mock_process.assert_called_once()


class TestProcessTranscriptJob:
    """Test process_transcript_job function."""

    @pytest.mark.asyncio
    async def test_process_transcript_job_success(self, sample_transcript_json):
        """Test successful transcript job processing."""
        job_id = str(uuid4())
        mock_ctx = MagicMock()

        mock_result = {
            "status": "success",
            "content_type": "transcript",
            "pipeline_result": {"test": "result"},
        }

        with patch(
            "marketing_project.worker.process_transcript", new_callable=AsyncMock
        ) as mock_process:
            mock_process.return_value = json.dumps(mock_result)

            with patch(
                "marketing_project.worker.get_job_manager"
            ) as mock_get_job_manager:
                mock_job_manager = AsyncMock()
                mock_job = MagicMock()
                mock_job.metadata = {}
                mock_job_manager.get_job = AsyncMock(return_value=mock_job)
                mock_job_manager.update_job_progress = AsyncMock()
                mock_job_manager._save_job_to_redis = AsyncMock()
                mock_get_job_manager.return_value = mock_job_manager

                result = await process_transcript_job(
                    mock_ctx, sample_transcript_json, job_id
                )

                assert result["status"] == "success"
                mock_process.assert_called_once()
