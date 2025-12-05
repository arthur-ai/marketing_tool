"""
Tests for worker functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.worker import (
    execute_single_step_job,
    process_blog_job,
    process_release_notes_job,
    process_transcript_job,
    resume_pipeline_job,
    retry_step_job,
)


@pytest.fixture
def mock_ctx():
    """Mock ARQ context."""
    return MagicMock()


@pytest.mark.asyncio
async def test_process_blog_job(mock_ctx):
    """Test process_blog_job function."""
    content_json = (
        '{"id": "test-1", "title": "Test", "content": "Content", "snippet": "Snippet"}'
    )

    with patch("marketing_project.worker.process_blog_post") as mock_process:
        mock_process.return_value = '{"status": "success", "data": {}}'

        result = await process_blog_job(mock_ctx, content_json, "test-job-1")

        assert isinstance(result, dict)
        assert result.get("status") == "success" or "data" in result


@pytest.mark.asyncio
async def test_process_release_notes_job(mock_ctx):
    """Test process_release_notes_job function."""
    content_json = '{"id": "test-1", "title": "Release v1.0", "content": "Content", "version": "1.0.0"}'

    with patch("marketing_project.worker.process_release_notes") as mock_process:
        mock_process.return_value = '{"status": "success", "data": {}}'

        result = await process_release_notes_job(mock_ctx, content_json, "test-job-1")

        assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_process_transcript_job(mock_ctx):
    """Test process_transcript_job function."""
    content_json = '{"id": "test-1", "title": "Podcast", "content": "Speaker 1: Hello"}'

    with patch("marketing_project.worker.process_transcript") as mock_process:
        mock_process.return_value = '{"status": "success", "data": {}}'

        result = await process_transcript_job(mock_ctx, content_json, "test-job-1")

        assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_resume_pipeline_job(mock_ctx):
    """Test resume_pipeline_job function."""
    context_data = {
        "context": {"seo_keywords": {"main_keyword": "test"}},
        "last_step": "seo_keywords",
        "last_step_number": 1,
        "original_content": {"id": "test-1", "title": "Test", "content": "Content"},
    }

    with patch("marketing_project.worker.FunctionPipeline") as mock_pipeline_class:
        with patch("marketing_project.worker.get_job_manager") as mock_job_mgr:
            with patch("marketing_project.services.function_pipeline.AsyncOpenAI"):
                mock_pipeline = MagicMock()
                mock_pipeline.resume_pipeline = AsyncMock(
                    return_value={"pipeline_status": "success"}
                )
                mock_pipeline_class.return_value = mock_pipeline

                mock_job = MagicMock()
                mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)
                mock_job_mgr.return_value.update_job_progress = AsyncMock()
                mock_job_mgr.return_value._save_job = AsyncMock()

                result = await resume_pipeline_job(
                    mock_ctx, "original-job-1", context_data, "test-job-1"
                )

                assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_retry_step_job(mock_ctx):
    """Test retry_step_job function."""
    input_data = {
        "content": {"id": "test-1", "title": "Test"},
        "prompt": "Extract keywords",
    }

    with patch(
        "marketing_project.services.step_retry_service.get_retry_service"
    ) as mock_get_service:
        mock_service = MagicMock()
        mock_service.retry_step = AsyncMock(
            return_value={"status": "success", "result": {"main_keyword": "test"}}
        )
        mock_get_service.return_value = mock_service

        result = await retry_step_job(
            mock_ctx, "seo_keywords", input_data, {}, "test-job-1", "approval-1"
        )

        assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_execute_single_step_job(mock_ctx):
    """Test execute_single_step_job function."""
    content_json = '{"id": "test-1", "title": "Test", "content": "Content"}'
    context = {"input_content": {"id": "test-1", "title": "Test"}}

    with patch("marketing_project.worker.FunctionPipeline") as mock_pipeline_class:
        mock_pipeline = MagicMock()
        mock_pipeline.execute_single_step = AsyncMock(
            return_value={"result": {"main_keyword": "test"}}
        )
        mock_pipeline_class.return_value = mock_pipeline

        result = await execute_single_step_job(
            mock_ctx, "seo_keywords", content_json, context, "test-job-1"
        )

        assert isinstance(result, dict)
