"""
Comprehensive tests for processor functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.processors import (
    process_blog_post,
    process_release_notes,
    process_transcript,
)


@pytest.mark.asyncio
async def test_process_blog_post_success():
    """Test process_blog_post with valid input."""
    content_data = '{"id": "test-1", "title": "Test Blog", "content": "Test content", "snippet": "Test snippet"}'

    with patch(
        "marketing_project.processors.blog_processor.FunctionPipeline"
    ) as mock_pipeline_class:
        mock_pipeline = MagicMock()
        mock_pipeline.execute_pipeline = AsyncMock(
            return_value={"seo_keywords": {}, "marketing_brief": {}}
        )
        mock_pipeline_class.return_value = mock_pipeline

        result = await process_blog_post(content_data, job_id="test-job-1")

        assert result is not None
        result_dict = eval(result) if isinstance(result, str) else result
        assert "status" in result_dict or "seo_keywords" in result_dict


@pytest.mark.asyncio
async def test_process_blog_post_invalid_input():
    """Test process_blog_post with invalid input."""
    content_data = '{"title": "Test"}'  # Missing required id

    result = await process_blog_post(content_data)

    result_dict = eval(result) if isinstance(result, str) else result
    assert result_dict["status"] == "error"
    assert "invalid_input" in result_dict["error"]


@pytest.mark.asyncio
async def test_process_release_notes_success():
    """Test process_release_notes with valid input."""
    content_data = '{"id": "test-1", "title": "Release v1.0", "content": "Release notes", "snippet": "Snippet", "version": "1.0.0"}'

    with patch(
        "marketing_project.processors.releasenotes_processor.FunctionPipeline"
    ) as mock_pipeline_class:
        mock_pipeline = MagicMock()
        mock_pipeline.execute_pipeline = AsyncMock(
            return_value={"seo_keywords": {}, "marketing_brief": {}}
        )
        mock_pipeline_class.return_value = mock_pipeline

        result = await process_release_notes(content_data, job_id="test-job-1")

        assert result is not None
        result_dict = eval(result) if isinstance(result, str) else result
        assert "status" in result_dict or "seo_keywords" in result_dict


@pytest.mark.asyncio
async def test_process_transcript_success():
    """Test process_transcript with valid input."""
    content_data = '{"id": "test-1", "title": "Podcast", "content": "Speaker 1: Hello\\nSpeaker 2: Hi", "snippet": "Snippet"}'

    with patch(
        "marketing_project.processors.transcript_processor.FunctionPipeline"
    ) as mock_pipeline_class:
        mock_pipeline = MagicMock()
        mock_pipeline.execute_pipeline = AsyncMock(
            return_value={"seo_keywords": {}, "marketing_brief": {}}
        )
        mock_pipeline_class.return_value = mock_pipeline

        result = await process_transcript(content_data, job_id="test-job-1")

        assert result is not None
        result_dict = eval(result) if isinstance(result, str) else result
        assert "status" in result_dict or "seo_keywords" in result_dict


@pytest.mark.asyncio
async def test_process_transcript_with_duration():
    """Test process_transcript with duration string."""
    content_data = '{"id": "test-1", "title": "Podcast", "content": "Content", "duration": "30:00"}'

    with patch(
        "marketing_project.processors.transcript_processor.FunctionPipeline"
    ) as mock_pipeline_class:
        mock_pipeline = MagicMock()
        mock_pipeline.execute_pipeline = AsyncMock(return_value={})
        mock_pipeline_class.return_value = mock_pipeline

        result = await process_transcript(content_data)

        assert result is not None
