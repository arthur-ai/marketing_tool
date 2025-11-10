"""
Tests for blog post processor.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from marketing_project.core.models import BlogPostContext
from marketing_project.processors.blog_processor import process_blog_post


@pytest.fixture
def sample_blog_data():
    """Sample blog post data for testing."""
    return {
        "id": "test-blog-1",
        "title": "Test Blog Post",
        "content": "This is a test blog post content with some sample text.",
        "snippet": "A test blog post snippet",
        "author": "Test Author",
        "tags": ["test", "blog"],
        "category": "testing",
    }


@pytest.fixture
def sample_blog_json(sample_blog_data):
    """Sample blog post as JSON string."""
    return json.dumps(sample_blog_data)


class TestBlogProcessor:
    """Test the process_blog_post function."""

    @pytest.mark.asyncio
    async def test_process_blog_post_success(self, sample_blog_json):
        """Test successful blog post processing."""
        job_id = str(uuid4())
        mock_pipeline_result = {
            "seo_keywords": {"primary": ["test", "blog"]},
            "marketing_brief": {"summary": "Test brief"},
            "article_generation": {"content": "Generated article"},
            "seo_optimization": {"optimized": True},
            "suggested_links": {"links": []},
            "content_formatting": {"formatted": True},
        }

        with patch(
            "marketing_project.processors.blog_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_blog_post(sample_blog_json, job_id=job_id)
            result = json.loads(result_json)

            assert result["status"] == "success"
            assert result["content_type"] == "blog_post"
            assert "pipeline_result" in result
            assert result["pipeline_result"] == mock_pipeline_result

    @pytest.mark.asyncio
    async def test_process_blog_post_invalid_json(self):
        """Test blog post processing with invalid JSON."""
        invalid_json = "not valid json {"

        result_json = await process_blog_post(invalid_json)
        result = json.loads(result_json)

        assert result["status"] == "error"
        assert result["error"] == "invalid_json"

    @pytest.mark.asyncio
    async def test_process_blog_post_invalid_input(self):
        """Test blog post processing with invalid input (missing required fields)."""
        invalid_data = {"id": "test", "title": ""}  # Missing content and snippet
        invalid_json = json.dumps(invalid_data)

        result_json = await process_blog_post(invalid_json)
        result = json.loads(result_json)

        assert result["status"] == "error"
        assert result["error"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_process_blog_post_with_output_content_type(self, sample_blog_json):
        """Test blog post processing with custom output_content_type."""
        job_id = str(uuid4())
        mock_pipeline_result = {"test": "result"}

        with patch(
            "marketing_project.processors.blog_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_blog_post(
                sample_blog_json, job_id=job_id, output_content_type="press_release"
            )
            result = json.loads(result_json)

            assert result["status"] == "success"
            # Verify output_content_type was passed to pipeline
            call_args = mock_pipeline.execute_pipeline.call_args
            assert call_args[1]["output_content_type"] == "press_release"

    @pytest.mark.asyncio
    async def test_process_blog_post_pipeline_failure(self, sample_blog_json):
        """Test blog post processing when pipeline fails."""
        job_id = str(uuid4())

        with patch(
            "marketing_project.processors.blog_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                side_effect=Exception("Pipeline error")
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_blog_post(sample_blog_json, job_id=job_id)
            result = json.loads(result_json)

            assert result["status"] == "error"
            assert result["error"] == "pipeline_failed"

    @pytest.mark.asyncio
    async def test_process_blog_post_approval_rejected(self, sample_blog_json):
        """Test blog post processing when approval is rejected."""
        job_id = str(uuid4())

        with patch(
            "marketing_project.processors.blog_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                side_effect=ValueError("Content rejected by user")
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_blog_post(sample_blog_json, job_id=job_id)
            result = json.loads(result_json)

            assert result["status"] == "error"
            assert result["error"] == "approval_rejected"

    @pytest.mark.asyncio
    async def test_process_blog_post_generates_job_id(self, sample_blog_json):
        """Test that job_id is generated if not provided."""
        mock_pipeline_result = {"test": "result"}

        with patch(
            "marketing_project.processors.blog_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_blog_post(sample_blog_json)
            result = json.loads(result_json)

            assert result["status"] == "success"
            # Verify pipeline was called with a job_id
            call_args = mock_pipeline.execute_pipeline.call_args
            assert call_args[1]["job_id"] is not None

    @pytest.mark.asyncio
    async def test_process_blog_post_with_dict_input(self, sample_blog_data):
        """Test that process_blog_post accepts dict input."""
        job_id = str(uuid4())
        mock_pipeline_result = {"test": "result"}

        with patch(
            "marketing_project.processors.blog_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_blog_post(sample_blog_data, job_id=job_id)
            result = json.loads(result_json)

            assert result["status"] == "success"
