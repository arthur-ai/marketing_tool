"""
End-to-end tests for function pipeline.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from marketing_project.core.models import BlogPostContext
from marketing_project.processors.blog_processor import process_blog_post
from marketing_project.services.function_pipeline import FunctionPipeline


@pytest.fixture
def sample_blog_post():
    """Sample blog post for testing."""
    return BlogPostContext(
        id="e2e-test-1",
        title="E2E Test Blog Post",
        content="This is a comprehensive test blog post for end-to-end testing of the function pipeline.",
        snippet="E2E test snippet",
        author="Test Author",
        tags=["test", "e2e"],
        category="testing",
    )


@pytest.mark.integration
class TestFunctionPipelineE2E:
    """End-to-end tests for function pipeline."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, sample_blog_post):
        """Test complete pipeline from processor to result."""
        job_id = str(uuid4())
        content_json = sample_blog_post.model_dump_json()

        # Mock all plugins
        mock_plugins = {}
        plugin_names = [
            "seo_keywords",
            "marketing_brief",
            "article_generation",
            "seo_optimization",
            "suggested_links",
            "content_formatting",
        ]

        for i, name in enumerate(plugin_names, 1):
            mock_plugin = MagicMock()
            mock_plugin.step_name = name
            mock_plugin.step_number = i
            mock_plugin.get_required_context_keys = lambda: []
            mock_plugin.validate_context = lambda ctx: True
            mock_plugin.execute = AsyncMock(
                return_value=MagicMock(
                    model_dump=lambda mode="json": {f"{name}_result": "test"}
                )
            )
            mock_plugins[name] = mock_plugin

        with patch(
            "marketing_project.processors.blog_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline_result = {
                "pipeline_status": "completed",
                "step_results": {
                    name: {f"{name}_result": "test"} for name in plugin_names
                },
                "metadata": {"job_id": job_id},
            }
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_blog_post(content_json, job_id=job_id)
            result = json.loads(result_json)

            assert result["status"] == "success"
            assert "pipeline_result" in result
            assert result["pipeline_result"]["pipeline_status"] == "completed"

    @pytest.mark.asyncio
    async def test_pipeline_with_job_tracking(self, sample_blog_post):
        """Test pipeline execution with job tracking."""
        job_id = str(uuid4())
        content_json = sample_blog_post.model_dump_json()

        with patch(
            "marketing_project.processors.blog_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline_result = {
                "pipeline_status": "completed",
                "step_results": {},
                "metadata": {"job_id": job_id},
            }
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            with patch(
                "marketing_project.processors.blog_processor.get_job_manager"
            ) as mock_get_job_manager:
                mock_job_manager = AsyncMock()
                mock_job = MagicMock()
                mock_job.metadata = {}
                mock_job_manager.get_job = AsyncMock(return_value=mock_job)
                mock_get_job_manager.return_value = mock_job_manager

                result_json = await process_blog_post(content_json, job_id=job_id)
                result = json.loads(result_json)

                assert result["status"] == "success"
                # Verify job manager was used
                mock_job_manager.get_job.assert_called()

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, sample_blog_post):
        """Test pipeline error handling."""
        job_id = str(uuid4())
        content_json = sample_blog_post.model_dump_json()

        with patch(
            "marketing_project.processors.blog_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                side_effect=Exception("Pipeline error")
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_blog_post(content_json, job_id=job_id)
            result = json.loads(result_json)

            assert result["status"] == "error"
            assert result["error"] == "pipeline_failed"
