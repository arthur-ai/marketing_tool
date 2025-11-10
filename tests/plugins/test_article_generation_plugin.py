"""
Tests for Article Generation plugin.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.pipeline_steps import ArticleGenerationResult
from marketing_project.plugins.article_generation.tasks import ArticleGenerationPlugin


@pytest.fixture
def article_generation_plugin():
    """Create ArticleGenerationPlugin instance."""
    return ArticleGenerationPlugin()


@pytest.fixture
def sample_context():
    """Sample context for plugin execution."""
    return {
        "input_content": {"id": "test-1", "title": "Test", "content": "Test content"},
        "seo_keywords": {"primary_keywords": ["test"]},
        "marketing_brief": {"summary": "Test brief"},
    }


class TestArticleGenerationPlugin:
    """Test ArticleGenerationPlugin."""

    def test_step_name(self, article_generation_plugin):
        """Test step_name property."""
        assert article_generation_plugin.step_name == "article_generation"

    def test_step_number(self, article_generation_plugin):
        """Test step_number property."""
        assert article_generation_plugin.step_number == 3

    def test_response_model(self, article_generation_plugin):
        """Test response_model property."""
        assert article_generation_plugin.response_model == ArticleGenerationResult

    @pytest.mark.asyncio
    async def test_execute(self, article_generation_plugin, sample_context):
        """Test plugin execution."""
        mock_pipeline = MagicMock()
        mock_result = ArticleGenerationResult(content="Generated article content")

        with patch.object(
            article_generation_plugin, "_call_function", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_result

            result = await article_generation_plugin.execute(
                sample_context, mock_pipeline, job_id="test-job"
            )

            assert isinstance(result, ArticleGenerationResult)
            mock_call.assert_called_once()
