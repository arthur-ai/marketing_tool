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
    from marketing_project.models.pipeline_steps import (
        MarketingBriefResult,
        SEOKeywordsResult,
    )

    return {
        "input_content": {"id": "test-1", "title": "Test", "content": "Test content"},
        "seo_keywords": SEOKeywordsResult(
            main_keyword="test",
            primary_keywords=["test"],
            search_intent="informational",
        ),
        "marketing_brief": MarketingBriefResult(
            target_audience=["Test audience"],
            key_messages=["Test message"],
            content_strategy="Test strategy",
        ),
    }


class TestArticleGenerationPlugin:
    """Test ArticleGenerationPlugin."""

    def test_step_name(self, article_generation_plugin):
        """Test step_name property."""
        assert article_generation_plugin.step_name == "article_generation"

    def test_step_number(self, article_generation_plugin):
        """Test step_number property."""
        assert article_generation_plugin.step_number == 4

    def test_response_model(self, article_generation_plugin):
        """Test response_model property."""
        assert article_generation_plugin.response_model == ArticleGenerationResult

    @pytest.mark.asyncio
    async def test_execute(self, article_generation_plugin, sample_context):
        """Test plugin execution."""
        mock_result = ArticleGenerationResult(
            article_title="Test Article",
            article_content="Generated article content",
            outline=["Section 1", "Section 2"],
            call_to_action="Learn more",
        )

        mock_pipeline = MagicMock()
        mock_pipeline._get_user_prompt = MagicMock(return_value="Test prompt")
        mock_pipeline._get_system_instruction = MagicMock(
            return_value="Test instruction"
        )
        mock_pipeline._call_function = AsyncMock(return_value=mock_result)

        result = await article_generation_plugin.execute(
            sample_context, mock_pipeline, job_id="test-job"
        )

        assert isinstance(result, ArticleGenerationResult)
        mock_pipeline._call_function.assert_called_once()
