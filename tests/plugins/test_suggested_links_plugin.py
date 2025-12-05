"""
Tests for Suggested Links plugin.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.pipeline_steps import SuggestedLinksResult
from marketing_project.plugins.suggested_links.tasks import SuggestedLinksPlugin


@pytest.fixture
def suggested_links_plugin():
    """Create SuggestedLinksPlugin instance."""
    return SuggestedLinksPlugin()


@pytest.fixture
def sample_context():
    """Sample context for plugin execution."""
    from marketing_project.models.pipeline_steps import (
        ArticleGenerationResult,
        SEOKeywordsResult,
        SEOOptimizationResult,
    )

    return {
        "input_content": {"id": "test-1", "title": "Test", "content": "Test content"},
        "article_generation": ArticleGenerationResult(
            article_title="Test Article",
            article_content="Generated article content",
            outline=["Section 1"],
            call_to_action="Learn more",
        ),
        "seo_keywords": SEOKeywordsResult(
            main_keyword="test",
            primary_keywords=["test"],
            search_intent="informational",
        ),
        "seo_optimization": SEOOptimizationResult(
            optimized_content="Optimized content",
            meta_title="Test Meta Title",
            meta_description="Test meta description",
            slug="test-slug",
        ),
    }


class TestSuggestedLinksPlugin:
    """Test SuggestedLinksPlugin."""

    def test_step_name(self, suggested_links_plugin):
        """Test step_name property."""
        assert suggested_links_plugin.step_name == "suggested_links"

    def test_step_number(self, suggested_links_plugin):
        """Test step_number property."""
        assert suggested_links_plugin.step_number == 6

    def test_response_model(self, suggested_links_plugin):
        """Test response_model property."""
        assert suggested_links_plugin.response_model == SuggestedLinksResult

    @pytest.mark.asyncio
    async def test_execute(self, suggested_links_plugin, sample_context):
        """Test plugin execution."""
        mock_result = SuggestedLinksResult(
            internal_links=[], total_suggestions=0, high_confidence_links=0
        )

        mock_pipeline = MagicMock()
        mock_pipeline._get_user_prompt = MagicMock(return_value="Test prompt")
        mock_pipeline._get_system_instruction = MagicMock(
            return_value="Test instruction"
        )
        mock_pipeline._call_function = AsyncMock(return_value=mock_result)

        result = await suggested_links_plugin.execute(
            sample_context, mock_pipeline, job_id="test-job"
        )

        assert isinstance(result, SuggestedLinksResult)
        mock_pipeline._call_function.assert_called_once()
