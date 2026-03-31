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
        HeaderStructure,
        KeywordMap,
        OGTags,
        ReadabilityOptimization,
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
            og_tags=OGTags(
                og_title="Test",
                og_description="Test description",
                og_image="https://example.com/img.jpg",
                og_type="article",
            ),
            confidence_score=0.9,
            seo_score=85.0,
            header_structure=HeaderStructure(),
            keyword_map=KeywordMap(),
            readability_optimization=ReadabilityOptimization(),
            modification_report=[],
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

        from marketing_project.services.arthur_prompt_client import ArthurPromptResult

        mock_arthur = ArthurPromptResult(
            system_content="Test instruction", user_template="Test prompt"
        )
        mock_pipeline = MagicMock()
        mock_pipeline._call_function = AsyncMock(return_value=mock_result)

        with patch(
            "marketing_project.services.arthur_prompt_client.fetch_arthur_prompt",
            new=AsyncMock(return_value=mock_arthur),
        ):
            result = await suggested_links_plugin.execute(
                sample_context, mock_pipeline, job_id="test-job"
            )

        assert isinstance(result, SuggestedLinksResult)
        mock_pipeline._call_function.assert_called_once()
