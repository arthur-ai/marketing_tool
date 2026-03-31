"""
Tests for SEO Optimization plugin.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.pipeline_steps import SEOOptimizationResult
from marketing_project.plugins.seo_optimization.tasks import SEOOptimizationPlugin


@pytest.fixture
def seo_optimization_plugin():
    """Create SEOOptimizationPlugin instance."""
    return SEOOptimizationPlugin()


@pytest.fixture
def sample_context():
    """Sample context for plugin execution."""
    from marketing_project.models.pipeline_steps import (
        ArticleGenerationResult,
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
        "article_generation": ArticleGenerationResult(
            article_title="Test Article",
            article_content="Generated article content",
            outline=["Section 1"],
            call_to_action="Learn more",
        ),
    }


class TestSEOOptimizationPlugin:
    """Test SEOOptimizationPlugin."""

    def test_step_name(self, seo_optimization_plugin):
        """Test step_name property."""
        assert seo_optimization_plugin.step_name == "seo_optimization"

    def test_step_number(self, seo_optimization_plugin):
        """Test step_number property."""
        assert seo_optimization_plugin.step_number == 5

    def test_response_model(self, seo_optimization_plugin):
        """Test response_model property."""
        assert seo_optimization_plugin.response_model == SEOOptimizationResult

    @pytest.mark.asyncio
    async def test_execute(self, seo_optimization_plugin, sample_context):
        """Test plugin execution."""
        from marketing_project.models.pipeline_steps import (
            HeaderStructure,
            KeywordMap,
            OGTags,
            ReadabilityOptimization,
        )

        mock_result = SEOOptimizationResult(
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
            result = await seo_optimization_plugin.execute(
                sample_context, mock_pipeline, job_id="test-job"
            )

        assert isinstance(result, SEOOptimizationResult)
        mock_pipeline._call_function.assert_called_once()
