"""
Tests for Content Formatting plugin.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.pipeline_steps import ContentFormattingResult
from marketing_project.plugins.content_formatting.tasks import ContentFormattingPlugin


@pytest.fixture
def content_formatting_plugin():
    """Create ContentFormattingPlugin instance."""
    return ContentFormattingPlugin()


@pytest.fixture
def sample_context():
    """Sample context for plugin execution."""
    from marketing_project.models.pipeline_steps import (
        HeaderStructure,
        KeywordMap,
        OGTags,
        ReadabilityOptimization,
        SEOOptimizationResult,
    )

    return {
        "input_content": {"id": "test-1", "title": "Test", "content": "Test content"},
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


class TestContentFormattingPlugin:
    """Test ContentFormattingPlugin."""

    def test_step_name(self, content_formatting_plugin):
        """Test step_name property."""
        assert content_formatting_plugin.step_name == "content_formatting"

    def test_step_number(self, content_formatting_plugin):
        """Test step_number property."""
        assert content_formatting_plugin.step_number == 8

    def test_response_model(self, content_formatting_plugin):
        """Test response_model property."""
        assert content_formatting_plugin.response_model == ContentFormattingResult

    @pytest.mark.asyncio
    async def test_execute(self, content_formatting_plugin, sample_context):
        """Test plugin execution."""
        mock_result = ContentFormattingResult(
            formatted_html="<p>Formatted</p>", formatted_markdown="# Formatted"
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
            result = await content_formatting_plugin.execute(
                sample_context, mock_pipeline, job_id="test-job"
            )

        assert isinstance(result, ContentFormattingResult)
        mock_pipeline._call_function.assert_called_once()
