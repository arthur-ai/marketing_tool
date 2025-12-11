"""
Comprehensive tests for SEO keywords plugin tasks.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.plugins.seo_keywords.tasks import SEOKeywordsPlugin


@pytest.fixture
def seo_keywords_plugin():
    """Create a SEOKeywordsPlugin instance."""
    return SEOKeywordsPlugin()


def test_step_name(seo_keywords_plugin):
    """Test step_name property."""
    assert seo_keywords_plugin.step_name == "seo_keywords"


def test_step_number(seo_keywords_plugin):
    """Test step_number property."""
    assert (
        seo_keywords_plugin.step_number == 2
    )  # SEO keywords is step 2 in the pipeline


def test_get_required_context_keys(seo_keywords_plugin):
    """Test get_required_context_keys method."""
    keys = seo_keywords_plugin.get_required_context_keys()

    assert isinstance(keys, list)
    assert "input_content" in keys


def test_validate_context(seo_keywords_plugin):
    """Test validate_context method."""
    context = {"input_content": {"id": "test", "title": "Test", "content": "Content"}}

    assert seo_keywords_plugin.validate_context(context) is True

    # Missing required key
    assert seo_keywords_plugin.validate_context({}) is False


@pytest.mark.asyncio
async def test_execute(seo_keywords_plugin):
    """Test execute method."""
    from marketing_project.services.function_pipeline import FunctionPipeline

    context = {
        "input_content": {"id": "test", "title": "Test", "content": "Test content"},
    }

    with patch(
        "marketing_project.services.function_pipeline.AsyncOpenAI"
    ) as mock_openai:
        mock_pipeline = FunctionPipeline()
        mock_pipeline._call_function = AsyncMock(
            return_value=MagicMock(
                main_keyword="test",
                primary_keywords=["test1", "test2"],
                confidence_score=0.9,
            )
        )

        result = await seo_keywords_plugin.execute(context, mock_pipeline, "test-job")

        assert result is not None
        assert hasattr(result, "main_keyword") or isinstance(result, dict)
