"""
Tests for SEO Keywords plugin.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.pipeline_steps import SEOKeywordsResult
from marketing_project.plugins.seo_keywords.tasks import SEOKeywordsPlugin


@pytest.fixture
def seo_keywords_plugin():
    """Create SEOKeywordsPlugin instance."""
    return SEOKeywordsPlugin()


@pytest.fixture
def sample_context():
    """Sample context for plugin execution."""
    return {
        "input_content": {
            "id": "test-1",
            "title": "Test Article",
            "content": "This is a test article about artificial intelligence and machine learning.",
            "snippet": "Test snippet",
        },
        "content_type": "blog_post",
    }


class TestSEOKeywordsPlugin:
    """Test SEOKeywordsPlugin."""

    def test_step_name(self, seo_keywords_plugin):
        """Test step_name property."""
        assert seo_keywords_plugin.step_name == "seo_keywords"

    def test_step_number(self, seo_keywords_plugin):
        """Test step_number property."""
        assert seo_keywords_plugin.step_number == 1

    def test_response_model(self, seo_keywords_plugin):
        """Test response_model property."""
        assert seo_keywords_plugin.response_model == SEOKeywordsResult

    def test_get_required_context_keys(self, seo_keywords_plugin):
        """Test get_required_context_keys method."""
        keys = seo_keywords_plugin.get_required_context_keys()
        assert "input_content" in keys

    def test_validate_context(self, seo_keywords_plugin, sample_context):
        """Test context validation."""
        assert seo_keywords_plugin.validate_context(sample_context) is True

    def test_validate_context_missing_key(self, seo_keywords_plugin):
        """Test context validation with missing key."""
        context = {}  # Missing input_content
        assert seo_keywords_plugin.validate_context(context) is False

    @pytest.mark.asyncio
    async def test_execute(self, seo_keywords_plugin, sample_context):
        """Test plugin execution."""
        mock_pipeline = MagicMock()
        mock_result = SEOKeywordsResult(
            primary_keywords=["artificial intelligence", "machine learning"],
            secondary_keywords=["AI", "ML"],
        )

        with patch.object(
            seo_keywords_plugin, "_call_function", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_result

            result = await seo_keywords_plugin.execute(
                sample_context, mock_pipeline, job_id="test-job"
            )

            assert isinstance(result, SEOKeywordsResult)
            assert len(result.primary_keywords) > 0
            mock_call.assert_called_once()
