"""
Tests for Marketing Brief plugin.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.pipeline_steps import MarketingBriefResult
from marketing_project.plugins.marketing_brief.tasks import MarketingBriefPlugin


@pytest.fixture
def marketing_brief_plugin():
    """Create MarketingBriefPlugin instance."""
    return MarketingBriefPlugin()


@pytest.fixture
def sample_context():
    """Sample context for plugin execution."""
    return {
        "input_content": {
            "id": "test-1",
            "title": "Test Article",
            "content": "Test content",
        },
        "seo_keywords": {
            "main_keyword": "test",
            "primary_keywords": ["test", "article"],
            "search_intent": "informational",
        },
        "content_type": "blog_post",
    }


class TestMarketingBriefPlugin:
    """Test MarketingBriefPlugin."""

    def test_step_name(self, marketing_brief_plugin):
        """Test step_name property."""
        assert marketing_brief_plugin.step_name == "marketing_brief"

    def test_step_number(self, marketing_brief_plugin):
        """Test step_number property."""
        assert marketing_brief_plugin.step_number == 3

    def test_response_model(self, marketing_brief_plugin):
        """Test response_model property."""
        assert marketing_brief_plugin.response_model == MarketingBriefResult

    def test_get_required_context_keys(self, marketing_brief_plugin):
        """Test get_required_context_keys method."""
        keys = marketing_brief_plugin.get_required_context_keys()
        assert "seo_keywords" in keys
        assert "content_type" in keys

    @pytest.mark.asyncio
    async def test_execute(self, marketing_brief_plugin, sample_context):
        """Test plugin execution."""
        mock_result = MarketingBriefResult(
            target_audience=["Test audience"],
            key_messages=["Test message"],
            content_strategy="Test strategy",
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
            result = await marketing_brief_plugin.execute(
                sample_context, mock_pipeline, job_id="test-job"
            )

        assert isinstance(result, MarketingBriefResult)
        mock_pipeline._call_function.assert_called_once()
