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
            "primary_keywords": ["test", "article"],
        },
    }


class TestMarketingBriefPlugin:
    """Test MarketingBriefPlugin."""

    def test_step_name(self, marketing_brief_plugin):
        """Test step_name property."""
        assert marketing_brief_plugin.step_name == "marketing_brief"

    def test_step_number(self, marketing_brief_plugin):
        """Test step_number property."""
        assert marketing_brief_plugin.step_number == 2

    def test_response_model(self, marketing_brief_plugin):
        """Test response_model property."""
        assert marketing_brief_plugin.response_model == MarketingBriefResult

    def test_get_required_context_keys(self, marketing_brief_plugin):
        """Test get_required_context_keys method."""
        keys = marketing_brief_plugin.get_required_context_keys()
        assert "input_content" in keys
        assert "seo_keywords" in keys

    @pytest.mark.asyncio
    async def test_execute(self, marketing_brief_plugin, sample_context):
        """Test plugin execution."""
        mock_pipeline = MagicMock()
        mock_result = MarketingBriefResult(summary="Test marketing brief")

        with patch.object(
            marketing_brief_plugin, "_call_function", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_result

            result = await marketing_brief_plugin.execute(
                sample_context, mock_pipeline, job_id="test-job"
            )

            assert isinstance(result, MarketingBriefResult)
            mock_call.assert_called_once()
