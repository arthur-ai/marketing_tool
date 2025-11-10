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
    return {
        "input_content": {"id": "test-1", "title": "Test", "content": "Test content"},
        "seo_optimization": {"optimized": True},
    }


class TestSuggestedLinksPlugin:
    """Test SuggestedLinksPlugin."""

    def test_step_name(self, suggested_links_plugin):
        """Test step_name property."""
        assert suggested_links_plugin.step_name == "suggested_links"

    def test_step_number(self, suggested_links_plugin):
        """Test step_number property."""
        assert suggested_links_plugin.step_number == 5

    def test_response_model(self, suggested_links_plugin):
        """Test response_model property."""
        assert suggested_links_plugin.response_model == SuggestedLinksResult

    @pytest.mark.asyncio
    async def test_execute(self, suggested_links_plugin, sample_context):
        """Test plugin execution."""
        mock_pipeline = MagicMock()
        mock_result = SuggestedLinksResult(links=[])

        with patch.object(
            suggested_links_plugin, "_call_function", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_result

            result = await suggested_links_plugin.execute(
                sample_context, mock_pipeline, job_id="test-job"
            )

            assert isinstance(result, SuggestedLinksResult)
            mock_call.assert_called_once()
