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
    return {
        "input_content": {"id": "test-1", "title": "Test", "content": "Test content"},
        "seo_optimization": {"optimized": True},
    }


class TestContentFormattingPlugin:
    """Test ContentFormattingPlugin."""

    def test_step_name(self, content_formatting_plugin):
        """Test step_name property."""
        assert content_formatting_plugin.step_name == "content_formatting"

    def test_step_number(self, content_formatting_plugin):
        """Test step_number property."""
        assert content_formatting_plugin.step_number == 6

    def test_response_model(self, content_formatting_plugin):
        """Test response_model property."""
        assert content_formatting_plugin.response_model == ContentFormattingResult

    @pytest.mark.asyncio
    async def test_execute(self, content_formatting_plugin, sample_context):
        """Test plugin execution."""
        mock_pipeline = MagicMock()
        mock_result = ContentFormattingResult(formatted_html="<p>Formatted</p>")

        with patch.object(
            content_formatting_plugin, "_call_function", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_result

            result = await content_formatting_plugin.execute(
                sample_context, mock_pipeline, job_id="test-job"
            )

            assert isinstance(result, ContentFormattingResult)
            mock_call.assert_called_once()
