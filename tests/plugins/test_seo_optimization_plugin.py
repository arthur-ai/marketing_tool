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
    return {
        "input_content": {"id": "test-1", "title": "Test", "content": "Test content"},
        "seo_keywords": {"primary_keywords": ["test"]},
        "marketing_brief": {"summary": "Test brief"},
        "article_generation": {"content": "Generated article"},
    }


class TestSEOOptimizationPlugin:
    """Test SEOOptimizationPlugin."""

    def test_step_name(self, seo_optimization_plugin):
        """Test step_name property."""
        assert seo_optimization_plugin.step_name == "seo_optimization"

    def test_step_number(self, seo_optimization_plugin):
        """Test step_number property."""
        assert seo_optimization_plugin.step_number == 4

    def test_response_model(self, seo_optimization_plugin):
        """Test response_model property."""
        assert seo_optimization_plugin.response_model == SEOOptimizationResult

    @pytest.mark.asyncio
    async def test_execute(self, seo_optimization_plugin, sample_context):
        """Test plugin execution."""
        mock_pipeline = MagicMock()
        mock_result = SEOOptimizationResult(optimized=True)

        with patch.object(
            seo_optimization_plugin, "_call_function", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = mock_result

            result = await seo_optimization_plugin.execute(
                sample_context, mock_pipeline, job_id="test-job"
            )

            assert isinstance(result, SEOOptimizationResult)
            mock_call.assert_called_once()
