"""
Tests for FunctionPipeline service.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from marketing_project.models.pipeline_steps import (
    ArticleGenerationResult,
    ContentFormattingResult,
    MarketingBriefResult,
    SEOKeywordsResult,
    SEOOptimizationResult,
    SuggestedLinksResult,
)
from marketing_project.services.function_pipeline import FunctionPipeline


@pytest.fixture
def sample_content_json():
    """Sample content as JSON string."""
    return json.dumps(
        {
            "id": "test-content-1",
            "title": "Test Content",
            "content": "This is test content for pipeline testing.",
            "snippet": "Test snippet",
        }
    )


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.parsed = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestFunctionPipelineInitialization:
    """Test FunctionPipeline initialization."""

    def test_init_defaults(self):
        """Test FunctionPipeline initialization with defaults."""
        pipeline = FunctionPipeline()
        assert pipeline.model == "gpt-4o-mini"
        assert pipeline.temperature == 0.7
        assert pipeline.lang == "en"
        assert pipeline.step_info == []

    def test_init_custom_params(self):
        """Test FunctionPipeline initialization with custom parameters."""
        pipeline = FunctionPipeline(model="gpt-4", temperature=0.5, lang="es")
        assert pipeline.model == "gpt-4"
        assert pipeline.temperature == 0.5
        assert pipeline.lang == "es"


class TestFunctionPipelineExecution:
    """Test FunctionPipeline execution."""

    @pytest.mark.asyncio
    async def test_execute_pipeline_success(self, sample_content_json):
        """Test successful pipeline execution."""
        job_id = str(uuid4())

        # Mock plugins
        mock_seo_keywords = MagicMock()
        mock_seo_keywords.step_name = "seo_keywords"
        mock_seo_keywords.step_number = 1
        mock_seo_keywords.execute = AsyncMock(
            return_value=SEOKeywordsResult(primary_keywords=["test", "content"])
        )
        mock_seo_keywords.get_required_context_keys = lambda: []
        mock_seo_keywords.validate_context = lambda ctx: True

        mock_marketing_brief = MagicMock()
        mock_marketing_brief.step_name = "marketing_brief"
        mock_marketing_brief.step_number = 2
        mock_marketing_brief.execute = AsyncMock(
            return_value=MarketingBriefResult(summary="Test brief")
        )
        mock_marketing_brief.get_required_context_keys = lambda: []
        mock_marketing_brief.validate_context = lambda ctx: True

        mock_article = MagicMock()
        mock_article.step_name = "article_generation"
        mock_article.step_number = 3
        mock_article.execute = AsyncMock(
            return_value=ArticleGenerationResult(content="Generated article")
        )
        mock_article.get_required_context_keys = lambda: []
        mock_article.validate_context = lambda ctx: True

        mock_seo_opt = MagicMock()
        mock_seo_opt.step_name = "seo_optimization"
        mock_seo_opt.step_number = 4
        mock_seo_opt.execute = AsyncMock(
            return_value=SEOOptimizationResult(optimized=True)
        )
        mock_seo_opt.get_required_context_keys = lambda: []
        mock_seo_opt.validate_context = lambda ctx: True

        mock_links = MagicMock()
        mock_links.step_name = "suggested_links"
        mock_links.step_number = 5
        mock_links.execute = AsyncMock(return_value=SuggestedLinksResult(links=[]))
        mock_links.get_required_context_keys = lambda: []
        mock_links.validate_context = lambda ctx: True

        mock_formatting = MagicMock()
        mock_formatting.step_name = "content_formatting"
        mock_formatting.step_number = 6
        mock_formatting.execute = AsyncMock(
            return_value=ContentFormattingResult(formatted_html="<p>Formatted</p>")
        )
        mock_formatting.get_required_context_keys = lambda: []
        mock_formatting.validate_context = lambda ctx: True

        with patch(
            "marketing_project.services.function_pipeline.get_plugin_registry"
        ) as mock_registry_func:
            mock_registry = MagicMock()
            mock_registry.validate_dependencies = lambda: (True, [])
            mock_registry.get_plugins_in_order = lambda: [
                mock_seo_keywords,
                mock_marketing_brief,
                mock_article,
                mock_seo_opt,
                mock_links,
                mock_formatting,
            ]
            mock_registry.get_plugin = lambda name: {
                "seo_keywords": mock_seo_keywords,
                "marketing_brief": mock_marketing_brief,
                "article_generation": mock_article,
                "seo_optimization": mock_seo_opt,
                "suggested_links": mock_links,
                "content_formatting": mock_formatting,
            }.get(name)
            mock_registry_func.return_value = mock_registry

            pipeline = FunctionPipeline()
            result = await pipeline.execute_pipeline(
                sample_content_json, job_id=job_id, content_type="blog_post"
            )

            assert result["pipeline_status"] == "completed"
            assert "step_results" in result
            assert len(result["step_results"]) == 6
            assert "seo_keywords" in result["step_results"]
            assert "content_formatting" in result["step_results"]
            assert result["metadata"]["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_execute_pipeline_invalid_json(self):
        """Test pipeline execution with invalid JSON."""
        pipeline = FunctionPipeline()

        with pytest.raises(ValueError, match="Invalid JSON input"):
            await pipeline.execute_pipeline("invalid json {", content_type="blog_post")

    @pytest.mark.asyncio
    async def test_execute_pipeline_dependency_validation_failure(
        self, sample_content_json
    ):
        """Test pipeline execution when dependency validation fails."""
        with patch(
            "marketing_project.services.function_pipeline.get_plugin_registry"
        ) as mock_registry_func:
            mock_registry = MagicMock()
            mock_registry.validate_dependencies = lambda: (
                False,
                ["Missing dependency: seo_keywords"],
            )
            mock_registry_func.return_value = mock_registry

            pipeline = FunctionPipeline()

            with pytest.raises(
                ValueError, match="Pipeline dependency validation failed"
            ):
                await pipeline.execute_pipeline(
                    sample_content_json, content_type="blog_post"
                )

    @pytest.mark.asyncio
    async def test_execute_pipeline_with_output_content_type(self, sample_content_json):
        """Test pipeline execution with custom output_content_type."""
        job_id = str(uuid4())

        # Mock a simple plugin
        mock_plugin = MagicMock()
        mock_plugin.step_name = "seo_keywords"
        mock_plugin.step_number = 1
        mock_plugin.execute = AsyncMock(
            return_value=SEOKeywordsResult(primary_keywords=["test"])
        )
        mock_plugin.get_required_context_keys = lambda: []
        mock_plugin.validate_context = lambda ctx: True

        with patch(
            "marketing_project.services.function_pipeline.get_plugin_registry"
        ) as mock_registry_func:
            mock_registry = MagicMock()
            mock_registry.validate_dependencies = lambda: (True, [])
            mock_registry.get_plugins_in_order = lambda: [mock_plugin]
            mock_registry.get_plugin = lambda name: mock_plugin
            mock_registry_func.return_value = mock_registry

            pipeline = FunctionPipeline()
            result = await pipeline.execute_pipeline(
                sample_content_json,
                job_id=job_id,
                content_type="blog_post",
                output_content_type="press_release",
            )

            # Verify output_content_type was used
            assert result["metadata"]["content_type"] == "blog_post"
            # Verify plugin was called with correct context
            call_args = mock_plugin.execute.call_args
            assert call_args[1]["context"]["output_content_type"] == "press_release"

    @pytest.mark.asyncio
    async def test_execute_pipeline_plugin_not_found(self, sample_content_json):
        """Test pipeline execution when plugin is not found."""
        with patch(
            "marketing_project.services.function_pipeline.get_plugin_registry"
        ) as mock_registry_func:
            mock_registry = MagicMock()
            mock_registry.validate_dependencies = lambda: (True, [])
            mock_plugin = MagicMock()
            mock_plugin.step_name = "seo_keywords"
            mock_plugin.step_number = 1
            mock_registry.get_plugins_in_order = lambda: [mock_plugin]
            mock_registry.get_plugin = lambda name: None  # Plugin not found
            mock_registry_func.return_value = mock_registry

            pipeline = FunctionPipeline()

            with pytest.raises(ValueError, match="Plugin not found for step"):
                await pipeline.execute_pipeline(
                    sample_content_json, content_type="blog_post"
                )

    @pytest.mark.asyncio
    async def test_execute_pipeline_context_validation_failure(
        self, sample_content_json
    ):
        """Test pipeline execution when context validation fails."""
        with patch(
            "marketing_project.services.function_pipeline.get_plugin_registry"
        ) as mock_registry_func:
            mock_registry = MagicMock()
            mock_registry.validate_dependencies = lambda: (True, [])
            mock_plugin = MagicMock()
            mock_plugin.step_name = "seo_keywords"
            mock_plugin.step_number = 1
            mock_plugin.get_required_context_keys = lambda: ["missing_key"]
            mock_plugin.validate_context = lambda ctx: False
            mock_registry.get_plugins_in_order = lambda: [mock_plugin]
            mock_registry.get_plugin = lambda name: mock_plugin
            mock_registry_func.return_value = mock_registry

            pipeline = FunctionPipeline()

            with pytest.raises(ValueError, match="Missing required context keys"):
                await pipeline.execute_pipeline(
                    sample_content_json, content_type="blog_post"
                )


class TestFunctionPipelineStepExecution:
    """Test individual step execution."""

    @pytest.mark.asyncio
    async def test_execute_step_with_plugin(self):
        """Test executing a single step with plugin."""
        pipeline = FunctionPipeline()

        mock_plugin = MagicMock()
        mock_plugin.step_name = "seo_keywords"
        mock_plugin.get_required_context_keys = lambda: []
        mock_plugin.validate_context = lambda ctx: True
        mock_plugin.execute = AsyncMock(
            return_value=SEOKeywordsResult(primary_keywords=["test"])
        )

        with patch(
            "marketing_project.services.function_pipeline.get_plugin_registry"
        ) as mock_registry_func:
            mock_registry = MagicMock()
            mock_registry.get_plugin = lambda name: mock_plugin
            mock_registry_func.return_value = mock_registry

            context = {"input_content": {"title": "Test"}}
            result = await pipeline._execute_step_with_plugin(
                "seo_keywords", context, job_id="test-job"
            )

            assert isinstance(result, SEOKeywordsResult)
            mock_plugin.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_step_plugin_not_found(self):
        """Test executing step when plugin is not found."""
        pipeline = FunctionPipeline()

        with patch(
            "marketing_project.services.function_pipeline.get_plugin_registry"
        ) as mock_registry_func:
            mock_registry = MagicMock()
            mock_registry.get_plugin = lambda name: None
            mock_registry_func.return_value = mock_registry

            with pytest.raises(ValueError, match="Plugin not found for step"):
                await pipeline._execute_step_with_plugin(
                    "nonexistent_step", {}, job_id="test-job"
                )
