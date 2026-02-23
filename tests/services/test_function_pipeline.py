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

    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    def test_init_defaults(self, mock_openai_class):
        """Test FunctionPipeline initialization with defaults."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        pipeline = FunctionPipeline()
        assert pipeline.model == "gpt-5.1"
        assert pipeline.temperature == 0.7
        assert pipeline.lang == "en"
        assert pipeline.step_info == []

    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    def test_init_custom_params(self, mock_openai_class):
        """Test FunctionPipeline initialization with custom parameters."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        pipeline = FunctionPipeline(model="gpt-4", temperature=0.5, lang="es")
        assert pipeline.model == "gpt-4"
        assert pipeline.temperature == 0.5
        assert pipeline.lang == "es"


class TestFunctionPipelineExecution:
    """Test FunctionPipeline execution."""

    @pytest.mark.asyncio
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_execute_pipeline_success(
        self, mock_openai_class, sample_content_json
    ):
        """Test successful pipeline execution."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        job_id = str(uuid4())

        # Mock plugins with proper required fields
        mock_seo_keywords = MagicMock()
        mock_seo_keywords.step_name = "seo_keywords"
        mock_seo_keywords.step_number = 1
        mock_seo_keywords.execute = AsyncMock(
            return_value=SEOKeywordsResult(
                main_keyword="test",
                primary_keywords=["test", "content"],
                search_intent="informational",
            )
        )
        mock_seo_keywords.get_required_context_keys = lambda: []
        mock_seo_keywords.validate_context = lambda ctx: True

        mock_marketing_brief = MagicMock()
        mock_marketing_brief.step_name = "marketing_brief"
        mock_marketing_brief.step_number = 2
        mock_marketing_brief.execute = AsyncMock(
            return_value=MarketingBriefResult(
                target_audience=["Test audience"],
                key_messages=["Test message"],
                content_strategy="Test strategy",
            )
        )
        mock_marketing_brief.get_required_context_keys = lambda: []
        mock_marketing_brief.validate_context = lambda ctx: True

        mock_article = MagicMock()
        mock_article.step_name = "article_generation"
        mock_article.step_number = 3
        mock_article.execute = AsyncMock(
            return_value=ArticleGenerationResult(
                article_title="Test Article",
                article_content="Generated article content",
                outline=["Section 1", "Section 2"],
                call_to_action="Learn more",
            )
        )
        mock_article.get_required_context_keys = lambda: []
        mock_article.validate_context = lambda ctx: True

        mock_seo_opt = MagicMock()
        mock_seo_opt.step_name = "seo_optimization"
        mock_seo_opt.step_number = 4
        mock_seo_opt.execute = AsyncMock(
            return_value=SEOOptimizationResult(
                optimized_content="Optimized content",
                meta_title="Test Meta Title",
                meta_description="Test meta description",
                slug="test-slug",
            )
        )
        mock_seo_opt.get_required_context_keys = lambda: []
        mock_seo_opt.validate_context = lambda ctx: True

        mock_links = MagicMock()
        mock_links.step_name = "suggested_links"
        mock_links.step_number = 5
        mock_links.execute = AsyncMock(
            return_value=SuggestedLinksResult(
                internal_links=[], total_suggestions=0, high_confidence_links=0
            )
        )
        mock_links.get_required_context_keys = lambda: []
        mock_links.validate_context = lambda ctx: True

        mock_formatting = MagicMock()
        mock_formatting.step_name = "content_formatting"
        mock_formatting.step_number = 6
        mock_formatting.execute = AsyncMock(
            return_value=ContentFormattingResult(
                formatted_html="<p>Formatted</p>", formatted_markdown="# Formatted"
            )
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
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_execute_pipeline_invalid_json(self, mock_openai_class):
        """Test pipeline execution with invalid JSON."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        pipeline = FunctionPipeline()

        with pytest.raises(ValueError, match="Invalid JSON input"):
            await pipeline.execute_pipeline("invalid json {", content_type="blog_post")

    @pytest.mark.asyncio
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_execute_pipeline_dependency_validation_failure(
        self, mock_openai_class, sample_content_json
    ):
        """Test pipeline execution when dependency validation fails."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        with (
            patch(
                "marketing_project.services.function_pipeline.get_plugin_registry"
            ) as mock_registry_func,
            patch(
                "marketing_project.services.job_manager.get_job_manager"
            ) as mock_job_manager,
            patch(
                "marketing_project.services.internal_docs_manager.get_internal_docs_manager"
            ) as mock_internal_docs,
            patch(
                "marketing_project.services.design_kit_manager.get_design_kit_manager"
            ) as mock_design_kit,
        ):
            mock_registry = MagicMock()
            mock_registry.validate_dependencies.return_value = (
                False,
                ["Missing dependency: seo_keywords"],
            )
            mock_registry_func.return_value = mock_registry

            mock_job_mgr = AsyncMock()
            mock_job_mgr.get_job = AsyncMock(return_value=None)
            mock_job_manager.return_value = mock_job_mgr

            # Mock internal docs and design kit managers to avoid Redis connection
            mock_internal_docs_mgr = AsyncMock()
            mock_internal_docs_mgr.get_active_config = AsyncMock(return_value=None)
            mock_internal_docs.return_value = mock_internal_docs_mgr

            mock_design_kit_mgr = AsyncMock()
            mock_design_kit_mgr.get_active_config = AsyncMock(return_value=None)
            mock_design_kit.return_value = mock_design_kit_mgr

            pipeline = FunctionPipeline()

            # Pipeline catches ValueError and returns error result, so check result instead
            result = await pipeline.execute_pipeline(
                sample_content_json, content_type="blog_post"
            )
            assert result["pipeline_status"] == "failed"
            assert "Pipeline dependency validation failed" in result.get(
                "metadata", {}
            ).get("error", "")

    @pytest.mark.asyncio
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_execute_pipeline_with_output_content_type(
        self, mock_openai_class, sample_content_json
    ):
        """Test pipeline execution with custom output_content_type."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        job_id = str(uuid4())

        # Mock a simple plugin
        mock_plugin = MagicMock()
        mock_plugin.step_name = "seo_keywords"
        mock_plugin.step_number = 1
        mock_plugin.execute = AsyncMock(
            return_value=SEOKeywordsResult(
                main_keyword="test",
                primary_keywords=["test"],
                search_intent="informational",
            )
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
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_execute_pipeline_plugin_not_found(
        self, mock_openai_class, sample_content_json
    ):
        """Test pipeline execution when plugin is not found."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        with (
            patch(
                "marketing_project.services.function_pipeline.get_plugin_registry"
            ) as mock_registry_func,
            patch(
                "marketing_project.services.job_manager.get_job_manager"
            ) as mock_job_manager,
            patch(
                "marketing_project.services.internal_docs_manager.get_internal_docs_manager"
            ) as mock_internal_docs,
            patch(
                "marketing_project.services.design_kit_manager.get_design_kit_manager"
            ) as mock_design_kit,
        ):
            mock_registry = MagicMock()
            mock_registry.validate_dependencies.return_value = (True, [])
            mock_plugin = MagicMock()
            mock_plugin.step_name = "seo_keywords"
            mock_plugin.step_number = 1
            mock_registry.get_plugins_in_order.return_value = [mock_plugin]
            mock_registry.get_plugin.return_value = None  # Plugin not found
            mock_registry_func.return_value = mock_registry

            mock_job_mgr = AsyncMock()
            mock_job_mgr.get_job = AsyncMock(return_value=None)
            mock_job_manager.return_value = mock_job_mgr

            # Mock internal docs and design kit managers to avoid Redis connection
            mock_internal_docs_mgr = AsyncMock()
            mock_internal_docs_mgr.get_active_config = AsyncMock(return_value=None)
            mock_internal_docs.return_value = mock_internal_docs_mgr

            mock_design_kit_mgr = AsyncMock()
            mock_design_kit_mgr.get_active_config = AsyncMock(return_value=None)
            mock_design_kit.return_value = mock_design_kit_mgr

            pipeline = FunctionPipeline()

            # Pipeline catches exceptions and returns error result, so check result instead
            result = await pipeline.execute_pipeline(
                sample_content_json, content_type="blog_post"
            )
            assert result["pipeline_status"] == "failed"
            assert "Plugin not found for step" in result.get("metadata", {}).get(
                "error", ""
            )

    @pytest.mark.asyncio
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_execute_pipeline_context_validation_failure(
        self, mock_openai_class, sample_content_json
    ):
        """Test pipeline execution when context validation fails."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        with (
            patch(
                "marketing_project.services.function_pipeline.get_plugin_registry"
            ) as mock_registry_func,
            patch(
                "marketing_project.services.job_manager.get_job_manager"
            ) as mock_job_manager,
            patch(
                "marketing_project.services.internal_docs_manager.get_internal_docs_manager"
            ) as mock_internal_docs,
            patch(
                "marketing_project.services.design_kit_manager.get_design_kit_manager"
            ) as mock_design_kit,
        ):
            mock_registry = MagicMock()
            mock_registry.validate_dependencies.return_value = (True, [])
            mock_plugin = MagicMock()
            mock_plugin.step_name = "seo_keywords"
            mock_plugin.step_number = 1
            mock_plugin.get_required_context_keys.return_value = ["missing_key"]
            mock_plugin.validate_context.return_value = False
            mock_registry.get_plugins_in_order.return_value = [mock_plugin]
            mock_registry.get_plugin.return_value = mock_plugin
            mock_registry_func.return_value = mock_registry

            mock_job_mgr = AsyncMock()
            mock_job_mgr.get_job = AsyncMock(return_value=None)
            mock_job_manager.return_value = mock_job_mgr

            # Mock internal docs and design kit managers to avoid Redis connection
            mock_internal_docs_mgr = AsyncMock()
            mock_internal_docs_mgr.get_active_config = AsyncMock(return_value=None)
            mock_internal_docs.return_value = mock_internal_docs_mgr

            mock_design_kit_mgr = AsyncMock()
            mock_design_kit_mgr.get_active_config = AsyncMock(return_value=None)
            mock_design_kit.return_value = mock_design_kit_mgr

            pipeline = FunctionPipeline()

            # Pipeline catches exceptions and returns error result, so check result instead
            result = await pipeline.execute_pipeline(
                sample_content_json, content_type="blog_post"
            )
            assert result["pipeline_status"] == "failed"
            assert "Missing required context keys" in result.get("metadata", {}).get(
                "error", ""
            )


class TestFunctionPipelineStepExecution:
    """Test individual step execution."""

    @pytest.mark.asyncio
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_execute_step_with_plugin(self, mock_openai_class):
        """Test executing a single step with plugin."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        pipeline = FunctionPipeline()

        mock_plugin = MagicMock()
        mock_plugin.step_name = "seo_keywords"
        mock_plugin.get_required_context_keys = lambda: []
        mock_plugin.validate_context = lambda ctx: True
        mock_plugin.execute = AsyncMock(
            return_value=SEOKeywordsResult(
                main_keyword="test",
                primary_keywords=["test"],
                search_intent="informational",
            )
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
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_execute_step_plugin_not_found(self, mock_openai_class):
        """Test executing step when plugin is not found."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

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

    @pytest.mark.asyncio
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_call_function(self, mock_openai_class):
        """Test _call_function method."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.parsed = SEOKeywordsResult(
            main_keyword="test",
            primary_keywords=["test"],
            search_intent="informational",
        )
        mock_response.choices = [mock_choice]
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client

        pipeline = FunctionPipeline()
        result = await pipeline._call_function(
            prompt="Test prompt",
            system_instruction="System instruction",
            response_model=SEOKeywordsResult,
            step_name="seo_keywords",
            step_number=1,
        )

        assert isinstance(result, SEOKeywordsResult)
        assert result.main_keyword == "test"

    @pytest.mark.asyncio
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_resume_pipeline(self, mock_openai_class):
        """Test resume_pipeline method."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        pipeline = FunctionPipeline()

        # Mock the plugins and registry
        with patch(
            "marketing_project.services.function_pipeline.get_plugin_registry"
        ) as mock_registry:
            mock_plugin = MagicMock()
            mock_plugin.step_name = "seo_keywords"
            mock_plugin.step_number = 2
            mock_plugin.execute = AsyncMock(
                return_value=SEOKeywordsResult(
                    main_keyword="test",
                    primary_keywords=["test"],
                    search_intent="informational",
                )
            )
            mock_registry_instance = MagicMock()
            mock_registry_instance.get_plugin.return_value = mock_plugin
            mock_registry_instance.get_all_plugins.return_value = {
                "seo_keywords": mock_plugin
            }
            mock_registry.return_value = mock_registry_instance

            with patch.object(
                pipeline, "execute_pipeline", new_callable=AsyncMock
            ) as mock_execute:
                mock_execute.return_value = {
                    "seo_keywords": {
                        "main_keyword": "test",
                        "primary_keywords": ["test"],
                        "search_intent": "informational",
                    }
                }

                result = await pipeline.resume_pipeline(
                    context_data={
                        "context": {
                            "seo_keywords": {},
                            "input_content": {"title": "Test"},
                        },
                        "original_content": {
                            "id": "test",
                            "title": "Test",
                            "content": "Test content",
                            "snippet": "Test snippet",
                        },
                        "last_step": "seo_keywords",
                        "last_step_number": 1,
                        "step_result": {},
                    },
                    content_type="blog_post",
                    job_id="test-job",
                )

                assert result is not None
                assert isinstance(result, dict)

    @pytest.mark.asyncio
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_execute_single_step(self, mock_openai_class):
        """Test execute_single_step method."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        pipeline = FunctionPipeline()

        with patch.object(
            pipeline, "_execute_step_with_plugin", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = SEOKeywordsResult(
                main_keyword="test",
                primary_keywords=["test"],
                search_intent="informational",
            )

            result = await pipeline.execute_single_step(
                step_name="seo_keywords",
                content_json='{"id": "test", "title": "Test", "content": "Test content", "snippet": "Test snippet"}',
                context={"input_content": {"title": "Test"}},
                job_id="test-job",
            )

            # Function returns a dict with execution metadata and result
            assert isinstance(result, dict)
            assert "result" in result or "step_result" in result
            # The actual step result is wrapped in the dict
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    @patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI")
    async def test_get_user_prompt(self, mock_openai_class):
        """Test _get_user_prompt method."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        pipeline = FunctionPipeline()
        context = {"input_content": {"title": "Test", "content": "Test content"}}

        prompt = pipeline._get_user_prompt("seo_keywords", context)

        assert prompt is not None
        assert isinstance(prompt, str)
        assert len(prompt) > 0
