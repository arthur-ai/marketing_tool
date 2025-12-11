"""
Tests for function pipeline edge cases and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.function_pipeline import FunctionPipeline


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch("marketing_project.services.function_pipeline.AsyncOpenAI") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def function_pipeline(mock_openai):
    """Create a FunctionPipeline instance."""
    return FunctionPipeline()


@pytest.mark.asyncio
async def test_execute_pipeline_with_pipeline_config(function_pipeline):
    """Test execute_pipeline with custom pipeline_config."""
    from marketing_project.models.pipeline_steps import PipelineConfig

    content_json = '{"id": "test-1", "title": "Test", "content": "Content"}'
    pipeline_config = PipelineConfig(
        default_model="gpt-4",
        default_temperature=0.5,
        default_max_retries=3,
    )

    with patch.object(
        function_pipeline, "_execute_step_with_plugin", new_callable=AsyncMock
    ) as mock_execute:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"main_keyword": "test"}
        mock_execute.return_value = mock_result

        result = await function_pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-job-1",
            pipeline_config=pipeline_config,
        )

        assert result is not None
        assert function_pipeline.model == "gpt-4"
        assert function_pipeline.temperature == 0.5


@pytest.mark.asyncio
async def test_execute_pipeline_with_internal_docs_config(function_pipeline):
    """Test execute_pipeline with internal docs config."""
    content_json = '{"id": "test-1", "title": "Test", "content": "Content"}'

    with patch(
        "marketing_project.services.internal_docs_manager.get_internal_docs_manager"
    ) as mock_manager:
        mock_config = MagicMock()
        mock_config.base_url = "https://example.com"
        mock_manager.return_value.get_active_config = AsyncMock(
            return_value=mock_config
        )

        with patch.object(
            function_pipeline, "_execute_step_with_plugin", new_callable=AsyncMock
        ) as mock_execute:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {"main_keyword": "test"}
            mock_execute.return_value = mock_result

            result = await function_pipeline.execute_pipeline(
                content_json=content_json,
                job_id="test-job-1",
            )

            assert result is not None


@pytest.mark.asyncio
async def test_execute_pipeline_with_design_kit_config(function_pipeline):
    """Test execute_pipeline with design kit config."""
    content_json = '{"id": "test-1", "title": "Test", "content": "Content"}'

    with patch(
        "marketing_project.services.design_kit_manager.get_design_kit_manager"
    ) as mock_manager:
        mock_config = MagicMock()
        mock_config.version = "1.0.0"
        mock_manager.return_value.get_active_config = AsyncMock(
            return_value=mock_config
        )

        with patch.object(
            function_pipeline, "_execute_step_with_plugin", new_callable=AsyncMock
        ) as mock_execute:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {"main_keyword": "test"}
            mock_execute.return_value = mock_result

            result = await function_pipeline.execute_pipeline(
                content_json=content_json,
                job_id="test-job-1",
            )

            assert result is not None


@pytest.mark.asyncio
async def test_resume_pipeline_with_pipeline_config(function_pipeline):
    """Test resume_pipeline with custom pipeline_config."""
    from marketing_project.models.pipeline_steps import PipelineConfig

    context_data = {
        "context": {"seo_keywords": {"main_keyword": "test"}},
        "last_step": "seo_keywords",
        "last_step_number": 1,
        "step_result": {"main_keyword": "test"},
        "original_content": {"id": "test-1", "title": "Test", "content": "Content"},
    }

    pipeline_config = PipelineConfig(
        default_model="gpt-4",
        default_temperature=0.5,
    )

    with patch.object(
        function_pipeline, "_execute_step_with_plugin", new_callable=AsyncMock
    ) as mock_execute:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"target_audience": "developers"}
        mock_execute.return_value = mock_result

        result = await function_pipeline.resume_pipeline(
            context_data=context_data,
            job_id="test-job-1",
            pipeline_config=pipeline_config,
        )

        assert result is not None
        assert function_pipeline.model == "gpt-4"


@pytest.mark.asyncio
async def test_execute_single_step_with_pipeline_config(function_pipeline):
    """Test execute_single_step with custom pipeline_config."""
    from marketing_project.models.pipeline_steps import PipelineConfig

    content_json = '{"id": "test-1", "title": "Test", "content": "Content"}'
    context = {"input_content": {"id": "test-1", "title": "Test"}}
    pipeline_config = PipelineConfig(
        default_model="gpt-4",
        default_temperature=0.5,
    )

    with patch(
        "marketing_project.services.function_pipeline.get_plugin_registry"
    ) as mock_registry:
        mock_plugin = MagicMock()
        mock_plugin.get_required_context_keys.return_value = ["input_content"]
        mock_plugin.step_name = "seo_keywords"
        mock_registry.return_value.get_plugin.return_value = mock_plugin

        with patch.object(
            function_pipeline, "_execute_step_with_plugin", new_callable=AsyncMock
        ) as mock_execute:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {"main_keyword": "test"}
            mock_execute.return_value = mock_result

            result = await function_pipeline.execute_single_step(
                step_name="seo_keywords",
                content_json=content_json,
                context=context,
                job_id="test-job-1",
                pipeline_config=pipeline_config,
            )

            assert result is not None
            assert function_pipeline.model == "gpt-4"
