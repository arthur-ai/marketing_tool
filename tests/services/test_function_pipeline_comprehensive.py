"""
Comprehensive tests for function pipeline service - covering execute_pipeline, resume_pipeline, execute_single_step.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.function_pipeline import FunctionPipeline


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch(
        "marketing_project.services.function_pipeline.pipeline.AsyncOpenAI"
    ) as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def function_pipeline(mock_openai):
    """Create a FunctionPipeline instance."""
    return FunctionPipeline()


@pytest.mark.asyncio
async def test_execute_pipeline_basic(function_pipeline, mock_openai):
    """Test execute_pipeline with basic content."""
    content_json = '{"id": "test-1", "title": "Test", "content": "Test content"}'

    with patch.object(
        function_pipeline, "_execute_step_with_plugin", new_callable=AsyncMock
    ) as mock_execute:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"main_keyword": "test"}
        mock_execute.return_value = mock_result

        result = await function_pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-job-1",
            content_type="blog_post",
        )

        assert result is not None
        assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_execute_pipeline_with_invalid_json(function_pipeline):
    """Test execute_pipeline with invalid JSON."""
    content_json = "invalid json"

    with pytest.raises(ValueError, match="Invalid JSON"):
        await function_pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-job-1",
        )


@pytest.mark.asyncio
async def test_resume_pipeline(function_pipeline):
    """Test resume_pipeline method."""
    context_data = {
        "context": {"seo_keywords": {"main_keyword": "test"}},
        "last_step": "seo_keywords",
        "last_step_number": 1,
        "step_result": {"main_keyword": "test"},
        "original_content": {"id": "test-1", "title": "Test", "content": "Content"},
    }

    with patch.object(
        function_pipeline, "_execute_step_with_plugin", new_callable=AsyncMock
    ) as mock_execute:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"target_audience": "developers"}
        mock_execute.return_value = mock_result

        result = await function_pipeline.resume_pipeline(
            context_data=context_data,
            job_id="test-job-1",
            content_type="blog_post",
        )

        assert result is not None
        assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_resume_pipeline_missing_original_content(function_pipeline):
    """Test resume_pipeline without original_content."""
    context_data = {
        "context": {},
        "last_step": "seo_keywords",
        "last_step_number": 1,
    }

    with pytest.raises(ValueError, match="original content not found"):
        await function_pipeline.resume_pipeline(
            context_data=context_data,
            job_id="test-job-1",
        )


@pytest.mark.asyncio
async def test_execute_single_step(function_pipeline):
    """Test execute_single_step method."""
    content_json = '{"id": "test-1", "title": "Test", "content": "Content"}'
    context = {
        "input_content": {"id": "test-1", "title": "Test"},
        "content_type": "blog_post",
    }

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
            )

            assert result is not None
            assert "result" in result


@pytest.mark.asyncio
async def test_execute_single_step_invalid_json(function_pipeline):
    """Test execute_single_step with invalid JSON."""
    with pytest.raises(ValueError, match="Invalid JSON"):
        await function_pipeline.execute_single_step(
            step_name="seo_keywords",
            content_json="invalid json",
            context={},
        )


@pytest.mark.asyncio
async def test_execute_single_step_not_found(function_pipeline):
    """Test execute_single_step with non-existent step."""
    with patch(
        "marketing_project.services.function_pipeline.get_plugin_registry"
    ) as mock_registry:
        mock_registry.return_value.get_plugin.return_value = None
        mock_registry.return_value.get_all_plugins.return_value = {}

        with pytest.raises(ValueError, match="not found"):
            await function_pipeline.execute_single_step(
                step_name="non_existent",
                content_json='{"id": "test"}',
                context={},
            )


@pytest.mark.asyncio
async def test_execute_single_step_missing_context(function_pipeline):
    """Test execute_single_step with missing required context."""
    with patch(
        "marketing_project.services.function_pipeline.get_plugin_registry"
    ) as mock_registry:
        mock_plugin = MagicMock()
        mock_plugin.get_required_context_keys.return_value = ["required_key"]
        mock_registry.return_value.get_plugin.return_value = mock_plugin

        with pytest.raises(ValueError, match="Missing required context keys"):
            await function_pipeline.execute_single_step(
                step_name="seo_keywords",
                content_json='{"id": "test"}',
                context={},
            )
