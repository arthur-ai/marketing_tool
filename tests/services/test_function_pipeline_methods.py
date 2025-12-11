"""
Tests for function pipeline helper methods.
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


def test_get_system_instruction(function_pipeline):
    """Test _get_system_instruction method."""
    instruction = function_pipeline._get_system_instruction("seo_keywords")

    assert isinstance(instruction, str)
    assert len(instruction) > 0


def test_get_user_prompt(function_pipeline):
    """Test _get_user_prompt method."""
    context = {
        "input_content": {"id": "test", "title": "Test", "content": "Content"},
    }

    prompt = function_pipeline._get_user_prompt("seo_keywords", context)

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_get_step_model(function_pipeline):
    """Test _get_step_model method."""
    model = function_pipeline._get_step_model("seo_keywords")

    assert isinstance(model, str)
    assert len(model) > 0


def test_get_step_temperature(function_pipeline):
    """Test _get_step_temperature method."""
    temp = function_pipeline._get_step_temperature("seo_keywords")

    assert isinstance(temp, (int, float))
    assert 0 <= temp <= 2


def test_get_step_max_retries(function_pipeline):
    """Test _get_step_max_retries method."""
    retries = function_pipeline._get_step_max_retries("seo_keywords")

    assert isinstance(retries, int)
    assert retries >= 0


@pytest.mark.asyncio
async def test_call_function_success(function_pipeline, mock_openai):
    """Test _call_function with successful response."""
    from marketing_project.models.pipeline_steps import SEOKeywordsResult

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = SEOKeywordsResult(
        main_keyword="test",
        primary_keywords=["test1"],
        search_intent="informational",
        confidence_score=0.9,
    )
    mock_openai.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

    result = await function_pipeline._call_function(
        prompt="Test prompt",
        system_instruction="Test instruction",
        response_model=SEOKeywordsResult,
        step_name="seo_keywords",
        step_number=1,
    )

    assert result is not None
    assert hasattr(result, "main_keyword")


@pytest.mark.asyncio
async def test_call_function_with_retry(function_pipeline, mock_openai):
    """Test _call_function with retry on failure."""
    from marketing_project.models.pipeline_steps import SEOKeywordsResult

    # First call fails, second succeeds
    mock_response_success = MagicMock()
    mock_response_success.choices = [MagicMock()]
    mock_response_success.choices[0].message.parsed = SEOKeywordsResult(
        main_keyword="test",
        primary_keywords=["test1"],
        search_intent="informational",
        confidence_score=0.9,
    )

    mock_openai.beta.chat.completions.parse = AsyncMock(
        side_effect=[Exception("Network error"), mock_response_success]
    )

    # Should retry and succeed
    result = await function_pipeline._call_function(
        prompt="Test prompt",
        system_instruction="Test instruction",
        response_model=SEOKeywordsResult,
        step_name="seo_keywords",
        step_number=1,
        context={},
    )

    assert result is not None


@pytest.mark.asyncio
async def test_execute_step_with_plugin_success(function_pipeline):
    """Test _execute_step_with_plugin with successful execution."""
    from marketing_project.models.pipeline_steps import SEOKeywordsResult

    with patch(
        "marketing_project.services.function_pipeline.get_plugin_registry"
    ) as mock_registry:
        mock_plugin = MagicMock()
        # step_name should be a property that returns a string
        type(mock_plugin).step_name = "seo_keywords"
        mock_plugin.step_number = 1
        mock_plugin.get_required_context_keys.return_value = ["input_content"]
        mock_plugin.validate_context.return_value = True
        mock_result = SEOKeywordsResult(
            main_keyword="test",
            primary_keywords=["test1"],
            search_intent="informational",
            confidence_score=0.9,
        )
        mock_plugin.execute = AsyncMock(return_value=mock_result)
        mock_registry.return_value.get_plugin.return_value = mock_plugin

        context = {
            "input_content": {"id": "test", "title": "Test", "content": "Content"},
        }

        # _execute_step_with_plugin takes step_name as string, not plugin object
        result = await function_pipeline._execute_step_with_plugin(
            "seo_keywords", context, "test-job"
        )

        assert result is not None
        assert hasattr(result, "main_keyword")
