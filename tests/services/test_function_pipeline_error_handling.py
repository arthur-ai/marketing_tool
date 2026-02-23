"""
Tests for function pipeline error handling and edge cases.
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
async def test_call_function_with_retry(function_pipeline, mock_openai):
    """Test _call_function with retry logic."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = MagicMock(
        main_keyword="test",
        primary_keywords=["test1"],
        confidence_score=0.9,
    )
    mock_openai.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

    from marketing_project.models.pipeline_steps import SEOKeywordsResult

    result = await function_pipeline._call_function(
        prompt="Test prompt",
        system_instruction="Test instruction",
        response_model=SEOKeywordsResult,
        step_name="seo_keywords",
        step_number=1,
    )

    assert result is not None


@pytest.mark.asyncio
async def test_call_function_with_approval_required(function_pipeline, mock_openai):
    """Test _call_function when approval is required."""
    from marketing_project.models.pipeline_steps import SEOKeywordsResult
    from marketing_project.processors.approval_helper import ApprovalRequiredException

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = MagicMock(
        main_keyword="test",
        primary_keywords=["test1"],
        confidence_score=0.3,  # Low confidence triggers approval
    )
    mock_openai.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

    with patch(
        "marketing_project.processors.approval_helper.check_and_create_approval_request"
    ) as mock_check:
        mock_check.side_effect = ApprovalRequiredException(
            approval_id="approval-1",
            job_id="test-job-1",
            step_name="seo_keywords",
            step_number=1,
        )

        with pytest.raises(ApprovalRequiredException):
            await function_pipeline._call_function(
                prompt="Test prompt",
                system_instruction="Test instruction",
                response_model=SEOKeywordsResult,
                step_name="seo_keywords",
                step_number=1,
                job_id="test-job-1",
            )


@pytest.mark.asyncio
async def test_execute_step_with_plugin_missing_context(function_pipeline):
    """Test _execute_step_with_plugin with missing context."""
    with patch(
        "marketing_project.services.function_pipeline.pipeline.get_plugin_registry"
    ) as mock_registry:
        mock_plugin = MagicMock()
        mock_plugin.get_required_context_keys.return_value = ["required_key"]
        mock_plugin.validate_context.return_value = False
        mock_registry.return_value.get_plugin.return_value = mock_plugin

        with pytest.raises(ValueError, match="Missing required context keys"):
            await function_pipeline._execute_step_with_plugin(
                "seo_keywords", {}, "test-job"
            )
