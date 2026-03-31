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
async def test_call_function_with_retry(function_pipeline):
    """Test _call_function with retry logic."""
    from marketing_project.models.pipeline_steps import SEOKeywordsResult

    expected = SEOKeywordsResult(
        main_keyword="test",
        primary_keywords=["test1"],
        search_intent="informational",
    )

    with patch.object(
        function_pipeline.llm_client,
        "call_with_retries",
        new_callable=AsyncMock,
        return_value=(expected, MagicMock()),
    ):
        result = await function_pipeline._call_function(
            prompt="Test prompt",
            system_instruction="Test instruction",
            response_model=SEOKeywordsResult,
            step_name="seo_keywords",
            step_number=1,
        )

    assert result is not None


@pytest.mark.asyncio
async def test_call_function_with_approval_required(function_pipeline):
    """Test _call_function returns ApprovalRequiredSentinel when approval is required."""
    from marketing_project.models.pipeline_steps import SEOKeywordsResult
    from marketing_project.processors.approval_helper import (
        ApprovalRequiredSentinel,
        ApprovalResult,
        ApprovalStatus,
    )

    expected = SEOKeywordsResult(
        main_keyword="test",
        primary_keywords=["test1"],
        search_intent="informational",
    )
    approval_result = ApprovalResult(
        status=ApprovalStatus.REQUIRED,
        approval_id="approval-1",
        job_id="test-job-1",
        step_name="seo_keywords",
        step_number=1,
    )

    with patch.object(
        function_pipeline.llm_client,
        "call_with_retries",
        new_callable=AsyncMock,
        return_value=(expected, MagicMock()),
    ):
        with patch(
            "marketing_project.services.function_pipeline.pipeline.check_step_approval",
            new_callable=AsyncMock,
            return_value=approval_result,
        ):
            result = await function_pipeline._call_function(
                prompt="Test prompt",
                system_instruction="Test instruction",
                response_model=SEOKeywordsResult,
                step_name="seo_keywords",
                step_number=1,
                job_id="test-job-1",
            )

    assert isinstance(result, ApprovalRequiredSentinel)
    assert result.approval_result.approval_id == "approval-1"


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
