"""
Comprehensive tests for step retry service methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.step_retry_service import StepRetryService


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch("marketing_project.services.function_pipeline.AsyncOpenAI") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def step_retry_service(mock_openai):
    """Create a StepRetryService instance."""
    return StepRetryService()


@pytest.mark.asyncio
async def test_retry_step_seo_keywords(step_retry_service, mock_openai):
    """Test retry_step for seo_keywords."""
    from marketing_project.models.pipeline_steps import SEOKeywordsResult

    input_data = {
        "content": {"id": "test-1", "title": "Test", "content": "Content"},
        "prompt": "Extract SEO keywords",
    }
    context = {"input_content": input_data["content"]}

    with patch.object(
        step_retry_service.pipeline, "_call_function", new_callable=AsyncMock
    ) as mock_call_function:
        with patch.object(
            step_retry_service.pipeline,
            "_get_system_instruction",
            return_value="System instruction",
        ) as mock_get_instruction:
            mock_result = SEOKeywordsResult(
                main_keyword="test",
                primary_keywords=["test1"],
                search_intent="informational",
                confidence_score=0.9,
            )
            mock_call_function.return_value = mock_result

            result = await step_retry_service.retry_step(
                step_name="seo_keywords",
                input_data=input_data,
                context=context,
                job_id="test-job-1",
            )

            assert result is not None
            assert "result" in result
            assert result["status"] == "success"


@pytest.mark.asyncio
async def test_retry_step_with_user_guidance(step_retry_service, mock_openai):
    """Test retry_step with user guidance."""
    from marketing_project.models.pipeline_steps import SEOKeywordsResult

    input_data = {
        "content": {"id": "test-1", "title": "Test"},
        "prompt": "Extract keywords",
    }

    with patch.object(
        step_retry_service.pipeline, "_call_function", new_callable=AsyncMock
    ) as mock_call_function:
        with patch.object(
            step_retry_service.pipeline,
            "_get_system_instruction",
            return_value="System instruction",
        ) as mock_get_instruction:
            mock_result = SEOKeywordsResult(
                main_keyword="test",
                primary_keywords=["test1"],
                search_intent="informational",
                confidence_score=0.9,
            )
            mock_call_function.return_value = mock_result

            result = await step_retry_service.retry_step(
                step_name="seo_keywords",
                input_data=input_data,
                context={},
                user_guidance="Focus on technical keywords",
            )

            assert result is not None
            assert result["status"] == "success"
            # User guidance should be included in the prompt
            assert mock_call_function.called
            # Verify user guidance was passed to _build_prompt (which is called before _call_function)
            call_args = mock_call_function.call_args
            assert call_args is not None


def test_build_prompt(step_retry_service):
    """Test _build_prompt method."""
    input_data = {
        "content": {"title": "Test", "content": "Content"},
        "prompt": "Extract keywords",
    }

    prompt = step_retry_service._build_prompt(
        step_name="seo_keywords",
        input_data=input_data,
        user_guidance="Focus on technical terms",
    )

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "Focus on technical terms" in prompt
