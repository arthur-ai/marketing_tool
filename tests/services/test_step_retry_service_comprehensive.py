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
        step_retry_service.pipeline, "execute_single_step", new_callable=AsyncMock
    ) as mock_execute:
        mock_result = SEOKeywordsResult(
            main_keyword="test",
            primary_keywords=["test1"],
            confidence_score=0.9,
        )
        mock_execute.return_value = {
            "result": mock_result,
            "step_name": "seo_keywords",
        }

        result = await step_retry_service.retry_step(
            step_name="seo_keywords",
            input_data=input_data,
            context=context,
            job_id="test-job-1",
        )

        assert result is not None
        assert "result" in result


@pytest.mark.asyncio
async def test_retry_step_with_user_guidance(step_retry_service, mock_openai):
    """Test retry_step with user guidance."""
    from marketing_project.models.pipeline_steps import SEOKeywordsResult

    input_data = {
        "content": {"id": "test-1", "title": "Test"},
        "prompt": "Extract keywords",
    }

    with patch.object(
        step_retry_service.pipeline, "execute_single_step", new_callable=AsyncMock
    ) as mock_execute:
        mock_result = SEOKeywordsResult(
            main_keyword="test",
            primary_keywords=["test1"],
            confidence_score=0.9,
        )
        mock_execute.return_value = {
            "result": mock_result,
            "step_name": "seo_keywords",
        }

        result = await step_retry_service.retry_step(
            step_name="seo_keywords",
            input_data=input_data,
            context={},
            user_guidance="Focus on technical keywords",
        )

        assert result is not None
        # User guidance should be included in the prompt
        assert mock_execute.called


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
