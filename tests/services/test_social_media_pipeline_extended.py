"""
Extended tests for social media pipeline - covering more methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.social_media_pipeline import SocialMediaPipeline


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch("marketing_project.services.social_media_pipeline.AsyncOpenAI") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def social_media_pipeline(mock_openai):
    """Create a SocialMediaPipeline instance."""
    return SocialMediaPipeline()


def test_fix_schema_additional_properties(social_media_pipeline):
    """Test _fix_schema_additional_properties method."""
    schema = {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "additionalProperties": True,
    }

    fixed = social_media_pipeline._fix_schema_additional_properties(schema)

    assert isinstance(fixed, dict)
    assert "additionalProperties" in fixed or "properties" in fixed


@pytest.mark.asyncio
async def test_call_function(social_media_pipeline, mock_openai):
    """Test _call_function method."""
    from marketing_project.models.pipeline_steps import SocialMediaPostResult

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    # _call_function uses chat.completions.create with response_format, not beta.chat.completions.parse
    # The response has choices[0].message.content which is parsed as JSON
    mock_response.choices[0].message.content = (
        '{"platform": "linkedin", "content": "Test post", "confidence_score": 0.9}'
    )
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = 100
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await social_media_pipeline._call_function(
        prompt="Test prompt",
        system_instruction="Test instruction",
        response_model=SocialMediaPostResult,
        step_name="social_media_post_generation",
        step_number=4,
        context={},
    )

    assert result is not None
    assert hasattr(result, "content") or isinstance(result, dict)


@pytest.mark.asyncio
async def test_execute_step_with_plugin(social_media_pipeline, mock_openai):
    """Test _execute_step_with_plugin method."""
    with patch(
        "marketing_project.services.social_media_pipeline.get_plugin_registry"
    ) as mock_registry:
        mock_plugin = MagicMock()
        mock_plugin.step_name = "social_media_post_generation"
        mock_plugin.step_number = 4
        mock_plugin.get_required_context_keys.return_value = ["input_content"]
        mock_plugin.validate_context.return_value = True
        mock_plugin.execute = AsyncMock(
            return_value=MagicMock(
                platform="linkedin",
                content="Test post",
                confidence_score=0.9,
            )
        )
        mock_registry.return_value.get_plugin.return_value = mock_plugin

        context = {
            "input_content": {"id": "test", "title": "Test", "content": "Content"},
        }

        result = await social_media_pipeline._execute_step_with_plugin(
            mock_plugin, context, "test-job"
        )

        assert result is not None
