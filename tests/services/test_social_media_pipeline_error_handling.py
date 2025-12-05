"""
Tests for social media pipeline error handling.
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


@pytest.mark.asyncio
async def test_execute_pipeline_invalid_json(social_media_pipeline):
    """Test execute_pipeline with invalid JSON."""
    with pytest.raises(ValueError, match="Invalid JSON"):
        await social_media_pipeline.execute_pipeline(
            content_json="invalid json",
            job_id="test-job-1",
        )


@pytest.mark.asyncio
async def test_execute_pipeline_platform_validation(social_media_pipeline):
    """Test execute_pipeline with invalid platform."""
    content_json = '{"id": "test-1", "title": "Test", "content": "Content"}'

    # Should handle invalid platform gracefully
    with patch.object(
        social_media_pipeline, "_execute_step_with_plugin", new_callable=AsyncMock
    ) as mock_execute:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"content": "Test post"}
        mock_execute.return_value = mock_result

        result = await social_media_pipeline.execute_pipeline(
            content_json=content_json,
            job_id="test-job-1",
            social_media_platform="invalid_platform",
        )

        # Should still execute but may have warnings
        assert result is not None


@pytest.mark.asyncio
async def test_execute_multi_platform_pipeline_invalid_platforms(social_media_pipeline):
    """Test execute_multi_platform_pipeline with invalid platforms."""
    content_json = '{"id": "test-1", "title": "Test", "content": "Content"}'

    # Empty platforms list
    with pytest.raises(ValueError, match="At least one platform"):
        await social_media_pipeline.execute_multi_platform_pipeline(
            content_json=content_json,
            platforms=[],
        )


def test_validate_content_length_exceeds_limit(social_media_pipeline):
    """Test _validate_content_length when content exceeds limit."""
    with patch.object(
        social_media_pipeline,
        "_get_platform_config",
        return_value={"character_limit": 3000},
    ):
        warnings = social_media_pipeline._validate_content_length(
            "a" * 3500, "linkedin"
        )

        assert isinstance(warnings, list)
        assert len(warnings) > 0  # Should have warnings for exceeding limit
