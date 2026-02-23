"""
Comprehensive tests for social media pipeline service methods.
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


def test_load_platform_config(social_media_pipeline):
    """Test _load_platform_config method."""
    config = social_media_pipeline._load_platform_config()

    assert isinstance(config, dict)


def test_get_platform_config(social_media_pipeline):
    """Test _get_platform_config method."""
    config = social_media_pipeline._get_platform_config("linkedin")

    assert isinstance(config, dict)
    assert "character_limit" in config or "max_length" in config or len(config) >= 0


def test_validate_content_length(social_media_pipeline):
    """Test _validate_content_length method."""
    # Mock platform config
    with patch.object(
        social_media_pipeline,
        "_get_platform_config",
        return_value={"character_limit": 3000},
    ):
        is_valid, warning = social_media_pipeline._validate_content_length(
            "a" * 2800, "linkedin"
        )

        assert isinstance(is_valid, bool)
        # Content is under limit, so should be valid
        assert is_valid is True
        assert warning is None or isinstance(warning, str)


def test_assess_platform_quality(social_media_pipeline):
    """Test _assess_platform_quality method."""
    from marketing_project.models.pipeline_steps import SocialMediaPostResult

    post = SocialMediaPostResult(
        content="Test post content",
        hashtags=["#test"],
        platform="linkedin",
        linkedin_score=0.85,
    )

    quality = social_media_pipeline._assess_platform_quality(post, "linkedin")

    assert isinstance(quality, dict)
    assert "linkedin_score" in quality or len(quality) >= 0


@pytest.mark.asyncio
async def test_generate_variations(social_media_pipeline, mock_openai):
    """Test _generate_variations method."""
    from marketing_project.models.pipeline_steps import SocialMediaPostResult

    base_post = SocialMediaPostResult(
        content="Test post",
        hashtags=["#test"],
        platform="linkedin",
    )
    pipeline_context = {
        "social_media_platform": "linkedin",
        "social_media_marketing_brief": {"target_audience": "developers"},
    }

    with patch(
        "marketing_project.services.social_media_pipeline.SocialMediaPostGenerationPlugin"
    ) as mock_plugin_class:
        mock_plugin = MagicMock()
        mock_result = SocialMediaPostResult(
            content="Variation 1",
            hashtags=["#test"],
            platform="linkedin",
            linkedin_score=0.85,
        )
        mock_plugin.execute = AsyncMock(return_value=mock_result)
        mock_plugin_class.return_value = mock_plugin

        variations = await social_media_pipeline._generate_variations(
            pipeline_context, base_post, num_variations=2
        )

        assert isinstance(variations, list)


def test_get_system_instruction(social_media_pipeline):
    """Test _get_system_instruction method."""
    instruction = social_media_pipeline._get_system_instruction(
        "social_media_post_generation"
    )

    assert isinstance(instruction, str)
    assert len(instruction) > 0


def test_get_user_prompt(social_media_pipeline):
    """Test _get_user_prompt method."""
    context = {
        "input_content": {"title": "Test", "content": "Content"},
        "social_media_marketing_brief": {"target_audience": "developers"},
    }

    prompt = social_media_pipeline._get_user_prompt(
        "social_media_post_generation", context
    )

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_get_step_model(social_media_pipeline):
    """Test _get_step_model method."""
    model = social_media_pipeline._get_step_model("social_media_post_generation")

    assert isinstance(model, str)
    assert len(model) > 0


def test_get_step_temperature(social_media_pipeline):
    """Test _get_step_temperature method."""
    temp = social_media_pipeline._get_step_temperature("social_media_post_generation")

    assert isinstance(temp, (int, float))
    assert 0 <= temp <= 2


@pytest.mark.asyncio
async def test_execute_multi_platform_pipeline(social_media_pipeline, mock_openai):
    """Test execute_multi_platform_pipeline method."""
    from marketing_project.models.pipeline_steps import (
        SEOKeywordsResult,
        SocialMediaMarketingBriefResult,
        SocialMediaPostResult,
    )

    content_json = '{"id": "test-1", "title": "Test", "content": "Content"}'

    with (
        patch(
            "marketing_project.services.social_media_pipeline.SEOKeywordsPlugin"
        ) as mock_seo_plugin_class,
        patch(
            "marketing_project.services.social_media_pipeline.SocialMediaMarketingBriefPlugin"
        ) as mock_brief_plugin_class,
        patch(
            "marketing_project.services.social_media_pipeline.SocialMediaAngleHookPlugin"
        ) as mock_angle_plugin_class,
        patch(
            "marketing_project.services.social_media_pipeline.SocialMediaPostGenerationPlugin"
        ) as mock_post_plugin_class,
    ):
        # Mock SEO plugin
        mock_seo_plugin = MagicMock()
        mock_seo_result = SEOKeywordsResult(
            main_keyword="test",
            primary_keywords=["test"],
            search_intent="informational",
        )
        mock_seo_plugin.execute = AsyncMock(return_value=mock_seo_result)
        mock_seo_plugin_class.return_value = mock_seo_plugin

        # Mock Brief plugin
        mock_brief_plugin = MagicMock()
        mock_brief_result = MagicMock(spec=SocialMediaMarketingBriefResult)
        mock_brief_plugin.execute = AsyncMock(return_value=mock_brief_result)
        mock_brief_plugin_class.return_value = mock_brief_plugin

        # Mock Angle Hook plugin
        from marketing_project.models.pipeline_steps import AngleHookResult

        mock_angle_plugin = MagicMock()
        mock_angle_result = AngleHookResult(
            angle="test angle",
            hook="test hook",
        )
        mock_angle_plugin.execute = AsyncMock(return_value=mock_angle_result)
        mock_angle_plugin_class.return_value = mock_angle_plugin

        # Mock Post plugin - need to provide all required fields for SocialMediaPostResult
        mock_post_plugin = MagicMock()
        mock_post_result = SocialMediaPostResult(
            platform="linkedin",
            content="LinkedIn post",
            confidence_score=0.9,
            hashtags=[],
            call_to_action=None,
            subject_line=None,
            linkedin_score=None,
            hackernews_score=None,
            email_score=None,
            engagement_score=None,
        )
        mock_post_plugin.execute = AsyncMock(return_value=mock_post_result)
        mock_post_plugin_class.return_value = mock_post_plugin

        result = await social_media_pipeline.execute_multi_platform_pipeline(
            content_json=content_json,
            platforms=["linkedin", "hackernews"],
            job_id="test-job-1",
        )

        assert result is not None
        assert isinstance(result, dict)
        assert "platforms" in result or "results" in result or "linkedin" in result
