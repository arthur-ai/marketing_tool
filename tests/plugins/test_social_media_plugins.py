"""
Tests for social media plugins: SocialMediaAngleHookPlugin and SocialMediaMarketingBriefPlugin.
Targets missed lines in:
  - plugins/social_media_angle_hook/tasks.py
  - plugins/social_media_marketing_brief/tasks.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.pipeline_steps import (
    AngleHookResult,
    SEOKeywordsResult,
    SocialMediaMarketingBriefResult,
)
from marketing_project.plugins.social_media_angle_hook.tasks import (
    SocialMediaAngleHookPlugin,
)
from marketing_project.plugins.social_media_marketing_brief.tasks import (
    SocialMediaMarketingBriefPlugin,
)

# ---------------------------------------------------------------------------
# SocialMediaAngleHookPlugin
# ---------------------------------------------------------------------------


@pytest.fixture
def angle_hook_plugin():
    return SocialMediaAngleHookPlugin()


@pytest.fixture
def angle_hook_context():
    return {
        "social_media_marketing_brief": {
            "platform": "linkedin",
            "target_audience": ["developers"],
            "key_messages": ["Test message"],
            "tone_and_voice": "professional",
            "content_strategy": "educational",
            "distribution_strategy": "organic",
        },
        "social_media_platform": "linkedin",
        "input_content": {"id": "test-1", "title": "Test", "content": "Test content"},
        "email_type": None,
    }


class TestSocialMediaAngleHookPlugin:
    def test_step_name(self, angle_hook_plugin):
        assert angle_hook_plugin.step_name == "social_media_angle_hook"

    def test_step_number(self, angle_hook_plugin):
        assert angle_hook_plugin.step_number == 3

    def test_response_model(self, angle_hook_plugin):
        assert angle_hook_plugin.response_model == AngleHookResult

    def test_get_required_context_keys(self, angle_hook_plugin):
        keys = angle_hook_plugin.get_required_context_keys()
        assert "social_media_marketing_brief" in keys
        assert "social_media_platform" in keys

    def test_build_prompt_context(self, angle_hook_plugin, angle_hook_context):
        """Test _build_prompt_context constructs correct dict (lines 44-58)."""
        result = angle_hook_plugin._build_prompt_context(angle_hook_context)
        assert "brief_result" in result
        assert result["platform"] == "linkedin"
        assert result["email_type"] is None
        assert "input_content" in result

    def test_build_prompt_context_default_platform(self, angle_hook_plugin):
        """Test _build_prompt_context uses default platform when missing (line 47)."""
        context = {
            "social_media_marketing_brief": {
                "platform": "linkedin",
                "target_audience": ["developers"],
                "key_messages": ["msg"],
                "tone_and_voice": "pro",
                "content_strategy": "edu",
                "distribution_strategy": "organic",
            },
            # no social_media_platform key
        }
        result = angle_hook_plugin._build_prompt_context(context)
        assert result["platform"] == "linkedin"

    def test_build_prompt_context_with_email_type(self, angle_hook_plugin):
        """Test _build_prompt_context includes email_type (line 48)."""
        context = {
            "social_media_marketing_brief": {
                "platform": "email",
                "target_audience": ["subscribers"],
                "key_messages": ["msg"],
                "tone_and_voice": "friendly",
                "content_strategy": "nurture",
                "distribution_strategy": "newsletter",
            },
            "social_media_platform": "email",
            "email_type": "newsletter",
        }
        result = angle_hook_plugin._build_prompt_context(context)
        assert result["email_type"] == "newsletter"

    @pytest.mark.asyncio
    async def test_execute_calls_execute_step(
        self, angle_hook_plugin, angle_hook_context
    ):
        """Test execute delegates to _execute_step (line 75)."""
        mock_result = AngleHookResult(
            platform="linkedin",
            angles=["angle1"],
            hooks=["hook1"],
            recommended_angle="angle1",
            recommended_hook="hook1",
            rationale="Test rationale",
        )

        from marketing_project.services.arthur_prompt_client import ArthurPromptResult

        mock_arthur = ArthurPromptResult(
            system_content="Test system", user_template="Test prompt"
        )
        mock_pipeline = MagicMock()
        mock_pipeline._call_function = AsyncMock(return_value=mock_result)

        with patch(
            "marketing_project.services.arthur_prompt_client.fetch_arthur_prompt",
            new=AsyncMock(return_value=mock_arthur),
        ):
            result = await angle_hook_plugin.execute(
                angle_hook_context, mock_pipeline, job_id="test-job"
            )

        assert isinstance(result, AngleHookResult)
        mock_pipeline._call_function.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_without_job_id(self, angle_hook_plugin, angle_hook_context):
        """Test execute works without job_id."""
        mock_result = AngleHookResult(
            platform="linkedin",
            angles=["angle1"],
            hooks=["hook1"],
            recommended_angle="angle1",
            recommended_hook="hook1",
            rationale="Test",
        )

        from marketing_project.services.arthur_prompt_client import ArthurPromptResult

        mock_arthur = ArthurPromptResult(system_content="sys", user_template="user")
        mock_pipeline = MagicMock()
        mock_pipeline._call_function = AsyncMock(return_value=mock_result)

        with patch(
            "marketing_project.services.arthur_prompt_client.fetch_arthur_prompt",
            new=AsyncMock(return_value=mock_arthur),
        ):
            result = await angle_hook_plugin.execute(angle_hook_context, mock_pipeline)

        assert isinstance(result, AngleHookResult)


# ---------------------------------------------------------------------------
# SocialMediaMarketingBriefPlugin
# ---------------------------------------------------------------------------


@pytest.fixture
def brief_plugin():
    return SocialMediaMarketingBriefPlugin()


@pytest.fixture
def brief_context():
    return {
        "seo_keywords": {
            "main_keyword": "test",
            "primary_keywords": ["test", "keyword"],
            "secondary_keywords": [],
            "long_tail_keywords": [],
            "search_intent": "informational",
        },
        "social_media_platform": "linkedin",
        "input_content": {"id": "test-1", "title": "Test", "content": "Test content"},
        "email_type": None,
    }


class TestSocialMediaMarketingBriefPlugin:
    def test_step_name(self, brief_plugin):
        assert brief_plugin.step_name == "social_media_marketing_brief"

    def test_step_number(self, brief_plugin):
        assert brief_plugin.step_number == 2

    def test_response_model(self, brief_plugin):
        assert brief_plugin.response_model == SocialMediaMarketingBriefResult

    def test_get_required_context_keys(self, brief_plugin):
        keys = brief_plugin.get_required_context_keys()
        assert "seo_keywords" in keys
        assert "social_media_platform" in keys

    def test_build_prompt_context(self, brief_plugin, brief_context):
        """Test _build_prompt_context constructs correct dict (lines 45-57)."""
        result = brief_plugin._build_prompt_context(brief_context)
        assert "seo_result" in result
        assert result["platform"] == "linkedin"
        assert result["email_type"] is None
        assert "input_content" in result

    def test_build_prompt_context_default_platform(self, brief_plugin):
        """Test _build_prompt_context defaults platform to linkedin (line 46)."""
        context = {
            "seo_keywords": {
                "main_keyword": "test",
                "primary_keywords": ["test"],
                "search_intent": "informational",
            },
            # no social_media_platform
        }
        result = brief_plugin._build_prompt_context(context)
        assert result["platform"] == "linkedin"

    def test_build_prompt_context_with_email_type(self, brief_plugin):
        """Test _build_prompt_context includes email_type (line 47)."""
        context = {
            "seo_keywords": {
                "main_keyword": "newsletter",
                "primary_keywords": ["newsletter"],
                "search_intent": "navigational",
            },
            "social_media_platform": "email",
            "email_type": "promotional",
        }
        result = brief_plugin._build_prompt_context(context)
        assert result["email_type"] == "promotional"

    @pytest.mark.asyncio
    async def test_execute_calls_execute_step(self, brief_plugin, brief_context):
        """Test execute delegates to _execute_step (line 74)."""
        mock_result = SocialMediaMarketingBriefResult(
            platform="linkedin",
            target_audience=["developers"],
            key_messages=["message"],
            tone_and_voice="professional",
            content_strategy="educational",
            distribution_strategy="organic",
        )

        from marketing_project.services.arthur_prompt_client import ArthurPromptResult

        mock_arthur = ArthurPromptResult(system_content="sys", user_template="user")
        mock_pipeline = MagicMock()
        mock_pipeline._call_function = AsyncMock(return_value=mock_result)

        with patch(
            "marketing_project.services.arthur_prompt_client.fetch_arthur_prompt",
            new=AsyncMock(return_value=mock_arthur),
        ):
            result = await brief_plugin.execute(
                brief_context, mock_pipeline, job_id="test-job"
            )

        assert isinstance(result, SocialMediaMarketingBriefResult)
        mock_pipeline._call_function.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_without_job_id(self, brief_plugin, brief_context):
        """Test execute works without optional job_id."""
        mock_result = SocialMediaMarketingBriefResult(
            platform="linkedin",
            target_audience=["devs"],
            key_messages=["msg"],
            tone_and_voice="pro",
            content_strategy="edu",
            distribution_strategy="organic",
        )

        from marketing_project.services.arthur_prompt_client import ArthurPromptResult

        mock_arthur = ArthurPromptResult(system_content="sys", user_template="user")
        mock_pipeline = MagicMock()
        mock_pipeline._call_function = AsyncMock(return_value=mock_result)

        with patch(
            "marketing_project.services.arthur_prompt_client.fetch_arthur_prompt",
            new=AsyncMock(return_value=mock_arthur),
        ):
            result = await brief_plugin.execute(brief_context, mock_pipeline)

        assert isinstance(result, SocialMediaMarketingBriefResult)
