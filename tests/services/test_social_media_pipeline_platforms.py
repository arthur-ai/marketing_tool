"""
Platform-specific tests for Social Media Pipeline.

Tests character limits, formatting, hashtag/mention handling, and content appropriateness
for each platform (LinkedIn, HackerNews, Email).
"""

from unittest.mock import patch

import pytest

from marketing_project.services.social_media_pipeline import SocialMediaPipeline


class TestPlatformCharacterLimits:
    """Test character limit validation for each platform."""

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_linkedin_character_limit(self, mock_openai):
        """Test LinkedIn character limit (3000 characters)."""
        pipeline = SocialMediaPipeline()
        # Clear cached config to ensure mock works
        pipeline._platform_config = None

        # Mock platform config to return proper limits
        with patch.object(pipeline, "_get_platform_config") as mock_config:
            mock_config.return_value = {
                "character_limit": 3000,
                "character_limit_warning": 2800,
            }

            # Content within limit
            content = "a" * 2000
            is_valid, warning = pipeline._validate_content_length(content, "linkedin")
            assert is_valid is True
            assert warning is None

            # Content approaching limit (must be > warning_threshold)
            content = "a" * 2801
            is_valid, warning = pipeline._validate_content_length(content, "linkedin")
            assert is_valid is True
            assert warning is not None
            assert "approaching" in warning.lower()

            # Content exceeding limit
            content = "a" * 3100
            is_valid, warning = pipeline._validate_content_length(content, "linkedin")
            assert is_valid is False
            assert warning is not None
            assert "exceeds" in warning.lower()

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_hackernews_character_limit(self, mock_openai):
        """Test HackerNews character limit (2000 characters)."""
        pipeline = SocialMediaPipeline()
        # Clear cached config to ensure mock works
        pipeline._platform_config = None

        # Mock platform config to return proper limits
        with patch.object(pipeline, "_get_platform_config") as mock_config:
            mock_config.return_value = {
                "character_limit": 2000,
                "character_limit_warning": 1900,
            }

            # Content within limit
            content = "a" * 1500
            is_valid, warning = pipeline._validate_content_length(content, "hackernews")
            assert is_valid is True
            assert warning is None

            # Content approaching limit (must be > warning_threshold)
            content = "a" * 1901
            is_valid, warning = pipeline._validate_content_length(content, "hackernews")
            assert is_valid is True
            assert warning is not None

            # Content exceeding limit
            content = "a" * 2100
            is_valid, warning = pipeline._validate_content_length(content, "hackernews")
            assert is_valid is False
            assert warning is not None

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_email_character_limit_newsletter(self, mock_openai):
        """Test email newsletter character limit (5000 characters)."""
        pipeline = SocialMediaPipeline()
        # Clear cached config to ensure mock works
        pipeline._platform_config = None

        # Mock platform config to return proper limits for newsletter
        with patch.object(pipeline, "_get_platform_config") as mock_config:
            mock_config.return_value = {
                "character_limit": 5000,
                "types": {
                    "newsletter": {
                        "character_limit": 5000,
                        "character_limit_warning": 4800,
                    }
                },
            }

            # Content approaching limit (must be > warning_threshold)
            content = "a" * 4801
            is_valid, warning = pipeline._validate_content_length(
                content, "email", email_type="newsletter"
            )
            assert is_valid is True
            assert warning is not None
            assert "approaching" in warning.lower()

            # Content within limit
            content = "a" * 4000
            is_valid, warning = pipeline._validate_content_length(
                content, "email", email_type="newsletter"
            )
            assert is_valid is True
            assert warning is None

            # Content exceeding limit
            content = "a" * 5100
            is_valid, warning = pipeline._validate_content_length(
                content, "email", email_type="newsletter"
            )
            assert is_valid is False
            assert warning is not None

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_email_character_limit_promotional(self, mock_openai):
        """Test email promotional character limit (3000 characters)."""
        pipeline = SocialMediaPipeline()
        # Clear cached config to ensure mock works
        pipeline._platform_config = None

        # Mock platform config to return proper limits for promotional
        with patch.object(pipeline, "_get_platform_config") as mock_config:
            mock_config.return_value = {
                "character_limit": 5000,
                "types": {
                    "promotional": {
                        "character_limit": 3000,
                        "character_limit_warning": 2800,
                    }
                },
            }

            # Content approaching limit (must be > warning_threshold)
            content = "a" * 2801
            is_valid, warning = pipeline._validate_content_length(
                content, "email", email_type="promotional"
            )
            assert is_valid is True
            assert warning is not None
            assert "approaching" in warning.lower()

            # Content within limit
            content = "a" * 2500
            is_valid, warning = pipeline._validate_content_length(
                content, "email", email_type="promotional"
            )
            assert is_valid is True
            assert warning is None

            # Content exceeding limit
            content = "a" * 3100
            is_valid, warning = pipeline._validate_content_length(
                content, "email", email_type="promotional"
            )
            assert is_valid is False
            assert warning is not None


class TestPlatformConfig:
    """Test platform configuration loading and access."""

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_load_platform_config(self, mock_openai):
        """Test loading platform configuration."""
        pipeline = SocialMediaPipeline()
        config = pipeline._load_platform_config()
        assert isinstance(config, dict)
        assert "platforms" in config or len(config) == 0  # Allow empty config

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_get_platform_config_linkedin(self, mock_openai):
        """Test getting LinkedIn platform configuration."""
        pipeline = SocialMediaPipeline()
        config = pipeline._get_platform_config("linkedin")
        # Config may be empty if file doesn't exist, so just check it's a dict
        assert isinstance(config, dict)

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_get_platform_config_hackernews(self, mock_openai):
        """Test getting HackerNews platform configuration."""
        pipeline = SocialMediaPipeline()
        config = pipeline._get_platform_config("hackernews")
        assert isinstance(config, dict)

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_get_platform_config_email(self, mock_openai):
        """Test getting email platform configuration."""
        pipeline = SocialMediaPipeline()
        config = pipeline._get_platform_config("email")
        assert isinstance(config, dict)

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_get_platform_config_invalid(self, mock_openai):
        """Test getting configuration for invalid platform."""
        pipeline = SocialMediaPipeline()
        config = pipeline._get_platform_config("invalid_platform")
        assert isinstance(config, dict)  # Should return empty dict


class TestPlatformSpecificTemplates:
    """Test platform-specific template selection."""

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_get_system_instruction_platform_specific(self, mock_openai):
        """Test that platform-specific templates are selected when available."""
        pipeline = SocialMediaPipeline()

        # Test LinkedIn-specific template
        context = {"social_media_platform": "linkedin"}
        instruction = pipeline._get_system_instruction(
            "social_media_marketing_brief", context=context
        )
        assert isinstance(instruction, str)
        assert len(instruction) > 0

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_get_system_instruction_fallback(self, mock_openai):
        """Test fallback to generic template when platform-specific not available."""
        pipeline = SocialMediaPipeline()

        # Test with invalid platform
        context = {"social_media_platform": "invalid"}
        instruction = pipeline._get_system_instruction(
            "social_media_marketing_brief", context=context
        )
        assert isinstance(instruction, str)
        assert len(instruction) > 0


class TestPlatformQualityAssessment:
    """Test platform-specific quality score assessment."""

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_assess_platform_quality_linkedin(self, mock_openai):
        """Test quality assessment for LinkedIn."""
        from marketing_project.models.pipeline_steps import SocialMediaPostResult

        pipeline = SocialMediaPipeline()
        result = SocialMediaPostResult(
            platform="linkedin",
            content="Test content",
            linkedin_score=85.0,
            engagement_score=80.0,
            confidence_score=0.9,
        )

        scores = pipeline._assess_platform_quality(result, "linkedin")
        assert "linkedin_score" in scores
        assert scores["linkedin_score"] == 85.0

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_assess_platform_quality_hackernews(self, mock_openai):
        """Test quality assessment for HackerNews."""
        from marketing_project.models.pipeline_steps import SocialMediaPostResult

        pipeline = SocialMediaPipeline()
        result = SocialMediaPostResult(
            platform="hackernews",
            content="Test content",
            hackernews_score=90.0,
            engagement_score=85.0,
            confidence_score=0.95,
        )

        scores = pipeline._assess_platform_quality(result, "hackernews")
        assert "hackernews_score" in scores
        assert scores["hackernews_score"] == 90.0

    @patch("marketing_project.services.social_media_pipeline.AsyncOpenAI")
    def test_assess_platform_quality_email(self, mock_openai):
        """Test quality assessment for Email."""
        from marketing_project.models.pipeline_steps import SocialMediaPostResult

        pipeline = SocialMediaPipeline()
        result = SocialMediaPostResult(
            platform="email",
            content="Test content",
            subject_line="Test Subject",
            email_score=75.0,
            engagement_score=70.0,
            confidence_score=0.85,
        )

        scores = pipeline._assess_platform_quality(result, "email")
        assert "email_score" in scores
        assert scores["email_score"] == 75.0


class TestPlatformContentAppropriateness:
    """Test content appropriateness for each platform."""

    def test_linkedin_professional_tone(self):
        """Test that LinkedIn content should be professional."""
        # This is a placeholder for future content analysis tests
        # Could use sentiment analysis or keyword detection
        pass

    def test_hackernews_technical_depth(self):
        """Test that HackerNews content should be technical."""
        # This is a placeholder for future content analysis tests
        pass

    def test_email_formatting(self):
        """Test that email content should have proper formatting."""
        # This is a placeholder for future content analysis tests
        pass
