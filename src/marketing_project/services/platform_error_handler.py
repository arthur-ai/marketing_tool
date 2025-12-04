"""
Platform-specific error handling utilities for social media pipeline.

Detects platform-specific errors (character limits, format issues) and provides
auto-fix capabilities and platform-specific error guidance.
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("marketing_project.services.platform_error_handler")


class PlatformErrorHandler:
    """Handler for platform-specific errors in social media pipeline."""

    @staticmethod
    def detect_platform_error(
        error: Exception, platform: str, content: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Detect if an error is platform-specific.

        Args:
            error: The exception that occurred
            platform: Platform name (linkedin, hackernews, email)
            content: Optional content that may have caused the error

        Returns:
            Tuple of (is_platform_error, error_type, error_details)
        """
        error_message = str(error).lower()

        # Character limit errors
        if any(
            keyword in error_message
            for keyword in ["character", "length", "limit", "too long", "exceeds"]
        ):
            if content:
                content_length = len(content)
                platform_limits = {
                    "linkedin": 3000,
                    "hackernews": 2000,
                    "email": 5000,
                }
                limit = platform_limits.get(platform, 3000)
                if content_length > limit:
                    return (
                        True,
                        "character_limit_exceeded",
                        {
                            "platform": platform,
                            "content_length": content_length,
                            "limit": limit,
                            "excess": content_length - limit,
                        },
                    )

        # Format errors
        if any(
            keyword in error_message
            for keyword in ["format", "invalid", "malformed", "structure"]
        ):
            return (True, "format_error", {"platform": platform})

        # Hashtag errors (LinkedIn specific)
        if platform == "linkedin" and any(
            keyword in error_message for keyword in ["hashtag", "tag"]
        ):
            return (True, "hashtag_error", {"platform": platform})

        # Subject line errors (Email specific)
        if platform == "email" and any(
            keyword in error_message
            for keyword in ["subject", "subject line", "header"]
        ):
            return (True, "subject_line_error", {"platform": platform})

        return (False, None, None)

    @staticmethod
    def auto_fix_content(
        content: str, error_type: str, error_details: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """
        Attempt to auto-fix content based on error type.

        Args:
            content: Content to fix
            error_type: Type of error detected
            error_details: Error details dictionary

        Returns:
            Tuple of (fixed_content, was_fixed)
        """
        if error_type == "character_limit_exceeded":
            limit = error_details.get("limit", 3000)
            excess = error_details.get("excess", 0)

            # Try to truncate intelligently (at sentence boundaries)
            if len(content) > limit:
                # Find last sentence boundary before limit
                truncated = content[: limit - 50]  # Leave room for ellipsis
                last_period = truncated.rfind(".")
                last_exclamation = truncated.rfind("!")
                last_question = truncated.rfind("?")

                cut_point = max(last_period, last_exclamation, last_question)
                if cut_point > limit * 0.8:  # Only use if we're keeping most content
                    fixed_content = content[: cut_point + 1] + "..."
                else:
                    # Just truncate at limit
                    fixed_content = content[: limit - 3] + "..."

                logger.info(
                    f"Auto-fixed character limit: truncated from {len(content)} to {len(fixed_content)} chars"
                )
                return (fixed_content, True)

        elif error_type == "hashtag_error":
            # Remove invalid hashtags (keep only alphanumeric)
            hashtag_pattern = r"#(\w+)"
            hashtags = re.findall(hashtag_pattern, content)
            valid_hashtags = [tag for tag in hashtags if tag.replace("_", "").isalnum()]
            if len(valid_hashtags) != len(hashtags):
                # Rebuild content with only valid hashtags
                content_without_hashtags = re.sub(r"#\w+", "", content).strip()
                if valid_hashtags:
                    fixed_content = (
                        content_without_hashtags
                        + "\n\n"
                        + " ".join(f"#{tag}" for tag in valid_hashtags[:5])
                    )
                else:
                    fixed_content = content_without_hashtags
                logger.info("Auto-fixed hashtag errors: removed invalid hashtags")
                return (fixed_content, True)

        elif error_type == "subject_line_error":
            # Ensure subject line is within 50-60 character limit
            if "subject_line" in error_details:
                subject = error_details["subject_line"]
                if len(subject) > 60:
                    fixed_subject = subject[:57] + "..."
                    logger.info(
                        f"Auto-fixed subject line: truncated from {len(subject)} to {len(fixed_subject)} chars"
                    )
                    return (fixed_subject, True)

        return (content, False)

    @staticmethod
    def get_error_guidance(
        error_type: str, platform: str, error_details: Dict[str, Any]
    ) -> str:
        """
        Get platform-specific error guidance for users.

        Args:
            error_type: Type of error
            platform: Platform name
            error_details: Error details

        Returns:
            Human-readable error guidance
        """
        if error_type == "character_limit_exceeded":
            limit = error_details.get("limit", 3000)
            excess = error_details.get("excess", 0)
            return f"Content exceeds {platform} limit of {limit} characters by {excess} characters. Consider shortening the content or splitting into multiple posts."

        elif error_type == "format_error":
            return f"Content format is invalid for {platform}. Please check platform-specific formatting requirements."

        elif error_type == "hashtag_error":
            return "LinkedIn hashtags must be alphanumeric (letters, numbers, underscores only). Remove special characters from hashtags."

        elif error_type == "subject_line_error":
            return "Email subject line should be 50-60 characters for optimal mobile display. Current subject line is too long."

        return f"Platform-specific error on {platform}. Please review content and try again."

    @staticmethod
    def should_retry_platform_error(error_type: str) -> bool:
        """
        Determine if a platform error should be retried after auto-fix.

        Args:
            error_type: Type of platform error

        Returns:
            True if error should be retried after auto-fix
        """
        # Retry after auto-fix for these error types
        retryable_errors = [
            "character_limit_exceeded",
            "hashtag_error",
            "subject_line_error",
        ]
        return error_type in retryable_errors
