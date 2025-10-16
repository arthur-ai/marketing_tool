"""
Tests for validation module.
"""

import pytest

from src.marketing_project.models.validation import (
    validate_api_key_format,
    validate_content_length,
)


class TestValidateContentLength:
    """Test content length validation."""

    def test_validate_content_length_valid(self):
        """Test valid content length."""
        content = "This is a valid content with sufficient length"
        assert validate_content_length(content) is True

    def test_validate_content_length_minimum(self):
        """Test minimum content length."""
        content = "1234567890"  # Exactly 10 characters
        assert validate_content_length(content) is True

    def test_validate_content_length_maximum(self):
        """Test maximum content length."""
        content = "x" * 100000  # Exactly 100000 characters
        assert validate_content_length(content) is True

    def test_validate_content_length_too_short(self):
        """Test content that is too short."""
        content = "short"  # Less than 10 characters
        assert validate_content_length(content) is False

    def test_validate_content_length_too_long(self):
        """Test content that is too long."""
        content = "x" * 100001  # More than 100000 characters
        assert validate_content_length(content) is False

    def test_validate_content_length_empty(self):
        """Test empty content."""
        content = ""
        assert validate_content_length(content) is False

    def test_validate_content_length_none(self):
        """Test None content."""
        content = None
        assert validate_content_length(content) is False


class TestValidateApiKeyFormat:
    """Test API key format validation."""

    def test_validate_api_key_format_valid(self):
        """Test valid API key format."""
        api_key = (
            "sk-1234567890abcdef1234567890abcdef"  # 32+ chars, alphanumeric with dashes
        )
        assert validate_api_key_format(api_key) is True

    def test_validate_api_key_format_with_underscores(self):
        """Test API key with underscores."""
        api_key = "sk_1234567890abcdef1234567890abcdef"  # 32+ chars, alphanumeric with underscores
        assert validate_api_key_format(api_key) is True

    def test_validate_api_key_format_mixed_separators(self):
        """Test API key with mixed separators."""
        api_key = "sk-1234567890abcdef_1234567890abcdef"  # 32+ chars, mixed separators
        assert validate_api_key_format(api_key) is True

    def test_validate_api_key_format_too_short(self):
        """Test API key that is too short."""
        api_key = "sk-1234567890abcdef"  # Less than 32 characters
        assert validate_api_key_format(api_key) is False

    def test_validate_api_key_format_invalid_characters(self):
        """Test API key with invalid characters."""
        api_key = "sk-1234567890abcdef1234567890abc@ef"  # Contains @ symbol
        assert validate_api_key_format(api_key) is False

    def test_validate_api_key_format_empty(self):
        """Test empty API key."""
        api_key = ""
        assert validate_api_key_format(api_key) is False

    def test_validate_api_key_format_none(self):
        """Test None API key."""
        api_key = None
        assert validate_api_key_format(api_key) is False
