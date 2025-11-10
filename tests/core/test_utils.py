"""
Tests for core utilities.
"""

import pytest

from marketing_project.core.models import (
    BlogPostContext,
    ReleaseNotesContext,
    TranscriptContext,
)
from marketing_project.core.utils import (
    convert_dict_to_content_context,
    validate_content_context,
)


class TestConvertDictToContentContext:
    """Test convert_dict_to_content_context function."""

    def test_convert_blog_post_dict(self):
        """Test converting blog post dictionary."""
        blog_dict = {
            "id": "test-123",
            "title": "Test Blog",
            "content": "Test content",
            "author": "Test Author",
            "category": "tech",
        }

        result = convert_dict_to_content_context(blog_dict)
        assert isinstance(result, BlogPostContext)
        assert result.id == "test-123"
        assert result.title == "Test Blog"

    def test_convert_transcript_dict(self):
        """Test converting transcript dictionary."""
        transcript_dict = {
            "id": "test-123",
            "title": "Test Transcript",
            "content": "Speaker 1: Hello",
            "speakers": ["Speaker 1"],
            "transcript_type": "podcast",
        }

        result = convert_dict_to_content_context(transcript_dict)
        assert isinstance(result, TranscriptContext)
        assert result.id == "test-123"
        assert result.title == "Test Transcript"

    def test_convert_release_notes_dict(self):
        """Test converting release notes dictionary."""
        release_dict = {
            "id": "test-123",
            "title": "Version 1.0.0",
            "content": "Release notes content",
            "version": "1.0.0",
        }

        result = convert_dict_to_content_context(release_dict)
        assert isinstance(result, ReleaseNotesContext)
        assert result.id == "test-123"
        assert result.version == "1.0.0"

    def test_convert_invalid_dict(self):
        """Test converting invalid dictionary."""
        invalid_dict = {"invalid": "data"}

        with pytest.raises((ValueError, KeyError, TypeError)):
            convert_dict_to_content_context(invalid_dict)


class TestValidateContentContext:
    """Test validate_content_context function."""

    def test_validate_valid_blog_post(self):
        """Test validation of valid blog post."""
        blog = BlogPostContext(
            id="test-123",
            title="Test Blog",
            content="Test content",
        )

        result = validate_content_context(blog)
        assert result is True

    def test_validate_valid_transcript(self):
        """Test validation of valid transcript."""
        transcript = TranscriptContext(
            id="test-123",
            title="Test Transcript",
            content="Speaker 1: Hello",
        )

        result = validate_content_context(transcript)
        assert result is True

    def test_validate_valid_release_notes(self):
        """Test validation of valid release notes."""
        release = ReleaseNotesContext(
            id="test-123",
            title="Version 1.0.0",
            content="Release notes",
        )

        result = validate_content_context(release)
        assert result is True
