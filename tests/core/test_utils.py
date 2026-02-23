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
    ensure_content_context,
    validate_content_for_processing,
)


class TestConvertDictToContentContext:
    """Test convert_dict_to_content_context function."""

    def test_convert_blog_post_dict(self):
        """Test converting blog post dictionary."""
        blog_dict = {
            "id": "test-123",
            "title": "Test Blog",
            "content": "Test content",
            "snippet": "Test snippet",
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
            "snippet": "Test snippet",
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
            "snippet": "Test snippet",
            "version": "1.0.0",
        }

        result = convert_dict_to_content_context(release_dict)
        assert isinstance(result, ReleaseNotesContext)
        assert result.id == "test-123"
        assert result.version == "2.0.0"

    def test_convert_invalid_dict(self):
        """Test converting invalid dictionary."""
        invalid_dict = {"invalid": "data"}

        with pytest.raises((ValueError, KeyError, TypeError)):
            convert_dict_to_content_context(invalid_dict)


class TestEnsureContentContext:
    """Test ensure_content_context function."""

    def test_ensure_content_context_with_dict(self):
        """Test ensuring content context from dict."""
        blog_dict = {
            "id": "test-123",
            "title": "Test Blog",
            "content": "Test content",
            "snippet": "Test snippet",
            "author": "Test Author",
        }

        result = ensure_content_context(blog_dict)
        assert isinstance(result, BlogPostContext)

    def test_ensure_content_context_with_object(self):
        """Test ensuring content context with existing object."""
        blog = BlogPostContext(
            id="test-123",
            title="Test Blog",
            content="Test content",
            snippet="Test snippet",
        )

        result = ensure_content_context(blog)
        assert result is blog


class TestValidateContentForProcessing:
    """Test validate_content_for_processing function."""

    def test_validate_valid_blog_post(self):
        """Test validation of valid blog post."""
        blog = BlogPostContext(
            id="test-123",
            title="Test Blog",
            content="Test content with enough words " * 10,  # Make it long enough
            snippet="Test snippet",
        )

        result = validate_content_for_processing(blog)
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

    def test_validate_invalid_blog_post_missing_title(self):
        """Test validation of blog post with missing title."""
        blog = BlogPostContext(
            id="test-123",
            title="",
            content="Test content",
            snippet="Test snippet",
        )

        result = validate_content_for_processing(blog)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

    def test_validate_valid_transcript(self):
        """Test validation of valid transcript."""
        transcript = TranscriptContext(
            id="test-123",
            title="Test Transcript",
            content="Speaker 1: Hello " * 20,  # Make it long enough
            snippet="Test snippet",
        )

        result = validate_content_for_processing(transcript)
        assert result["is_valid"] is True

    def test_validate_valid_release_notes(self):
        """Test validation of valid release notes."""
        release = ReleaseNotesContext(
            id="test-123",
            title="Version 1.0.0",
            content="Release notes " * 20,  # Make it long enough
            snippet="Test snippet",
            version="1.0.0",  # Required field
        )

        result = validate_content_for_processing(release)
        assert result["is_valid"] is True
