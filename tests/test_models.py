"""
Tests for the core models.
"""

from datetime import datetime

import pytest

from marketing_project.core.models import AppContext, ContentContext, EmailContext
from marketing_project.models.content_models import (
    BlogPostContext,
    ReleaseNotesContext,
    TranscriptContext,
)


def test_email_context():
    """Test EmailContext model (legacy support)."""
    email = EmailContext(
        id="test-123",
        thread_id="thread-456",
        subject="Test Email",
        snippet="This is a test email",
        body="This is the full body of the test email",
    )
    assert email.id == "test-123"
    assert email.thread_id == "thread-456"
    assert email.subject == "Test Email"
    assert email.snippet == "This is a test email"
    assert email.body == "This is the full body of the test email"


def test_base_content_context():
    """Test ContentContext base model (using BlogPostContext as example)."""
    # ContentContext is now a Union type, so we test with a concrete implementation
    content = BlogPostContext(
        id="content-123",
        title="Test Content",
        content="This is the full content",
        snippet="This is a snippet",
        created_at=datetime.now(),
        source_url="https://example.com",
        metadata={"author": "Test Author"},
    )
    assert content.id == "content-123"
    assert content.title == "Test Content"
    assert content.content == "This is the full content"
    assert content.snippet == "This is a snippet"
    assert content.source_url == "https://example.com"
    assert content.metadata["author"] == "Test Author"


def test_transcript_context():
    """Test TranscriptContext model."""
    transcript = TranscriptContext(
        id="transcript-123",
        title="Podcast Episode 1",
        content="Speaker 1: Hello everyone. Speaker 2: Welcome to the show.",
        snippet="A conversation about technology",
        speakers=["Speaker 1", "Speaker 2"],
        duration=2730,  # 45:30 in seconds (int, not string)
        transcript_type="podcast",
    )
    assert transcript.id == "transcript-123"
    assert transcript.title == "Podcast Episode 1"
    assert transcript.transcript_type == "podcast"
    assert len(transcript.speakers) == 2
    assert transcript.duration == 2730


def test_blog_post_context():
    """Test BlogPostContext model."""
    blog_post = BlogPostContext(
        id="blog-123",
        title="How to Use AI",
        content="This is a comprehensive guide about AI...",
        snippet="A guide about artificial intelligence",
        author="John Doe",
        tags=["AI", "Technology", "Guide"],
        category="Technology",
        word_count=1500,
    )
    assert blog_post.id == "blog-123"
    assert blog_post.title == "How to Use AI"
    assert blog_post.author == "John Doe"
    assert "AI" in blog_post.tags
    assert blog_post.category == "Technology"
    assert blog_post.word_count == 1500


def test_release_notes_context():
    """Test ReleaseNotesContext model."""
    release_notes = ReleaseNotesContext(
        id="release-123",
        title="Version 2.0.0 Release Notes",
        content="Major update with new features and improvements...",
        snippet="Version 2.0.0 is now available",
        version="2.0.0",
        release_date=datetime.now(),
        changes=["Added new dashboard", "Improved performance"],
        features=["New user interface", "Enhanced security"],
        bug_fixes=["Fixed login issue", "Resolved memory leak"],
    )
    assert release_notes.id == "release-123"
    assert release_notes.version == "2.0.0"
    assert len(release_notes.changes) == 2
    assert len(release_notes.features) == 2
    assert len(release_notes.bug_fixes) == 2


def test_app_context_with_transcript():
    """Test AppContext model with TranscriptContext."""
    transcript = TranscriptContext(
        id="transcript-123",
        title="Test Podcast",
        content="Full transcript content",
        snippet="Test snippet",
    )
    context = AppContext(
        content=transcript, labels={"category": "technology"}, content_type="transcript"
    )
    assert context.content.id == "transcript-123"
    assert context.content_type == "transcript"
    assert context.labels["category"] == "technology"


def test_app_context_with_blog_post():
    """Test AppContext model with BlogPostContext."""
    blog_post = BlogPostContext(
        id="blog-123",
        title="Test Blog Post",
        content="Full blog content",
        snippet="Test snippet",
    )
    context = AppContext(content=blog_post, content_type="blog_post")
    assert context.content.id == "blog-123"
    assert context.content_type == "blog_post"


def test_app_context_with_release_notes():
    """Test AppContext model with ReleaseNotesContext."""
    release_notes = ReleaseNotesContext(
        id="release-123",
        title="Test Release",
        content="Full release content",
        snippet="Test snippet",
        version="1.0.0",
    )
    context = AppContext(content=release_notes, content_type="release_notes")
    assert context.content.id == "release-123"
    assert context.content_type == "release_notes"
    assert context.content.version == "2.0.0"
