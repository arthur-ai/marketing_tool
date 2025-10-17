"""
Tests for the content parsers.
"""

from datetime import datetime

import pytest

from marketing_project.core.parsers import (
    clean_text,
    extract_metadata_from_content,
    parse_blog_post,
    parse_datetime,
    parse_release_notes,
    parse_transcript,
)


def test_parse_datetime():
    """Test datetime parsing functionality."""
    # Test various datetime formats
    assert parse_datetime("2024-01-15") is not None
    assert parse_datetime("January 15, 2024") is not None
    assert parse_datetime("15/01/2024") is not None
    assert parse_datetime("invalid date") is None


def test_clean_text():
    """Test text cleaning functionality."""
    # Test HTML cleaning
    html_text = "<p>Hello <strong>world</strong>!</p>"
    cleaned = clean_text(html_text)
    assert cleaned == "Hello world!"

    # Test unicode normalization
    unicode_text = "Héllo wörld!"
    cleaned = clean_text(unicode_text)
    assert "Héllo" in cleaned

    # Test whitespace normalization
    messy_text = "  Hello    world  \n\n  "
    cleaned = clean_text(messy_text)
    assert cleaned == "Hello world"


def test_parse_transcript():
    """Test transcript parsing functionality."""
    transcript_content = """
    Speaker 1: Welcome to our podcast!
    Speaker 2: Thanks for having me.
    [00:30] Speaker 1: Let's talk about AI.
    Speaker 2: That's a great topic.
    """

    result = parse_transcript(transcript_content, "podcast")

    assert "cleaned_content" in result
    assert "speakers" in result
    assert "timestamps" in result
    assert "Speaker 1" in result["speakers"]
    assert "Speaker 2" in result["speakers"]
    assert "00:30" in result["timestamps"]
    assert result["content_type"] == "podcast"
    assert result["word_count"] > 0


def test_parse_blog_post():
    """Test blog post parsing functionality."""
    blog_content = """
    # How to Use AI in Marketing

    This is a comprehensive guide about AI.

    ## Key Points
    - AI is transforming marketing
    - Automation is key
    - #AI #Marketing #Automation

    Check out https://example.com for more info.
    """

    result = parse_blog_post(blog_content, {"title": "Test Blog Post"})

    assert "cleaned_content" in result
    assert "title" in result
    assert "headings" in result
    assert "tags" in result
    assert "links" in result
    assert "How to Use AI in Marketing" in result["headings"]
    assert "Key Points" in result["headings"]
    assert "AI" in result["tags"]
    assert "https://example.com" in result["links"]
    assert result["word_count"] > 0
    assert result["reading_time"] is not None


def test_parse_release_notes():
    """Test release notes parsing functionality."""
    release_content = """
    # Version 2.0.0 Release Notes

    Released on 2024-01-15

    ## New Features
    - Added new dashboard
    - Enhanced security features

    ## Bug Fixes
    - Fixed login issue
    - Resolved memory leak

    ## Breaking Changes
    - Removed deprecated API
    """

    result = parse_release_notes(release_content, "2.0.0")

    assert "cleaned_content" in result
    assert "version" in result
    assert "features" in result
    assert "bug_fixes" in result
    assert "breaking_changes" in result
    assert result["version"] == "2.0.0"
    assert "Added new dashboard" in result["features"]
    assert "Fixed login issue" in result["bug_fixes"]
    assert "Removed deprecated API" in result["breaking_changes"]
    assert result["word_count"] > 0


def test_extract_metadata_from_content():
    """Test metadata extraction for different content types."""
    # Test transcript
    transcript_content = "Speaker 1: Hello world!"
    result = extract_metadata_from_content(transcript_content, "transcript")
    assert "cleaned_content" in result
    assert "speakers" in result

    # Test blog post
    blog_content = "# Test Blog Post\nThis is content."
    result = extract_metadata_from_content(blog_content, "blog_post")
    assert "cleaned_content" in result
    assert "title" in result

    # Test release notes
    release_content = "# Version 1.0.0\nNew features added."
    result = extract_metadata_from_content(release_content, "release_notes")
    assert "cleaned_content" in result
    assert "version" in result

    # Test unknown type
    unknown_content = "Some random content."
    result = extract_metadata_from_content(unknown_content, "unknown")
    assert "cleaned_content" in result
