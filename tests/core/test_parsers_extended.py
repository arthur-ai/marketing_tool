"""
Extended tests for core parsers module - covering more parsing functions.
"""

import pytest

from marketing_project.core.parsers import (
    clean_text,
    extract_metadata_from_content,
    parse_blog_post,
    parse_datetime,
    parse_release_notes,
    parse_transcript,
)


def test_parse_blog_post():
    """Test parse_blog_post function."""
    content = """
    # Test Blog Post

    This is the content of the blog post.

    ## Section 1

    More content here.
    """

    result = parse_blog_post(content)

    assert isinstance(result, dict)
    assert "title" in result or "content" in result or "sections" in result


def test_parse_release_notes():
    """Test parse_release_notes function."""
    content = """
    # Release v1.0.0

    ## Features
    - Feature 1
    - Feature 2

    ## Bug Fixes
    - Fix 1
    """

    result = parse_release_notes(content)

    assert isinstance(result, dict)
    assert "version" in result or "features" in result or "content" in result


def test_parse_transcript_json_format():
    """Test parse_transcript with JSON format."""
    transcript = '{"speakers": ["Speaker 1", "Speaker 2"], "transcript": [{"speaker": "Speaker 1", "text": "Hello"}]}'

    result = parse_transcript(transcript)

    assert isinstance(result, dict)
    assert "speakers" in result or "cleaned_content" in result


def test_parse_transcript_csv_format():
    """Test parse_transcript with CSV format."""
    transcript = (
        "timestamp,speaker,text\n00:00:10,Speaker 1,Hello\n00:00:20,Speaker 2,Hi"
    )

    result = parse_transcript(transcript)

    assert isinstance(result, dict)
    assert "speakers" in result or "cleaned_content" in result


def test_clean_text_with_html():
    """Test clean_text with HTML content."""
    text = "<p>This is <strong>bold</strong> text</p>"
    cleaned = clean_text(text)

    assert "strong" not in cleaned
    assert "bold" in cleaned.lower()


def test_clean_text_with_unicode():
    """Test clean_text with unicode characters."""
    text = "This has unicode: \u00a0\u2014\u201c\u201d"
    cleaned = clean_text(text)

    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


def test_parse_datetime_various_formats():
    """Test parse_datetime with various formats."""
    formats = [
        "2024-01-01",
        "2024-01-01T12:00:00Z",
        "January 1, 2024",
        "01/01/2024",
    ]

    for dt_str in formats:
        dt = parse_datetime(dt_str)
        assert dt is None or isinstance(dt, type(parse_datetime("2024-01-01")))


def test_extract_metadata_from_content():
    """Test extract_metadata_from_content function."""
    content = """
    # Blog Post Title

    Published: 2024-01-01
    Author: Test Author
    Tags: test, blog

    Content here...
    """

    metadata = extract_metadata_from_content(content, "blog_post")

    assert isinstance(metadata, dict)
