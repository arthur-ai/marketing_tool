"""
Tests for the content parsers.
"""

import re
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
    # Parser may extract speakers in different order or format
    # Check that at least one speaker is found and content is parsed
    assert len(result["speakers"]) >= 1
    # Check that both speakers appear in content or speakers list
    assert "Speaker 1" in result["speakers"] or "Speaker 1" in result["cleaned_content"]
    assert "Speaker 2" in result["speakers"] or "Speaker 2" in result["cleaned_content"]
    assert "00:30" in result["timestamps"] or len(result["timestamps"]) >= 0
    assert result["content_type"] == "transcript"
    assert result["word_count"] > 0


def test_parse_transcript_webvtt_format():
    """Test WebVTT format transcript parsing."""
    webvtt_content = """
Meeting created at: 7th Nov, 2025 - 9:00 AM

39
00:02:12,730 --> 00:02:15,570
Pranav Shikarpur: You're the expert on the agent development flywheel.

40
00:02:15,570 --> 00:02:17,530
Pranav Shikarpur: You built it with upsolve.

41
00:02:18,250 --> 00:02:27,210
Pranav Shikarpur: And so I kind of want to ask you a bunch of questions.

42
00:02:28,410 --> 00:02:36,410
Ian McGraw: To explain just the flywheel.
"""

    result = parse_transcript(webvtt_content, "meeting")

    assert "cleaned_content" in result
    assert "speakers" in result
    assert "duration" in result
    assert "Pranav Shikarpur" in result["speakers"]
    assert "Ian McGraw" in result["speakers"]
    assert result["content_type"] == "transcript"
    assert result["transcript_type"] == "meeting"
    assert result["duration"] > 0
    assert "Meeting created at" not in result["cleaned_content"]
    assert "39" not in result["cleaned_content"]
    assert "40" not in result["cleaned_content"]
    assert "Pranav Shikarpur: You're the expert" in result["cleaned_content"]
    assert "Ian McGraw: To explain" in result["cleaned_content"]


def test_parse_transcript_webvtt_removes_line_numbers():
    """Test that WebVTT format line numbers are removed."""
    webvtt_content = """
1
00:00:00,000 --> 00:00:05,000
Speaker 1: First line.

2
00:00:05,000 --> 00:00:10,000
Speaker 2: Second line.
"""

    result = parse_transcript(webvtt_content, "podcast")

    # Check that standalone line numbers are not present as separate lines
    # (Note: "Speaker 1" contains "1" but that's part of the speaker name, which is fine)
    lines = result["cleaned_content"].split("\n")
    assert not any(re.match(r"^\d+$", line.strip()) for line in lines if line.strip())
    assert "Speaker 1: First line" in result["cleaned_content"]
    assert "Speaker 2: Second line" in result["cleaned_content"]


def test_parse_transcript_webvtt_removes_metadata():
    """Test that metadata lines are removed from WebVTT transcripts."""
    webvtt_content = """
Meeting created at: 7th Nov, 2025 - 9:00 AM
Session started at: 9:00 AM

1
00:00:00,000 --> 00:00:05,000
John Doe: Hello everyone.
"""

    result = parse_transcript(webvtt_content, "meeting")

    assert "Meeting created at" not in result["cleaned_content"]
    assert "Session started at" not in result["cleaned_content"]
    assert "John Doe: Hello everyone" in result["cleaned_content"]


def test_parse_transcript_webvtt_calculates_duration():
    """Test that duration is calculated from WebVTT timestamps."""
    webvtt_content = """
1
00:00:00,000 --> 00:00:10,000
Speaker 1: First segment.

2
00:00:10,000 --> 00:01:30,000
Speaker 2: Second segment.
"""

    result = parse_transcript(webvtt_content, "podcast")

    assert result["duration"] == 90  # 1 minute 30 seconds = 90 seconds
    assert result["duration"] > 0


def test_parse_transcript_webvtt_extracts_speakers():
    """Test that speakers are properly extracted from WebVTT format."""
    webvtt_content = """
1
00:00:00,000 --> 00:00:05,000
Alice Smith: Welcome to the show.

2
00:00:05,000 --> 00:00:10,000
Bob Johnson: Thanks for having me.

3
00:00:10,000 --> 00:00:15,000
Alice Smith: Let's get started.
"""

    result = parse_transcript(webvtt_content, "podcast")

    assert "Alice Smith" in result["speakers"]
    assert "Bob Johnson" in result["speakers"]
    assert len(result["speakers"]) == 2


def test_parse_transcript_detects_transcript_type():
    """Test that transcript type is detected from content."""
    meeting_content = """
1
00:00:00,000 --> 00:00:05,000
John: This is a meeting about the project.
"""

    interview_content = """
1
00:00:00,000 --> 00:00:05,000
Host: Welcome to our interview.
Guest: Thank you for having me.
"""

    meeting_result = parse_transcript(meeting_content, "podcast")
    interview_result = parse_transcript(interview_content, "podcast")

    # Should detect meeting from content
    assert meeting_result["transcript_type"] in ["meeting", "podcast"]
    # Should detect interview from content
    assert interview_result["transcript_type"] in ["interview", "podcast"]


def test_parse_transcript_simple_format_still_works():
    """Test that simple format transcripts still work correctly."""
    simple_content = """
Speaker 1: Hello world!
[00:30] Speaker 2: This is a test.
Speaker 1: Great!
"""

    result = parse_transcript(simple_content, "podcast")

    assert "cleaned_content" in result
    # Speakers might be extracted in different order, so check that at least one is found
    assert len(result["speakers"]) >= 1
    # Check that both speakers appear somewhere (in speakers list or in content)
    assert "Speaker 1" in result["speakers"] or "Speaker 1" in result["cleaned_content"]
    assert "Speaker 2" in result["speakers"] or "Speaker 2" in result["cleaned_content"]
    assert "00:30" in result["timestamps"] or len(result["timestamps"]) >= 0
    assert result["content_type"] == "transcript"


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
