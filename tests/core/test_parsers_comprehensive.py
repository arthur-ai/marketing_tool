"""
Comprehensive tests for core parsers module.
"""

import pytest

from marketing_project.core.parsers import clean_text, parse_datetime, parse_transcript


def test_parse_transcript_simple():
    """Test parse_transcript with simple format."""
    transcript = """
    Speaker 1: Hello, welcome to the podcast.
    Speaker 2: Thanks for having me.
    """

    result = parse_transcript(transcript)

    assert isinstance(result, dict)
    assert "speakers" in result
    assert "cleaned_content" in result


def test_parse_transcript_with_timestamps():
    """Test parse_transcript with timestamps."""
    transcript = """
    [00:00:10] Speaker 1: Hello
    [00:00:20] Speaker 2: Hi there
    """

    result = parse_transcript(transcript)

    assert isinstance(result, dict)
    assert "timestamps" in result or "cleaned_content" in result


def test_clean_text():
    """Test clean_text function."""
    text = "  This   is   a   test  \n\n  text  "
    cleaned = clean_text(text)

    assert cleaned == "This is a test text"
    assert "\n" not in cleaned
    assert "  " not in cleaned


def test_parse_datetime():
    """Test parse_datetime function."""
    from datetime import datetime

    # Test ISO format
    dt_str = "2024-01-01T12:00:00Z"
    dt = parse_datetime(dt_str)

    assert isinstance(dt, datetime)

    # Test date only
    dt_str = "2024-01-01"
    dt = parse_datetime(dt_str)

    assert isinstance(dt, datetime)
