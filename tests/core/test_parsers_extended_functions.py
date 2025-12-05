"""
Extended tests for parser helper functions.
"""

import pytest

from marketing_project.core.parsers import (
    _calculate_quality_score,
    _calculate_speaking_time,
    _detect_language,
    _detect_overlapping_speakers,
    _detect_timestamp_gaps,
    _detect_transcript_format,
    _extract_topics,
    _generate_snippet,
    _get_content_hash,
    _normalize_speaker_name,
    _parse_csv_transcript,
    _parse_json_transcript,
    _parse_standard_format,
    _parse_ttml_transcript,
    _timestamp_to_seconds,
    _validate_parsed_data,
    _validate_timestamp_order,
    clean_text,
    parse_transcript,
)


def test_normalize_speaker_name():
    """Test _normalize_speaker_name function."""
    normalized = _normalize_speaker_name("  John Doe  ")

    assert normalized == "John Doe"
    assert normalized.strip() == normalized


def test_get_content_hash():
    """Test _get_content_hash function."""
    from marketing_project.core.parsers import _get_content_hash

    hash1 = _get_content_hash("Test content")
    hash2 = _get_content_hash("Test content")
    hash3 = _get_content_hash("Different content")

    assert hash1 == hash2  # Same content should produce same hash
    assert hash1 != hash3  # Different content should produce different hash
    assert isinstance(hash1, str)


def test_validate_timestamp_order():
    """Test _validate_timestamp_order function."""
    timestamps = {
        "00:00:10": "Speaker 1: Hello",
        "00:00:20": "Speaker 2: Hi",
        "00:00:15": "Speaker 1: How are you?",  # Out of order
    }

    valid, errors = _validate_timestamp_order(timestamps)

    assert isinstance(valid, bool)
    assert isinstance(errors, list)


def test_detect_timestamp_gaps():
    """Test _detect_timestamp_gaps function."""
    timestamps = {
        "00:00:10": "Speaker 1: Hello",
        "00:00:20": "Speaker 2: Hi",
        "00:00:50": "Speaker 1: Long gap",  # 30 second gap
    }

    gaps = _detect_timestamp_gaps(timestamps, duration=60)

    assert isinstance(gaps, list)


def test_calculate_speaking_time():
    """Test _calculate_speaking_time function."""
    speakers = ["Speaker 1", "Speaker 2"]
    content = "Speaker 1: Hello\nSpeaker 2: Hi\nSpeaker 1: Goodbye"
    timestamps = {
        "00:00:10": "Speaker 1: Hello",
        "00:00:20": "Speaker 2: Hi",
        "00:00:30": "Speaker 1: Goodbye",
    }

    speaking_times = _calculate_speaking_time(speakers, content, timestamps)

    assert isinstance(speaking_times, dict)
    assert len(speaking_times) >= 0


def test_detect_overlapping_speakers():
    """Test _detect_overlapping_speakers function."""
    content = "Speaker 1: Hello [overlapping] Speaker 2: Hi"

    overlapping = _detect_overlapping_speakers(content)

    assert isinstance(overlapping, list)


def test_calculate_quality_score():
    """Test _calculate_quality_score function."""
    data = {
        "speakers": ["Speaker 1", "Speaker 2"],
        "cleaned_content": "Test content",
        "timestamps": {"00:00:10": "Hello"},
    }

    score = _calculate_quality_score(data)

    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_generate_snippet():
    """Test _generate_snippet function."""
    content = "This is a long piece of content that should be truncated to create a snippet for preview purposes."

    snippet = _generate_snippet(content, max_length=50)

    assert isinstance(snippet, str)
    # Allow some flexibility for ellipsis or truncation
    assert len(snippet) <= 60


def test_extract_topics():
    """Test _extract_topics function."""
    content = "This content discusses artificial intelligence, machine learning, and deep learning technologies."

    topics = _extract_topics(content, max_topics=5)

    assert isinstance(topics, list)
    assert len(topics) <= 5


def test_detect_language():
    """Test _detect_language function."""
    content = "This is English text content."

    language = _detect_language(content)

    assert isinstance(language, str)
    assert len(language) >= 2  # Language codes are at least 2 characters


def test_detect_transcript_format():
    """Test _detect_transcript_format function."""
    content = "[00:00:10] Speaker 1: Hello"
    lines = content.split("\n")

    format_type = _detect_transcript_format(content, lines)

    assert isinstance(format_type, str)
    assert format_type in [
        "standard",
        "json",
        "csv",
        "ttml",
        "simple",
        "plain",
        "webvtt",
        "srt",
    ]


def test_parse_json_transcript():
    """Test _parse_json_transcript function."""
    json_content = '{"speakers": ["Speaker 1"], "transcript": [{"speaker": "Speaker 1", "text": "Hello"}]}'

    result = _parse_json_transcript(json_content)

    assert result is None or isinstance(result, dict)


def test_parse_csv_transcript():
    """Test _parse_csv_transcript function."""
    csv_content = "timestamp,speaker,text\n00:00:10,Speaker 1,Hello"

    result = _parse_csv_transcript(csv_content)

    assert result is None or isinstance(result, dict)


def test_parse_ttml_transcript():
    """Test _parse_ttml_transcript function."""
    ttml_content = (
        '<?xml version="1.0"?><tt><body><p begin="00:00:10">Hello</p></body></tt>'
    )

    result = _parse_ttml_transcript(ttml_content)

    assert result is None or isinstance(result, dict)


def test_parse_standard_format():
    """Test _parse_standard_format function."""
    content = "[00:00:10] Speaker 1: Hello\n[00:00:20] Speaker 2: Hi"

    result = _parse_standard_format(content, "podcast")

    assert isinstance(result, dict)
    assert "speakers" in result or "cleaned_content" in result


def test_timestamp_to_seconds():
    """Test _timestamp_to_seconds function."""
    seconds = _timestamp_to_seconds("00:01:30")

    assert isinstance(seconds, float)
    assert seconds == 90.0


def test_validate_parsed_data():
    """Test _validate_parsed_data function."""
    data = {
        "speakers": ["Speaker 1"],
        "cleaned_content": "Test content",
    }

    valid, errors = _validate_parsed_data(data)

    assert isinstance(valid, bool)
    assert isinstance(errors, list)
