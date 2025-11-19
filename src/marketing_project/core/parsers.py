"""
Content parsing utilities for Marketing Project.

This module provides parsing functions for different content types including
transcripts, blog posts, and release notes. It handles text extraction,
cleaning, and formatting for content processing.

Functions:
    parse_datetime: Parse datetime strings from various formats
    parse_transcript: Parse and clean transcript content
    parse_blog_post: Parse and clean blog post content
    parse_release_notes: Parse and clean release notes content
    clean_text: General text cleaning and normalization
"""

import logging
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from dateparser import parse

logger = logging.getLogger("marketing_project.core.parsers")


def parse_datetime(text: str) -> Optional[datetime]:
    """
    Parse datetime strings from various formats.

    Args:
        text: String containing date/time information

    Returns:
        datetime object or None if parsing fails
    """
    try:
        return parse(text)
    except Exception as e:
        logger.warning(f"Failed to parse datetime '{text}': {e}")
        return None


def clean_text(raw_text: str) -> str:
    """
    Clean and normalize text content.

    Args:
        raw_text: Raw text content

    Returns:
        Cleaned and normalized text
    """
    if not raw_text:
        return ""

    # Use BeautifulSoup to convert HTML to plain text if needed
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text(separator=" ")

    # Normalize unicode (NFKC = Compatibility Decomposition, then Composition)
    text = unicodedata.normalize("NFKC", text)

    # Remove control/non-printable characters (except basic whitespace)
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]+", "", text)

    # Replace specific unicode whitespace
    text = text.replace("\u00a0", " ")  # Non-breaking space -> regular space

    # Collapse extra whitespace
    text = " ".join(text.split())

    # Fix spaces before punctuation
    text = re.sub(r"\s+([.!?])", r"\1", text)

    return text


def parse_transcript(raw_content: str, content_type: str = "podcast") -> Dict[str, Any]:
    """
    Parse transcript content and extract structured information.
    Supports WebVTT format, SRT format, and simple transcript formats.

    Args:
        raw_content: Raw transcript text
        content_type: Type of transcript (podcast, video, meeting, etc.)

    Returns:
        Dictionary with parsed transcript data including:
        - cleaned_content: Cleaned transcript text
        - speakers: List of speaker names
        - timestamps: Dictionary of timestamps
        - duration: Duration in seconds (if calculable)
        - word_count: Word count
        - transcript_type: Detected transcript type
    """
    # For transcripts, we need to preserve line breaks for speaker detection
    # Use BeautifulSoup to convert HTML to plain text if needed, but preserve line breaks
    soup = BeautifulSoup(raw_content, "html.parser")
    text = soup.get_text(separator="\n")

    # Normalize unicode and clean up
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF\n]+", "", text)
    text = text.replace("\u00a0", " ")

    # Split into lines for processing
    raw_lines = text.split("\n")

    # Detect format: WebVTT, SRT, or simple
    is_webvtt = False
    is_srt = False
    webvtt_timestamp_pattern = (
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
    )
    srt_timestamp_pattern = (
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
    )

    for line in raw_lines[:20]:  # Check first 20 lines for format detection
        if re.search(webvtt_timestamp_pattern, line):
            is_webvtt = True
            break
        if re.search(srt_timestamp_pattern, line) and re.match(
            r"^\d+$",
            raw_lines[raw_lines.index(line) - 1] if raw_lines.index(line) > 0 else "",
        ):
            is_srt = True
            break

    # Process lines based on format
    processed_lines = []
    speakers = set()
    timestamps = {}
    line_numbers_to_remove = set()
    last_timestamp_seconds = 0

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Remove metadata lines (e.g., "Meeting created at: ...", "Session started at: ...")
        if re.match(
            r"^(Meeting|Session|Recording|Created).*(created|started).*at:",
            line,
            re.IGNORECASE,
        ):
            i += 1
            continue

        # Handle WebVTT/SRT format
        if is_webvtt or is_srt:
            # Check if this is a line number (standalone numeric line)
            if re.match(r"^\d+$", line) and i + 1 < len(raw_lines):
                # Check if next line is a timestamp
                next_line = raw_lines[i + 1].strip()
                if re.search(webvtt_timestamp_pattern, next_line) or re.search(
                    srt_timestamp_pattern, next_line
                ):
                    line_numbers_to_remove.add(i)
                    i += 1
                    continue

            # Check if this is a timestamp line
            timestamp_match = re.search(webvtt_timestamp_pattern, line) or re.search(
                srt_timestamp_pattern, line
            )
            if timestamp_match:
                # Extract start and end timestamps
                start_time = timestamp_match.group(1).replace(",", ".")
                end_time = timestamp_match.group(2).replace(",", ".")

                # Convert to seconds for duration calculation
                try:
                    start_seconds = _timestamp_to_seconds(start_time)
                    end_seconds = _timestamp_to_seconds(end_time)
                    last_timestamp_seconds = max(last_timestamp_seconds, end_seconds)

                    # Store timestamp
                    timestamps[start_time] = ""
                except ValueError:
                    pass

                # Skip timestamp line, next line should be content
                i += 1
                if i < len(raw_lines):
                    content_line = raw_lines[i].strip()
                    if content_line:
                        # Extract speaker if present (format: "Speaker Name: content")
                        speaker_match = re.match(
                            r"^([A-Za-z][A-Za-z0-9\s]{0,48}?):\s*(.+)$", content_line
                        )
                        if speaker_match:
                            speaker = speaker_match.group(1).strip()
                            content = speaker_match.group(2).strip()
                            speakers.add(speaker)
                            processed_lines.append(f"{speaker}: {content}")
                        else:
                            processed_lines.append(content_line)
                i += 1
                continue

        # Handle simple format (existing logic)
        # Check for line numbers (standalone numeric lines followed by content)
        if re.match(r"^\d+$", line) and i + 1 < len(raw_lines):
            next_line = raw_lines[i + 1].strip()
            # If next line doesn't look like a timestamp or structured content, skip the number
            if not re.search(r"\[.*\]|\(.*\)|-->", next_line):
                line_numbers_to_remove.add(i)
                i += 1
                continue

        # Extract speakers (look for patterns like "Speaker 1:", "John:", etc.)
        # Handle lines with timestamps like "[00:30] Speaker 2: content"
        # First try to match with timestamp prefix
        speaker_with_timestamp_pattern = (
            r"^[\[\(].*?[\]\)]\s*([A-Za-z][A-Za-z0-9\s]{0,48}?):\s*(.+)$"
        )
        speaker_match = re.match(speaker_with_timestamp_pattern, line)

        if not speaker_match:
            # Try without timestamp prefix
            speaker_pattern = r"^([A-Za-z][A-Za-z0-9\s]{0,48}?):\s*(.+)$"
            speaker_match = re.match(speaker_pattern, line)

        if speaker_match:
            speaker = speaker_match.group(1).strip()
            content = speaker_match.group(2).strip()
            if len(speaker) < 50 and content:  # Reasonable speaker name length
                speakers.add(speaker)
                processed_lines.append(line)
        else:
            # Regular content line (not starting with speaker pattern)
            # Check if it's a standalone number line (should have been caught earlier)
            if not (re.match(r"^\d+$", line) and i + 1 < len(raw_lines)):
                processed_lines.append(line)

        i += 1

    # Build cleaned content
    cleaned_content = "\n".join(processed_lines)

    # Extract timestamps from simple format (look for patterns like [00:30], (1:23), etc.)
    if not is_webvtt and not is_srt:
        simple_timestamp_pattern = r"[\[\(](\d{1,2}:\d{2}(?::\d{2})?)[\]\)]"
        for line in processed_lines:
            matches = re.findall(simple_timestamp_pattern, line)
            for timestamp in matches:
                if timestamp not in timestamps:
                    timestamps[timestamp] = line.strip()

    # Calculate duration
    duration_seconds = None
    if last_timestamp_seconds > 0:
        duration_seconds = int(last_timestamp_seconds)
    elif timestamps:
        # Try to estimate from last timestamp in simple format
        try:
            last_ts = max(
                timestamps.keys(),
                key=lambda x: _simple_timestamp_to_seconds(x) if ":" in x else 0,
            )
            duration_seconds = _simple_timestamp_to_seconds(last_ts)
        except (ValueError, KeyError):
            pass

    # If no duration found, estimate from word count (~150 words per minute)
    if duration_seconds is None:
        word_count = len(cleaned_content.split())
        duration_seconds = int((word_count / 150) * 60) if word_count > 0 else 0
    else:
        word_count = len(cleaned_content.split())

    # Detect transcript type from content if not specified
    transcript_type = content_type
    if content_type == "podcast":  # Default, try to detect
        content_lower = cleaned_content.lower()
        if any(word in content_lower for word in ["meeting", "call", "zoom", "teams"]):
            transcript_type = "meeting"
        elif any(word in content_lower for word in ["interview", "guest", "host"]):
            transcript_type = "interview"
        elif any(word in content_lower for word in ["video", "youtube", "recording"]):
            transcript_type = "video"

    return {
        "cleaned_content": cleaned_content,
        "speakers": list(speakers),
        "timestamps": timestamps,
        "duration": duration_seconds,
        "word_count": word_count,
        "transcript_type": transcript_type,
        "content_type": "transcript",
    }


def _timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert WebVTT/SRT timestamp (HH:MM:SS.mmm) to seconds.

    Args:
        timestamp: Timestamp string in format HH:MM:SS.mmm or HH:MM:SS,mmm

    Returns:
        Total seconds as float
    """
    # Normalize comma to dot
    timestamp = timestamp.replace(",", ".")

    parts = timestamp.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])

    return hours * 3600 + minutes * 60 + seconds


def _simple_timestamp_to_seconds(timestamp: str) -> int:
    """
    Convert simple timestamp (MM:SS or HH:MM:SS) to seconds.

    Args:
        timestamp: Timestamp string in format MM:SS or HH:MM:SS

    Returns:
        Total seconds as int
    """
    parts = timestamp.split(":")
    if len(parts) == 2:
        # MM:SS format
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # HH:MM:SS format
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def parse_blog_post(
    raw_content: str, metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Parse blog post content and extract structured information.

    Args:
        raw_content: Raw blog post HTML/text
        metadata: Optional metadata dictionary

    Returns:
        Dictionary with parsed blog post data
    """
    # For blog posts, we need to preserve line breaks for heading detection
    # Use BeautifulSoup to convert HTML to plain text if needed, but preserve line breaks
    soup = BeautifulSoup(raw_content, "html.parser")
    text = soup.get_text(separator="\n")

    # Normalize unicode and clean up
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF\n]+", "", text)
    text = text.replace("\u00a0", " ")

    # Clean up whitespace but preserve line structure
    lines = []
    for line in text.split("\n"):
        cleaned_line = " ".join(line.split())
        if cleaned_line:
            lines.append(cleaned_line)

    cleaned_content = "\n".join(lines)

    # Extract title if not provided in metadata
    title = ""
    if metadata and "title" in metadata:
        title = metadata["title"]
    else:
        # Try to extract title from content (first heading or first line)
        for line in lines[:5]:  # Check first 5 lines
            if len(line.strip()) > 10 and len(line.strip()) < 200:
                title = line.strip()
                break

    # Extract headings (look for patterns like # Heading, ## Heading, etc.)
    heading_pattern = r"^#{1,6}\s+(.+)$"
    headings = []
    for line in lines:
        match = re.match(heading_pattern, line.strip())
        if match:
            headings.append(match.group(1).strip())

    # Extract links
    link_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    links = re.findall(link_pattern, cleaned_content)

    # Extract tags (look for #hashtag patterns)
    tag_pattern = r"#([a-zA-Z0-9_]+)"
    tags = re.findall(tag_pattern, cleaned_content)

    # Calculate reading time (average 200 words per minute)
    word_count = len(cleaned_content.split())
    reading_time = f"{max(1, word_count // 200)} min"

    return {
        "cleaned_content": cleaned_content,
        "title": title,
        "headings": headings,
        "links": links,
        "tags": tags,
        "word_count": word_count,
        "reading_time": reading_time,
        "metadata": metadata or {},
    }


def parse_release_notes(
    raw_content: str, version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse release notes content and extract structured information.

    Args:
        raw_content: Raw release notes text
        version: Optional version string

    Returns:
        Dictionary with parsed release notes data
    """
    # For release notes, we need to preserve line breaks for section detection
    # Use BeautifulSoup to convert HTML to plain text if needed, but preserve line breaks
    soup = BeautifulSoup(raw_content, "html.parser")
    text = soup.get_text(separator="\n")

    # Normalize unicode and clean up
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF\n]+", "", text)
    text = text.replace("\u00a0", " ")

    # Clean up whitespace but preserve line structure
    lines = []
    for line in text.split("\n"):
        cleaned_line = " ".join(line.split())
        if cleaned_line:
            lines.append(cleaned_line)

    cleaned_content = "\n".join(lines)

    # Extract version if not provided
    if not version:
        version_pattern = r"v?(\d+\.\d+\.\d+)"
        version_match = re.search(version_pattern, cleaned_content)
        if version_match:
            version = version_match.group(1)

    # Extract different types of changes
    changes = []
    features = []
    bug_fixes = []
    breaking_changes = []

    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section headers (handle markdown format like ## New Features)
        if re.match(r"^#+\s*(features?|new|added)", line, re.IGNORECASE):
            current_section = "features"
        elif re.match(r"^#+\s*(fixes?|bugs?|issues?)", line, re.IGNORECASE):
            current_section = "bug_fixes"
        elif re.match(r"^#+\s*(breaking|breaking changes?)", line, re.IGNORECASE):
            current_section = "breaking"
        elif re.match(r"^#+\s*(changes?|updates?)", line, re.IGNORECASE):
            current_section = "changes"
        elif line.startswith("-") or line.startswith("*") or line.startswith("•"):
            # Extract bullet point
            item = re.sub(r"^[-*•]\s*", "", line).strip()
            if item:
                if current_section == "features":
                    features.append(item)
                elif current_section == "bug_fixes":
                    bug_fixes.append(item)
                elif current_section == "breaking":
                    breaking_changes.append(item)
                else:
                    changes.append(item)

    # Extract release date if present
    date_pattern = r"(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})"
    date_match = re.search(date_pattern, cleaned_content)
    release_date = date_match.group(1) if date_match else None

    return {
        "cleaned_content": cleaned_content,
        "version": version,
        "release_date": release_date,
        "changes": changes,
        "features": features,
        "bug_fixes": bug_fixes,
        "breaking_changes": breaking_changes,
        "word_count": len(cleaned_content.split()),
    }


def extract_metadata_from_content(content: str, content_type: str) -> Dict[str, Any]:
    """
    Extract metadata from content based on its type.

    Args:
        content: Content text
        content_type: Type of content (transcript, blog_post, release_notes)

    Returns:
        Dictionary with extracted metadata
    """
    if content_type == "transcript":
        return parse_transcript(content)
    elif content_type == "blog_post":
        return parse_blog_post(content)
    elif content_type == "release_notes":
        return parse_release_notes(content)
    else:
        return {"cleaned_content": clean_text(content)}
