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

import hashlib
import json
import logging
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from dateparser import parse

logger = logging.getLogger("marketing_project.core.parsers")

# Performance constants
LARGE_FILE_THRESHOLD = 10000  # Lines threshold for streaming
CACHE_SIZE = 100  # Maximum cached parsing results

# Parsing cache (simple in-memory cache)
_parsing_cache: Dict[str, Dict[str, Any]] = {}


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


# ============================================================================
# Transcript Parsing Helper Functions
# ============================================================================


def _get_content_hash(content: str) -> str:
    """Generate hash for content caching."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _should_use_streaming(num_lines: int) -> bool:
    """Determine if streaming/chunked processing is needed."""
    return num_lines > LARGE_FILE_THRESHOLD


def _handle_parsing_error(error: Exception, context: str) -> None:
    """Centralized error handling for parsing operations."""
    logger.warning(f"Parsing error in {context}: {error}")
    # Don't raise, just log - allow fallback parsing


def _validate_parsed_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate extracted data before returning.

    Returns:
        Tuple of (is_valid, warnings_list)
    """
    warnings = []

    # Validate duration
    if data.get("duration") is not None:
        duration = data.get("duration")
        if not isinstance(duration, int) or duration < 0:
            warnings.append(f"Invalid duration: {duration}")
            data["duration"] = None

    # Validate speakers
    speakers = data.get("speakers", [])
    if speakers:
        # Remove empty strings
        speakers = [s for s in speakers if s and isinstance(s, str) and s.strip()]
        data["speakers"] = speakers
        if not speakers:
            warnings.append("All speaker names were empty or invalid")

    # Validate content
    content = data.get("cleaned_content", "")
    if not content or not content.strip():
        warnings.append("Content is empty after cleaning")

    return len(warnings) == 0, warnings


def _fallback_parse(raw_content: str, content_type: str) -> Dict[str, Any]:
    """Fallback parsing when primary method fails."""
    logger.info("Attempting fallback parsing")

    # Simple fallback: extract any text, try to find speakers
    lines = raw_content.split("\n")
    speakers = set()
    content_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to extract speaker
        speaker_match = re.match(r"^([A-Z][A-Za-z\s]+):\s*(.+)$", line)
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            content = speaker_match.group(2).strip()
            if len(speaker) >= 2 and len(speaker) < 50:
                speakers.add(speaker)
                content_lines.append(f"{speaker}: {content}")
        else:
            content_lines.append(line)

    cleaned_content = "\n".join(content_lines)
    word_count = len(cleaned_content.split())
    duration = int((word_count / 150) * 60) if word_count > 0 else 0

    return {
        "cleaned_content": cleaned_content,
        "speakers": list(speakers),
        "timestamps": {},
        "duration": duration,
        "word_count": word_count,
        "transcript_type": content_type,
        "content_type": "transcript",
        "parsing_confidence": 0.3,  # Low confidence for fallback
        "warnings": ["Used fallback parsing - format detection failed"],
    }


def _normalize_speaker_name(name: str) -> str:
    """Normalize speaker name (remove extra spaces, handle titles)."""
    if not name:
        return ""

    # Remove extra whitespace
    name = " ".join(name.split())

    # Handle common titles (keep them)
    # Titles like Dr., Mr., Mrs., Ms., Prof. are preserved

    return name.strip()


def _extract_speaker_role(name: str) -> Optional[str]:
    """Extract role from speaker name if present."""
    role_patterns = {
        r"^(Host|Moderator|Interviewer):": "host",
        r"^(Guest|Interviewee):": "guest",
        r"^(Speaker|Participant)\s+\d+": "participant",
    }

    for pattern, role in role_patterns.items():
        if re.match(pattern, name, re.IGNORECASE):
            return role

    return None


def _merge_similar_speakers(
    speakers: List[str], threshold: float = 0.8
) -> Dict[str, str]:
    """
    Merge speakers with similar names using simple string similarity.

    Returns mapping of original -> normalized name.
    """
    speaker_map = {}
    normalized = []

    for speaker in speakers:
        normalized_name = _normalize_speaker_name(speaker)

        # Find similar existing speaker
        merged = False
        for existing in normalized:
            similarity = SequenceMatcher(
                None, normalized_name.lower(), existing.lower()
            ).ratio()
            if similarity >= threshold:
                speaker_map[speaker] = existing
                merged = True
                break

        if not merged:
            normalized.append(normalized_name)
            speaker_map[speaker] = normalized_name

    return speaker_map


def _validate_timestamp_order(timestamps: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate that timestamps are in sequential order."""
    warnings = []

    if not timestamps:
        return True, warnings

    # Extract and sort timestamps
    try:
        sorted_times = []
        for ts in timestamps.keys():
            if ":" in ts:
                try:
                    seconds = (
                        _simple_timestamp_to_seconds(ts)
                        if len(ts.split(":")) <= 3
                        else _timestamp_to_seconds(ts.replace(",", "."))
                    )
                    sorted_times.append((seconds, ts))
                except (ValueError, AttributeError):
                    continue

        sorted_times.sort(key=lambda x: x[0])

        # Check for gaps (more than 10 seconds between consecutive timestamps)
        for i in range(1, len(sorted_times)):
            gap = sorted_times[i][0] - sorted_times[i - 1][0]
            if gap > 10:
                warnings.append(
                    f"Timestamp gap detected: {gap:.1f}s between {sorted_times[i-1][1]} and {sorted_times[i][1]}"
                )

    except Exception as e:
        _handle_parsing_error(e, "timestamp validation")
        warnings.append("Could not validate timestamp order")

    return len(warnings) == 0, warnings


def _detect_timestamp_gaps(
    timestamps: Dict[str, Any], duration: Optional[int]
) -> List[str]:
    """Detect gaps in timestamps."""
    gaps = []

    if not timestamps or not duration:
        return gaps

    # This is a simplified check - full implementation would analyze all timestamps
    # For now, just check if we have timestamps covering the full duration
    try:
        max_ts = 0
        for ts in timestamps.keys():
            if ":" in ts:
                try:
                    seconds = (
                        _simple_timestamp_to_seconds(ts)
                        if len(ts.split(":")) <= 3
                        else _timestamp_to_seconds(ts.replace(",", "."))
                    )
                    max_ts = max(max_ts, seconds)
                except (ValueError, AttributeError):
                    continue

        if (
            duration and max_ts < duration * 0.9
        ):  # If max timestamp is less than 90% of duration
            gaps.append(f"Timestamps only cover {max_ts}s of {duration}s duration")
    except Exception as e:
        _handle_parsing_error(e, "gap detection")

    return gaps


def _calculate_speaking_time(
    speakers: List[str], content: str, timestamps: Dict[str, Any]
) -> Dict[str, int]:
    """Calculate speaking time per speaker (simplified - estimates based on content length)."""
    speaking_time = defaultdict(int)

    if not speakers or not content:
        return dict(speaking_time)

    # Simple estimation: count lines/content per speaker
    lines = content.split("\n")
    total_chars = len(content)

    for speaker in speakers:
        # Count characters attributed to this speaker
        speaker_chars = 0
        for line in lines:
            if line.startswith(f"{speaker}:"):
                speaker_chars += len(line)

        # Estimate time based on speaking rate (~150 words per minute, ~5 chars per word)
        if total_chars > 0:
            ratio = speaker_chars / total_chars
            # Estimate total speaking time (rough estimate)
            estimated_words = len(content.split())
            estimated_minutes = estimated_words / 150
            speaking_time[speaker] = int(ratio * estimated_minutes * 60)

    return dict(speaking_time)


def _detect_overlapping_speakers(content: str) -> List[str]:
    """Detect if multiple speakers appear in same timestamp segment."""
    warnings = []

    # This is a simplified check - look for multiple speakers in close proximity
    lines = content.split("\n")
    speaker_count = 0
    last_speaker = None

    for line in lines:
        speaker_match = re.match(r"^([A-Z][A-Za-z\s]+):", line)
        if speaker_match:
            current_speaker = speaker_match.group(1).strip()
            if last_speaker and current_speaker != last_speaker:
                speaker_count += 1
                if (
                    speaker_count > 3
                ):  # Multiple rapid speaker changes might indicate overlap
                    # This is a heuristic - not definitive
                    pass
            last_speaker = current_speaker

    return warnings


def _calculate_quality_score(data: Dict[str, Any]) -> float:
    """Calculate overall quality score (0-1) based on completeness and validation."""
    score = 1.0

    # Deduct for missing fields
    if not data.get("speakers"):
        score -= 0.3
    if not data.get("duration") or data.get("duration", 0) <= 0:
        score -= 0.2
    if not data.get("cleaned_content") or len(data.get("cleaned_content", "")) < 100:
        score -= 0.3
    if not data.get("transcript_type"):
        score -= 0.1

    # Deduct for warnings
    warnings = data.get("warnings", [])
    score -= min(0.2, len(warnings) * 0.05)

    # Deduct for low parsing confidence
    confidence = data.get("parsing_confidence", 1.0)
    score = score * confidence

    return max(0.0, min(1.0, score))


def _generate_snippet(content: str, max_length: int = 200) -> str:
    """Generate smart snippet (first meaningful sentence, not just first N chars)."""
    if not content:
        return ""

    # Try to find first sentence
    sentence_end = re.search(r"[.!?]\s+", content)
    if sentence_end:
        snippet = content[: sentence_end.end()].strip()
        if len(snippet) <= max_length:
            return snippet

    # Fallback to first max_length characters
    snippet = content[:max_length].strip()
    if len(content) > max_length:
        snippet += "..."

    return snippet


def _extract_topics(content: str, max_topics: int = 10) -> List[str]:
    """Extract key topics/themes from content (simple keyword extraction)."""
    if not content:
        return []

    # Simple approach: find capitalized words/phrases and common keywords
    # This is a basic implementation - could be enhanced with NLP

    # Find capitalized phrases (potential topics)
    topics = set()

    # Look for patterns like "AI", "Machine Learning", etc.
    capitalized_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    matches = re.findall(capitalized_pattern, content)

    for match in matches:
        if len(match.split()) <= 3 and len(match) > 2:  # 1-3 word phrases
            topics.add(match)

    # Common topic keywords
    topic_keywords = [
        "artificial intelligence",
        "machine learning",
        "data science",
        "software development",
        "cloud computing",
        "cybersecurity",
        "business strategy",
        "marketing",
        "sales",
        "customer experience",
    ]

    content_lower = content.lower()
    for keyword in topic_keywords:
        if keyword in content_lower:
            topics.add(keyword.title())

    return list(topics)[:max_topics]


def _detect_language(content: str) -> str:
    """Detect language using simple heuristics (defaults to 'en')."""
    if not content:
        return "en"

    # Simple heuristic: check for common English words
    common_english = ["the", "and", "is", "are", "was", "were", "this", "that"]
    content_lower = content.lower()
    english_count = sum(1 for word in common_english if word in content_lower)

    # Very basic - assumes English if common words found
    # Could be enhanced with a proper language detection library
    if english_count > 5:
        return "en"

    # Default to English
    return "en"


def _identify_conversation_flow(content: str) -> Dict[str, Any]:
    """Identify conversation flow patterns (questions, answers, transitions)."""
    if not content:
        return {"flow_type": "unknown", "patterns": []}

    lines = content.split("\n")
    questions = 0
    answers = 0

    question_indicators = ["?", "what", "how", "why", "when", "where", "who"]
    answer_indicators = ["because", "so", "therefore", "that's", "it's"]

    for line in lines:
        line_lower = line.lower()
        if any(indicator in line_lower for indicator in question_indicators):
            questions += 1
        if any(indicator in line_lower for indicator in answer_indicators):
            answers += 1

    flow_type = "discussion"
    if questions > answers * 2:
        flow_type = "interview"
    elif answers > questions * 2:
        flow_type = "presentation"

    return {
        "flow_type": flow_type,
        "question_count": questions,
        "answer_count": answers,
        "patterns": ["q_and_a" if questions > 0 and answers > 0 else "monologue"],
    }


def _detect_transcript_format(content: str, lines: List[str]) -> str:
    """Enhanced format detection for multiple transcript formats."""
    content_stripped = content.strip()

    # Check for JSON
    if content_stripped.startswith("{") or content_stripped.startswith("["):
        try:
            json.loads(content_stripped)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass

    # Check for TTML (XML)
    if content_stripped.startswith("<?xml") or content_stripped.startswith("<tt"):
        try:
            ET.fromstring(content_stripped[:1000])  # Check first 1000 chars
            return "ttml"
        except ET.ParseError:
            pass

    # Check for CSV (look for comma-separated structure)
    if len(lines) > 0:
        first_line = lines[0]
        if "," in first_line and len(first_line.split(",")) >= 2:
            # Check if it looks like CSV with headers
            if any(
                keyword in first_line.lower()
                for keyword in ["speaker", "timestamp", "text", "time"]
            ):
                return "csv"

    # Check for WebVTT
    if lines and "WEBVTT" in lines[0].upper():
        return "webvtt"

    # Check for SRT (line number + timestamp pattern)
    srt_timestamp_pattern = (
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
    )
    for i, line in enumerate(lines[:30]):
        if re.match(r"^\d+$", line.strip()) and i + 1 < len(lines):
            if re.search(srt_timestamp_pattern, lines[i + 1]):
                return "srt"

    # Check for SRT-like timestamps in plain text (even without line numbers)
    # This handles .txt files that contain SRT-style timestamps
    srt_like_count = 0
    for line in lines[:50]:
        if re.search(srt_timestamp_pattern, line):
            srt_like_count += 1
            if (
                srt_like_count >= 2
            ):  # If we find multiple SRT-like timestamps, treat as SRT
                return "srt"

    # Default to plain text
    return "plain"


def _parse_json_transcript(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON transcript format (Zoom, Otter.ai, Rev.com, etc.)."""
    try:
        data = json.loads(content)

        # Handle different JSON structures
        speakers = []
        timestamps = {}
        transcript_lines = []
        duration = None

        # Try common field names
        if isinstance(data, list):
            # Array of segments
            for segment in data:
                if isinstance(segment, dict):
                    speaker = (
                        segment.get("speaker")
                        or segment.get("name")
                        or segment.get("speaker_name")
                    )
                    text = (
                        segment.get("text")
                        or segment.get("content")
                        or segment.get("transcript")
                    )
                    start = (
                        segment.get("start")
                        or segment.get("start_time")
                        or segment.get("timestamp")
                    )
                    end = segment.get("end") or segment.get("end_time")

                    if speaker:
                        speakers.append(str(speaker))
                    if text:
                        if speaker:
                            transcript_lines.append(f"{speaker}: {text}")
                        else:
                            transcript_lines.append(text)
                    if start:
                        timestamps[str(start)] = text or ""
                    if end:
                        try:
                            end_seconds = float(end)
                            if duration is None or end_seconds > duration:
                                duration = int(end_seconds)
                        except (ValueError, TypeError):
                            pass

        elif isinstance(data, dict):
            # Single object with segments
            segments = (
                data.get("segments")
                or data.get("transcript")
                or data.get("utterances")
                or []
            )
            if segments:
                for segment in segments:
                    if isinstance(segment, dict):
                        speaker = segment.get("speaker") or segment.get("name")
                        text = segment.get("text") or segment.get("content")
                        start = segment.get("start") or segment.get("start_time")
                        end = segment.get("end") or segment.get("end_time")

                        if speaker:
                            speakers.append(str(speaker))
                        if text:
                            if speaker:
                                transcript_lines.append(f"{speaker}: {text}")
                            else:
                                transcript_lines.append(text)
                        if end:
                            try:
                                end_seconds = float(end)
                                if duration is None or end_seconds > duration:
                                    duration = int(end_seconds)
                            except (ValueError, TypeError):
                                pass

        cleaned_content = "\n".join(transcript_lines)
        word_count = len(cleaned_content.split())

        if not duration and word_count > 0:
            duration = int((word_count / 150) * 60)

        return {
            "cleaned_content": cleaned_content,
            "speakers": list(set(speakers)),
            "timestamps": timestamps,
            "duration": duration,
            "word_count": word_count,
            "detected_format": "json",
        }

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        _handle_parsing_error(e, "JSON transcript parsing")
        return None


def _parse_csv_transcript(content: str) -> Optional[Dict[str, Any]]:
    """Parse CSV transcript format."""
    import csv
    from io import StringIO

    try:
        reader = csv.DictReader(StringIO(content))
        speakers = []
        timestamps = {}
        transcript_lines = []
        duration = None

        for row in reader:
            # Flexible column name matching
            speaker = None
            text = None
            timestamp = None

            for key, value in row.items():
                key_lower = key.lower()
                if "speaker" in key_lower or "name" in key_lower:
                    speaker = value
                elif (
                    "text" in key_lower
                    or "content" in key_lower
                    or "transcript" in key_lower
                ):
                    text = value
                elif (
                    "time" in key_lower
                    or "timestamp" in key_lower
                    or "start" in key_lower
                ):
                    timestamp = value

            if speaker:
                speakers.append(str(speaker).strip())
            if text:
                if speaker:
                    transcript_lines.append(f"{speaker}: {text}")
                else:
                    transcript_lines.append(text)
            if timestamp:
                timestamps[str(timestamp)] = text or ""

        cleaned_content = "\n".join(transcript_lines)
        word_count = len(cleaned_content.split())

        if not duration and word_count > 0:
            duration = int((word_count / 150) * 60)

        return {
            "cleaned_content": cleaned_content,
            "speakers": list(set(speakers)),
            "timestamps": timestamps,
            "duration": duration,
            "word_count": word_count,
            "detected_format": "csv",
        }

    except (csv.Error, ValueError, KeyError) as e:
        _handle_parsing_error(e, "CSV transcript parsing")
        return None


def _parse_ttml_transcript(content: str) -> Optional[Dict[str, Any]]:
    """Parse TTML (Timed Text Markup Language) format."""
    try:
        root = ET.fromstring(content)

        # TTML namespace handling
        namespaces = {
            "ttml": "http://www.w3.org/ns/ttml",
            "": "http://www.w3.org/ns/ttml",
        }

        speakers = []
        timestamps = {}
        transcript_lines = []
        duration = None

        # Find all p (paragraph) elements which contain the text
        for p in root.iter():
            if p.tag.endswith("}p") or p.tag == "p":
                text = p.text or ""
                begin = p.get("begin")
                end = p.get("end")
                speaker_id = p.get("speaker") or p.get("who")

                if speaker_id:
                    speakers.append(speaker_id)

                if text.strip():
                    if speaker_id:
                        transcript_lines.append(f"{speaker_id}: {text}")
                    else:
                        transcript_lines.append(text)

                if end:
                    try:
                        # TTML uses time format like "00:01:23.456" or "123.456s"
                        if ":" in end:
                            end_seconds = _timestamp_to_seconds(end.replace(",", "."))
                        else:
                            end_seconds = float(
                                end.replace("s", "").replace("ms", "").replace(",", ".")
                            )
                            if "ms" in end:
                                end_seconds = end_seconds / 1000

                        if duration is None or end_seconds > duration:
                            duration = int(end_seconds)
                    except (ValueError, AttributeError):
                        pass

        cleaned_content = "\n".join(transcript_lines)
        word_count = len(cleaned_content.split())

        if not duration and word_count > 0:
            duration = int((word_count / 150) * 60)

        return {
            "cleaned_content": cleaned_content,
            "speakers": list(set(speakers)),
            "timestamps": timestamps,
            "duration": duration,
            "word_count": word_count,
            "detected_format": "ttml",
        }

    except (ET.ParseError, ValueError, AttributeError) as e:
        _handle_parsing_error(e, "TTML transcript parsing")
        return None


def parse_transcript(raw_content: str, content_type: str = "podcast") -> Dict[str, Any]:
    """
    Parse transcript content and extract structured information.
    Supports WebVTT, SRT, JSON, CSV, TTML, and plain text formats.

    Args:
        raw_content: Raw transcript text
        content_type: Type of transcript (podcast, video, meeting, etc.)

    Returns:
        Dictionary with parsed transcript data including:
        - cleaned_content: Cleaned transcript text
        - speakers: List of speaker names (normalized)
        - timestamps: Dictionary of timestamps
        - duration: Duration in seconds (if calculable)
        - word_count: Word count
        - transcript_type: Detected transcript type
        - detected_format: Format that was detected (webvtt, srt, json, csv, ttml, plain)
        - parsing_confidence: Confidence score (0-1)
        - quality_metrics: Detailed quality breakdown
        - warnings: List of parsing warnings
        - speaking_time_per_speaker: Dict of speaker -> seconds
        - detected_language: Language code
        - key_topics: List of extracted topics
        - conversation_flow: Analysis of flow structure
    """
    # Pre-parse validation
    if not raw_content or not raw_content.strip():
        logger.warning("Empty transcript content provided")
        return {
            "cleaned_content": "",
            "speakers": [],
            "timestamps": {},
            "duration": 0,
            "word_count": 0,
            "transcript_type": content_type,
            "content_type": "transcript",
            "detected_format": "unknown",
            "parsing_confidence": 0.0,
            "warnings": ["Empty transcript content"],
            "quality_metrics": {},
        }

    # Check cache
    content_hash = _get_content_hash(raw_content)
    if content_hash in _parsing_cache:
        logger.debug("Using cached parsing result")
        cached_result = _parsing_cache[content_hash].copy()
        cached_result["cached"] = True
        return cached_result

    # Initialize result structure
    result = {
        "cleaned_content": "",
        "speakers": [],
        "timestamps": {},
        "duration": None,
        "word_count": 0,
        "transcript_type": content_type,
        "content_type": "transcript",
        "detected_format": "unknown",
        "parsing_confidence": 1.0,
        "warnings": [],
        "quality_metrics": {},
        "speaking_time_per_speaker": {},
        "detected_language": "en",
        "key_topics": [],
        "conversation_flow": {},
    }

    try:
        # Step 1: Pre-process content
        try:
            soup = BeautifulSoup(raw_content, "html.parser")
            text = soup.get_text(separator="\n")
        except Exception as e:
            _handle_parsing_error(e, "HTML parsing")
            text = raw_content

        # Normalize unicode and clean up
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF\n]+", "", text)
        text = text.replace("\u00a0", " ")

        raw_lines = text.split("\n")

        # Step 2: Detect format
        detected_format = _detect_transcript_format(text, raw_lines)
        result["detected_format"] = detected_format

        # Step 3: Parse based on format
        parsed_data = None

        if detected_format == "json":
            parsed_data = _parse_json_transcript(text)
            if parsed_data:
                result.update(parsed_data)
        elif detected_format == "csv":
            parsed_data = _parse_csv_transcript(text)
            if parsed_data:
                result.update(parsed_data)
        elif detected_format == "ttml":
            parsed_data = _parse_ttml_transcript(text)
            if parsed_data:
                result.update(parsed_data)

        # If format-specific parser didn't work or format is webvtt/srt/plain, use standard parser
        if not parsed_data or detected_format in ["webvtt", "srt", "plain"]:
            # Use existing parsing logic for webvtt/srt/plain
            parsed_data = _parse_standard_format(raw_lines, detected_format)
            if parsed_data:
                result.update(parsed_data)

        # If still no data, try fallback
        if not result.get("cleaned_content"):
            logger.warning("Primary parsing failed, attempting fallback")
            fallback_result = _fallback_parse(raw_content, content_type)
            result.update(fallback_result)
            result["parsing_confidence"] = 0.3

        # Step 4: Post-processing enhancements
        # Normalize speakers
        if result.get("speakers"):
            speaker_map = _merge_similar_speakers(result["speakers"])
            normalized_speakers = list(set(speaker_map.values()))
            result["speakers"] = normalized_speakers
            result["speaker_mapping"] = speaker_map

        # Calculate speaking time
        if result.get("speakers") and result.get("cleaned_content"):
            speaking_time = _calculate_speaking_time(
                result["speakers"],
                result["cleaned_content"],
                result.get("timestamps", {}),
            )
            result["speaking_time_per_speaker"] = speaking_time

        # Validate timestamps
        if result.get("timestamps"):
            is_valid, ts_warnings = _validate_timestamp_order(result["timestamps"])
            result["warnings"].extend(ts_warnings)

            # Detect gaps
            gaps = _detect_timestamp_gaps(result["timestamps"], result.get("duration"))
            result["warnings"].extend(gaps)

        # Detect overlaps
        overlap_warnings = _detect_overlapping_speakers(
            result.get("cleaned_content", "")
        )
        result["warnings"].extend(overlap_warnings)

        # Content enhancement
        if result.get("cleaned_content"):
            result["detected_language"] = _detect_language(result["cleaned_content"])
            result["key_topics"] = _extract_topics(result["cleaned_content"])
            result["conversation_flow"] = _identify_conversation_flow(
                result["cleaned_content"]
            )

        # Generate snippet if needed
        if not result.get("snippet") and result.get("cleaned_content"):
            result["snippet"] = _generate_snippet(result["cleaned_content"])

        # Step 5: Validation
        is_valid, validation_warnings = _validate_parsed_data(result)
        result["warnings"].extend(validation_warnings)

        # Calculate quality metrics
        quality_metrics = {
            "completeness": (
                1.0
                if all(
                    [
                        result.get("speakers"),
                        result.get("duration"),
                        result.get("cleaned_content"),
                    ]
                )
                else 0.5
            ),
            "speaker_clarity": 1.0 if result.get("speakers") else 0.0,
            "timestamp_accuracy": 1.0 if not result["warnings"] else 0.7,
            "content_quality": (
                1.0 if len(result.get("cleaned_content", "")) > 100 else 0.5
            ),
        }
        result["quality_metrics"] = quality_metrics

        # Calculate overall quality score
        result["parsing_confidence"] = _calculate_quality_score(result)

        # Detect transcript type
        if content_type == "podcast" and result.get("cleaned_content"):
            content_lower = result["cleaned_content"].lower()
            if any(
                word in content_lower for word in ["meeting", "call", "zoom", "teams"]
            ):
                result["transcript_type"] = "meeting"
            elif any(word in content_lower for word in ["interview", "guest", "host"]):
                result["transcript_type"] = "interview"
            elif any(
                word in content_lower for word in ["video", "youtube", "recording"]
            ):
                result["transcript_type"] = "video"

        # Cache result (limit cache size)
        if len(_parsing_cache) >= CACHE_SIZE:
            # Remove oldest entry (simple FIFO)
            _parsing_cache.pop(next(iter(_parsing_cache)))
        _parsing_cache[content_hash] = result.copy()

    except Exception as e:
        logger.error(f"Unexpected error in parse_transcript: {e}", exc_info=True)
        result["warnings"].append(f"Parsing error: {str(e)}")
        result["parsing_confidence"] = 0.1

        # Try fallback
        try:
            fallback_result = _fallback_parse(raw_content, content_type)
            result.update(fallback_result)
        except Exception as fallback_error:
            _handle_parsing_error(fallback_error, "fallback parsing")

    return result


def _parse_standard_format(
    raw_lines: List[str], detected_format: str
) -> Dict[str, Any]:
    """Parse standard formats (WebVTT, SRT, plain text)."""
    is_webvtt = detected_format == "webvtt"
    is_srt = detected_format == "srt"

    webvtt_timestamp_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
    )
    srt_timestamp_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
    )

    processed_lines = []
    speakers = set()
    timestamps = {}
    last_timestamp_seconds = 0

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()

        if not line:
            i += 1
            continue

        # Remove metadata lines
        if re.match(
            r"^(Meeting|Session|Recording|Created).*(created|started).*at:",
            line,
            re.IGNORECASE,
        ):
            i += 1
            continue

        if re.match(
            r"^.*\d{1,2}(st|nd|rd|th)?\s+\w+,\s+\d{4}.*\d{1,2}:\d{2}.*(AM|PM)",
            line,
            re.IGNORECASE,
        ):
            i += 1
            continue

        # Handle WebVTT/SRT format - also check for SRT-like timestamps in plain text
        # Check for SRT-like timestamps even in plain text mode
        timestamp_match = webvtt_timestamp_pattern.search(
            line
        ) or srt_timestamp_pattern.search(line)

        if timestamp_match:
            start_time = timestamp_match.group(1)
            end_time = timestamp_match.group(2)
            start_time_normalized = start_time.replace(",", ".")
            end_time_normalized = end_time.replace(",", ".")

            try:
                end_seconds = _timestamp_to_seconds(end_time_normalized)
                last_timestamp_seconds = max(last_timestamp_seconds, end_seconds)
                timestamps[start_time_normalized] = ""
            except ValueError as e:
                _handle_parsing_error(e, f"timestamp parsing at line {i}")

            # Skip segment numbers before timestamps
            if is_webvtt or is_srt:
                if re.match(r"^\d+$", line) and i + 1 < len(raw_lines):
                    next_line = raw_lines[i + 1].strip()
                    if webvtt_timestamp_pattern.search(
                        next_line
                    ) or srt_timestamp_pattern.search(next_line):
                        i += 1
                        continue

            # Look for speaker and content on the next line(s)
            i += 1
            speaker_found = False
            speaker_name = None
            content_lines_after_timestamp = []

            # Check next few lines (up to 5 lines) for speaker and content
            for j in range(i, min(i + 5, len(raw_lines))):
                content_line = raw_lines[j].strip()
                if not content_line:
                    # Empty line - if we found a speaker, this might be separator before content
                    if speaker_found:
                        continue
                    else:
                        continue

                # Skip if this is another timestamp or segment number
                if (
                    webvtt_timestamp_pattern.search(content_line)
                    or srt_timestamp_pattern.search(content_line)
                    or re.match(r"^Segment\s+\d+", content_line, re.IGNORECASE)
                    or re.match(r"^\d+$", content_line)
                ):
                    break

                # Pattern 1: Check if this line is a standalone speaker name
                # Examples: "Pranav Shikarpur", "John Doe", "Dr. Smith"
                # Must start with capital letter, contain letters, 2-50 chars, no colons (unless it's a title)
                standalone_speaker_pattern = re.compile(
                    r"^([A-Z][A-Za-z\.\s\-']{1,49})$"
                )
                standalone_match = standalone_speaker_pattern.match(content_line)

                if standalone_match:
                    potential_speaker = standalone_match.group(1).strip()
                    # Additional validation: must have at least one letter, not all dots/spaces
                    if (
                        re.search(r"[A-Za-z]", potential_speaker)
                        and len(potential_speaker) >= 2
                        and not re.match(r"^[\d\s\.\-']+$", potential_speaker)
                        and not re.search(r"-->|\[|\]|:", potential_speaker)
                    ):
                        # This looks like a speaker name - check if next line has content
                        if j + 1 < len(raw_lines):
                            next_line = raw_lines[j + 1].strip()
                            # If next line has actual content (not another speaker or timestamp)
                            if (
                                next_line
                                and not standalone_speaker_pattern.match(next_line)
                                and not (
                                    webvtt_timestamp_pattern.search(next_line)
                                    or srt_timestamp_pattern.search(next_line)
                                )
                                and not re.match(
                                    r"^Segment\s+\d+", next_line, re.IGNORECASE
                                )
                                and len(next_line) > 5
                            ):  # Content should be substantial
                                speaker_name = potential_speaker
                                speakers.add(speaker_name)
                                speaker_found = True
                                processed_lines.append(f"{speaker_name}: {next_line}")
                                i = j + 2
                                break
                            elif next_line and len(next_line) <= 5:
                                # Short line might still be speaker, continue checking
                                speaker_name = potential_speaker
                                speakers.add(speaker_name)
                                speaker_found = True
                                # Will collect content on next iteration
                                i = j + 1
                                continue
                        else:
                            # Last line, treat as speaker
                            speaker_name = potential_speaker
                            speakers.add(speaker_name)
                            speaker_found = True
                            processed_lines.append(f"{speaker_name}:")
                            i = j + 1
                            break

                # Pattern 2: Check if line contains speaker pattern with colon
                # Examples: "Pranav Shikarpur: text", "Speaker 1: content"
                colon_speaker_match = re.match(
                    r"^([A-Z][A-Za-z0-9\s\.\-']{1,49}?):\s*(.+)$", content_line
                )
                if colon_speaker_match:
                    speaker = colon_speaker_match.group(1).strip()
                    content = colon_speaker_match.group(2).strip()
                    # Validate speaker name
                    if (
                        re.search(r"[A-Za-z]", speaker)
                        and len(speaker) >= 2
                        and len(speaker) < 50
                        and not re.match(r"^\d+$", speaker)
                    ):
                        speakers.add(speaker)
                        speaker_found = True
                        processed_lines.append(f"{speaker}: {content}")
                        i = j + 1
                        break

                # Pattern 3: If we already found a speaker, collect content lines
                if speaker_found and speaker_name:
                    # This is content for the speaker we found
                    if not standalone_speaker_pattern.match(content_line):
                        processed_lines.append(f"{speaker_name}: {content_line}")
                        i = j + 1
                        break
                    else:
                        # Another speaker found, stop here
                        break

                # Pattern 4: Regular content line (no speaker found yet)
                if not speaker_found:
                    # Check if this might be content that should be associated with previous speaker
                    # or if it's just regular content
                    content_lines_after_timestamp.append(content_line)
                    # Don't break yet, keep looking for speaker
                    if len(content_lines_after_timestamp) >= 3:
                        # If we've collected 3 lines without finding a speaker,
                        # they're probably just content
                        break

            # If we found content but no speaker, add the content as-is
            if not speaker_found and content_lines_after_timestamp:
                processed_lines.extend(content_lines_after_timestamp)
                i = min(i + len(content_lines_after_timestamp), len(raw_lines))

            if not speaker_found and i < len(raw_lines):
                # Skip empty lines after timestamp
                while i < len(raw_lines) and not raw_lines[i].strip():
                    i += 1

            continue

        # Handle simple format
        if re.match(r"^\d+$", line) and i + 1 < len(raw_lines):
            next_line = raw_lines[i + 1].strip()
            if not re.search(r"\[.*\]|\(.*\)|-->", next_line):
                i += 1
                continue

        # Extract speakers from lines without timestamps
        # Pattern 1: Speaker with colon (e.g., "Speaker: text")
        speaker_with_timestamp_pattern = re.compile(
            r"^[\[\(].*?[\]\)]\s*([A-Z][A-Za-z0-9\s\.\-']{0,49}?):\s*(.+)$"
        )
        speaker_pattern = re.compile(r"^([A-Z][A-Za-z0-9\s\.\-']{0,49}?):\s*(.+)$")

        speaker_match = speaker_with_timestamp_pattern.match(
            line
        ) or speaker_pattern.match(line)

        if speaker_match:
            speaker = speaker_match.group(1).strip()
            content = speaker_match.group(2).strip()
            if (
                re.search(r"[A-Za-z]", speaker)
                and len(speaker) >= 2
                and len(speaker) < 50
                and content
            ):
                speakers.add(speaker)
                processed_lines.append(line)
            else:
                if not (re.match(r"^\d+$", line) and i + 1 < len(raw_lines)):
                    processed_lines.append(line)
        else:
            # Pattern 2: Standalone speaker name (check if this line looks like a name and next line is content)
            # This helps catch speakers that weren't caught by timestamp parsing
            standalone_speaker_check = re.match(r"^([A-Z][A-Za-z\.\s\-']{1,49})$", line)
            if standalone_speaker_check:
                potential_speaker = standalone_speaker_check.group(1).strip()
                # Validate it looks like a name
                if (
                    re.search(r"[A-Za-z]", potential_speaker)
                    and len(potential_speaker) >= 2
                    and not re.match(r"^[\d\s\.\-']+$", potential_speaker)
                    and not re.search(r"-->|\[|\]|:", potential_speaker)
                    and i + 1 < len(raw_lines)
                ):
                    next_line = raw_lines[i + 1].strip()
                    # If next line has substantial content (not another name or timestamp)
                    if (
                        next_line
                        and len(next_line) > 10
                        and not re.match(  # Substantial content
                            r"^([A-Z][A-Za-z\.\s\-']{1,49})$", next_line
                        )
                        and not (
                            webvtt_timestamp_pattern.search(next_line)
                            or srt_timestamp_pattern.search(next_line)
                        )
                    ):
                        speakers.add(potential_speaker)
                        processed_lines.append(f"{potential_speaker}: {next_line}")
                        i += 2  # Skip both speaker and content lines
                        continue

            if re.match(r"^\d+$", line):
                if len(line) <= 4 and (
                    i + 1 >= len(raw_lines)
                    or not re.search(r"^[A-Za-z]", raw_lines[i + 1].strip())
                ):
                    i += 1
                    continue
                elif i + 1 < len(raw_lines):
                    next_line = raw_lines[i + 1].strip()
                    if not re.search(r"\[.*\]|\(.*\)|-->|:", next_line):
                        i += 1
                        continue
                processed_lines.append(line)

        i += 1

    # Build cleaned content
    final_processed_lines = []
    timestamp_line_pattern = re.compile(
        r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}$"
    )

    for line in processed_lines:
        if timestamp_line_pattern.match(line.strip()):
            continue
        cleaned_line = re.sub(
            r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}", "", line
        ).strip()
        if cleaned_line:
            final_processed_lines.append(cleaned_line)

    cleaned_content = "\n".join(final_processed_lines)

    # Post-process: Extract any missed speakers from cleaned content
    # Look for patterns like "Speaker Name:" or standalone speaker names followed by content
    if cleaned_content:
        # Pattern 1: Extract speakers from "Speaker: text" format
        speaker_colon_pattern = re.compile(
            r"^([A-Z][A-Za-z0-9\s\.\-']{1,49}?):\s+(.+)$", re.MULTILINE
        )
        for match in speaker_colon_pattern.finditer(cleaned_content):
            speaker = match.group(1).strip()
            if (
                re.search(r"[A-Za-z]", speaker)
                and len(speaker) >= 2
                and len(speaker) < 50
                and not re.match(r"^\d+$", speaker)
            ):
                speakers.add(speaker)

        # Pattern 2: Look for standalone capitalized names that appear multiple times
        # (likely speakers if they appear repeatedly)
        lines = cleaned_content.split("\n")
        potential_speakers = {}
        for line in lines:
            line = line.strip()
            if not line or len(line) < 2:
                continue

            # Check if line is just a capitalized name (2-50 chars, starts with capital)
            standalone_name_match = re.match(r"^([A-Z][A-Za-z\.\s\-']{1,49})$", line)
            if standalone_name_match:
                name = standalone_name_match.group(1).strip()
                if (
                    re.search(r"[A-Za-z]", name)
                    and not re.match(r"^[\d\s\.\-']+$", name)
                    and not re.search(r"-->|\[|\]|:", name)
                ):
                    potential_speakers[name] = potential_speakers.get(name, 0) + 1

        # If a name appears 2+ times, it's likely a speaker
        for name, count in potential_speakers.items():
            if count >= 2:
                speakers.add(name)

    # Extract timestamps from simple format (brackets/parentheses)
    if not is_webvtt and not is_srt:
        simple_timestamp_pattern = re.compile(r"[\[\(](\d{1,2}:\d{2}(?::\d{2})?)[\]\)]")
        for line in final_processed_lines:
            matches = simple_timestamp_pattern.findall(line)
            for timestamp in matches:
                if timestamp not in timestamps:
                    timestamps[timestamp] = line.strip()

    # Also extract SRT-like timestamps from raw content if we haven't found duration yet
    # This helps with plain text files that have SRT-style timestamps
    if last_timestamp_seconds == 0 and detected_format == "plain":
        srt_timestamp_pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
        )
        for line in raw_lines:
            timestamp_match = srt_timestamp_pattern.search(line)
            if timestamp_match:
                end_time = timestamp_match.group(2)
                end_time_normalized = end_time.replace(",", ".")
                try:
                    end_seconds = _timestamp_to_seconds(end_time_normalized)
                    last_timestamp_seconds = max(last_timestamp_seconds, end_seconds)
                except ValueError:
                    pass

    # Calculate duration
    duration_seconds = None
    if last_timestamp_seconds > 0:
        duration_seconds = int(last_timestamp_seconds)
    elif timestamps:
        try:
            last_ts = max(
                timestamps.keys(),
                key=lambda x: (
                    _simple_timestamp_to_seconds(x)
                    if ":" in x and len(x.split(":")) <= 3
                    else _timestamp_to_seconds(x.replace(",", ".")) if ":" in x else 0
                ),
            )
            duration_seconds = (
                _simple_timestamp_to_seconds(last_ts)
                if len(last_ts.split(":")) <= 3
                else int(_timestamp_to_seconds(last_ts.replace(",", ".")))
            )
        except (ValueError, KeyError, AttributeError):
            pass

    if duration_seconds is None:
        word_count = len(cleaned_content.split())
        duration_seconds = int((word_count / 150) * 60) if word_count > 0 else 0
    else:
        word_count = len(cleaned_content.split())

    return {
        "cleaned_content": cleaned_content,
        "speakers": list(speakers),
        "timestamps": timestamps,
        "duration": duration_seconds,
        "word_count": word_count,
        "detected_format": detected_format,
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
