"""
Content analysis plugin tasks for Marketing Project.

This module provides general content analysis functions that work across
all content types for routing and processing decisions.

Functions:
    analyze_content_type: Analyzes the content type and returns agent name
    extract_content_metadata: Extracts metadata from any content type
    validate_content_structure: Validates content has required fields
    route_to_appropriate_agent: Routes content to appropriate agent
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union

from marketing_project.core.models import AppContext, ContentContext, EmailContext
from marketing_project.core.utils import (
    create_standard_task_result,
    ensure_content_context,
    extract_content_metadata_for_pipeline,
    validate_content_for_processing,
)
from marketing_project.models.content_models import (
    BlogPostContext,
    ReleaseNotesContext,
    TranscriptContext,
)

logger = logging.getLogger("marketing_project.plugins.content_analysis")


def analyze_content_type(content: ContentContext) -> str:
    """
    Analyzes the content type and returns the appropriate agent name.

    Args:
        content: Content context object

    Returns:
        str: Agent name to route to
    """
    if isinstance(content, TranscriptContext):
        return "transcripts_agent"
    elif isinstance(content, BlogPostContext):
        return "blog_agent"
    elif isinstance(content, ReleaseNotesContext):
        return "releasenotes_agent"
    elif isinstance(content, EmailContext):
        return "email_agent"
    else:
        return "general_agent"


def extract_content_metadata(content: ContentContext) -> Dict[str, Any]:
    """
    Extracts metadata from content for routing decisions.

    Args:
        content: Content context object

    Returns:
        Dict[str, Any]: Extracted metadata
    """
    metadata = {
        "content_type": type(content).__name__,
        "id": content.id,
        "title": content.title,
        "has_snippet": bool(content.snippet),
        "has_metadata": bool(content.metadata),
    }

    # Add type-specific metadata
    if isinstance(content, TranscriptContext):
        metadata.update(
            {
                "speakers": content.speakers,
                "duration": content.duration,
                "transcript_type": content.transcript_type,
            }
        )
    elif isinstance(content, BlogPostContext):
        metadata.update(
            {
                "author": content.author,
                "tags": content.tags,
                "category": content.category,
                "word_count": content.word_count,
            }
        )
    elif isinstance(content, ReleaseNotesContext):
        metadata.update(
            {
                "version": content.version,
                "changes_count": len(content.changes),
                "features_count": len(content.features),
                "bug_fixes_count": len(content.bug_fixes),
            }
        )

    return metadata


def validate_content_structure(content: ContentContext) -> bool:
    """
    Validates that the content has the required structure for processing.

    Args:
        content: Content context object

    Returns:
        bool: True if content is valid, False otherwise
    """
    required_fields = ["id", "title", "content", "snippet"]

    for field in required_fields:
        if not hasattr(content, field) or not getattr(content, field):
            logger.error(f"Content missing required field: {field}")
            return False

    return True


def analyze_content_for_pipeline(
    content: Union[ContentContext, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyzes content for the new marketing pipeline workflow.

    Args:
        content: Content context object or dictionary

    Returns:
        Dict[str, Any]: Comprehensive content analysis for pipeline
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Validate content
        validation = validate_content_for_processing(content_obj)
        if not validation["is_valid"]:
            return create_standard_task_result(
                success=False,
                error=f"Content validation failed: {', '.join(validation['issues'])}",
                task_name="analyze_content_for_pipeline",
            )
        # Extract metadata
        metadata = extract_content_metadata_for_pipeline(content_obj)

        analysis = {
            "content_type": metadata["content_type"],
            "content_quality": {},
            "seo_potential": {},
            "marketing_value": {},
            "processing_recommendations": validation.get("warnings", []),
            "pipeline_ready": validation["is_valid"],
        }

        # Analyze content quality
        content_text = content_obj.content or ""
        word_count = len(content_text.split())

        analysis["content_quality"] = {
            "word_count": word_count,
            "has_title": bool(content_obj.title),
            "has_snippet": bool(content_obj.snippet),
            "has_metadata": bool(content_obj.metadata),
            "readability_score": calculate_basic_readability(content_text),
            "completeness_score": assess_content_completeness(content_obj),
        }

        # Analyze SEO potential
        analysis["seo_potential"] = {
            "has_keywords": bool(extract_potential_keywords(content_text)),
            "title_optimization": assess_title_seo(content_obj.title or ""),
            "content_structure": assess_content_structure(content_text),
            "internal_linking_potential": assess_linking_potential(content_text),
        }

        # Analyze marketing value
        analysis["marketing_value"] = {
            "engagement_potential": assess_engagement_potential(content_text),
            "conversion_potential": assess_conversion_potential(content_text),
            "shareability": assess_shareability(content_text),
            "target_audience_appeal": assess_audience_appeal(content_text),
        }

        # Generate processing recommendations
        recommendations = list(validation.get("warnings", []))

        if word_count < 500:
            recommendations.append("Content is too short - consider expanding")
        elif word_count > 3000:
            recommendations.append(
                "Content is very long - consider breaking into series"
            )

        if not content_obj.title:
            recommendations.append("Add a compelling title")

        if analysis["content_quality"]["readability_score"] < 60:
            recommendations.append("Improve readability with shorter sentences")

        if not analysis["seo_potential"]["has_keywords"]:
            recommendations.append("Add relevant keywords for SEO")

        analysis["processing_recommendations"] = recommendations

        # Determine if content is ready for pipeline
        critical_issues = [
            word_count < 100,
            not content_obj.title,
            analysis["content_quality"]["readability_score"] < 30,
        ]

        analysis["pipeline_ready"] = not any(critical_issues)

        return create_standard_task_result(
            success=True,
            data=analysis,
            task_name="analyze_content_for_pipeline",
            metadata=metadata,
        )

    except Exception as e:
        logger.error(f"Error in analyze_content_for_pipeline: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Analysis failed: {str(e)}",
            task_name="analyze_content_for_pipeline",
        )


def calculate_basic_readability(text: str) -> float:
    """
    Calculate basic readability score for content.

    Args:
        text: Content text to analyze

    Returns:
        float: Readability score (0-100)
    """
    if not text:
        return 0

    words = text.split()
    sentences = text.count(".") + text.count("!") + text.count("?")

    if sentences == 0 or len(words) == 0:
        return 0

    avg_sentence_length = len(words) / sentences
    avg_syllables = sum(estimate_syllables(word) for word in words) / len(words)

    # Simplified Flesch Reading Ease
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
    return max(0, min(100, score))


def assess_content_completeness(content: ContentContext) -> float:
    """
    Assess how complete the content is.

    Args:
        content: Content context object

    Returns:
        float: Completeness score (0-100)
    """
    score = 0
    max_score = 100

    # Check required fields
    if content.title:
        score += 20
    if content.content:
        score += 30
    if content.snippet:
        score += 15
    if content.metadata:
        score += 15

    # Check content quality indicators
    if content.content and len(content.content.split()) > 200:
        score += 10
    if content.content and "introduction" in content.content.lower():
        score += 5
    if content.content and "conclusion" in content.content.lower():
        score += 5

    return min(score, max_score)


def extract_potential_keywords(text: str) -> List[str]:
    """
    Extract potential keywords from content.

    Args:
        text: Content text to analyze

    Returns:
        List[str]: List of potential keywords
    """
    if not text:
        return []

    # Simple keyword extraction
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    stop_words = {
        "this",
        "that",
        "with",
        "have",
        "will",
        "from",
        "they",
        "know",
        "want",
        "been",
        "good",
        "much",
        "some",
        "time",
        "very",
        "when",
        "come",
        "here",
        "just",
        "like",
    }

    filtered_words = [word for word in words if word not in stop_words]
    word_freq = {}

    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Return most frequent words
    return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]


def assess_title_seo(title: str) -> Dict[str, Any]:
    """
    Assess title SEO potential.

    Args:
        title: Title to assess

    Returns:
        Dict[str, Any]: Title SEO assessment
    """
    if not title:
        return {"score": 0, "issues": ["No title provided"]}

    score = 0
    issues = []

    # Check length
    if 30 <= len(title) <= 60:
        score += 30
    else:
        issues.append(f"Title length ({len(title)}) should be 30-60 characters")

    # Check for power words
    power_words = [
        "ultimate",
        "complete",
        "guide",
        "best",
        "expert",
        "proven",
        "essential",
    ]
    if any(word in title.lower() for word in power_words):
        score += 20

    # Check for numbers
    if re.search(r"\d+", title):
        score += 15

    # Check for emotional words
    emotional_words = ["amazing", "incredible", "shocking", "secret", "revealed"]
    if any(word in title.lower() for word in emotional_words):
        score += 15

    # Check for question format
    if title.endswith("?"):
        score += 10

    # Check for colon format
    if ":" in title:
        score += 10

    return {"score": min(score, 100), "issues": issues}


def assess_content_structure(text: str) -> Dict[str, Any]:
    """
    Assess content structure quality.

    Args:
        text: Content text to assess

    Returns:
        Dict[str, Any]: Structure assessment
    """
    if not text:
        return {"score": 0, "issues": ["No content provided"]}

    score = 0
    issues = []

    # Check for headings
    heading_count = text.count("#")
    if heading_count > 0:
        score += 25
    else:
        issues.append("No headings found")

    # Check for lists
    list_indicators = ["- ", "* ", "1. ", "2. "]
    if any(indicator in text for indicator in list_indicators):
        score += 20
    else:
        issues.append("No lists found")

    # Check for links
    if "](" in text:
        score += 15
    else:
        issues.append("No links found")

    # Check for images
    if "![" in text:
        score += 10
    else:
        issues.append("No images found")

    # Check paragraph structure
    paragraphs = text.split("\n\n")
    if len(paragraphs) >= 3:
        score += 15
    else:
        issues.append("Too few paragraphs")

    # Check for call-to-action
    cta_indicators = ["learn more", "get started", "read more", "contact us"]
    if any(cta in text.lower() for cta in cta_indicators):
        score += 15
    else:
        issues.append("No call-to-action found")

    return {"score": min(score, 100), "issues": issues}


def assess_linking_potential(text: str) -> Dict[str, Any]:
    """
    Assess internal linking potential.

    Args:
        text: Content text to assess

    Returns:
        Dict[str, Any]: Linking potential assessment
    """
    if not text:
        return {"score": 0, "opportunities": []}

    opportunities = []
    score = 0

    # Look for linking opportunities
    link_indicators = [
        "learn more",
        "read more",
        "see also",
        "related to",
        "for more information",
        "additional resources",
        "further reading",
        "next steps",
        "get started",
    ]

    for indicator in link_indicators:
        if indicator in text.lower():
            opportunities.append(f"Add internal link to '{indicator}'")
            score += 10

    # Check for topic mentions that could be linked
    topic_mentions = ["tutorial", "guide", "best practices", "examples", "case study"]
    for topic in topic_mentions:
        if topic in text.lower():
            opportunities.append(f"Link to {topic} content")
            score += 5

    return {"score": min(score, 100), "opportunities": opportunities}


def assess_engagement_potential(text: str) -> float:
    """
    Assess content engagement potential.

    Args:
        text: Content text to assess

    Returns:
        float: Engagement score (0-100)
    """
    if not text:
        return 0

    score = 0

    # Check for questions
    if "?" in text:
        score += 20

    # Check for personal pronouns
    personal_pronouns = ["you", "your", "we", "our", "us"]
    if any(pronoun in text.lower() for pronoun in personal_pronouns):
        score += 15

    # Check for emotional words
    emotional_words = ["amazing", "incredible", "shocking", "exciting", "fantastic"]
    if any(word in text.lower() for word in emotional_words):
        score += 15

    # Check for storytelling elements
    story_indicators = ["story", "example", "case", "experience", "journey"]
    if any(indicator in text.lower() for indicator in story_indicators):
        score += 20

    # Check for actionable content
    action_words = ["how to", "step by step", "tutorial", "guide", "tips"]
    if any(phrase in text.lower() for phrase in action_words):
        score += 30

    return min(score, 100)


def assess_conversion_potential(text: str) -> float:
    """
    Assess content conversion potential.

    Args:
        text: Content text to assess

    Returns:
        float: Conversion score (0-100)
    """
    if not text:
        return 0

    score = 0

    # Check for call-to-action
    cta_indicators = [
        "learn more",
        "get started",
        "read more",
        "contact us",
        "subscribe",
    ]
    if any(cta in text.lower() for cta in cta_indicators):
        score += 30

    # Check for value propositions
    value_indicators = ["benefit", "advantage", "solution", "improve", "increase"]
    if any(indicator in text.lower() for indicator in value_indicators):
        score += 25

    # Check for social proof
    proof_indicators = ["testimonial", "review", "rating", "customer", "user"]
    if any(indicator in text.lower() for indicator in proof_indicators):
        score += 20

    # Check for urgency
    urgency_indicators = ["now", "today", "limited", "exclusive", "urgent"]
    if any(indicator in text.lower() for indicator in urgency_indicators):
        score += 15

    # Check for trust signals
    trust_indicators = ["guarantee", "secure", "trusted", "certified", "verified"]
    if any(indicator in text.lower() for indicator in trust_indicators):
        score += 10

    return min(score, 100)


def assess_shareability(text: str) -> float:
    """
    Assess content shareability potential.

    Args:
        text: Content text to assess

    Returns:
        float: Shareability score (0-100)
    """
    if not text:
        return 0

    score = 0

    # Check for quotable content
    if '"' in text:
        score += 20

    # Check for lists (highly shareable)
    list_indicators = ["- ", "* ", "1. ", "2. "]
    if any(indicator in text for indicator in list_indicators):
        score += 25

    # Check for statistics
    if re.search(r"\d+%", text) or re.search(r"\d+ out of \d+", text):
        score += 20

    # Check for controversial or trending topics
    trending_indicators = ["trending", "viral", "popular", "hot", "buzz"]
    if any(indicator in text.lower() for indicator in trending_indicators):
        score += 15

    # Check for emotional content
    emotional_indicators = ["shocking", "amazing", "incredible", "unbelievable"]
    if any(indicator in text.lower() for indicator in emotional_indicators):
        score += 20

    return min(score, 100)


def assess_audience_appeal(text: str) -> float:
    """
    Assess target audience appeal.

    Args:
        text: Content text to assess

    Returns:
        float: Audience appeal score (0-100)
    """
    if not text:
        return 0

    score = 0

    # Check for beginner-friendly content
    beginner_indicators = ["beginner", "introduction", "getting started", "basics"]
    if any(indicator in text.lower() for indicator in beginner_indicators):
        score += 25

    # Check for expert content
    expert_indicators = ["advanced", "expert", "professional", "enterprise"]
    if any(indicator in text.lower() for indicator in expert_indicators):
        score += 25

    # Check for practical content
    practical_indicators = ["how to", "tutorial", "guide", "step by step", "tips"]
    if any(indicator in text.lower() for indicator in practical_indicators):
        score += 25

    # Check for industry-specific content
    industry_indicators = ["industry", "market", "business", "professional"]
    if any(indicator in text.lower() for indicator in industry_indicators):
        score += 25

    return min(score, 100)


def estimate_syllables(word: str) -> int:
    """
    Estimate syllable count for a word.

    Args:
        word: Word to analyze

    Returns:
        int: Estimated syllable count
    """
    word = word.lower().strip(".,!?")
    if not word:
        return 0

    vowels = "aeiouy"
    syllable_count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel

    # Handle silent 'e'
    if word.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    return max(1, syllable_count)
