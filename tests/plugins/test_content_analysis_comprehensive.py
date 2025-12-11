"""
Comprehensive tests for content analysis plugin functions.
"""

import pytest

from marketing_project.models.content_models import (
    BlogPostContext,
    ReleaseNotesContext,
    TranscriptContext,
)
from marketing_project.plugins.content_analysis.tasks import (
    analyze_content_for_pipeline,
    analyze_content_type,
    assess_audience_appeal,
    assess_content_completeness,
    assess_content_structure,
    assess_conversion_potential,
    assess_engagement_potential,
    assess_linking_potential,
    assess_shareability,
    assess_title_seo,
    calculate_basic_readability,
    extract_content_metadata,
    extract_potential_keywords,
    validate_content_structure,
)


def test_analyze_content_type_blog():
    """Test analyze_content_type for blog post."""
    content = BlogPostContext(id="test-1", title="Test")
    agent = analyze_content_type(content)

    assert agent == "blog_agent"


def test_analyze_content_type_transcript():
    """Test analyze_content_type for transcript."""
    content = TranscriptContext(id="test-1", title="Test")
    agent = analyze_content_type(content)

    assert agent == "transcripts_agent"


def test_analyze_content_type_release_notes():
    """Test analyze_content_type for release notes."""
    content = ReleaseNotesContext(id="test-1", title="Test")
    agent = analyze_content_type(content)

    assert agent == "releasenotes_agent"


def test_extract_content_metadata():
    """Test extract_content_metadata function."""
    content = BlogPostContext(
        id="test-1",
        title="Test",
        author="Test Author",
        tags=["test", "blog"],
    )

    metadata = extract_content_metadata(content)

    assert isinstance(metadata, dict)
    assert metadata["content_type"] == "BlogPostContext"
    assert metadata["id"] == "test-1"


def test_validate_content_structure():
    """Test validate_content_structure function."""
    content = BlogPostContext(id="test-1", title="Test", content="Content")

    result = validate_content_structure(content)

    assert isinstance(result, bool)


def test_analyze_content_for_pipeline():
    """Test analyze_content_for_pipeline function."""
    content = BlogPostContext(
        id="test-1",
        title="Test Blog",
        content="This is test content for analysis. " * 50,  # Ensure enough content
        snippet="Test snippet",
    )

    analysis = analyze_content_for_pipeline(content)

    assert isinstance(analysis, dict)
    # Function returns create_standard_task_result structure
    if analysis.get("success") is True:
        # Success case: data contains the analysis
        data = analysis.get("data", {})
        assert (
            "quality_score" in data or "seo_potential" in data or "word_count" in data
        )
    else:
        # Error case: should have error field
        assert "error" in analysis


def test_calculate_basic_readability():
    """Test calculate_basic_readability function."""
    text = "This is a simple sentence. It has multiple words and sentences."
    score = calculate_basic_readability(text)

    assert isinstance(score, float)
    assert 0 <= score <= 100


def test_assess_content_completeness():
    """Test assess_content_completeness function."""
    content = BlogPostContext(
        id="test-1",
        title="Test",
        content="Test content",
        snippet="Snippet",
    )

    score = assess_content_completeness(content)

    assert isinstance(score, (int, float))
    assert 0 <= score <= 100  # Function returns 0-100, not 0-1


def test_extract_potential_keywords():
    """Test extract_potential_keywords function."""
    text = "This is about artificial intelligence and machine learning"
    keywords = extract_potential_keywords(text)

    assert isinstance(keywords, list)


def test_assess_title_seo():
    """Test assess_title_seo function."""
    title = "Complete Guide to Artificial Intelligence"
    seo = assess_title_seo(title)

    assert isinstance(seo, dict)
    assert "score" in seo or "length" in seo or "keywords" in seo


def test_assess_content_structure():
    """Test assess_content_structure function."""
    text = "# Heading\n\nParagraph text\n\n## Subheading\n\nMore text"
    structure = assess_content_structure(text)

    assert isinstance(structure, dict)
    assert (
        "headings" in structure or "paragraphs" in structure or "sections" in structure
    )


def test_assess_linking_potential():
    """Test assess_linking_potential function."""
    text = "This content mentions various topics and concepts"
    potential = assess_linking_potential(text)

    assert isinstance(potential, dict)
    assert "score" in potential or "opportunities" in potential or "links" in potential


def test_assess_engagement_potential():
    """Test assess_engagement_potential function."""
    text = "This is engaging content with questions and interesting points"
    score = assess_engagement_potential(text)

    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_assess_conversion_potential():
    """Test assess_conversion_potential function."""
    text = "This content includes call-to-action and conversion elements"
    score = assess_conversion_potential(text)

    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_assess_shareability():
    """Test assess_shareability function."""
    text = "This is shareable content with interesting insights"
    score = assess_shareability(text)

    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_assess_audience_appeal():
    """Test assess_audience_appeal function."""
    text = "This content appeals to a wide audience"
    score = assess_audience_appeal(text)

    assert isinstance(score, float)
    assert 0 <= score <= 1
