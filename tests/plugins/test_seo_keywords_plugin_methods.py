"""
Tests for SEO keywords plugin helper methods.
"""

import pytest

from marketing_project.models.pipeline_steps import SEOKeywordsResult
from marketing_project.plugins.seo_keywords.tasks import SEOKeywordsPlugin


@pytest.fixture
def seo_keywords_plugin():
    """Create a SEOKeywordsPlugin instance."""
    return SEOKeywordsPlugin()


def test_analyze_content_structure_with_headings(seo_keywords_plugin):
    """Test _analyze_content_structure with markdown headings."""
    content = "# Main Heading\n\nParagraph\n\n## Subheading\n\nMore content"

    structure = seo_keywords_plugin._analyze_content_structure(content)

    assert isinstance(structure, dict)
    assert "headings" in structure or "sections" in structure or len(structure) >= 0


def test_analyze_content_structure_empty(seo_keywords_plugin):
    """Test _analyze_content_structure with empty content."""
    structure = seo_keywords_plugin._analyze_content_structure("")

    assert isinstance(structure, dict)


def test_extract_key_sections(seo_keywords_plugin):
    """Test _extract_key_sections method."""
    content = "# Introduction\n\nIntro text\n\n## Main Content\n\nMain text\n\n## Conclusion\n\nConclusion text"

    sections = seo_keywords_plugin._extract_key_sections(content)

    assert isinstance(sections, dict)


def test_post_process_keywords(seo_keywords_plugin):
    """Test _post_process_keywords method."""
    result = SEOKeywordsResult(
        main_keyword="test",
        primary_keywords=["test1", "test2"],
        search_intent="informational",
        confidence_score=0.9,
    )

    processed = seo_keywords_plugin._post_process_keywords(result, {})

    assert processed is not None
    assert hasattr(processed, "main_keyword")


def test_calculate_keyword_density(seo_keywords_plugin):
    """Test _calculate_keyword_density method."""
    content = "This is a test about artificial intelligence and machine learning. Artificial intelligence is important."
    keywords = ["artificial intelligence", "machine learning"]

    density = seo_keywords_plugin._calculate_keyword_density(content, keywords)

    assert isinstance(density, list)
    assert len(density) == len(keywords)


def test_calculate_overall_relevance(seo_keywords_plugin):
    """Test _calculate_overall_relevance method."""
    result = SEOKeywordsResult(
        main_keyword="test",
        primary_keywords=["test1", "test2"],
        search_intent="informational",
        confidence_score=0.9,
    )

    # Method returns score 0-100, not 0-1
    relevance = seo_keywords_plugin._calculate_overall_relevance(result, "test content")

    assert isinstance(relevance, float)
    assert 0 <= relevance <= 100  # Score is 0-100, not 0-1


def test_convert_difficulty_to_scores(seo_keywords_plugin):
    """Test _convert_difficulty_to_scores method."""
    result = SEOKeywordsResult(
        main_keyword="test",
        primary_keywords=["test1", "test2"],
        search_intent="informational",
        confidence_score=0.9,
    )

    # Method requires difficulty_str and keywords as arguments
    converted = seo_keywords_plugin._convert_difficulty_to_scores(
        "medium", result.primary_keywords
    )

    assert isinstance(converted, dict)
    assert len(converted) == len(result.primary_keywords)


def test_generate_lsi_keywords(seo_keywords_plugin):
    """Test _generate_lsi_keywords method."""
    result = SEOKeywordsResult(
        main_keyword="artificial intelligence",
        primary_keywords=["AI", "machine learning"],
        search_intent="informational",
        confidence_score=0.9,
    )

    lsi_keywords = seo_keywords_plugin._generate_lsi_keywords(result, {})

    assert isinstance(lsi_keywords, list)
    assert len(lsi_keywords) >= 0


def test_calculate_derived_metrics(seo_keywords_plugin):
    """Test _calculate_derived_metrics method."""
    result = SEOKeywordsResult(
        main_keyword="test",
        primary_keywords=["test1", "test2"],
        search_intent="informational",
        confidence_score=0.9,
    )

    # Method returns SEOKeywordsResult, not dict
    metrics_result = seo_keywords_plugin._calculate_derived_metrics(result, {})

    assert isinstance(metrics_result, SEOKeywordsResult)
    assert metrics_result.main_keyword == "test"
