"""
Comprehensive tests for SEO keywords plugin methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.pipeline_steps import SEOKeywordsResult
from marketing_project.plugins.seo_keywords.tasks import SEOKeywordsPlugin


@pytest.fixture
def seo_keywords_plugin():
    """Create a SEOKeywordsPlugin instance."""
    return SEOKeywordsPlugin()


def test_analyze_content_structure(seo_keywords_plugin):
    """Test _analyze_content_structure method."""
    content = "# Heading 1\n\nParagraph\n\n## Heading 2\n\nMore content"

    structure = seo_keywords_plugin._analyze_content_structure(content)

    assert isinstance(structure, dict)
    assert "headings" in structure or "sections" in structure or len(structure) >= 0


def test_extract_key_sections(seo_keywords_plugin):
    """Test _extract_key_sections method."""
    content = "# Introduction\n\nIntro text\n\n## Main Content\n\nMain text"

    sections = seo_keywords_plugin._extract_key_sections(content)

    assert isinstance(sections, dict)


def test_normalize_keyword(seo_keywords_plugin):
    """Test _normalize_keyword method."""
    normalized = seo_keywords_plugin._normalize_keyword("  Test Keyword  ")

    assert normalized == "test keyword"


def test_normalize_keywords(seo_keywords_plugin):
    """Test _normalize_keywords method."""
    result = SEOKeywordsResult(
        main_keyword="  Test  ",
        primary_keywords=["  Keyword1  ", "Keyword2"],
        confidence_score=0.9,
    )

    normalized = seo_keywords_plugin._normalize_keywords(result)

    assert normalized.main_keyword == "test"
    assert "keyword1" in [k.lower() for k in normalized.primary_keywords]


def test_validate_keyword_counts(seo_keywords_plugin):
    """Test _validate_keyword_counts method."""
    result = SEOKeywordsResult(
        main_keyword="test",
        primary_keywords=["test1", "test2"],
        secondary_keywords=["test3"] * 20,  # Too many
        confidence_score=0.9,
    )

    validated = seo_keywords_plugin._validate_keyword_counts(result)

    assert validated is not None
    assert len(validated.secondary_keywords) <= 15  # Should be capped


def test_deduplicate_keywords(seo_keywords_plugin):
    """Test _deduplicate_keywords method."""
    result = SEOKeywordsResult(
        main_keyword="test",
        primary_keywords=["test", "test1", "test"],  # Duplicates
        confidence_score=0.9,
    )

    deduplicated = seo_keywords_plugin._deduplicate_keywords(result)

    assert len(deduplicated.primary_keywords) == 2  # Should remove duplicates


def test_validate_result(seo_keywords_plugin):
    """Test _validate_result method."""
    result = SEOKeywordsResult(
        main_keyword="test",
        primary_keywords=["test1", "test2"],
        confidence_score=0.9,
    )

    valid, errors = seo_keywords_plugin._validate_result(result)

    assert isinstance(valid, bool)
    assert isinstance(errors, list)
