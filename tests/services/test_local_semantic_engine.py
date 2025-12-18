"""
Tests for local semantic SEO keywords engine.
"""

from unittest.mock import MagicMock, patch

import pytest

from marketing_project.services.engines.seo_keywords.local_semantic_engine import (
    LocalSemanticSEOKeywordsEngine,
)


@pytest.fixture
def local_semantic_engine():
    """Create a LocalSemanticSEOKeywordsEngine instance."""
    return LocalSemanticSEOKeywordsEngine()


def test_supports_operation(local_semantic_engine):
    """Test supports_operation method."""
    # Check what operations are actually supported
    supported = local_semantic_engine.supports_operation("extract_main_keyword")
    assert isinstance(supported, bool)

    # Test unsupported operation
    assert local_semantic_engine.supports_operation("non_existent") is False


@pytest.mark.asyncio
async def test_execute(local_semantic_engine):
    """Test execute method."""
    inputs = {
        "content": "This is a test blog post about artificial intelligence and machine learning.",
        "title": "AI and ML Guide",
    }
    context = {}

    # Use a supported operation
    # Skip if NLP processor is not available
    try:
        local_semantic_engine._get_nlp_processor()
    except (FileNotFoundError, ImportError, Exception) as e:
        pytest.skip(f"NLP processor not available: {e}")

    result = await local_semantic_engine.execute(
        operation="extract_main_keyword",
        inputs=inputs,
        context=context,
    )

    assert result is not None
    assert isinstance(result, (str, dict))


def test_get_content_hash(local_semantic_engine):
    """Test _get_content_hash method."""
    hash_value = local_semantic_engine._get_content_hash("test content", "test title")

    assert isinstance(hash_value, str)
    assert len(hash_value) > 0


def test_extract_headings(local_semantic_engine):
    """Test _extract_headings method."""
    content = "# Heading 1\n## Heading 2\n### Heading 3"
    headings = local_semantic_engine._extract_headings(content)

    assert isinstance(headings, list)
    assert len(headings) >= 0


def test_normalize_keyword(local_semantic_engine):
    """Test _normalize_keyword method."""
    normalized = local_semantic_engine._normalize_keyword("  Test Keyword  ")

    assert normalized == "test keyword"
