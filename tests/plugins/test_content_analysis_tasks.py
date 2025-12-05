"""
Tests for content analysis plugin tasks.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.plugins.content_analysis.tasks import (
    analyze_content_for_pipeline,
)


def test_analyze_content_for_pipeline():
    """Test analyze_content_for_pipeline function."""
    from marketing_project.models.content_models import BlogPostContext

    content = BlogPostContext(
        id="test-1",
        title="Test Blog",
        content="This is test content for analysis.",
        snippet="Test snippet",
    )

    analysis = analyze_content_for_pipeline(content)

    assert isinstance(analysis, dict)
    assert (
        "quality_score" in analysis
        or "seo_potential" in analysis
        or "word_count" in analysis
    )
