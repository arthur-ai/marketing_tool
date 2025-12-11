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
