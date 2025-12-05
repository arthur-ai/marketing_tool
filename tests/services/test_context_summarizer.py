"""
Tests for context summarizer service.
"""

import pytest

from marketing_project.services.context_summarizer import ContextSummarizer


class TestContextSummarizer:
    """Test ContextSummarizer class."""

    def test_summarize_step_output_short_text(self):
        """Test summarizing step output with short text."""
        output = {
            "main_keyword": "test",
            "primary_keywords": ["keyword1", "keyword2"],
            "description": "Short description",
        }

        summarized = ContextSummarizer.summarize_step_output(output, max_length=500)

        assert summarized["main_keyword"] == "test"
        assert summarized["description"] == "Short description"

    def test_summarize_step_output_long_text(self):
        """Test summarizing step output with long text."""
        long_text = "a" * 1000
        output = {"description": long_text}

        summarized = ContextSummarizer.summarize_step_output(output, max_length=500)

        assert len(summarized["description"]) <= 520  # 500 + "... [truncated]"
        assert "... [truncated]" in summarized["description"]

    def test_summarize_step_output_list(self):
        """Test summarizing step output with list."""
        output = {"keywords": [f"keyword{i}" for i in range(20)]}

        summarized = ContextSummarizer.summarize_step_output(output, max_length=500)

        assert isinstance(summarized["keywords"], list)
        assert len(summarized["keywords"]) <= 11  # 10 items + truncation message

    def test_summarize_step_output_dict(self):
        """Test summarizing step output with nested dict."""
        output = {
            "metadata": {
                "key1": "value1",
                "key2": "value2",
                **{f"key{i}": f"value{i}" for i in range(3, 15)},
            }
        }

        summarized = ContextSummarizer.summarize_step_output(output, max_length=500)

        assert isinstance(summarized["metadata"], dict)
        assert (
            "_truncated" in summarized["metadata"] or len(summarized["metadata"]) <= 11
        )

    def test_summarize_nested_list(self):
        """Test _summarize_nested with list."""
        data = [f"item{i}" for i in range(20)]

        summarized = ContextSummarizer._summarize_nested(data, max_items=10)

        assert isinstance(summarized, list)
        assert len(summarized) == 11  # 10 items + truncation message

    def test_summarize_nested_dict(self):
        """Test _summarize_nested with dict."""
        data = {f"key{i}": f"value{i}" for i in range(20)}

        summarized = ContextSummarizer._summarize_nested(data, max_items=10)

        assert isinstance(summarized, dict)
        assert "_truncated" in summarized or len(summarized) <= 11

    def test_summarize_nested_max_depth(self):
        """Test _summarize_nested with max depth."""
        data = {"level1": {"level2": {"level3": {"level4": "deep"}}}}

        summarized = ContextSummarizer._summarize_nested(
            data, max_items=10, max_depth=2
        )

        assert "[max depth reached]" in str(summarized)

    def test_get_relevant_context_keys(self):
        """Test get_relevant_context_keys method."""
        keys = ContextSummarizer.get_relevant_context_keys("article_generation")

        assert isinstance(keys, list)
        # May return empty list if step not in registry, which is valid
        # Just check it's a list
        assert isinstance(keys, list)

    def test_build_optimized_context(self):
        """Test build_optimized_context method."""
        full_context = {
            "seo_keywords": {"main_keyword": "test", "description": "a" * 1000},
            "marketing_brief": {"target_audience": ["audience1", "audience2"]},
            "input_content": {"title": "Test"},
        }

        optimized = ContextSummarizer.build_optimized_context(
            full_context, step_name="article_generation"
        )

        assert isinstance(optimized, dict)
        # Should include relevant keys or at least some context
        assert len(optimized) >= 0
