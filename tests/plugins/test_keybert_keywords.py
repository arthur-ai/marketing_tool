"""
Tests for KeyBERT-based keyword extraction functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from marketing_project.core.models import BlogPostContext
from marketing_project.plugins.seo_keywords.tasks import (
    extract_keywords_advanced,
    extract_keywords_with_keybert,
)


class TestKeyBERTKeywordExtraction:
    """Test cases for KeyBERT-based keyword extraction."""

    @pytest.fixture
    def sample_content(self):
        """Sample content for testing."""
        return BlogPostContext(
            id="test-content",
            title="Machine Learning for Marketing",
            content="Machine learning is revolutionizing marketing strategies. AI-powered tools help businesses analyze customer behavior and optimize campaigns. Data-driven marketing approaches are becoming essential for competitive advantage.",
            snippet="Learn how machine learning transforms marketing",
        )

    @pytest.fixture
    def empty_content(self):
        """Empty content for testing edge cases."""
        return BlogPostContext(
            id="empty-content", title="Empty Test", content="", snippet=""
        )

    @pytest.fixture
    def title_only_content(self):
        """Content with only title for testing."""
        return BlogPostContext(
            id="title-only", title="Marketing Automation Guide", content="", snippet=""
        )

    @patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", False)
    def test_extract_keywords_with_keybert_not_available(self, sample_content):
        """Test KeyBERT extraction when library is not available."""
        result = extract_keywords_with_keybert(sample_content)

        assert result["success"] is False
        assert "KeyBERT library not available" in result["error"]
        assert result["task_name"] == "extract_keywords_with_keybert"

    @patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", True)
    @patch("marketing_project.plugins.seo_keywords.tasks.KeyBERT")
    def test_extract_keywords_with_keybert_mmr_method(
        self, mock_keybert_class, sample_content
    ):
        """Test KeyBERT extraction with MMR method."""
        # Mock the KeyBERT instance and its extract_keywords method
        mock_keybert_instance = MagicMock()
        mock_keybert_class.return_value = mock_keybert_instance
        mock_keybert_instance.extract_keywords.return_value = [
            ("machine learning", 0.95),
            ("marketing", 0.88),
            ("data-driven", 0.82),
            ("customer behavior", 0.75),
            ("optimize campaigns", 0.70),
        ]

        result = extract_keywords_with_keybert(
            sample_content, max_keywords=5, method="mmr"
        )

        assert result["success"] is True
        assert result["data"]["total_keywords_found"] == 5
        assert result["data"]["method"] == "mmr"
        assert result["data"]["extraction_library"] == "keybert"

        keywords = result["data"]["keywords"]
        assert len(keywords) == 5
        assert keywords[0]["keyword"] == "machine learning"
        assert keywords[0]["score"] == 0.95
        assert keywords[0]["method"] == "mmr"
        assert keywords[0]["in_title"] is True  # "machine learning" is in title

    @patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", True)
    @patch("marketing_project.plugins.seo_keywords.tasks.KeyBERT")
    def test_extract_keywords_with_keybert_maxsum_method(
        self, mock_keybert_class, sample_content
    ):
        """Test KeyBERT extraction with maxsum method."""
        # Mock the KeyBERT instance and its extract_keywords method
        mock_keybert_instance = MagicMock()
        mock_keybert_class.return_value = mock_keybert_instance
        mock_keybert_instance.extract_keywords.return_value = [
            ("ai-powered tools", 0.92),
            ("marketing strategies", 0.85),
            ("competitive advantage", 0.78),
        ]

        result = extract_keywords_with_keybert(
            sample_content, max_keywords=3, method="maxsum"
        )

        assert result["success"] is True
        assert result["data"]["method"] == "maxsum"
        assert result["data"]["total_keywords_found"] == 3

        keywords = result["data"]["keywords"]
        assert keywords[0]["keyword"] == "ai-powered tools"
        assert keywords[0]["score"] == 0.92

    @patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", True)
    @patch("marketing_project.plugins.seo_keywords.tasks.KeyBERT")
    def test_extract_keywords_with_keybert_maximal_marginal_relevance_method(
        self, mock_keybert_class, sample_content
    ):
        """Test KeyBERT extraction with maximal_marginal_relevance method."""
        # Mock the KeyBERT instance and its extract_keywords method
        mock_keybert_instance = MagicMock()
        mock_keybert_class.return_value = mock_keybert_instance
        mock_keybert_instance.extract_keywords.return_value = [
            ("marketing automation", 0.88),
            ("customer analytics", 0.82),
        ]

        result = extract_keywords_with_keybert(sample_content, method="maximal_marginal_relevance")

        assert result["success"] is True
        assert result["data"]["method"] == "maximal_marginal_relevance"
        assert result["data"]["total_keywords_found"] == 2

    @patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", True)
    @patch("marketing_project.plugins.seo_keywords.tasks.KeyBERT")
    def test_extract_keywords_with_keybert_invalid_method(
        self, mock_keybert_class, sample_content
    ):
        """Test KeyBERT extraction with invalid method falls back to mmr."""
        # Mock the KeyBERT instance and its extract_keywords method
        mock_keybert_instance = MagicMock()
        mock_keybert_class.return_value = mock_keybert_instance
        mock_keybert_instance.extract_keywords.return_value = [("test keyword", 0.5)]

        result = extract_keywords_with_keybert(sample_content, method="invalid_method")

        assert result["success"] is True
        # Should fall back to mmr method
        mock_keybert_instance.extract_keywords.assert_called_once()
        call_args = mock_keybert_instance.extract_keywords.call_args
        assert call_args[1]["use_mmr"] is True  # Should use mmr as fallback

    def test_extract_keywords_with_keybert_empty_content(self, empty_content):
        """Test KeyBERT extraction with empty content."""
        result = extract_keywords_with_keybert(empty_content)

        assert (
            result["success"] is True
        )  # Our implementation handles empty content gracefully
        assert result["data"]["total_keywords_found"] == 0
        assert result["data"]["keywords"] == []

    def test_extract_keywords_with_keybert_title_only(self, title_only_content):
        """Test KeyBERT extraction with title only content."""
        result = extract_keywords_with_keybert(title_only_content)

        assert result["success"] is True
        assert result["data"]["total_keywords_found"] == 0
        assert result["data"]["keywords"] == []

    @patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", True)
    @patch("marketing_project.plugins.seo_keywords.tasks.KeyBERT")
    def test_extract_keywords_with_keybert_exception_handling(
        self, mock_keybert_class, sample_content
    ):
        """Test KeyBERT extraction exception handling."""
        # Mock the KeyBERT instance to raise an exception
        mock_keybert_instance = MagicMock()
        mock_keybert_class.return_value = mock_keybert_instance
        mock_keybert_instance.extract_keywords.side_effect = Exception("KeyBERT processing error")

        result = extract_keywords_with_keybert(sample_content)

        assert result["success"] is False
        assert "KeyBERT keyword extraction failed" in result["error"]
        assert "KeyBERT processing error" in result["error"]

    @patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", False)
    def test_extract_keywords_advanced_not_available(self, sample_content):
        """Test advanced extraction when KeyBERT is not available."""
        result = extract_keywords_advanced(sample_content)

        assert result["success"] is False
        assert "KeyBERT library not available" in result["error"]

    @patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", True)
    @patch("marketing_project.plugins.seo_keywords.tasks.extract_keywords_with_keybert")
    def test_extract_keywords_advanced_single_method(
        self, mock_extract, sample_content
    ):
        """Test advanced extraction with single method."""
        # Mock successful extraction for mmr method
        mock_extract.return_value = {
            "success": True,
            "data": {
                "keywords": [
                    {
                        "keyword": "machine learning",
                        "score": 0.8,
                        "ngram_length": 2,
                        "in_title": True,
                    },
                    {
                        "keyword": "marketing",
                        "score": 0.7,
                        "ngram_length": 1,
                        "in_title": True,
                    },
                ]
            },
        }

        result = extract_keywords_advanced(
            sample_content, methods=["mmr"], max_keywords=2
        )

        assert result["success"] is True
        assert result["data"]["total_keywords_found"] == 2
        assert result["data"]["methods_used"] == ["mmr"]
        assert result["data"]["combined_results"] is True

        keywords = result["data"]["keywords"]
        assert len(keywords) == 2
        assert keywords[0]["keyword"] == "machine learning"
        assert keywords[0]["total_score"] == 0.8
        assert keywords[0]["confidence"] == 1.0  # Single method coverage

    @patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", True)
    @patch("marketing_project.plugins.seo_keywords.tasks.extract_keywords_with_keybert")
    def test_extract_keywords_advanced_multiple_methods(
        self, mock_extract, sample_content
    ):
        """Test advanced extraction with multiple methods."""

        # Mock different results for different methods
        def mock_extract_side_effect(content, max_keywords, method):
            if method == "mmr":
                return {
                    "success": True,
                    "data": {
                        "keywords": [
                            {
                                "keyword": "machine learning",
                                "score": 0.8,
                                "ngram_length": 2,
                                "in_title": True,
                            },
                            {
                                "keyword": "marketing",
                                "score": 0.7,
                                "ngram_length": 1,
                                "in_title": True,
                            },
                        ]
                    },
                }
            elif method == "maxsum":
                return {
                    "success": True,
                    "data": {
                        "keywords": [
                            {
                                "keyword": "machine learning",
                                "score": 0.9,
                                "ngram_length": 2,
                                "in_title": True,
                            },
                            {
                                "keyword": "ai-powered",
                                "score": 0.6,
                                "ngram_length": 1,
                                "in_title": False,
                            },
                        ]
                    },
                }
            else:  # maximal_marginal_relevance
                return {
                    "success": True,
                    "data": {
                        "keywords": [
                            {
                                "keyword": "marketing strategies",
                                "score": 0.85,
                                "ngram_length": 2,
                                "in_title": False,
                            }
                        ]
                    },
                }

        mock_extract.side_effect = mock_extract_side_effect

        result = extract_keywords_advanced(
            sample_content, methods=["mmr", "maxsum", "maximal_marginal_relevance"], max_keywords=3
        )

        assert result["success"] is True
        assert result["data"]["total_keywords_found"] == 3
        assert result["data"]["methods_used"] == ["mmr", "maxsum", "maximal_marginal_relevance"]

        keywords = result["data"]["keywords"]
        # Should be sorted by total_score
        assert keywords[0]["keyword"] == "machine learning"  # Highest combined score
        assert (
            abs(keywords[0]["total_score"] - 0.85) < 0.01
        )  # Average of 0.8 and 0.9 (with floating point precision)
        assert keywords[0]["confidence"] == 2 / 3  # Found in 2 out of 3 methods

    @patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", True)
    @patch("marketing_project.plugins.seo_keywords.tasks.extract_keywords_with_keybert")
    def test_extract_keywords_advanced_all_methods_fail(
        self, mock_extract, sample_content
    ):
        """Test advanced extraction when all methods fail."""
        # Mock all methods to fail
        mock_extract.return_value = {"success": False, "error": "Method failed"}

        result = extract_keywords_advanced(
            sample_content, methods=["mmr", "maxsum"]
        )

        assert result["success"] is True  # Now returns success with empty results
        assert result["data"]["total_keywords_found"] == 0
        assert result["data"]["keywords"] == []

    @patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", True)
    @patch("marketing_project.plugins.seo_keywords.tasks.extract_keywords_with_keybert")
    def test_extract_keywords_advanced_without_combining(
        self, mock_extract, sample_content
    ):
        """Test advanced extraction without combining results."""
        # Mock successful extraction
        mock_extract.return_value = {
            "success": True,
            "data": {
                "keywords": [
                    {
                        "keyword": "test keyword",
                        "score": 0.5,
                        "ngram_length": 2,
                        "in_title": False,
                    }
                ]
            },
        }

        result = extract_keywords_advanced(
            sample_content, methods=["mmr"], combine_results=False
        )

        assert result["success"] is True
        assert result["data"]["combined_results"] is False
        # Keywords should not have combined scoring when combine_results=False
        keywords = result["data"]["keywords"]
        # When not combining, keywords should have individual method results
        assert "methods" in keywords[0]  # Should have methods field
        assert "scores" in keywords[0]  # Should have scores dict

    def test_extract_keywords_advanced_default_methods(self, sample_content):
        """Test advanced extraction with default methods."""
        with (
            patch("marketing_project.plugins.seo_keywords.tasks.KEYBERT_AVAILABLE", True),
            patch(
                "marketing_project.plugins.seo_keywords.tasks.extract_keywords_with_keybert"
            ) as mock_extract,
        ):

            mock_extract.return_value = {"success": True, "data": {"keywords": []}}

            result = extract_keywords_advanced(sample_content)

            # Should use default methods
            assert result["success"] is True
            assert result["data"]["methods_used"] == ["mmr", "maxsum"]
