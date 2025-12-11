"""
Tests for SEO metrics provider service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.engines.seo_keywords.seo_metrics_provider import (
    SEOMetricsProvider,
)


@pytest.fixture
def seo_metrics_provider():
    """Create a SEOMetricsProvider instance."""
    return SEOMetricsProvider()


@pytest.mark.asyncio
async def test_get_keyword_difficulty(seo_metrics_provider):
    """Test get_keyword_difficulty method."""
    keywords = ["artificial intelligence", "machine learning"]

    difficulty = await seo_metrics_provider.get_keyword_difficulty(keywords)

    assert isinstance(difficulty, dict)
    assert len(difficulty) == len(keywords)


def test_calculate_base_difficulty(seo_metrics_provider):
    """Test _calculate_base_difficulty method."""
    difficulty = seo_metrics_provider._calculate_base_difficulty("test keyword")

    assert isinstance(difficulty, float)
    assert 0 <= difficulty <= 100


@pytest.mark.asyncio
async def test_get_search_volume(seo_metrics_provider):
    """Test get_search_volume method."""
    keywords = ["test keyword", "another keyword"]

    volume = await seo_metrics_provider.get_search_volume(keywords)

    assert isinstance(volume, dict)
    assert len(volume) == len(keywords)


@pytest.mark.asyncio
async def test_get_keyword_metadata(seo_metrics_provider):
    """Test get_keyword_metadata method."""
    keywords = ["test keyword"]

    metadata = await seo_metrics_provider.get_keyword_metadata(keywords)

    assert isinstance(metadata, list)
    assert len(metadata) == len(keywords)


@pytest.mark.asyncio
async def test_analyze_serp_with_llm(seo_metrics_provider):
    """Test analyze_serp_with_llm method."""
    keyword = "test keyword"  # Method takes a single keyword string, not a list

    with patch.object(seo_metrics_provider, "pipeline") as mock_pipeline:
        from marketing_project.services.engines.seo_keywords.seo_metrics_provider import (
            SERPAnalysisResult,
        )

        mock_result = SERPAnalysisResult(
            result_count_estimate="1000000",
            typical_domains=["example.com"],
            competition_level="medium",
        )
        mock_pipeline.execute_single_step = AsyncMock(
            return_value={"result": mock_result}
        )

        serp_data = await seo_metrics_provider.analyze_serp_with_llm(keyword)

        assert isinstance(serp_data, dict)


def test_get_fallback_serp_analysis(seo_metrics_provider):
    """Test _get_fallback_serp_analysis method."""
    analysis = seo_metrics_provider._get_fallback_serp_analysis("test keyword")

    assert isinstance(analysis, dict)
    assert "result_count_estimate" in analysis or "competition_level" in analysis


@pytest.mark.asyncio
async def test_get_domain_authorities(seo_metrics_provider):
    """Test _get_domain_authorities method."""
    domains = ["example.com", "test.com"]

    with patch("aiohttp.ClientSession") as mock_session:
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={"response": [{"domain": "example.com", "rank": 50}]}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_response
        mock_session.return_value = mock_session

        authorities = await seo_metrics_provider._get_domain_authorities(domains)

        assert isinstance(authorities, dict)
