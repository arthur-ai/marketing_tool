"""
Tests for trend integration service.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from marketing_project.services.trend_integration import (
    TREND_CACHE_TTL,
    TrendIntegrationService,
)


@pytest.fixture
def trend_service():
    """Create a TrendIntegrationService instance."""
    return TrendIntegrationService()


@pytest.mark.asyncio
async def test_get_trending_hashtags_cached(trend_service):
    """Test getting trending hashtags from cache."""
    # Set up cache
    cache_key = "hashtags:linkedin"
    trend_service._trend_cache[cache_key] = {"hashtags": ["ai", "tech", "marketing"]}
    trend_service._cache_timestamps[cache_key] = datetime.now()

    result = await trend_service.get_trending_hashtags("linkedin", limit=5)

    assert isinstance(result, list)
    assert len(result) <= 5


@pytest.mark.asyncio
async def test_get_trending_hashtags_expired_cache(trend_service):
    """Test getting trending hashtags with expired cache."""
    # Set up expired cache
    cache_key = "hashtags:linkedin"
    trend_service._trend_cache[cache_key] = {"hashtags": ["old"]}
    trend_service._cache_timestamps[cache_key] = datetime.now() - timedelta(
        seconds=TREND_CACHE_TTL + 1
    )

    with patch.object(
        trend_service, "_fetch_trending_hashtags", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = ["new", "trends"]

        result = await trend_service.get_trending_hashtags("linkedin")

        assert mock_fetch.called
        assert isinstance(result, list)


@pytest.mark.asyncio
async def test_get_trending_hashtags_fetch_error(trend_service):
    """Test getting trending hashtags when fetch fails."""
    with patch.object(
        trend_service, "_fetch_trending_hashtags", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.side_effect = Exception("API Error")

        result = await trend_service.get_trending_hashtags("linkedin")

        assert result == []
        assert isinstance(result, list)


@pytest.mark.asyncio
async def test_fetch_trending_hashtags(trend_service):
    """Test _fetch_trending_hashtags method."""
    result = await trend_service._fetch_trending_hashtags("linkedin", limit=10)

    assert isinstance(result, list)
    # Currently returns empty list (placeholder)


@pytest.mark.asyncio
async def test_get_trending_topics_cached(trend_service):
    """Test getting trending topics from cache."""
    cache_key = "topics:tech"
    trend_service._trend_cache[cache_key] = {
        "topics": [{"title": "AI Trends", "category": "tech"}]
    }
    trend_service._cache_timestamps[cache_key] = datetime.now()

    result = await trend_service.get_trending_topics("tech", limit=5)

    assert isinstance(result, list)
    assert len(result) <= 5


@pytest.mark.asyncio
async def test_get_trending_topics_no_category(trend_service):
    """Test getting trending topics without category."""
    result = await trend_service.get_trending_topics(None, limit=5)

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_get_trending_topics_fetch_error(trend_service):
    """Test getting trending topics when fetch fails."""
    with patch.object(
        trend_service, "_fetch_trending_topics", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.side_effect = Exception("API Error")

        result = await trend_service.get_trending_topics("tech")

        assert result == []


@pytest.mark.asyncio
async def test_get_trending_topics(trend_service):
    """Test getting trending topics."""
    with patch.object(
        trend_service, "_fetch_trending_topics", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = [
            {"title": "AI Trends", "category": "tech", "relevance_score": 0.9}
        ]

        result = await trend_service.get_trending_topics("tech", limit=5)

        assert isinstance(result, list)
        mock_fetch.assert_called_once()
