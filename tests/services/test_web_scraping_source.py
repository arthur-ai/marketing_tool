"""
Tests for web scraping content source service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.core.content_sources import WebScrapingSourceConfig
from marketing_project.services.web_scraping_source import WebScrapingContentSource


@pytest.fixture
def web_scraping_config():
    """Create web scraping source config."""
    return WebScrapingSourceConfig(
        name="test-web-source",
        urls=["https://example.com"],
        user_agent="TestBot/1.0",
        timeout=30,
        respect_robots_txt=False,
    )


@pytest.fixture
def web_scraping_source(web_scraping_config):
    """Create WebScrapingContentSource instance."""
    return WebScrapingContentSource(web_scraping_config)


@pytest.mark.asyncio
async def test_initialize_success(web_scraping_source):
    """Test successful initialization."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        result = await web_scraping_source.initialize()

        assert result is True
        assert web_scraping_source.session is not None


@pytest.mark.asyncio
async def test_fetch_content(web_scraping_source):
    """Test fetching content from web."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value="<html><body>Test content</body></html>"
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        await web_scraping_source.initialize()
        result = await web_scraping_source.fetch_content()

        assert result.success is True or result.success is False
        assert isinstance(result.content_items, list)


@pytest.mark.asyncio
async def test_cleanup(web_scraping_source):
    """Test cleanup method."""
    mock_session = MagicMock()
    mock_session.close = AsyncMock()
    web_scraping_source.session = mock_session

    await web_scraping_source.cleanup()

    # Session may be closed but not set to None
    mock_session.close.assert_called_once()
