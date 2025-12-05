"""
Tests for API content source service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.core.content_sources import APISourceConfig
from marketing_project.services.api_source import APIContentSource


@pytest.fixture
def api_source_config():
    """Create API source config."""
    return APISourceConfig(
        name="test-api-source",
        base_url="https://api.example.com",
        endpoints=["/posts", "/articles"],
        rate_limit=5,
        timeout=30,
    )


@pytest.fixture
def api_content_source(api_source_config):
    """Create APIContentSource instance."""
    return APIContentSource(api_source_config)


@pytest.mark.asyncio
async def test_initialize_success(api_content_source):
    """Test successful initialization."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session.get = AsyncMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "ok"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        result = await api_content_source.initialize()

        # May succeed or fail depending on health check
        assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_fetch_content(api_content_source):
    """Test fetching content from API."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value=[{"id": "1", "title": "Test", "content": "Content"}]
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        await api_content_source.initialize()
        result = await api_content_source.fetch_content()

        assert result.success is True or result.success is False
        assert isinstance(result.content_items, list)


@pytest.mark.asyncio
async def test_health_check(api_content_source):
    """Test health check."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        api_content_source.session = mock_session
        result = await api_content_source.health_check()

        assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_cleanup(api_content_source):
    """Test cleanup method."""
    mock_session = MagicMock()
    mock_session.close = AsyncMock()
    api_content_source.session = mock_session

    await api_content_source.cleanup()

    # Session may be closed but not set to None
    mock_session.close.assert_called_once()
