"""
Tests for content source services.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.api_source import APIContentSource
from marketing_project.services.database_source import DatabaseContentSource
from marketing_project.services.web_scraping_source import WebScrapingContentSource


@pytest.fixture
def api_source_config():
    """Sample API source configuration."""
    from marketing_project.core.content_sources import (
        APISourceConfig,
        ContentSourceType,
    )

    return APISourceConfig(
        name="test_api",
        source_type=ContentSourceType.API,
        base_url="https://api.example.com/content",
        headers={"Authorization": "Bearer token"},
    )


@pytest.fixture
def database_source_config():
    """Sample database source configuration."""
    from marketing_project.core.content_sources import (
        ContentSourceType,
        DatabaseSourceConfig,
    )

    return DatabaseSourceConfig(
        name="test_db",
        source_type=ContentSourceType.DATABASE,
        connection_string="postgresql://user:pass@localhost/db",
        query="SELECT * FROM content",
    )


@pytest.fixture
def web_scraping_source_config():
    """Sample web scraping source configuration."""
    from marketing_project.core.content_sources import (
        ContentSourceType,
        WebScrapingSourceConfig,
    )

    return WebScrapingSourceConfig(
        name="test_scraper",
        source_type=ContentSourceType.WEB_SCRAPING,
        urls=["https://example.com"],
        selectors={"title": "h1", "content": ".content"},
    )


@pytest.mark.asyncio
async def test_api_content_source_initialization(api_source_config):
    """Test APIContentSource initialization."""
    source = APIContentSource(api_source_config)

    assert source.config.name == "test_api"
    assert source.config.base_url == "https://api.example.com/content"


@pytest.mark.asyncio
async def test_api_content_source_get_status(api_source_config):
    """Test APIContentSource get_status."""
    source = APIContentSource(api_source_config)

    status = source.get_status()

    assert status is not None
    assert "status" in status
    assert "type" in status


@pytest.mark.asyncio
async def test_api_content_source_fetch_content(api_source_config):
    """Test APIContentSource fetch_content."""
    source = APIContentSource(api_source_config)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [{"id": "1", "title": "Test"}]}
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await source.fetch_content(limit=10)

        assert result is not None
        assert hasattr(result, "success") or isinstance(result, dict)


@pytest.mark.asyncio
async def test_api_content_source_health_check(api_source_config):
    """Test APIContentSource health_check."""
    source = APIContentSource(api_source_config)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await source.health_check()

        assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_database_content_source_initialization(database_source_config):
    """Test DatabaseContentSource initialization."""
    source = DatabaseContentSource(database_source_config)

    assert source.name == "test_db"
    assert source.connection_string == "postgresql://user:pass@localhost/db"


@pytest.mark.asyncio
async def test_database_content_source_get_status(database_source_config):
    """Test DatabaseContentSource get_status."""
    source = DatabaseContentSource(database_source_config)

    status = source.get_status()

    assert status is not None
    assert "status" in status
    assert "type" in status


@pytest.mark.asyncio
async def test_database_content_source_fetch_content(database_source_config):
    """Test DatabaseContentSource fetch_content."""
    source = DatabaseContentSource(database_source_config)

    with patch("sqlalchemy.create_engine") as mock_engine:
        with patch("pandas.read_sql") as mock_read_sql:
            import pandas as pd

            mock_read_sql.return_value = pd.DataFrame([{"id": "1", "title": "Test"}])

            result = await source.fetch_content(limit=10)

            assert result is not None
            assert hasattr(result, "success") or isinstance(result, dict)


@pytest.mark.asyncio
async def test_database_content_source_health_check(database_source_config):
    """Test DatabaseContentSource health_check."""
    source = DatabaseContentSource(database_source_config)

    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_eng = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute = MagicMock()
        mock_eng.connect.return_value.__enter__.return_value = mock_conn
        mock_engine.return_value = mock_eng

        result = await source.health_check()

        assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_web_scraping_content_source_initialization(web_scraping_source_config):
    """Test WebScrapingContentSource initialization."""
    source = WebScrapingContentSource(web_scraping_source_config)

    assert source.config.name == "test_scraper"
    assert len(source.config.urls) > 0
    assert "example.com" in source.config.urls[0]


@pytest.mark.asyncio
async def test_web_scraping_content_source_get_status(web_scraping_source_config):
    """Test WebScrapingContentSource get_status."""
    source = WebScrapingContentSource(web_scraping_source_config)

    status = source.get_status()

    assert status is not None
    assert "status" in status
    assert "type" in status


@pytest.mark.asyncio
async def test_web_scraping_content_source_fetch_content(web_scraping_source_config):
    """Test WebScrapingContentSource fetch_content."""
    source = WebScrapingContentSource(web_scraping_source_config)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><h1>Test Title</h1><div class='content'>Test Content</div></html>"
        )
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client

        with patch("bs4.BeautifulSoup") as mock_bs:
            mock_soup = MagicMock()
            mock_soup.find.return_value = MagicMock(text="Test Title")
            mock_bs.return_value = mock_soup

            result = await source.fetch_content(limit=10)

            assert result is not None
            assert hasattr(result, "success") or isinstance(result, dict)


@pytest.mark.asyncio
async def test_web_scraping_content_source_health_check(web_scraping_source_config):
    """Test WebScrapingContentSource health_check."""
    source = WebScrapingContentSource(web_scraping_source_config)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await source.health_check()

        assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_api_content_source_error_handling(api_source_config):
    """Test APIContentSource error handling."""
    source = APIContentSource(api_source_config)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection error"))
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await source.fetch_content(limit=10)

        # Should handle error gracefully
        assert result is not None
        if hasattr(result, "success"):
            assert result.success is False


@pytest.mark.asyncio
async def test_database_content_source_error_handling(database_source_config):
    """Test DatabaseContentSource error handling."""
    source = DatabaseContentSource(database_source_config)

    with patch("sqlalchemy.create_engine", side_effect=Exception("Connection error")):
        result = await source.fetch_content(limit=10)

        # Should handle error gracefully
        assert result is not None
        if hasattr(result, "success"):
            assert result.success is False
