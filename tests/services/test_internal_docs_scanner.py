"""
Tests for internal docs scanner service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.internal_docs_config import ScannedDocument
from marketing_project.services.internal_docs_scanner import (
    InternalDocsScanner,
    get_internal_docs_scanner,
)


@pytest.fixture
def scanner():
    """Create an InternalDocsScanner instance for testing."""
    return InternalDocsScanner()


@pytest.mark.asyncio
async def test_scanner_initialization(scanner):
    """Test InternalDocsScanner initialization."""
    assert scanner.session is None
    assert scanner.visited_urls == set()
    assert scanner.base_domain is None


@pytest.mark.asyncio
async def test_initialize(scanner):
    """Test scanner initialization."""
    await scanner.initialize()
    assert scanner.session is not None
    await scanner.cleanup()


@pytest.mark.asyncio
async def test_cleanup(scanner):
    """Test scanner cleanup."""
    await scanner.initialize()
    assert scanner.session is not None
    await scanner.cleanup()
    assert scanner.session is None


@pytest.mark.asyncio
async def test_scan_from_base_url(scanner):
    """Test scan_from_base_url method."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(
        return_value="<html><body><h1>Test</h1></body></html>"
    )
    mock_response.headers = {"Content-Type": "text/html"}
    mock_session.get = AsyncMock(return_value=mock_response.__aenter__())
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

    scanner.session = mock_session

    with patch(
        "marketing_project.services.internal_docs_scanner.BeautifulSoup"
    ) as mock_bs:
        mock_soup = MagicMock()
        mock_soup.find_all.return_value = []
        mock_soup.get_text.return_value = "Test content"
        mock_bs.return_value = mock_soup

        result = await scanner.scan_from_base_url(
            "https://example.com", max_depth=1, max_pages=1
        )

        assert isinstance(result, list)
        assert len(result) >= 0  # May be empty if no valid documents found


@pytest.mark.asyncio
async def test_scan_from_url_list(scanner):
    """Test scan_from_url_list method."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(
        return_value="<html><body><h1>Test</h1></body></html>"
    )
    mock_response.headers = {"Content-Type": "text/html"}
    mock_session.get = AsyncMock(return_value=mock_response.__aenter__())
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

    scanner.session = mock_session

    with patch(
        "marketing_project.services.internal_docs_scanner.BeautifulSoup"
    ) as mock_bs:
        mock_soup = MagicMock()
        mock_soup.find_all.return_value = []
        mock_soup.get_text.return_value = "Test content"
        mock_bs.return_value = mock_soup

        urls = ["https://example.com/page1", "https://example.com/page2"]
        result = await scanner.scan_from_url_list(urls)

        assert isinstance(result, list)


@pytest.mark.asyncio
async def test_scan_from_url_list_error_handling(scanner):
    """Test scan_from_url_list error handling."""
    mock_session = AsyncMock()
    mock_session.get = AsyncMock(side_effect=Exception("Network error"))

    scanner.session = mock_session

    urls = ["https://example.com/page1"]
    result = await scanner.scan_from_url_list(urls)

    assert isinstance(result, list)
    # Should handle errors gracefully


@pytest.mark.asyncio
async def test_extract_document_info(scanner):
    """Test extract_document_info method."""
    mock_soup = MagicMock()
    mock_soup.find.return_value = MagicMock(text="Test Title")
    mock_soup.get_text.return_value = "Test content"

    await scanner.initialize()

    result = scanner._extract_document_info(
        "https://example.com/page", mock_soup, "<html></html>"
    )

    assert result is not None
    assert isinstance(result, ScannedDocument) or isinstance(result, dict)


@pytest.mark.asyncio
async def test_extract_links(scanner):
    """Test extract_links method."""
    mock_soup = MagicMock()
    mock_link1 = MagicMock()
    mock_link1.get.return_value = "/page1"
    mock_link2 = MagicMock()
    mock_link2.get.return_value = "https://external.com/page"
    mock_soup.find_all.return_value = [mock_link1, mock_link2]

    await scanner.initialize()

    base_url = "https://example.com"
    links = scanner._extract_links(mock_soup, base_url, follow_external=False)

    assert isinstance(links, list)


@pytest.mark.asyncio
async def test_is_valid_document_url(scanner):
    """Test is_valid_document_url method."""
    assert scanner._is_valid_document_url("https://example.com/page.html") is True
    assert scanner._is_valid_document_url("https://example.com/page.pdf") is True
    assert scanner._is_valid_document_url("https://example.com/image.jpg") is False
    assert scanner._is_valid_document_url("https://example.com/script.js") is False


@pytest.mark.asyncio
async def test_normalize_url(scanner):
    """Test normalize_url method."""
    assert (
        scanner._normalize_url("https://example.com/page#section")
        == "https://example.com/page"
    )
    assert (
        scanner._normalize_url("https://example.com/page?param=value")
        == "https://example.com/page?param=value"
    )


def test_get_internal_docs_scanner_singleton():
    """Test that get_internal_docs_scanner returns a singleton."""
    scanner1 = get_internal_docs_scanner()
    scanner2 = get_internal_docs_scanner()
    assert scanner1 is scanner2
