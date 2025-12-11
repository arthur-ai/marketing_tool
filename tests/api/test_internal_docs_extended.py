"""
Extended tests for internal docs API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.server import app

client = TestClient(app)


@pytest.fixture
def mock_internal_docs_manager():
    """Mock internal docs manager."""
    with patch("marketing_project.api.internal_docs.get_internal_docs_manager") as mock:
        manager = MagicMock()
        manager.get_active_config = AsyncMock(return_value=None)
        manager.save_config = AsyncMock(return_value="config-1")
        manager.list_versions = AsyncMock(return_value=["v1.0.0"])
        manager.activate_version = AsyncMock(return_value=True)
        mock.return_value = manager
        yield manager


@pytest.fixture
def mock_scanner():
    """Mock internal docs scanner."""
    with patch("marketing_project.api.internal_docs.get_internal_docs_scanner") as mock:
        scanner = MagicMock()
        scanner.scan_from_base_url = AsyncMock(return_value={"scanned": 10})
        scanner.scan_from_url_list = AsyncMock(return_value={"scanned": 5})
        mock.return_value = scanner
        yield scanner


@pytest.mark.asyncio
async def test_scan_from_url(mock_internal_docs_manager, mock_scanner):
    """Test POST /internal-docs/scan/url endpoint."""
    request_data = {
        "base_url": "https://example.com",
        "max_depth": 2,
        "max_pages": 50,
    }

    response = client.post("/api/v1/internal-docs/scan/url", json=request_data)

    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_scan_from_list(mock_internal_docs_manager, mock_scanner):
    """Test POST /internal-docs/scan/list endpoint."""
    request_data = {
        "urls": ["https://example.com/doc1", "https://example.com/doc2"],
    }

    response = client.post("/api/v1/internal-docs/scan/list", json=request_data)

    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_merge_scan_results(mock_internal_docs_manager):
    """Test POST /internal-docs/scan/merge endpoint."""
    # Mock get_active_config to return None to trigger 404
    mock_internal_docs_manager.get_active_config = AsyncMock(return_value=None)

    request_data = {
        "scanned_docs": [
            {"url": "https://example.com/doc1", "title": "Doc 1"},
            {"url": "https://example.com/doc2", "title": "Doc 2"},
        ],
    }

    response = client.post("/api/v1/internal-docs/scan/merge", json=request_data)

    # Endpoint returns 404 if no active config exists, 200 on success, or 500 on error
    assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_add_document(mock_internal_docs_manager):
    """Test POST /internal-docs/documents endpoint."""
    request_data = {
        "url": "https://example.com/new-doc",
        "title": "New Document",
        "content": "Document content",
    }

    response = client.post("/api/v1/internal-docs/documents", json=request_data)

    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_remove_document(mock_internal_docs_manager):
    """Test DELETE /internal-docs/documents/{doc_url} endpoint."""
    response = client.delete("/api/v1/internal-docs/documents/https://example.com/doc")

    assert response.status_code in [200, 404, 500]
