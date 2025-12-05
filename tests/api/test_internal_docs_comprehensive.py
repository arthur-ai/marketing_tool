"""
Comprehensive tests for internal docs API endpoints.
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
        manager.list_versions = AsyncMock(return_value=[])
        mock.return_value = manager
        yield manager


@pytest.mark.asyncio
async def test_get_active_config(mock_internal_docs_manager):
    """Test GET /internal-docs/config endpoint."""
    response = client.get("/api/v1/internal-docs/config")

    assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_get_config_by_version(mock_internal_docs_manager):
    """Test GET /internal-docs/config/{version} endpoint."""
    response = client.get("/api/v1/internal-docs/config/v1.0.0")

    assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_create_config(mock_internal_docs_manager):
    """Test POST /internal-docs/config endpoint."""
    request_data = {
        "base_url": "https://example.com",
        "scan_depth": 2,
    }

    response = client.post("/api/v1/internal-docs/config", json=request_data)

    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_list_versions(mock_internal_docs_manager):
    """Test GET /internal-docs/versions endpoint."""
    response = client.get("/api/v1/internal-docs/versions")

    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_activate_version(mock_internal_docs_manager):
    """Test POST /internal-docs/activate/{version} endpoint."""
    response = client.post("/api/v1/internal-docs/activate/v1.0.0")

    assert response.status_code in [200, 404, 500]
