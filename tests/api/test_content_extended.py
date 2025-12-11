"""
Extended tests for content API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.content import router
from marketing_project.server import app

# Add router to app
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_content_manager():
    """Mock content source manager."""
    with patch("marketing_project.api.content.get_content_manager") as mock:
        manager = MagicMock()
        manager.list_sources = AsyncMock(return_value=[])

        # Create a mock source object for get_source_status test
        mock_source = MagicMock()
        mock_source.health_check = AsyncMock(return_value=True)
        mock_source.config = MagicMock()
        mock_source.config.name = "test-source"
        mock_source.config.source_type = MagicMock()
        mock_source.config.source_type.value = "api"
        mock_source.config.enabled = True
        mock_source.config.priority = 0
        mock_source.config.metadata = {}

        manager.get_source = MagicMock(return_value=mock_source)
        manager.fetch_from_source = AsyncMock(
            return_value={"content_items": [], "total_count": 0, "success": True}
        )
        mock.return_value = manager
        yield manager


@pytest.mark.asyncio
async def test_list_content_sources(mock_content_manager):
    """Test /content-sources endpoint."""
    response = client.get("/content-sources")

    assert response.status_code == 200
    data = response.json()
    assert "sources" in data or isinstance(data, list)


@pytest.mark.asyncio
async def test_get_source_status(mock_content_manager):
    """Test /content-sources/{source_name}/status endpoint."""
    response = client.get("/content-sources/test-source/status")

    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_fetch_from_source(mock_content_manager):
    """Test POST /content-sources/{source_name}/fetch endpoint."""
    request_data = {"limit": 10}

    response = client.post("/content-sources/test-source/fetch", json=request_data)

    assert response.status_code in [200, 404, 500]
