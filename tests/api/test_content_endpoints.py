"""
Tests for content source API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from marketing_project.api.content import router


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_content_source():
    """Mock content source."""
    source = Mock()
    source.config.name = "test_source"
    source.config.source_type.value = "file"
    source.config.enabled = True
    source.config.priority = 1
    source.config.metadata = {"path": "/test/path"}
    source.health_check = AsyncMock(return_value=True)
    return source


class TestListContentSources:
    """Test the /content-sources endpoint."""

    @patch('marketing_project.api.content.content_manager')
    def test_list_content_sources_success(
        self, 
        mock_manager, 
        client, 
        mock_content_source
    ):
        """Test successful listing of content sources."""
        # Setup mocks
        mock_manager.get_all_sources.return_value = [mock_content_source]
        
        response = client.get("/content-sources")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["sources"]) == 1
        assert data["sources"][0]["name"] == "test_source"
        assert data["sources"][0]["status"] == "healthy"
        assert data["sources"][0]["healthy"] is True

    @patch('marketing_project.api.content.content_manager')
    def test_list_content_sources_with_error(
        self, 
        mock_manager, 
        client
    ):
        """Test listing content sources with health check error."""
        # Setup mocks
        source = Mock()
        source.config.name = "error_source"
        source.config.source_type.value = "api"
        source.config.enabled = True
        source.config.priority = 1
        source.config.metadata = {}
        source.health_check = AsyncMock(side_effect=Exception("Connection failed"))
        mock_manager.get_all_sources.return_value = [source]
        
        response = client.get("/content-sources")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["sources"][0]["status"] == "error"
        assert "error" in data["sources"][0]["metadata"]

    @patch('marketing_project.api.content.content_manager')
    def test_list_content_sources_manager_error(
        self, 
        mock_manager, 
        client
    ):
        """Test listing content sources when manager fails."""
        # Setup mocks
        mock_manager.get_all_sources.side_effect = Exception("Manager error")
        
        response = client.get("/content-sources")
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to list content sources" in data["detail"]


class TestGetSourceStatus:
    """Test the /content-sources/{source_name}/status endpoint."""

    @patch('marketing_project.api.content.content_manager')
    def test_get_source_status_success(
        self, 
        mock_manager, 
        client, 
        mock_content_source
    ):
        """Test successful source status retrieval."""
        # Setup mocks
        mock_manager.get_source.return_value = mock_content_source
        
        response = client.get("/content-sources/test_source/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["source"]["name"] == "test_source"
        assert data["source"]["status"] == "healthy"

    @patch('marketing_project.api.content.content_manager')
    def test_get_source_status_not_found(
        self, 
        mock_manager, 
        client
    ):
        """Test source status when source not found."""
        # Setup mocks
        mock_manager.get_source.return_value = None
        
        response = client.get("/content-sources/nonexistent/status")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    @patch('marketing_project.api.content.content_manager')
    def test_get_source_status_error(
        self, 
        mock_manager, 
        client, 
        mock_content_source
    ):
        """Test source status when health check fails."""
        # Setup mocks
        mock_content_source.health_check = AsyncMock(return_value=False)
        mock_manager.get_source.return_value = mock_content_source
        
        response = client.get("/content-sources/test_source/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["source"]["status"] == "unhealthy"


class TestFetchFromSource:
    """Test the /content-sources/{source_name}/fetch endpoint."""

    @patch('marketing_project.api.content.content_manager')
    def test_fetch_from_source_success(
        self, 
        mock_manager, 
        client, 
        mock_content_source
    ):
        """Test successful content fetching."""
        from marketing_project.core.content_sources import ContentSourceResult
        
        # Setup mocks
        fetch_result = ContentSourceResult(
            success=True,
            content_items=[
                {"id": "item1", "title": "Test Item 1", "content": "Content 1"},
                {"id": "item2", "title": "Test Item 2", "content": "Content 2"}
            ],
            total_count=2,
            source_name="test_source"
        )
        mock_content_source.fetch_content = AsyncMock(return_value=fetch_result)
        mock_manager.get_source.return_value = mock_content_source
        
        response = client.post("/content-sources/test_source/fetch?limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_count"] == 2
        assert len(data["content_items"]) == 2

    @patch('marketing_project.api.content.content_manager')
    def test_fetch_from_source_not_found(
        self, 
        mock_manager, 
        client
    ):
        """Test fetching from non-existent source."""
        # Setup mocks
        mock_manager.get_source.return_value = None
        
        response = client.post("/content-sources/nonexistent/fetch")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]

    @patch('marketing_project.api.content.content_manager')
    def test_fetch_from_source_error(
        self, 
        mock_manager, 
        client, 
        mock_content_source
    ):
        """Test fetching when source fails."""
        # Setup mocks
        mock_content_source.fetch_content = AsyncMock(side_effect=Exception("Fetch failed"))
        mock_manager.get_source.return_value = mock_content_source
        
        response = client.post("/content-sources/test_source/fetch")
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to fetch content" in data["detail"]
