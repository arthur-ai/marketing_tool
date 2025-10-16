"""
Tests for system API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from marketing_project.api.system import router


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestSystemInfo:
    """Test the /system/info endpoint."""

    @patch('marketing_project.api.system.PIPELINE_SPEC', {"test": "spec"})
    @patch('marketing_project.api.system.PROMPTS_DIR', "/test/prompts")
    @patch('os.path.exists')
    @patch('os.getenv')
    def test_get_system_info_success(self, mock_getenv, mock_exists, client):
        """Test successful system info retrieval."""
        # Setup mocks
        mock_getenv.side_effect = lambda key, default=None: {
            "DEBUG": "false",
            "LOG_LEVEL": "INFO",
            "TEMPLATE_VERSION": "v1"
        }.get(key, default)
        mock_exists.return_value = True
        
        response = client.get("/system/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "marketing-project"
        assert data["version"] == "1.0.0"
        assert data["environment"]["debug"] is False
        assert data["configuration"]["pipeline_loaded"] is True
        assert data["configuration"]["prompts_dir_exists"] is True

    @patch('marketing_project.api.system.PIPELINE_SPEC', None)
    @patch('marketing_project.api.system.PROMPTS_DIR', "/nonexistent/prompts")
    @patch('os.path.exists')
    @patch('os.getenv')
    def test_get_system_info_with_missing_config(self, mock_getenv, mock_exists, client):
        """Test system info with missing configuration."""
        # Setup mocks
        mock_getenv.side_effect = lambda key, default=None: {
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
            "TEMPLATE_VERSION": "v2"
        }.get(key, default)
        mock_exists.return_value = False
        
        response = client.get("/system/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["environment"]["debug"] is True
        assert data["configuration"]["pipeline_loaded"] is False
        assert data["configuration"]["prompts_dir_exists"] is False

    @patch('marketing_project.api.system.PIPELINE_SPEC', {"test": "spec"})
    @patch('marketing_project.api.system.PROMPTS_DIR', "/test/prompts")
    @patch('os.path.exists')
    @patch('os.getenv')
    def test_get_system_info_error(self, mock_getenv, mock_exists, client):
        """Test system info with error."""
        # Setup mocks
        mock_getenv.side_effect = Exception("Environment error")
        
        response = client.get("/system/info")
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to retrieve system information" in data["detail"]