"""
Tests for system API endpoints.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.system import router
from marketing_project.middleware.keycloak_auth import get_current_user
from tests.utils.keycloak_test_helpers import create_user_context


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    mock_user = create_user_context(roles=["admin"])
    app.dependency_overrides[get_current_user] = lambda: mock_user
    return TestClient(app)


class TestSystemInfo:
    """Test the /system/info endpoint."""

    @patch("marketing_project.api.system.PIPELINE_SPEC", {"test": "spec"})
    @patch("marketing_project.api.system.PROMPTS_DIR", "/test/prompts")
    @patch("marketing_project.api.system.os.path.exists")
    @patch("marketing_project.api.system.os.getenv")
    def test_get_system_info_success(self, mock_getenv, mock_exists, client):
        """Test successful system info retrieval."""
        # Setup mocks
        mock_getenv.side_effect = lambda key, default=None: {
            "DEBUG": "false",
            "LOG_LEVEL": "INFO",
            "TEMPLATE_VERSION": "v1",
        }.get(key, default)
        mock_exists.return_value = True

        response = client.get("/system/info")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "marketing-project"
        assert data["version"] == "2.0.0"
        assert data["environment"]["debug"] is False
        assert data["configuration"]["pipeline_loaded"] is True
        assert data["configuration"]["prompts_dir_exists"] is True

    @patch("marketing_project.api.system.PIPELINE_SPEC", None)
    @patch("marketing_project.api.system.PROMPTS_DIR", "/nonexistent/prompts")
    @patch("marketing_project.api.system.os.path.exists")
    @patch("marketing_project.api.system.os.getenv")
    def test_get_system_info_with_missing_config(
        self, mock_getenv, mock_exists, client
    ):
        """Test system info with missing configuration."""
        # Setup mocks
        mock_getenv.side_effect = lambda key, default=None: {
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
            "TEMPLATE_VERSION": "v2",
        }.get(key, default)
        mock_exists.return_value = False

        response = client.get("/system/info")

        assert response.status_code == 200
        data = response.json()
        assert data["environment"]["debug"] is True
        assert data["configuration"]["pipeline_loaded"] is False
        assert data["configuration"]["prompts_dir_exists"] is False

    @patch("marketing_project.api.system.PIPELINE_SPEC", {"test": "spec"})
    @patch("marketing_project.api.system.PROMPTS_DIR", "/test/prompts")
    @patch("marketing_project.api.system.os.path.exists")
    @patch("marketing_project.api.system.os.getenv")
    def test_get_system_info_error(self, mock_getenv, mock_exists, client):
        """Test system info with error."""
        # Setup mocks
        mock_getenv.side_effect = Exception("Environment error")

        response = client.get("/system/info")

        assert response.status_code == 500
        data = response.json()
        assert "Failed to retrieve system information" in data["detail"]
