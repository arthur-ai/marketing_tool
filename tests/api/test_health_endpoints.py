"""
Tests for health and readiness check endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from marketing_project.api.health import router


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestHealthCheck:
    """Test the /health endpoint."""

    @patch('marketing_project.api.health.os.path.exists')
    @patch('marketing_project.api.health.PIPELINE_SPEC', {'test': 'config'})
    def test_health_check_success(self, mock_exists, client):
        """Test successful health check."""
        # Setup mocks
        mock_exists.return_value = True
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "marketing-project"
        assert data["version"] == "1.0.0"
        assert data["checks"]["config_loaded"] is True
        assert data["checks"]["prompts_dir_exists"] is True

    @patch('marketing_project.api.health.os.path.exists')
    @patch('marketing_project.api.health.PIPELINE_SPEC', None)
    def test_health_check_config_not_loaded(self, mock_exists, client):
        """Test health check when config is not loaded."""
        # Setup mocks
        mock_exists.return_value = True
        
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["config_loaded"] is False

    @patch('marketing_project.api.health.os.path.exists')
    @patch('marketing_project.api.health.PIPELINE_SPEC', {'test': 'config'})
    def test_health_check_prompts_dir_missing(self, mock_exists, client):
        """Test health check when prompts directory is missing."""
        # Setup mocks
        mock_exists.return_value = False
        
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["prompts_dir_exists"] is False

    @patch('marketing_project.api.health.os.path.exists')
    @patch('marketing_project.api.health.PIPELINE_SPEC', None)
    def test_health_check_both_fail(self, mock_exists, client):
        """Test health check when both checks fail."""
        # Setup mocks
        mock_exists.return_value = False
        
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["config_loaded"] is False
        assert data["checks"]["prompts_dir_exists"] is False

    @patch('marketing_project.api.health.os.path.exists')
    def test_health_check_exception(self, mock_exists, client):
        """Test health check when exception occurs."""
        # Setup mocks
        mock_exists.side_effect = Exception("File system error")
        
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "error" in data


class TestReadinessCheck:
    """Test the /ready endpoint."""

    @patch('marketing_project.api.health.os.path.exists')
    @patch('marketing_project.api.health.PIPELINE_SPEC', {'test': 'config'})
    def test_readiness_check_success(self, mock_exists, client):
        """Test successful readiness check."""
        # Setup mocks
        mock_exists.return_value = True
        
        response = client.get("/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["service"] == "marketing-project"
        assert data["checks"]["config_loaded"] is True
        assert data["checks"]["prompts_dir_exists"] is True

    @patch('marketing_project.api.health.os.path.exists')
    @patch('marketing_project.api.health.PIPELINE_SPEC', None)
    def test_readiness_check_config_not_loaded(self, mock_exists, client):
        """Test readiness check when config is not loaded."""
        # Setup mocks
        mock_exists.return_value = True
        
        response = client.get("/ready")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["config_loaded"] is False

    @patch('marketing_project.api.health.os.path.exists')
    @patch('marketing_project.api.health.PIPELINE_SPEC', {'test': 'config'})
    def test_readiness_check_prompts_dir_missing(self, mock_exists, client):
        """Test readiness check when prompts directory is missing."""
        # Setup mocks
        mock_exists.return_value = False
        
        response = client.get("/ready")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["prompts_dir_exists"] is False

    @patch('marketing_project.api.health.os.path.exists')
    def test_readiness_check_exception(self, mock_exists, client):
        """Test readiness check when exception occurs."""
        # Setup mocks
        mock_exists.side_effect = Exception("File system error")
        
        response = client.get("/ready")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert "error" in data


class TestNoAuthentication:
    """Test that health endpoints don't require authentication."""

    def test_health_check_no_auth_required(self, client):
        """Test that health check doesn't require authentication."""
        with patch('marketing_project.api.health.PIPELINE_SPEC', {'test': 'config'}), \
             patch('marketing_project.api.health.os.path.exists', return_value=True):
            response = client.get("/health")
            assert response.status_code == 200

    def test_readiness_check_no_auth_required(self, client):
        """Test that readiness check doesn't require authentication."""
        with patch('marketing_project.api.health.PIPELINE_SPEC', {'test': 'config'}), \
             patch('marketing_project.api.health.os.path.exists', return_value=True):
            response = client.get("/ready")
            assert response.status_code == 200
