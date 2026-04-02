"""
Tests for Profound Settings API endpoints.
Covers missed lines in api/profound_settings.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.middleware.rbac import require_roles
from marketing_project.models.profound_models import ProfoundSettingsResponse
from marketing_project.models.user_context import UserContext
from marketing_project.server import app

client = TestClient(app)


def _admin_user():
    return UserContext(
        user_id="admin-1",
        email="admin@example.com",
        username="admin",
        roles=["admin"],
        realm_roles=["admin"],
        client_roles=[],
    )


@pytest.fixture(autouse=True)
def override_auth():
    """Override auth to inject an admin user for all tests in this module."""
    admin = _admin_user()

    def _require_admin():
        return admin

    def _get_user():
        return admin

    app.dependency_overrides[require_roles(["admin"])] = _require_admin
    app.dependency_overrides[get_current_user] = _get_user
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_profound_manager():
    with patch(
        "marketing_project.api.profound_settings.get_profound_settings_manager"
    ) as mock:
        mgr = MagicMock()
        mgr.to_response = AsyncMock(
            return_value=ProfoundSettingsResponse(
                is_enabled=True,
                has_api_key=True,
                default_category_id="cat-1",
                created_at=None,
                updated_at=None,
            )
        )
        mgr.upsert = AsyncMock(return_value=None)
        mgr.delete = AsyncMock(return_value=True)
        mgr.get_credentials = AsyncMock(return_value=("test-api-key", "cat-1"))
        mock.return_value = mgr
        yield mgr


@pytest.mark.asyncio
async def test_get_profound_settings(mock_profound_manager):
    """Test GET /api/v1/settings/profound returns settings (lines 38-39)."""
    response = client.get("/api/v1/settings/profound")

    assert response.status_code == 200
    data = response.json()
    assert "is_enabled" in data
    assert "has_api_key" in data


@pytest.mark.asyncio
async def test_update_profound_settings(mock_profound_manager):
    """Test PUT /api/v1/settings/profound updates settings (lines 48-50)."""
    payload = {"is_enabled": True, "api_key": "new-key", "default_category_id": "cat-2"}
    response = client.put("/api/v1/settings/profound", json=payload)

    assert response.status_code == 200
    mock_profound_manager.upsert.assert_called_once()
    mock_profound_manager.to_response.assert_called()


@pytest.mark.asyncio
async def test_delete_profound_settings_success(mock_profound_manager):
    """Test DELETE /api/v1/settings/profound when record exists (lines 58-64)."""
    mock_profound_manager.delete.return_value = True
    response = client.delete("/api/v1/settings/profound")

    assert response.status_code == 204


@pytest.mark.asyncio
async def test_delete_profound_settings_not_found(mock_profound_manager):
    """Test DELETE /api/v1/settings/profound when no record exists (lines 60-64)."""
    mock_profound_manager.delete.return_value = False
    response = client.delete("/api/v1/settings/profound")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_test_profound_connection_no_api_key(mock_profound_manager):
    """Test POST /settings/profound/test when no API key configured (lines 75-79)."""
    mock_profound_manager.get_credentials.return_value = (None, None)
    response = client.post("/api/v1/settings/profound/test")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "No Profound API key" in data["message"]


@pytest.mark.asyncio
async def test_test_profound_connection_no_category_id(mock_profound_manager):
    """Test POST /settings/profound/test when no category_id configured (lines 81-88)."""
    mock_profound_manager.get_credentials.return_value = ("test-api-key", None)
    response = client.post("/api/v1/settings/profound/test")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert (
        "no default_category_id" in data["message"].lower()
        or "category" in data["message"].lower()
    )


@pytest.mark.asyncio
async def test_test_profound_connection_success(mock_profound_manager):
    """Test POST /settings/profound/test when connection succeeds (lines 90-99)."""
    mock_profound_manager.get_credentials.return_value = ("test-api-key", "cat-1")

    with patch(
        "marketing_project.services.profound_client.ProfoundClient.get_category_personas",
        new=AsyncMock(return_value=[{"name": "Persona 1"}, {"name": "Persona 2"}]),
    ):
        response = client.post("/api/v1/settings/profound/test")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["personas_count"] == 2


@pytest.mark.asyncio
async def test_test_profound_connection_failure(mock_profound_manager):
    """Test POST /settings/profound/test when connection fails (lines 100-105)."""
    mock_profound_manager.get_credentials.return_value = ("test-api-key", "cat-1")

    with patch(
        "marketing_project.services.profound_client.ProfoundClient.get_category_personas",
        new=AsyncMock(side_effect=Exception("Connection refused")),
    ):
        response = client.post("/api/v1/settings/profound/test")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "Connection failed" in data["message"] or "failed" in data["message"].lower()
