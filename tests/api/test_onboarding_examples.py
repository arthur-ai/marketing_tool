"""
Tests for onboarding examples API endpoints and related validation.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.models.onboarding_models import (
    OnboardingExampleCreateRequest,
    OnboardingExampleResponse,
    OnboardingExamplesListResponse,
    OnboardingExampleUpdateRequest,
)
from marketing_project.models.user_context import UserContext
from marketing_project.server import app

client = TestClient(app)


def _make_example(id=1, title="Test", job_type="blog", is_active=True):
    return OnboardingExampleResponse(
        id=id,
        title=title,
        description=None,
        job_type=job_type,
        input_data={},
        display_order=0,
        is_active=is_active,
    )


def _admin_user():
    return UserContext(
        user_id="admin-1",
        email="admin@example.com",
        username="admin",
        roles=["admin"],
        realm_roles=["admin"],
        client_roles=[],
    )


def _regular_user():
    return UserContext(
        user_id="user-1",
        email="user@example.com",
        username="user",
        roles=[],
        realm_roles=[],
        client_roles=[],
    )


# ---------------------------------------------------------------------------
# Validator tests (no DB / HTTP needed)
# ---------------------------------------------------------------------------


def test_create_request_small_input_data_passes():
    req = OnboardingExampleCreateRequest(
        title="Test", job_type="blog", input_data={"key": "value"}
    )
    assert req.input_data == {"key": "value"}


def test_create_request_empty_input_data_passes():
    req = OnboardingExampleCreateRequest(title="Test", job_type="blog")
    assert req.input_data == {}


def test_create_request_oversized_input_data_raises():
    from pydantic import ValidationError

    large_data = {"data": "x" * 70_000}
    with pytest.raises(ValidationError, match="input_data must not exceed"):
        OnboardingExampleCreateRequest(
            title="Test", job_type="blog", input_data=large_data
        )


def test_update_request_none_input_data_passes():
    req = OnboardingExampleUpdateRequest(title="New Title")
    assert req.input_data is None


def test_update_request_oversized_input_data_raises():
    from pydantic import ValidationError

    large_data = {"data": "x" * 70_000}
    with pytest.raises(ValidationError, match="input_data must not exceed"):
        OnboardingExampleUpdateRequest(input_data=large_data)


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_manager():
    with patch(
        "marketing_project.api.onboarding_examples.get_onboarding_examples_manager"
    ) as mock:
        mgr = MagicMock()
        mgr.list_active = AsyncMock(
            return_value=[_make_example(1), _make_example(2, title="Second")]
        )
        mgr.count_active = AsyncMock(return_value=2)
        mgr.list_all = AsyncMock(
            return_value=[
                _make_example(1),
                _make_example(2, is_active=False),
            ]
        )
        mgr.count_all = AsyncMock(return_value=2)
        mgr.get = AsyncMock(return_value=_make_example(1))
        mgr.create = AsyncMock(return_value=_make_example(3, title="Created"))
        mgr.update = AsyncMock(return_value=_make_example(1, title="Updated"))
        mgr.delete = AsyncMock(return_value=True)
        mock.return_value = mgr
        yield mgr


@pytest.fixture
def override_auth_admin():
    """Override auth to inject an admin user."""
    from marketing_project.middleware.keycloak_auth import get_current_user
    from marketing_project.middleware.rbac import require_roles

    admin = _admin_user()

    def _require_admin_override():
        return admin

    def _get_user_override():
        return admin

    app.dependency_overrides[require_roles(["admin"])] = _require_admin_override
    app.dependency_overrides[get_current_user] = _get_user_override
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def override_auth_user():
    """Override auth to inject a regular (non-admin) user."""
    from marketing_project.middleware.keycloak_auth import get_current_user

    user = _regular_user()

    def _get_user_override():
        return user

    app.dependency_overrides[get_current_user] = _get_user_override
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_active_examples_returns_list(mock_manager, override_auth_user):
    response = client.get("/api/v1/onboarding-examples")
    assert response.status_code == 200
    data = response.json()
    assert "examples" in data
    assert data["total"] == 2


@pytest.mark.asyncio
async def test_list_active_examples_pagination_params(mock_manager, override_auth_user):
    response = client.get("/api/v1/onboarding-examples?limit=5&offset=10")
    assert response.status_code == 200
    mock_manager.list_active.assert_called_once_with(limit=5, offset=10)


@pytest.mark.asyncio
async def test_list_active_examples_requires_auth():
    """Without any auth override, endpoint should return 401, 403, or 500 (Keycloak unreachable)."""
    response = client.get("/api/v1/onboarding-examples")
    assert response.status_code in (401, 403, 422, 500)


@pytest.mark.asyncio
async def test_list_all_admin_returns_all(mock_manager, override_auth_admin):
    response = client.get("/api/v1/onboarding-examples/admin")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2


@pytest.mark.asyncio
async def test_create_example_admin(mock_manager, override_auth_admin):
    payload = {"title": "New", "job_type": "blog", "input_data": {"x": 1}}
    response = client.post("/api/v1/onboarding-examples/admin", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Created"


@pytest.mark.asyncio
async def test_create_example_oversized_input_rejected(override_auth_admin):
    """Large input_data should fail Pydantic validation before hitting the service."""
    payload = {
        "title": "Bomb",
        "job_type": "blog",
        "input_data": {"x": "y" * 70_000},
    }
    response = client.post("/api/v1/onboarding-examples/admin", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_example_found(mock_manager, override_auth_admin):
    response = client.get("/api/v1/onboarding-examples/admin/1")
    assert response.status_code == 200
    assert response.json()["id"] == 1


@pytest.mark.asyncio
async def test_get_example_not_found(mock_manager, override_auth_admin):
    mock_manager.get.return_value = None
    response = client.get("/api/v1/onboarding-examples/admin/999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_example(mock_manager, override_auth_admin):
    payload = {"title": "Updated"}
    response = client.patch("/api/v1/onboarding-examples/admin/1", json=payload)
    assert response.status_code == 200
    assert response.json()["title"] == "Updated"


@pytest.mark.asyncio
async def test_update_example_not_found(mock_manager, override_auth_admin):
    mock_manager.update.return_value = None
    response = client.patch(
        "/api/v1/onboarding-examples/admin/999", json={"title": "X"}
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_example(mock_manager, override_auth_admin):
    response = client.delete("/api/v1/onboarding-examples/admin/1")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_delete_example_not_found(mock_manager, override_auth_admin):
    mock_manager.delete.return_value = False
    response = client.delete("/api/v1/onboarding-examples/admin/999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_admin_endpoints_deny_non_admin(override_auth_user):
    """Non-admin user should be denied from admin endpoints."""
    response = client.get("/api/v1/onboarding-examples/admin")
    assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Error path tests to cover missed lines (65-67, 85-87, 109-111, 134-136, 157-159)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_active_examples_service_error(override_auth_user):
    """Test list_active_examples returns 500 when service raises (lines 46-50)."""
    with patch(
        "marketing_project.api.onboarding_examples.get_onboarding_examples_manager"
    ) as mock:
        mgr = MagicMock()
        mgr.list_active = AsyncMock(side_effect=Exception("DB error"))
        mgr.count_active = AsyncMock(return_value=0)
        mock.return_value = mgr

        response = client.get("/api/v1/onboarding-examples")
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_list_all_examples_service_error(override_auth_admin):
    """Test list_all_examples returns 500 when service raises (lines 65-69)."""
    with patch(
        "marketing_project.api.onboarding_examples.get_onboarding_examples_manager"
    ) as mock:
        mgr = MagicMock()
        mgr.list_all = AsyncMock(side_effect=Exception("DB error"))
        mgr.count_all = AsyncMock(return_value=0)
        mock.return_value = mgr

        response = client.get("/api/v1/onboarding-examples/admin")
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_create_example_service_error(override_auth_admin):
    """Test create_example returns 500 when service raises (lines 85-89)."""
    with patch(
        "marketing_project.api.onboarding_examples.get_onboarding_examples_manager"
    ) as mock:
        mgr = MagicMock()
        mgr.create = AsyncMock(side_effect=Exception("DB write error"))
        mock.return_value = mgr

        payload = {"title": "New Example", "job_type": "blog", "input_data": {}}
        response = client.post("/api/v1/onboarding-examples/admin", json=payload)
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_get_example_service_error(override_auth_admin):
    """Test get_example returns 500 when service raises (lines 109-113)."""
    with patch(
        "marketing_project.api.onboarding_examples.get_onboarding_examples_manager"
    ) as mock:
        mgr = MagicMock()
        mgr.get = AsyncMock(side_effect=Exception("DB read error"))
        mock.return_value = mgr

        response = client.get("/api/v1/onboarding-examples/admin/1")
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_update_example_service_error(override_auth_admin):
    """Test update_example returns 500 when service raises (lines 134-138)."""
    with patch(
        "marketing_project.api.onboarding_examples.get_onboarding_examples_manager"
    ) as mock:
        mgr = MagicMock()
        mgr.update = AsyncMock(side_effect=Exception("DB write error"))
        mock.return_value = mgr

        response = client.patch(
            "/api/v1/onboarding-examples/admin/1", json={"title": "Updated"}
        )
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_delete_example_service_error(override_auth_admin):
    """Test delete_example returns 500 when service raises (lines 157-161)."""
    with patch(
        "marketing_project.api.onboarding_examples.get_onboarding_examples_manager"
    ) as mock:
        mgr = MagicMock()
        mgr.delete = AsyncMock(side_effect=Exception("DB error"))
        mock.return_value = mgr

        response = client.delete("/api/v1/onboarding-examples/admin/1")
        assert response.status_code == 500
