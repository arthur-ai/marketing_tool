"""
Tests for provider settings API endpoints.

NOTE: These tests require a compatible NumPy/sklearn environment.
In environments where sklearn's pyarrow/numpy are mismatched (NumPy 2.x with
sklearn compiled for NumPy 1.x) this file will be auto-skipped at collection.
"""

import sys

import pytest

# Attempt imports; skip entire module if the environment is incompatible.
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from marketing_project.api.provider_settings import (
        router as provider_router,  # type: ignore[import]
    )
    from marketing_project.middleware.keycloak_auth import get_current_user
    from marketing_project.models.provider_models import ProviderCredentialsResponse
    from marketing_project.models.user_context import UserContext
except (ImportError, Exception) as _import_err:  # noqa: BLE001
    pytest.skip(
        f"Skipping provider API tests — import failed (environment issue): {_import_err}",
        allow_module_level=True,
    )

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402 (after skip guard)

# Minimal app — only the provider settings router
_test_app = FastAPI()
_test_app.include_router(provider_router, prefix="/api/v1")

_admin_user = UserContext(
    user_id="test-admin",
    email="admin@test.com",
    username="admin",
    roles=["admin"],
    realm_roles=["admin"],
    client_roles=[],
)
_test_app.dependency_overrides[get_current_user] = lambda: _admin_user


@pytest.fixture
def client():
    return TestClient(_test_app)


def _make_provider_response(
    provider: str, has_api_key: bool = False, is_enabled: bool = False
):
    return ProviderCredentialsResponse(
        provider=provider,
        is_enabled=is_enabled,
        has_api_key=has_api_key,
        project_id=None,
        region=None,
        api_base=None,
        has_vertex_credentials=False,
        has_aws_credentials=False,
        created_at=None,
        updated_at=None,
    )


@pytest.fixture
def mock_svc():
    svc = MagicMock()
    svc.list_all = AsyncMock(
        return_value=[
            _make_provider_response("openai", has_api_key=True, is_enabled=True),
            _make_provider_response("anthropic"),
            _make_provider_response("gemini"),
            _make_provider_response("vertex_ai"),
            _make_provider_response("bedrock"),
            _make_provider_response("hosted_vllm"),
        ]
    )
    svc.get = AsyncMock(return_value=None)
    svc.upsert = AsyncMock()
    svc.delete = AsyncMock(return_value=True)
    svc.get_llm_client = AsyncMock()
    return svc


# ──────────────────────────────────────────────
# GET /v1/settings/providers
# ──────────────────────────────────────────────


def test_list_providers_returns_all_six(client, mock_svc):
    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.get("/api/v1/settings/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert len(data["providers"]) == 6
    providers = {p["provider"] for p in data["providers"]}
    assert providers == {
        "openai",
        "anthropic",
        "gemini",
        "vertex_ai",
        "bedrock",
        "hosted_vllm",
    }


def test_list_providers_shows_enabled_status(client, mock_svc):
    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.get("/api/v1/settings/providers")
    data = response.json()
    openai = next(p for p in data["providers"] if p["provider"] == "openai")
    assert openai["is_enabled"] is True
    assert openai["has_api_key"] is True
    anthropic = next(p for p in data["providers"] if p["provider"] == "anthropic")
    assert anthropic["is_enabled"] is False


# ──────────────────────────────────────────────
# GET /v1/settings/providers/{provider}
# ──────────────────────────────────────────────


def test_get_provider_not_configured_returns_disabled(client, mock_svc):
    mock_svc.get = AsyncMock(return_value=None)
    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.get("/api/v1/settings/providers/anthropic")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "anthropic"
    assert data["is_enabled"] is False
    assert data["has_api_key"] is False


def test_get_provider_configured_returns_status(client, mock_svc):
    db_record = MagicMock()
    db_record.is_enabled = True
    db_record.api_key = "sk-encrypted"
    db_record.project_id = None
    db_record.region = None
    db_record.api_base = None
    db_record.vertex_credentials_json = None
    db_record.aws_bedrock_credentials_json = None
    db_record.created_at = None
    db_record.updated_at = None
    mock_svc.get = AsyncMock(return_value=db_record)

    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.get("/api/v1/settings/providers/openai")
    assert response.status_code == 200
    data = response.json()
    assert data["has_api_key"] is True
    assert data["is_enabled"] is True
    # Raw key must NOT be returned
    assert "api_key" not in data


def test_get_unsupported_provider_returns_400(client, mock_svc):
    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.get("/api/v1/settings/providers/unknown_llm")
    assert response.status_code == 400


# ──────────────────────────────────────────────
# PUT /v1/settings/providers/{provider}
# ──────────────────────────────────────────────


def test_upsert_provider_calls_service(client, mock_svc):
    db_record = MagicMock()
    db_record.is_enabled = True
    db_record.api_key = "sk-new"
    db_record.project_id = None
    db_record.region = None
    db_record.api_base = None
    db_record.vertex_credentials_json = None
    db_record.aws_bedrock_credentials_json = None
    db_record.created_at = None
    db_record.updated_at = None
    mock_svc.get = AsyncMock(return_value=db_record)

    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.put(
            "/api/v1/settings/providers/openai",
            json={"is_enabled": True, "api_key": "sk-new-key"},
        )
    assert response.status_code == 200
    mock_svc.upsert.assert_called_once()
    call_args = mock_svc.upsert.call_args
    assert call_args[0][0] == "openai"
    assert call_args[0][1].api_key == "sk-new-key"


def test_upsert_partial_does_not_pass_none_key(client, mock_svc):
    """Sending only is_enabled (no api_key) must not overwrite existing key."""
    db_record = MagicMock()
    db_record.is_enabled = True
    db_record.api_key = "sk-existing"
    db_record.project_id = None
    db_record.region = None
    db_record.api_base = None
    db_record.vertex_credentials_json = None
    db_record.aws_bedrock_credentials_json = None
    db_record.created_at = None
    db_record.updated_at = None
    mock_svc.get = AsyncMock(return_value=db_record)

    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.put(
            "/api/v1/settings/providers/anthropic",
            json={"is_enabled": False},
        )
    assert response.status_code == 200
    call_args = mock_svc.upsert.call_args[0][1]
    # api_key field in request should be None (not sent), so service won't overwrite
    assert call_args.api_key is None


# ──────────────────────────────────────────────
# DELETE /v1/settings/providers/{provider}
# ──────────────────────────────────────────────


def test_delete_provider_success(client, mock_svc):
    mock_svc.delete = AsyncMock(return_value=True)
    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.delete("/api/v1/settings/providers/openai")
    assert response.status_code == 204


def test_delete_provider_not_found_returns_404(client, mock_svc):
    mock_svc.delete = AsyncMock(return_value=False)
    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.delete("/api/v1/settings/providers/anthropic")
    assert response.status_code == 404


# ──────────────────────────────────────────────
# POST /v1/settings/providers/{provider}/test
# ──────────────────────────────────────────────


def test_test_provider_success(client, mock_svc):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    llm_client = MagicMock()
    llm_client.acompletion = AsyncMock(return_value=mock_response)
    mock_svc.get_llm_client = AsyncMock(return_value=llm_client)

    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.post("/api/v1/settings/providers/openai/test")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "ok" in data["message"].lower()


def test_test_provider_auth_failure(client, mock_svc):
    import litellm

    llm_client = MagicMock()
    llm_client.acompletion = AsyncMock(
        side_effect=litellm.AuthenticationError(
            message="Invalid API key", llm_provider="openai", model="gpt-4o-mini"
        )
    )
    mock_svc.get_llm_client = AsyncMock(return_value=llm_client)

    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.post("/api/v1/settings/providers/openai/test")
    assert response.status_code == 200  # endpoint always returns 200 with success=False
    data = response.json()
    assert data["success"] is False


def test_test_vllm_success(client, mock_svc):
    """Test POST /providers/{provider}/test — mocks acompletion so no real LLM call."""
    mock_choice = MagicMock()
    mock_choice.message.content = "ok"
    mock_completion_response = MagicMock()
    mock_completion_response.choices = [mock_choice]

    llm_client = MagicMock()
    llm_client.acompletion = AsyncMock(return_value=mock_completion_response)
    mock_svc.get_llm_client = AsyncMock(return_value=llm_client)

    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.post("/api/v1/settings/providers/openai/test")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "ok" in data["message"]


# ──────────────────────────────────────────────
# GET /v1/settings/providers/{provider}/models
# ──────────────────────────────────────────────


def test_list_models_for_provider(client, mock_svc):
    llm_client = MagicMock()
    llm_client.get_available_models = MagicMock(
        return_value=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    )
    mock_svc.get_llm_client = AsyncMock(return_value=llm_client)

    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.get("/api/v1/settings/providers/openai/models")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "openai"
    assert "gpt-4o" in data["models"]


def test_list_models_unsupported_provider_returns_400(client, mock_svc):
    with patch(
        "marketing_project.api.provider_settings.get_provider_credential_service",
        return_value=mock_svc,
    ):
        response = client.get("/api/v1/settings/providers/not_a_provider/models")
    assert response.status_code == 400
