"""
Unit tests for ProfoundSettingsManager and the async get_profound_client() factory.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.profound_models import ProfoundSettingsRequest

# ---------------------------------------------------------------------------
# ProfoundSettingsManager.get_credentials
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_credentials_returns_db_values_when_record_exists():
    """DB record with api_key takes precedence over env vars."""
    from marketing_project.services.profound_settings_manager import (
        ProfoundSettingsManager,
    )

    mock_record = MagicMock()
    mock_record.is_enabled = True
    mock_record.api_key = "db-api-key"
    mock_record.default_category_id = "db-cat-uuid"

    mgr = ProfoundSettingsManager()
    with patch.object(mgr, "get", AsyncMock(return_value=mock_record)):
        api_key, cat_id = await mgr.get_credentials()

    assert api_key == "db-api-key"
    assert cat_id == "db-cat-uuid"


@pytest.mark.asyncio
async def test_get_credentials_returns_none_when_db_disabled():
    """Disabled DB record falls back to env vars (or None)."""
    from marketing_project.services.profound_settings_manager import (
        ProfoundSettingsManager,
    )

    mock_record = MagicMock()
    mock_record.is_enabled = False
    mock_record.api_key = "db-api-key"
    mock_record.default_category_id = "db-cat-uuid"

    mgr = ProfoundSettingsManager()
    with patch.object(mgr, "get", AsyncMock(return_value=mock_record)):
        with patch.dict("os.environ", {}, clear=True):
            api_key, cat_id = await mgr.get_credentials()

    # Disabled record → env-var fallback (no env vars set → None)
    assert api_key is None
    assert cat_id is None


@pytest.mark.asyncio
async def test_get_credentials_falls_back_to_env_when_no_db_record():
    """No DB record → use PROFOUND_API_KEY + PROFOUND_CATEGORY_ID env vars."""
    from marketing_project.services.profound_settings_manager import (
        ProfoundSettingsManager,
    )

    mgr = ProfoundSettingsManager()
    with patch.object(mgr, "get", AsyncMock(return_value=None)):
        with patch.dict(
            "os.environ",
            {"PROFOUND_API_KEY": "env-key", "PROFOUND_CATEGORY_ID": "env-cat"},
        ):
            api_key, cat_id = await mgr.get_credentials()

    assert api_key == "env-key"
    assert cat_id == "env-cat"


@pytest.mark.asyncio
async def test_get_credentials_returns_none_when_no_config():
    """No DB record, no env vars → (None, None); pipeline uses default keywords."""
    from marketing_project.services.profound_settings_manager import (
        ProfoundSettingsManager,
    )

    mgr = ProfoundSettingsManager()
    with patch.object(mgr, "get", AsyncMock(return_value=None)):
        with patch.dict("os.environ", {}, clear=True):
            api_key, cat_id = await mgr.get_credentials()

    assert api_key is None
    assert cat_id is None


# ---------------------------------------------------------------------------
# to_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_response_never_exposes_raw_api_key():
    """to_response() returns has_api_key=True but not the key itself."""
    from marketing_project.services.profound_settings_manager import (
        ProfoundSettingsManager,
    )

    mock_record = MagicMock()
    mock_record.is_enabled = True
    mock_record.api_key = "super-secret"
    mock_record.default_category_id = "uuid-123"
    mock_record.created_at = None
    mock_record.updated_at = None

    mgr = ProfoundSettingsManager()
    with patch.object(mgr, "get", AsyncMock(return_value=mock_record)):
        resp = await mgr.to_response()

    assert resp.has_api_key is True
    assert not hasattr(resp, "api_key") or resp.__dict__.get("api_key") is None
    assert resp.default_category_id == "uuid-123"
    assert resp.is_enabled is True


@pytest.mark.asyncio
async def test_to_response_returns_empty_when_no_record():
    from marketing_project.services.profound_settings_manager import (
        ProfoundSettingsManager,
    )

    mgr = ProfoundSettingsManager()
    with patch.object(mgr, "get", AsyncMock(return_value=None)):
        resp = await mgr.to_response()

    assert resp.has_api_key is False
    assert resp.is_enabled is False
    assert resp.default_category_id is None


# ---------------------------------------------------------------------------
# async get_profound_client() factory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_profound_client_returns_configured_client():
    """get_profound_client() builds a ProfoundClient using DB credentials."""
    from marketing_project.services.profound_client import get_profound_client

    with patch(
        "marketing_project.services.profound_client.get_profound_settings_manager"
    ) as mock_get_mgr:
        mock_mgr = MagicMock()
        mock_mgr.get_credentials = AsyncMock(return_value=("db-key", "db-cat-id"))
        mock_get_mgr.return_value = mock_mgr

        client, cat_id = await get_profound_client()

    assert client.is_configured()
    assert client.api_key == "db-key"
    assert cat_id == "db-cat-id"


@pytest.mark.asyncio
async def test_get_profound_client_returns_unconfigured_client_when_no_key():
    """get_profound_client() returns unconfigured client when no API key anywhere."""
    from marketing_project.services.profound_client import get_profound_client

    with patch(
        "marketing_project.services.profound_client.get_profound_settings_manager"
    ) as mock_get_mgr:
        mock_mgr = MagicMock()
        mock_mgr.get_credentials = AsyncMock(return_value=(None, None))
        mock_get_mgr.return_value = mock_mgr

        client, cat_id = await get_profound_client()

    assert not client.is_configured()
    assert cat_id is None


@pytest.mark.asyncio
async def test_get_profound_client_falls_back_to_env_on_db_error():
    """get_profound_client() falls back to env vars if DB lookup raises."""
    from marketing_project.services.profound_client import get_profound_client

    with patch(
        "marketing_project.services.profound_client.get_profound_settings_manager",
        side_effect=RuntimeError("DB unavailable"),
    ):
        with patch.dict(
            "os.environ",
            {"PROFOUND_API_KEY": "env-fallback", "PROFOUND_CATEGORY_ID": "env-cat"},
        ):
            client, cat_id = await get_profound_client()

    assert client.api_key == "env-fallback"
    assert cat_id == "env-cat"
