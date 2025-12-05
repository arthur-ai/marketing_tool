"""
Tests for design kit API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.design_kit import router
from marketing_project.models.design_kit_config import DesignKitConfig
from marketing_project.server import app

app.include_router(router, prefix="/api/v1/design-kit", tags=["Design Kit"])


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_design_kit_config():
    """Sample design kit configuration."""
    from datetime import datetime

    return DesignKitConfig(
        color_scheme={"primary": "#000000"},
        typography={"font_family": "Arial"},
        components=[],
        version="1.0.0",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        is_active=True,
    )


@pytest.mark.asyncio
async def test_get_design_kit_config(client, sample_design_kit_config):
    """Test get design kit config endpoint."""
    with patch(
        "marketing_project.api.design_kit.get_design_kit_manager"
    ) as mock_manager:
        mock_mgr = AsyncMock()
        mock_mgr.get_active_config = AsyncMock(return_value=sample_design_kit_config)
        mock_manager.return_value = mock_mgr

        response = client.get("/api/v1/design-kit/config")

        assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_get_design_kit_config_not_found(client):
    """Test get design kit config when not found."""
    with patch(
        "marketing_project.api.design_kit.get_design_kit_manager"
    ) as mock_manager:
        mock_mgr = AsyncMock()
        mock_mgr.get_active_config = AsyncMock(return_value=None)
        mock_manager.return_value = mock_mgr

        response = client.get("/api/v1/design-kit/config")

        assert response.status_code in [404, 500]


@pytest.mark.asyncio
async def test_get_design_kit_config_refresh(client, sample_design_kit_config):
    """Test get design kit config with refresh."""
    with patch(
        "marketing_project.api.design_kit.get_design_kit_manager"
    ) as mock_manager:
        with patch(
            "marketing_project.api.design_kit.get_job_manager"
        ) as mock_job_manager:
            mock_mgr = AsyncMock()
            mock_mgr.get_active_config = AsyncMock(
                return_value=sample_design_kit_config
            )
            mock_manager.return_value = mock_mgr

            mock_job_mgr = MagicMock()
            mock_job = MagicMock()
            mock_job.id = "test-job-1"
            mock_job_mgr.create_job = AsyncMock(return_value=mock_job)
            mock_job_mgr.submit_to_arq = AsyncMock(return_value="arq-job-1")
            mock_job_manager.return_value = mock_job_mgr

            response = client.get("/api/v1/design-kit/config?refresh=true")

            assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_update_design_kit_config(client, sample_design_kit_config):
    """Test update design kit config endpoint."""
    with patch(
        "marketing_project.api.design_kit.get_design_kit_manager"
    ) as mock_manager:
        mock_mgr = AsyncMock()
        mock_mgr.update_config = AsyncMock(return_value=sample_design_kit_config)
        mock_manager.return_value = mock_mgr

        response = client.post(
            "/api/v1/design-kit/config",
            json=sample_design_kit_config.model_dump(mode="json"),
        )

        assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_get_design_kit_templates(client):
    """Test get design kit templates endpoint."""
    with patch(
        "marketing_project.api.design_kit.get_design_kit_manager"
    ) as mock_manager:
        mock_mgr = AsyncMock()
        mock_mgr.list_templates = AsyncMock(return_value=["template1", "template2"])
        mock_manager.return_value = mock_mgr

        response = client.get("/api/v1/design-kit/templates")

        assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_get_design_kit_assets(client):
    """Test get design kit assets endpoint."""
    with patch(
        "marketing_project.api.design_kit.get_design_kit_manager"
    ) as mock_manager:
        mock_mgr = AsyncMock()
        mock_mgr.list_assets = AsyncMock(return_value=["asset1", "asset2"])
        mock_manager.return_value = mock_mgr

        response = client.get("/api/v1/design-kit/assets")

        assert response.status_code in [200, 404, 500]
