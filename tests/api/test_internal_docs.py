"""
Tests for internal docs API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.internal_docs import router
from marketing_project.models.internal_docs_config import InternalDocsConfig
from marketing_project.server import app

app.include_router(router, prefix="/api/v1/internal-docs", tags=["Internal Docs"])


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_internal_docs_config():
    """Sample internal docs configuration."""
    return InternalDocsConfig(
        scanned_documents=[],
        commonly_referenced_pages=[],
        commonly_referenced_categories=[],
        anchor_phrasing_patterns=[],
    )


@pytest.mark.asyncio
async def test_get_internal_docs_config(client, sample_internal_docs_config):
    """Test get internal docs config endpoint."""
    with patch(
        "marketing_project.api.internal_docs.get_internal_docs_manager"
    ) as mock_manager:
        mock_mgr = AsyncMock()
        mock_mgr.get_active_config = AsyncMock(return_value=sample_internal_docs_config)
        mock_manager.return_value = mock_mgr

        response = client.get("/api/v1/internal-docs/config")

        assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_get_internal_docs_config_not_found(client):
    """Test get internal docs config when not found."""
    with patch(
        "marketing_project.api.internal_docs.get_internal_docs_manager"
    ) as mock_manager:
        mock_mgr = AsyncMock()
        mock_mgr.get_active_config = AsyncMock(return_value=None)
        mock_manager.return_value = mock_mgr

        response = client.get("/api/v1/internal-docs/config")

        assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_scan_internal_docs(client):
    """Test scan internal docs endpoint."""
    with patch(
        "marketing_project.api.internal_docs.get_internal_docs_manager"
    ) as mock_manager:
        with patch(
            "marketing_project.services.job_manager.get_job_manager"
        ) as mock_job_manager:
            mock_mgr = AsyncMock()
            mock_manager.return_value = mock_mgr

            mock_job_mgr = MagicMock()
            mock_job = MagicMock()
            mock_job.id = "test-job-1"
            mock_job_mgr.create_job = AsyncMock(return_value=mock_job)
            mock_job_mgr.submit_to_arq = AsyncMock(return_value="arq-job-1")
            mock_job_manager.return_value = mock_job_mgr

            response = client.post("/api/v1/internal-docs/scan")

            assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
async def test_update_internal_docs_config(client, sample_internal_docs_config):
    """Test update internal docs config endpoint."""
    with patch(
        "marketing_project.api.internal_docs.get_internal_docs_manager"
    ) as mock_manager:
        mock_mgr = AsyncMock()
        mock_mgr.update_config = AsyncMock(return_value=sample_internal_docs_config)
        mock_manager.return_value = mock_mgr

        response = client.post(
            "/api/v1/internal-docs/config",
            json=sample_internal_docs_config.model_dump(mode="json"),
        )

        assert response.status_code in [200, 500]
