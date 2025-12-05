"""
Tests for settings API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.settings import router
from marketing_project.server import app

# Add router to app
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_settings_manager():
    """Mock pipeline settings manager."""
    with patch("marketing_project.api.settings.get_pipeline_settings_manager") as mock:
        manager = MagicMock()
        manager.load_settings = AsyncMock(return_value=None)
        manager.save_settings = AsyncMock()
        mock.return_value = manager
        yield manager


@pytest.mark.asyncio
async def test_get_pipeline_settings_default(mock_settings_manager):
    """Test getting default pipeline settings."""
    response = client.get("/v1/settings/pipeline")

    assert response.status_code == 200
    data = response.json()
    assert "pipeline_config" in data
    assert "optional_steps" in data
    assert data["pipeline_config"]["default_model"] == "gpt-5.1"


@pytest.mark.asyncio
async def test_get_pipeline_settings_existing(mock_settings_manager):
    """Test getting existing pipeline settings."""
    from marketing_project.services.pipeline_settings_manager import PipelineSettings

    mock_settings = PipelineSettings(
        pipeline_config={"default_model": "gpt-4", "default_temperature": 0.5},
        optional_steps=["suggested_links"],
        retry_strategy={"max_retries": 3},
    )
    mock_settings_manager.load_settings = AsyncMock(return_value=mock_settings)

    response = client.get("/v1/settings/pipeline")

    assert response.status_code == 200
    data = response.json()
    assert data["pipeline_config"]["default_model"] == "gpt-4"
    assert data["optional_steps"] == ["suggested_links"]


@pytest.mark.asyncio
async def test_save_pipeline_settings(mock_settings_manager):
    """Test saving pipeline settings."""
    request_data = {
        "pipeline_config": {
            "default_model": "gpt-4",
            "default_temperature": 0.7,
        },
        "optional_steps": ["suggested_links", "design_kit"],
        "retry_strategy": {"max_retries": 3},
    }

    response = client.post("/v1/settings/pipeline", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["pipeline_config"]["default_model"] == "gpt-4"
    assert "suggested_links" in data["optional_steps"]
    mock_settings_manager.save_settings.assert_called_once()


@pytest.mark.asyncio
async def test_save_pipeline_settings_error(mock_settings_manager):
    """Test saving pipeline settings with error."""
    mock_settings_manager.save_settings = AsyncMock(side_effect=Exception("DB Error"))

    request_data = {
        "pipeline_config": {},
        "optional_steps": [],
    }

    response = client.post("/v1/settings/pipeline", json=request_data)

    assert response.status_code == 500
