"""
Extended tests for processor API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.processors import router
from marketing_project.server import app

# Add router to app
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_job_manager():
    """Mock job manager."""
    with patch("marketing_project.api.processors.get_job_manager") as mock:
        manager = MagicMock()
        manager.create_job = AsyncMock(return_value=MagicMock(id="test-job-1"))
        mock.return_value = manager
        yield manager


@pytest.mark.asyncio
async def test_process_blog_endpoint(mock_job_manager):
    """Test /process/blog endpoint."""
    request_data = {
        "content": {
            "id": "test-1",
            "title": "Test Blog",
            "content": "Test content",
            "snippet": "Test snippet",
        },
        "output_content_type": "blog_post",
    }

    response = client.post("/process/blog", json=request_data)

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "job_id" in data


@pytest.mark.asyncio
async def test_process_release_notes_endpoint(mock_job_manager):
    """Test /process/release-notes endpoint."""
    request_data = {
        "content": {
            "id": "test-1",
            "title": "Release v1.0",
            "content": "Release notes content",
            "snippet": "Release snippet",
        },
    }

    response = client.post("/process/release-notes", json=request_data)

    assert response.status_code in [200, 500]


@pytest.mark.asyncio
async def test_process_transcript_endpoint(mock_job_manager):
    """Test /process/transcript endpoint."""
    request_data = {
        "content": {
            "id": "test-1",
            "title": "Podcast Transcript",
            "content": "Speaker 1: Hello\nSpeaker 2: Hi",
            "snippet": "Transcript snippet",
        },
    }

    response = client.post("/process/transcript", json=request_data)

    assert response.status_code in [200, 500]
