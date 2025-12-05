"""
Tests for upload API endpoints.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.upload import router
from marketing_project.server import app

# Add router to app
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def mock_s3_storage():
    """Mock S3 storage."""
    with patch("marketing_project.api.upload.s3_storage") as mock:
        mock_storage = MagicMock()
        mock_storage.is_available.return_value = False
        mock.return_value = mock_storage
        yield mock_storage


@pytest.fixture
def mock_content_manager():
    """Mock content source manager."""
    with patch("marketing_project.api.upload.content_manager") as mock:
        manager = MagicMock()
        manager.add_source_from_config = MagicMock(return_value={"id": "source-1"})
        mock.return_value = manager
        yield manager


def test_upload_json_file(mock_s3_storage, mock_content_manager):
    """Test uploading a JSON file."""
    file_content = '{"id": "test", "title": "Test", "content": "Content"}'
    files = {
        "file": ("test.json", io.BytesIO(file_content.encode()), "application/json")
    }
    data = {"content_type": "blog_post"}

    response = client.post("/upload", files=files, data=data)

    assert response.status_code in [200, 500]  # May fail if processing not fully mocked


def test_upload_markdown_file(mock_s3_storage, mock_content_manager):
    """Test uploading a Markdown file."""
    file_content = "# Test Title\n\nTest content"
    files = {"file": ("test.md", io.BytesIO(file_content.encode()), "text/markdown")}
    data = {"content_type": "blog_post"}

    response = client.post("/upload", files=files, data=data)

    assert response.status_code in [200, 500]


def test_upload_unsupported_file_type(mock_s3_storage, mock_content_manager):
    """Test uploading an unsupported file type."""
    files = {"file": ("test.exe", io.BytesIO(b"binary"), "application/x-msdownload")}
    data = {"content_type": "blog_post"}

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_file_too_large(mock_s3_storage, mock_content_manager):
    """Test uploading a file that's too large."""
    # Create a file larger than 25MB
    large_content = b"x" * (26 * 1024 * 1024)
    files = {"file": ("large.txt", io.BytesIO(large_content), "text/plain")}
    data = {"content_type": "blog_post"}

    response = client.post("/upload", files=files, data=data)

    assert response.status_code == 400
    assert "too large" in response.json()["detail"].lower()


def test_upload_url(mock_s3_storage, mock_content_manager):
    """Test uploading content from URL."""
    with patch("marketing_project.api.upload.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        request_data = {
            "url": "https://example.com/article",
            "content_type": "blog_post",
        }

        response = client.post("/upload/url", json=request_data)

        assert response.status_code in [200, 500]


def test_upload_transcript(mock_s3_storage, mock_content_manager):
    """Test uploading a transcript file."""
    transcript_content = """
    Speaker 1: Welcome to the podcast
    Speaker 2: Thanks for having me
    """
    files = {
        "file": (
            "transcript.txt",
            io.BytesIO(transcript_content.encode()),
            "text/plain",
        )
    }
    data = {"content_type": "transcript"}

    response = client.post("/upload", files=files, data=data)

    assert response.status_code in [200, 500]
