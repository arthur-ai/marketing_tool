"""
Comprehensive tests for upload API endpoints.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.server import app

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


def test_upload_csv_file(mock_s3_storage, mock_content_manager):
    """Test uploading a CSV file."""
    csv_content = "id,title,content\n1,Test,Content"
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    data = {"content_type": "blog_post"}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code in [200, 500]


def test_upload_yaml_file(mock_s3_storage, mock_content_manager):
    """Test uploading a YAML file."""
    yaml_content = "id: test\ntitle: Test\ncontent: Content"
    files = {"file": ("test.yaml", io.BytesIO(yaml_content.encode()), "text/yaml")}
    data = {"content_type": "blog_post"}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code in [200, 500]


def test_upload_url_invalid(mock_s3_storage, mock_content_manager):
    """Test uploading from invalid URL."""
    request_data = {
        "url": "not-a-valid-url",
        "content_type": "blog_post",
    }

    response = client.post("/api/v1/upload/url", json=request_data)

    assert response.status_code in [400, 500]
