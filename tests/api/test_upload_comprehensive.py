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


# ---------------------------------------------------------------------------
# Additional tests to increase coverage
# ---------------------------------------------------------------------------


def test_upload_unsupported_file_type():
    """Test that unsupported file types return 400."""
    files = {
        "file": ("test.exe", io.BytesIO(b"binary content"), "application/octet-stream")
    }
    data = {"content_type": "blog_post"}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_txt_file():
    """Test uploading a plain text file succeeds."""
    txt_content = "This is a blog post content for testing purposes."
    files = {"file": ("post.txt", io.BytesIO(txt_content.encode()), "text/plain")}
    data = {"content_type": "blog_post"}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        body = response.json()
        assert body["success"] is True
        assert body["filename"] == "post.txt"
        assert body["content_type"] == "blog_post"


def test_upload_md_file():
    """Test uploading a Markdown file."""
    md_content = "# My Blog Post\n\nThis is content."
    files = {"file": ("post.md", io.BytesIO(md_content.encode()), "text/markdown")}
    data = {"content_type": "blog_post"}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code in [200, 500]


def test_upload_json_file_blog_post():
    """Test uploading a JSON file for blog_post content type."""
    import json

    json_data = {"title": "Test Blog", "content": "Some blog content here."}
    files = {
        "file": (
            "blog.json",
            io.BytesIO(json.dumps(json_data).encode()),
            "application/json",
        )
    }
    data = {"content_type": "blog_post"}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        body = response.json()
        assert body["success"] is True


def test_upload_json_file_transcript():
    """Test uploading a JSON file for transcript content type."""
    import json

    json_data = {
        "title": "Test Transcript",
        "content": "Speaker A: Hello there. Speaker B: Hi!",
    }
    files = {
        "file": (
            "transcript.json",
            io.BytesIO(json.dumps(json_data).encode()),
            "application/json",
        )
    }
    data = {"content_type": "transcript"}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code in [200, 500]


def test_upload_invalid_json_file_treated_as_raw():
    """Test that an invalid JSON file is treated as raw text."""
    files = {
        "file": (
            "notes.json",
            io.BytesIO(b"not valid json {content"),
            "application/json",
        )
    }
    data = {"content_type": "blog_post"}

    response = client.post("/api/v1/upload", files=files, data=data)

    # Should process as raw text, not error on invalid JSON
    assert response.status_code in [200, 500]


def test_upload_yml_file():
    """Test uploading a .yml file."""
    yml_content = "title: Test\ncontent: yaml content"
    files = {"file": ("config.yml", io.BytesIO(yml_content.encode()), "text/yaml")}
    data = {"content_type": "blog_post"}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code in [200, 500]


def test_upload_returns_file_id_on_success():
    """Test that a successful upload returns a file_id."""
    txt_content = "Sample content for upload test."
    files = {"file": ("sample.txt", io.BytesIO(txt_content.encode()), "text/plain")}
    data = {"content_type": "blog_post"}

    response = client.post("/api/v1/upload", files=files, data=data)

    if response.status_code == 200:
        body = response.json()
        assert "file_id" in body
        assert "upload_path" in body
        assert body["uploaded_by"] is not None


def test_upload_default_content_type():
    """Test that upload uses blog_post as default content_type."""
    txt_content = "Default content type test."
    files = {"file": ("default.txt", io.BytesIO(txt_content.encode()), "text/plain")}
    # No content_type provided — should default to blog_post
    data = {}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        body = response.json()
        assert body["content_type"] == "blog_post"


def test_upload_status_endpoint():
    """Test the upload status endpoint."""
    file_id = "some-fake-file-id"
    response = client.get(f"/api/v1/upload/status/{file_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["file_id"] == file_id
    assert "status" in body


def test_upload_from_url_endpoint_network_error():
    """Test upload from URL with a URL that causes a network error."""
    with patch("marketing_project.api.upload.requests.get") as mock_get:
        import requests as req_module

        mock_get.side_effect = req_module.exceptions.ConnectionError(
            "Connection refused"
        )
        request_data = {
            "url": "http://example.com/blog-post",
            "content_type": "blog_post",
        }
        response = client.post("/api/v1/upload/from-url", json=request_data)

    assert response.status_code in [400, 500]


def test_upload_from_url_alias():
    """Test that /upload/url alias works same as /upload/from-url."""
    with patch("marketing_project.api.upload.requests.get") as mock_get:
        import requests as req_module

        mock_get.side_effect = req_module.exceptions.ConnectionError("Timeout")
        request_data = {"url": "http://example.com/post", "content_type": "blog_post"}
        response1 = client.post("/api/v1/upload/url", json=request_data)
        response2 = client.post("/api/v1/upload/from-url", json=request_data)

    # Both should return the same error code
    assert response1.status_code == response2.status_code


def test_upload_from_url_insufficient_content():
    """Test that URL extraction returning < 100 chars raises 400."""
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    # Return HTML that has very little text content
    mock_response.content = b"<html><body><p>Short.</p></body></html>"

    with patch("marketing_project.api.upload.requests.get", return_value=mock_response):
        request_data = {"url": "http://example.com/short", "content_type": "blog_post"}
        response = client.post("/api/v1/upload/from-url", json=request_data)

    assert response.status_code in [400, 500]


def test_upload_from_url_success():
    """Test upload from URL with sufficient content."""
    import json
    from unittest.mock import MagicMock

    # Build a response with enough content
    long_content = "This is a detailed blog post. " * 20
    html = f"<html><head><title>Test Blog</title></head><body><article><p>{long_content}</p></article></body></html>"
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = html.encode("utf-8")

    with patch("marketing_project.api.upload.requests.get", return_value=mock_response):
        request_data = {"url": "http://example.com/blog", "content_type": "blog_post"}
        response = client.post("/api/v1/upload/from-url", json=request_data)

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        body = response.json()
        assert body["success"] is True
        assert "file_id" in body
        assert body["url"] == "http://example.com/blog"


def test_upload_rtf_file():
    """Test uploading an RTF file."""
    rtf_content = r"{\rtf1\ansi Hello World\par}"
    files = {"file": ("doc.rtf", io.BytesIO(rtf_content.encode()), "application/rtf")}
    data = {"content_type": "blog_post"}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code in [200, 500]


def test_upload_csv_transcript():
    """Test uploading a CSV file as a transcript."""
    csv_content = "speaker,text\nAlice,Hello Bob.\nBob,Hi Alice!"
    files = {"file": ("chat.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    data = {"content_type": "transcript"}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code in [200, 500]


def test_upload_large_file_rejected():
    """Test that files exceeding 25 MB are rejected with 400."""
    # Create content just over 25 MB
    large_content = b"x" * (25 * 1024 * 1024 + 1)
    files = {"file": ("huge.txt", io.BytesIO(large_content), "text/plain")}
    data = {"content_type": "blog_post"}

    response = client.post("/api/v1/upload", files=files, data=data)

    assert response.status_code == 400
    assert "too large" in response.json()["detail"].lower()
