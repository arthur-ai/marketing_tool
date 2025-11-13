"""
Tests for Upload API endpoints.
"""

import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from marketing_project.api.upload import router


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


@pytest.fixture
def sample_json_file():
    """Sample JSON file content."""
    return json.dumps(
        {
            "id": "test-123",
            "title": "Test Blog Post",
            "content": "This is test content",
            "author": "Test Author",
        }
    )


@pytest.fixture
def sample_text_file():
    """Sample text file content."""
    return "This is a test text file content."


@pytest.mark.asyncio
class TestUploadAPI:
    """Test suite for Upload API endpoints."""

    async def test_upload_json_file_success(self, client, sample_json_file):
        """Test successful JSON file upload."""
        with patch(
            "marketing_project.api.upload.process_uploaded_file"
        ) as mock_process:
            mock_process.return_value = "content/blog_posts/test_file.json"

            files = {
                "file": (
                    "test.json",
                    io.BytesIO(sample_json_file.encode()),
                    "application/json",
                )
            }
            data = {"content_type": "blog_post"}

            response = client.post("/api/v1/upload", files=files, data=data)
            assert response.status_code == 200
            data_response = response.json()
            assert data_response["success"] is True
            assert "file_id" in data_response

    async def test_upload_text_file_success(self, client, sample_text_file):
        """Test successful text file upload."""
        with patch(
            "marketing_project.api.upload.process_uploaded_file"
        ) as mock_process:
            mock_process.return_value = "content/blog_posts/test_file.json"

            files = {
                "file": (
                    "test.txt",
                    io.BytesIO(sample_text_file.encode()),
                    "text/plain",
                )
            }
            data = {"content_type": "blog_post"}

            response = client.post("/api/v1/upload", files=files, data=data)
            assert response.status_code == 200

    async def test_upload_unsupported_file_type(self, client):
        """Test upload with unsupported file type."""
        files = {
            "file": (
                "test.exe",
                io.BytesIO(b"binary content"),
                "application/x-msdownload",
            )
        }
        data = {"content_type": "blog_post"}

        response = client.post("/api/v1/upload", files=files, data=data)
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    async def test_upload_file_too_large(self, client):
        """Test upload with file too large."""
        # Create a file larger than 25MB
        large_content = b"x" * (26 * 1024 * 1024)  # 26MB
        files = {"file": ("large.txt", io.BytesIO(large_content), "text/plain")}
        data = {"content_type": "blog_post"}

        response = client.post("/api/v1/upload", files=files, data=data)
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()

    async def test_upload_from_url_success(self, client):
        """Test successful URL extraction."""
        with patch(
            "marketing_project.api.upload.extract_blog_content_from_url"
        ) as mock_extract:
            mock_extract.return_value = {
                "id": "extracted-123",
                "title": "Extracted Blog Post",
                "content": "This is extracted content from URL"
                * 10,  # Make it long enough
                "author": "Extracted Author",
                "word_count": 100,
            }

            request_data = {
                "url": "https://example.com/blog-post",
                "content_type": "blog_post",
            }

            response = client.post("/api/v1/upload/from-url", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "file_id" in data

    async def test_upload_from_url_insufficient_content(self, client):
        """Test URL extraction with insufficient content."""
        with patch(
            "marketing_project.api.upload.extract_blog_content_from_url"
        ) as mock_extract:
            mock_extract.return_value = {
                "id": "extracted-123",
                "title": "Short Post",
                "content": "Short",  # Too short
                "word_count": 1,
            }

            request_data = {
                "url": "https://example.com/blog-post",
                "content_type": "blog_post",
            }

            response = client.post("/api/v1/upload/from-url", json=request_data)
            assert response.status_code == 400
            assert "sufficient content" in response.json()["detail"].lower()

    async def test_upload_from_url_fetch_error(self, client):
        """Test URL extraction with fetch error."""
        with patch(
            "marketing_project.api.upload.extract_blog_content_from_url"
        ) as mock_extract:
            import requests

            mock_extract.side_effect = requests.exceptions.RequestException(
                "Connection error"
            )

            request_data = {
                "url": "https://example.com/blog-post",
                "content_type": "blog_post",
            }

            response = client.post("/api/v1/upload/from-url", json=request_data)
            # Connection errors return 500 (internal server error) per the implementation
            assert response.status_code == 500

    async def test_get_upload_status(self, client):
        """Test getting upload status."""
        response = client.get("/api/v1/upload/status/test-file-id")
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "status" in data

    async def test_extract_text_from_docx(self, client):
        """Test DOCX text extraction."""
        # This would require python-docx to be installed
        # For now, we'll test the error case
        with patch("marketing_project.api.upload.DOCX_AVAILABLE", False):
            import pytest

            from marketing_project.api.upload import extract_text_from_docx

            with pytest.raises(Exception):
                extract_text_from_docx("test.docx")

    async def test_extract_text_from_pdf(self, client):
        """Test PDF text extraction."""
        # This would require PyPDF2 to be installed
        # For now, we'll test the error case
        with patch("marketing_project.api.upload.PDF_AVAILABLE", False):
            import pytest

            from marketing_project.api.upload import extract_text_from_pdf

            with pytest.raises(Exception):
                extract_text_from_pdf("test.pdf")

    async def test_extract_text_from_csv(self, client):
        """Test CSV text extraction."""
        from marketing_project.api.upload import extract_text_from_csv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2,col3\n")
            f.write("val1,val2,val3\n")
            f.write("val4,val5,val6\n")
            temp_path = f.name

        try:
            result = extract_text_from_csv(temp_path)
            assert "Headers:" in result
            assert "Row 1:" in result
        finally:
            os.unlink(temp_path)

    async def test_extract_text_from_rtf(self, client):
        """Test RTF text extraction."""
        from marketing_project.api.upload import extract_text_from_rtf

        with tempfile.NamedTemporaryFile(mode="w", suffix=".rtf", delete=False) as f:
            f.write("{\\rtf1\\ansi This is RTF content}")
            temp_path = f.name

        try:
            result = extract_text_from_rtf(temp_path)
            assert "This is RTF content" in result
        finally:
            os.unlink(temp_path)
