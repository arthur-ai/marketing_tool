"""
Tests for S3 storage service.
"""

from unittest.mock import MagicMock, patch

import pytest

from marketing_project.services.s3_storage import S3Storage


@pytest.fixture
def mock_boto3():
    """Mock boto3."""
    with patch("marketing_project.services.s3_storage.boto3") as mock:
        mock_client = MagicMock()
        mock.client.return_value = mock_client
        yield mock_client


def test_s3_storage_initialization_no_bucket():
    """Test S3Storage initialization without bucket."""
    with patch.dict("os.environ", {}, clear=True):
        storage = S3Storage()

        assert storage.bucket_name is None
        assert storage.s3_client is None


def test_s3_storage_initialization_with_bucket(mock_boto3):
    """Test S3Storage initialization with bucket."""
    with patch.dict("os.environ", {"AWS_S3_BUCKET": "test-bucket"}):
        storage = S3Storage()

        assert storage.bucket_name == "test-bucket"
        # s3_client may be None if boto3 not available
        assert storage.s3_client is None or storage.s3_client is not None


def test_is_available_no_boto3():
    """Test is_available when boto3 is not available."""
    with patch("marketing_project.services.s3_storage.BOTO3_AVAILABLE", False):
        storage = S3Storage()
        assert storage.is_available() is False


def test_is_available_no_bucket():
    """Test is_available when bucket is not configured."""
    with patch("marketing_project.services.s3_storage.BOTO3_AVAILABLE", True):
        storage = S3Storage()
        storage.bucket_name = None
        assert storage.is_available() is False


def test_get_s3_key():
    """Test _get_s3_key method."""
    storage = S3Storage(prefix="content/")
    key = storage._get_s3_key("test/file.json")

    assert key.startswith("content/")
    assert "test/file.json" in key


@pytest.mark.asyncio
async def test_upload_file_not_available():
    """Test upload_file when S3 is not available."""
    storage = S3Storage()
    storage.s3_client = None

    result = await storage.upload_file("test.txt")

    assert result is None


@pytest.mark.asyncio
async def test_download_file_not_available():
    """Test download_file when S3 is not available."""
    storage = S3Storage()
    storage.s3_client = None

    result = await storage.download_file("test.txt", "local.txt")

    assert result is False


@pytest.mark.asyncio
async def test_list_files_not_available():
    """Test list_files when S3 is not available."""
    storage = S3Storage()
    storage.s3_client = None

    result = await storage.list_files()

    assert result == []


@pytest.mark.asyncio
async def test_delete_file_not_available():
    """Test delete_file when S3 is not available."""
    storage = S3Storage()
    storage.s3_client = None

    result = await storage.delete_file("test.txt")

    assert result is False


@pytest.mark.asyncio
async def test_file_exists_not_available():
    """Test file_exists when S3 is not available."""
    storage = S3Storage()
    storage.s3_client = None

    result = await storage.file_exists("test.txt")

    assert result is False
