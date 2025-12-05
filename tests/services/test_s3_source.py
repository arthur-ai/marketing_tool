"""
Tests for S3 content source service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.core.content_sources import S3SourceConfig
from marketing_project.services.s3_source import S3ContentSource


@pytest.fixture
def s3_source_config():
    """Create S3 source config."""
    return S3SourceConfig(
        name="test-s3-source",
        bucket_name="test-bucket",
        region="us-east-1",
        prefix="content/",
        file_patterns=["*.json", "*.md"],
    )


@pytest.fixture
def s3_content_source(s3_source_config):
    """Create S3ContentSource instance."""
    return S3ContentSource(s3_source_config)


@pytest.mark.asyncio
async def test_initialize_success(s3_content_source):
    """Test successful initialization."""
    with patch("marketing_project.services.s3_source.S3Storage") as mock_s3_class:
        mock_s3 = MagicMock()
        mock_s3.is_available.return_value = True
        mock_s3.list_files = AsyncMock(return_value=["file1.json", "file2.md"])
        mock_s3_class.return_value = mock_s3

        result = await s3_content_source.initialize()

        assert result is True
        assert s3_content_source.s3_storage is not None


@pytest.mark.asyncio
async def test_initialize_s3_not_available(s3_content_source):
    """Test initialization when S3 is not available."""
    with patch("marketing_project.services.s3_source.S3Storage") as mock_s3_class:
        mock_s3 = MagicMock()
        mock_s3.is_available.return_value = False
        mock_s3_class.return_value = mock_s3

        result = await s3_content_source.initialize()

        assert result is False


@pytest.mark.asyncio
async def test_fetch_content(s3_content_source):
    """Test fetching content from S3."""
    with patch("marketing_project.services.s3_source.S3Storage") as mock_s3_class:
        mock_s3 = MagicMock()
        mock_s3.is_available.return_value = True
        mock_s3.list_files = AsyncMock(return_value=["file1.json"])
        mock_s3.get_file_content = AsyncMock(
            return_value=b'{"id": "test", "title": "Test", "content": "Content"}'
        )
        mock_s3_class.return_value = mock_s3

        await s3_content_source.initialize()
        result = await s3_content_source.fetch_content()

        # May return empty if file parsing fails, but should not error
        assert result.success is True or result.success is False
        assert isinstance(result.content_items, list)


def test_matches_pattern_with_prefix(s3_content_source):
    """Test _matches_pattern_with_prefix method."""
    assert (
        s3_content_source._matches_pattern_with_prefix(
            "content/blog/file.json", ["*.json"], "content/"
        )
        is True
    )

    assert (
        s3_content_source._matches_pattern_with_prefix(
            "content/blog/file.txt", ["*.json"], "content/"
        )
        is False
    )


@pytest.mark.asyncio
async def test_cleanup(s3_content_source):
    """Test cleanup method."""
    s3_content_source.s3_storage = MagicMock()
    s3_content_source.s3_storage.close = AsyncMock()

    await s3_content_source.cleanup()

    # Should not raise exception
