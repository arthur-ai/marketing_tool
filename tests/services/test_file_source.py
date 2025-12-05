"""
Tests for file content source service.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from marketing_project.core.content_sources import FileSourceConfig
from marketing_project.services.file_source import FileContentSource


@pytest.fixture
def temp_file():
    """Create a temporary file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"id": "test", "title": "Test", "content": "Content"}')
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def file_source_config(temp_file):
    """Create file source config."""
    return FileSourceConfig(
        name="test-file-source",
        file_paths=[temp_file],
        file_patterns=[],
    )


@pytest.fixture
def file_content_source(file_source_config):
    """Create FileContentSource instance."""
    return FileContentSource(file_source_config)


@pytest.mark.asyncio
async def test_initialize_success(file_content_source, temp_file):
    """Test successful initialization."""
    result = await file_content_source.initialize()

    assert result is True


@pytest.mark.asyncio
async def test_fetch_content(file_content_source, temp_file):
    """Test fetching content from files."""
    await file_content_source.initialize()
    result = await file_content_source.fetch_content()

    assert result.success is True
    assert len(result.content_items) > 0


@pytest.mark.asyncio
async def test_fetch_content_with_limit(file_content_source, temp_file):
    """Test fetching content with limit."""
    await file_content_source.initialize()
    result = await file_content_source.fetch_content(limit=1)

    assert result.success is True
    assert len(result.content_items) <= 1
