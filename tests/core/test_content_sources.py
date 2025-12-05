"""
Tests for content sources base classes.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from marketing_project.core.content_sources import (
    ContentSource,
    ContentSourceManager,
    ContentSourceStatus,
    ContentSourceType,
    FileSourceConfig,
)


class TestContentSource(ContentSource):
    """Test content source implementation."""

    async def initialize(self) -> bool:
        return True

    async def fetch_content(self, limit=None):
        from marketing_project.core.content_sources import ContentSourceResult

        return ContentSourceResult(
            source_name=self.config.name,
            content_items=[],
            total_count=0,
            success=True,
        )

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def content_source():
    """Create a test content source."""
    config = FileSourceConfig(
        name="test-source",
        source_type=ContentSourceType.FILE,
        file_paths=["test.txt"],
    )
    return TestContentSource(config)


@pytest.fixture
def content_source_manager():
    """Create a ContentSourceManager instance."""
    return ContentSourceManager()


@pytest.mark.asyncio
async def test_content_source_initialize(content_source):
    """Test content source initialization."""
    result = await content_source.initialize()

    assert result is True


@pytest.mark.asyncio
async def test_content_source_fetch_content(content_source):
    """Test content source fetch_content."""
    result = await content_source.fetch_content()

    assert result.success is True
    assert result.source_name == "test-source"


@pytest.mark.asyncio
async def test_content_source_health_check(content_source):
    """Test content source health_check."""
    result = await content_source.health_check()

    assert result is True


def test_content_source_get_status(content_source):
    """Test content source get_status."""
    status = content_source.get_status()

    assert isinstance(status, dict)
    assert "name" in status or "status" in status


@pytest.mark.asyncio
async def test_content_source_manager_add_source(content_source_manager):
    """Test ContentSourceManager.add_source method."""
    source = TestContentSource(
        FileSourceConfig(name="test", source_type=ContentSourceType.FILE, file_paths=[])
    )

    result = await content_source_manager.add_source(source)

    assert result is True


@pytest.mark.asyncio
async def test_content_source_manager_remove_source(content_source_manager):
    """Test ContentSourceManager.remove_source method."""
    source = TestContentSource(
        FileSourceConfig(name="test", source_type=ContentSourceType.FILE, file_paths=[])
    )
    await content_source_manager.add_source(source)

    result = await content_source_manager.remove_source("test")

    assert result is True


@pytest.mark.asyncio
async def test_content_source_manager_fetch_all_content(content_source_manager):
    """Test ContentSourceManager.fetch_all_content method."""
    source = TestContentSource(
        FileSourceConfig(name="test", source_type=ContentSourceType.FILE, file_paths=[])
    )
    await content_source_manager.add_source(source)

    result = await content_source_manager.fetch_all_content()

    assert result is not None
    assert isinstance(result, dict) or isinstance(result, list)


@pytest.mark.asyncio
async def test_content_source_manager_get_source_status(content_source_manager):
    """Test ContentSourceManager.get_source_status method."""
    source = TestContentSource(
        FileSourceConfig(name="test", source_type=ContentSourceType.FILE, file_paths=[])
    )
    await content_source_manager.add_source(source)

    status = await content_source_manager.get_source_status()

    assert isinstance(status, dict)


@pytest.mark.asyncio
async def test_content_source_manager_get_all_sources(content_source_manager):
    """Test ContentSourceManager.get_all_sources method."""
    source = TestContentSource(
        FileSourceConfig(name="test", source_type=ContentSourceType.FILE, file_paths=[])
    )
    await content_source_manager.add_source(source)

    sources = content_source_manager.get_all_sources()

    assert isinstance(sources, list)
    assert len(sources) >= 1


@pytest.mark.asyncio
async def test_content_source_manager_get_source(content_source_manager):
    """Test ContentSourceManager.get_source method."""
    source = TestContentSource(
        FileSourceConfig(name="test", source_type=ContentSourceType.FILE, file_paths=[])
    )
    await content_source_manager.add_source(source)

    retrieved = content_source_manager.get_source("test")

    assert retrieved is not None
    assert retrieved.config.name == "test"
