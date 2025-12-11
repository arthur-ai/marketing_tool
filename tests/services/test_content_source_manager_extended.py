"""
Extended tests for content source manager methods.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from marketing_project.core.content_sources import (
    ContentSourceStatus,
    ContentSourceType,
    FileSourceConfig,
)
from marketing_project.services.content_source_factory import ContentSourceManager


class TestContentSource:
    """Test content source implementation."""

    def __init__(self, config):
        self.config = config
        self.status = ContentSourceStatus.ACTIVE
        self.error_count = 0
        self.last_run = None

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

    async def cleanup(self) -> None:
        pass

    def get_status(self) -> dict:
        return {"name": self.config.name, "status": self.status.value}


@pytest.fixture
def content_source_manager():
    """Create a ContentSourceManager instance."""
    return ContentSourceManager()


@pytest.mark.asyncio
async def test_start_polling(content_source_manager):
    """Test start_polling method."""
    source = TestContentSource(
        FileSourceConfig(
            name="test",
            source_type=ContentSourceType.FILE,
            file_paths=[],
            polling_interval=60,
        )
    )
    await content_source_manager.add_source(source)

    await content_source_manager.start_polling()

    assert content_source_manager.running is True


@pytest.mark.asyncio
async def test_stop_polling(content_source_manager):
    """Test stop_polling method."""
    source = TestContentSource(
        FileSourceConfig(
            name="test",
            source_type=ContentSourceType.FILE,
            file_paths=[],
            polling_interval=60,
        )
    )
    await content_source_manager.add_source(source)
    await content_source_manager.start_polling()

    await content_source_manager.stop_polling()

    assert content_source_manager.running is False


@pytest.mark.asyncio
async def test_fetch_content_as_models(content_source_manager):
    """Test fetch_content_as_models method."""
    source = TestContentSource(
        FileSourceConfig(name="test", source_type=ContentSourceType.FILE, file_paths=[])
    )
    await content_source_manager.add_source(source)

    models = await content_source_manager.fetch_content_as_models(limit_per_source=10)

    assert isinstance(models, list)


@pytest.mark.asyncio
async def test_get_content_models_by_type(content_source_manager):
    """Test get_content_models_by_type method."""
    source = TestContentSource(
        FileSourceConfig(name="test", source_type=ContentSourceType.FILE, file_paths=[])
    )
    await content_source_manager.add_source(source)

    models = await content_source_manager.get_content_models_by_type(
        "blog_post", limit_per_source=10
    )

    assert isinstance(models, list)


@pytest.mark.asyncio
async def test_search_content_models(content_source_manager):
    """Test search_content_models method."""
    source = TestContentSource(
        FileSourceConfig(name="test", source_type=ContentSourceType.FILE, file_paths=[])
    )
    await content_source_manager.add_source(source)

    results = await content_source_manager.search_content_models("test query", limit=10)

    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_restart_failed_sources(content_source_manager):
    """Test restart_failed_sources method."""
    source = TestContentSource(
        FileSourceConfig(name="test", source_type=ContentSourceType.FILE, file_paths=[])
    )
    source.status = ContentSourceStatus.ERROR
    await content_source_manager.add_source(source)

    restarted = await content_source_manager.restart_failed_sources()

    assert isinstance(restarted, dict)


def test_clear_cache(content_source_manager):
    """Test clear_cache method."""
    content_source_manager.clear_cache()

    # Should not raise exception
    assert True


def test_set_cache_ttl(content_source_manager):
    """Test set_cache_ttl method."""
    content_source_manager.set_cache_ttl(3600)

    # Should not raise exception
    assert True
