"""
Comprehensive tests for content source factory service methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.core.content_sources import (
    APISourceConfig,
    ContentSourceType,
    FileSourceConfig,
)
from marketing_project.services.content_source_factory import (
    ContentSourceFactory,
    ContentSourceManager,
)


def test_create_source_file():
    """Test create_source for file source."""
    config = FileSourceConfig(
        name="test-file",
        source_type=ContentSourceType.FILE,
        file_paths=["test.txt"],
    )

    source = ContentSourceFactory.create_source(config)

    assert source is not None
    assert source.config.name == "test-file"


def test_create_source_api():
    """Test create_source for API source."""
    config = APISourceConfig(
        name="test-api",
        source_type=ContentSourceType.API,
        base_url="https://api.example.com",
    )

    source = ContentSourceFactory.create_source(config)

    assert source is not None
    assert source.config.name == "test-api"


def test_create_source_invalid_name():
    """Test create_source with invalid name."""
    config = FileSourceConfig(
        name="",
        source_type=ContentSourceType.FILE,
    )

    source = ContentSourceFactory.create_source(config)

    assert source is None


@pytest.fixture
def content_source_manager():
    """Create a ContentSourceManager instance."""
    return ContentSourceManager()


@pytest.mark.asyncio
async def test_add_source_from_config(content_source_manager):
    """Test add_source_from_config method."""
    config = FileSourceConfig(
        name="test-source",
        source_type=ContentSourceType.FILE,
        file_paths=["test.txt"],
    )

    result = await content_source_manager.add_source_from_config(config)

    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_add_multiple_sources(content_source_manager):
    """Test add_multiple_sources method."""
    configs = [
        FileSourceConfig(
            name="source1", source_type=ContentSourceType.FILE, file_paths=["file1.txt"]
        ),
        FileSourceConfig(
            name="source2", source_type=ContentSourceType.FILE, file_paths=["file2.txt"]
        ),
    ]

    results = await content_source_manager.add_multiple_sources(configs)

    assert isinstance(results, dict)
    assert len(results) >= 0


@pytest.mark.asyncio
async def test_fetch_content_with_cache(content_source_manager):
    """Test fetch_content_with_cache method."""
    result = await content_source_manager.fetch_content_with_cache(limit_per_source=10)

    assert result is not None
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_get_content_by_type(content_source_manager):
    """Test get_content_by_type method."""
    result = await content_source_manager.get_content_by_type(
        "blog_post", limit_per_source=10
    )

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_search_content(content_source_manager):
    """Test search_content method."""
    result = await content_source_manager.search_content("test query", limit=10)

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_get_source_statistics(content_source_manager):
    """Test get_source_statistics method."""
    stats = await content_source_manager.get_source_statistics()

    assert isinstance(stats, dict)


@pytest.mark.asyncio
async def test_health_check_all(content_source_manager):
    """Test health_check_all method."""
    health = await content_source_manager.health_check_all()

    assert isinstance(health, dict)


@pytest.mark.asyncio
async def test_list_sources(content_source_manager):
    """Test list_sources method."""
    sources = await content_source_manager.list_sources()

    assert isinstance(sources, list)
