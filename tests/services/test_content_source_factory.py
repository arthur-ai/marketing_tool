"""
Tests for content source factory and manager.

This module tests the content source factory, manager, and integration
with the marketing pipeline.
"""

import asyncio
import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from marketing_project.core.content_sources import (
    APISourceConfig,
    ContentSourceType,
    FileSourceConfig,
)
from marketing_project.core.models import BlogPostContext, TranscriptContext
from marketing_project.services.content_source_config_loader import (
    ContentSourceConfigLoader,
)
from marketing_project.services.content_source_factory import (
    ContentSourceFactory,
    ContentSourceManager,
)


class TestContentSourceFactory:
    """Test the ContentSourceFactory class."""

    def test_create_file_source(self):
        """Test creating a file source."""
        config = FileSourceConfig(
            name="test_file",
            source_type=ContentSourceType.FILE,
            file_paths=["test/"],
            watch_directory=False,
        )

        source = ContentSourceFactory.create_source(config)
        assert source is not None
        assert source.config.name == "test_file"

    def test_create_api_source(self):
        """Test creating an API source."""
        config = APISourceConfig(
            name="test_api",
            source_type=ContentSourceType.API,
            base_url="https://api.example.com",
            endpoints=["/content"],
        )

        source = ContentSourceFactory.create_source(config)
        assert source is not None
        assert source.config.name == "test_api"

    def test_create_uploaded_file_source(self):
        """Test creating an uploaded file source."""
        source = ContentSourceFactory.create_uploaded_file_source(
            "test_uploads", "uploads/"
        )
        assert source is not None
        assert source.config.name == "test_uploads"


class TestContentSourceManager:
    """Test the ContentSourceManager class."""

    @pytest.fixture
    async def manager(self):
        """Create a content source manager for testing."""
        manager = ContentSourceManager()
        yield manager
        await manager.cleanup()

    @pytest.fixture
    def sample_blog_post(self):
        """Create a sample blog post for testing."""
        return {
            "id": "test_blog_001",
            "title": "Test Blog Post",
            "content": "This is a test blog post content.",
            "snippet": "Test snippet",
            "author": "Test Author",
            "tags": ["test", "blog"],
            "category": "Testing",
            "created_at": datetime.now().isoformat(),
            "source_url": "https://example.com/test-blog",
        }

    @pytest.fixture
    def sample_transcript(self):
        """Create a sample transcript for testing."""
        return {
            "id": "test_transcript_001",
            "title": "Test Podcast Episode",
            "content": "This is a test transcript content.",
            "snippet": "Test transcript snippet",
            "speakers": ["Host", "Guest"],
            "duration": "30:00",
            "transcript_type": "podcast",
            "created_at": datetime.now().isoformat(),
            "source_url": "https://example.com/test-podcast",
        }

    async def test_add_source(self, manager, tmp_path):
        """Test adding a content source."""
        # Create a temporary directory with some test files
        test_dir = tmp_path / "test_content"
        test_dir.mkdir()
        (test_dir / "test_file.json").write_text(
            '{"title": "Test", "content": "Test content"}'
        )

        config = FileSourceConfig(
            name="test_source",
            source_type=ContentSourceType.FILE,
            file_paths=[str(test_dir)],
            file_patterns=[str(test_dir / "*.json")],  # Add pattern to find the file
        )

        success = await manager.add_source_from_config(config)
        assert success is True
        assert "test_source" in manager.sources

    async def test_fetch_content_as_models(
        self, manager, sample_blog_post, sample_transcript
    ):
        """Test fetching content as models."""
        # Mock a content source that returns sample data
        mock_source = Mock()
        mock_source.status = "active"
        mock_source.error_count = 0
        mock_source.fetch_content = AsyncMock(
            return_value=Mock(
                success=True, content_items=[sample_blog_post, sample_transcript]
            )
        )
        mock_source.cleanup = AsyncMock(return_value=None)

        manager.sources["test_source"] = mock_source

        # Test fetching content as models
        content_models = await manager.fetch_content_as_models()

        assert len(content_models) == 2
        assert any(isinstance(model, BlogPostContext) for model in content_models)
        assert any(isinstance(model, TranscriptContext) for model in content_models)

    async def test_search_content_models(self, manager, sample_blog_post):
        """Test searching content models."""
        # Mock a content source
        mock_source = Mock()
        mock_source.status = "active"
        mock_source.error_count = 0
        mock_source.fetch_content = AsyncMock(
            return_value=Mock(success=True, content_items=[sample_blog_post])
        )
        mock_source.cleanup = AsyncMock(return_value=None)

        manager.sources["test_source"] = mock_source

        # Test searching
        results = await manager.search_content_models("test")
        assert len(results) == 1
        assert results[0].title == "Test Blog Post"

    async def test_get_content_models_by_type(
        self, manager, sample_blog_post, sample_transcript
    ):
        """Test getting content models by type."""
        # Mock a content source
        mock_source = Mock()
        mock_source.status = "active"
        mock_source.error_count = 0
        mock_source.fetch_content = AsyncMock(
            return_value=Mock(
                success=True, content_items=[sample_blog_post, sample_transcript]
            )
        )
        mock_source.cleanup = AsyncMock(return_value=None)

        manager.sources["test_source"] = mock_source

        # Test getting blog posts
        blog_posts = await manager.get_content_models_by_type("blog_post")
        assert len(blog_posts) == 1
        assert isinstance(blog_posts[0], BlogPostContext)

        # Test getting transcripts
        transcripts = await manager.get_content_models_by_type("transcript")
        assert len(transcripts) == 1
        assert isinstance(transcripts[0], TranscriptContext)

    async def test_health_check_all(self, manager):
        """Test health checking all sources."""
        # Mock sources
        healthy_source = Mock()
        healthy_source.health_check = AsyncMock(return_value=True)
        healthy_source.cleanup = AsyncMock(return_value=None)

        unhealthy_source = Mock()
        unhealthy_source.health_check = AsyncMock(return_value=False)
        unhealthy_source.cleanup = AsyncMock(return_value=None)

        manager.sources = {"healthy": healthy_source, "unhealthy": unhealthy_source}

        health_status = await manager.health_check_all()

        assert health_status["healthy"] is True
        assert health_status["unhealthy"] is False

    async def test_get_source_statistics(self, manager):
        """Test getting source statistics."""
        from marketing_project.core.content_sources import ContentSourceStatus

        # Mock sources
        mock_source = Mock()
        mock_source.status = ContentSourceStatus.ACTIVE
        mock_source.get_status.return_value = {
            "status": "active",
            "type": "file",
            "enabled": True,
        }
        mock_source.fetch_content = AsyncMock(
            return_value=Mock(success=True, total_count=5)
        )
        mock_source.cleanup = AsyncMock(return_value=None)

        manager.sources = {"test_source": mock_source}

        stats = await manager.get_source_statistics()

        assert stats["total_sources"] == 1
        assert stats["active_sources"] == 1
        assert stats["total_content_items"] == 5


class TestContentSourceConfigLoader:
    """Test the ContentSourceConfigLoader class."""

    def test_load_from_environment(self):
        """Test loading configurations from environment variables."""
        with patch.dict(
            os.environ,
            {
                "CONTENT_DIR": "test_content/",
                "CONTENT_API_URL": "https://api.test.com",
                "CONTENT_API_KEY": "test_key",
            },
        ):
            loader = ContentSourceConfigLoader()
            configs = loader._load_from_environment()

            assert len(configs) >= 2  # At least file and API sources
            assert any(config["type"] == "file" for config in configs)
            assert any(config["type"] == "api" for config in configs)

    def test_substitute_env_vars(self):
        """Test environment variable substitution."""
        loader = ContentSourceConfigLoader()

        config = {
            "base_url": "${CONTENT_API_URL}",
            "nested": {"key": "${CONTENT_API_KEY}"},
            "list": ["${CONTENT_DIR}", "static_value"],
        }

        with patch.dict(
            os.environ,
            {
                "CONTENT_API_URL": "https://api.test.com",
                "CONTENT_API_KEY": "test_key",
                "CONTENT_DIR": "test_content/",
            },
        ):
            result = loader._substitute_env_vars(config)

            assert result["base_url"] == "https://api.test.com"
            assert result["nested"]["key"] == "test_key"
            assert result["list"][0] == "test_content/"
            assert result["list"][1] == "static_value"

    def test_create_source_configs(self):
        """Test creating source configuration objects."""
        with patch.dict(os.environ, {"CONTENT_DIR": "test_content/"}):
            loader = ContentSourceConfigLoader()
            source_configs = loader.create_source_configs()

            assert len(source_configs) >= 1
            assert any(
                isinstance(config, FileSourceConfig) for config in source_configs
            )


class TestContentSourceIntegration:
    """Test content source integration with the marketing pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_integration(self, tmp_path):
        """Test integration with the marketing pipeline."""
        # Create test content files
        content_dir = tmp_path / "content"
        content_dir.mkdir()

        blog_file = content_dir / "test_blog.json"
        blog_file.write_text(
            """{
            "id": "test_blog_001",
            "title": "Test Blog Post",
            "content": "This is a test blog post for pipeline integration.",
            "snippet": "Test snippet",
            "author": "Test Author",
            "tags": ["test", "integration"],
            "category": "Testing"
        }"""
        )

        # Test the pipeline integration
        from marketing_project.services.content_source_config_loader import (
            ContentSourceConfigLoader,
        )
        from marketing_project.services.content_source_factory import (
            ContentSourceManager,
        )

        manager = ContentSourceManager()

        # Create a file source for the test content
        config = FileSourceConfig(
            name="test_pipeline_source",
            source_type=ContentSourceType.FILE,
            file_paths=[str(content_dir)],
            file_patterns=[str(content_dir / "*.json")],
        )

        await manager.add_source_from_config(config)

        # Fetch content as models
        content_models = await manager.fetch_content_as_models()

        assert len(content_models) == 1
        assert isinstance(content_models[0], BlogPostContext)
        assert content_models[0].title == "Test Blog Post"

        await manager.cleanup()


@pytest.mark.asyncio
async def test_content_source_error_handling():
    """Test error handling in content sources."""
    manager = ContentSourceManager()

    # Test with invalid configuration
    invalid_config = FileSourceConfig(
        name="invalid_source",
        source_type=ContentSourceType.FILE,
        file_paths=["/nonexistent/path/"],
    )

    success = await manager.add_source_from_config(invalid_config)
    assert success is False

    # Test fetching from invalid source
    results = await manager.fetch_all_content()
    assert len(results) == 0

    await manager.cleanup()
