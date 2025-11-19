"""
Content source factory for Marketing Project.

This module provides a factory for creating content sources based on configuration
and a manager for orchestrating multiple content sources.

Classes:
    ContentSourceFactory: Creates content sources from configuration
    ContentSourceManager: Manages multiple content sources
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from marketing_project.core.content_sources import ContentSource
from marketing_project.core.content_sources import (
    ContentSourceManager as BaseContentSourceManager,
)
from marketing_project.core.content_sources import (
    ContentSourceResult,
    ContentSourceStatus,
    ContentSourceType,
    DatabaseSourceConfig,
    FileSourceConfig,
    S3SourceConfig,
    SourceConfig,
)
from marketing_project.core.models import ContentContext
from marketing_project.core.utils import convert_dict_to_content_context
from marketing_project.models.content_models import (
    BlogPostContext,
    ReleaseNotesContext,
    TranscriptContext,
)
from marketing_project.services.api_source import (
    APIContentSource,
    RSSContentSource,
    WebhookContentSource,
)
from marketing_project.services.database_source import (
    MongoDBContentSource,
    RedisContentSource,
    SQLContentSource,
)
from marketing_project.services.file_source import (
    DirectoryWatcherSource,
    FileContentSource,
    UploadedFileSource,
)
from marketing_project.services.s3_source import S3ContentSource
from marketing_project.services.web_scraping_source import (
    BeautifulSoupScrapingSource,
    SeleniumScrapingSource,
    WebScrapingContentSource,
)

logger = logging.getLogger("marketing_project.services.content_source_factory")


class ContentSourceFactory:
    """Factory for creating content sources from configuration."""

    @staticmethod
    def create_source(config: SourceConfig) -> Optional[ContentSource]:
        """Create a content source from configuration."""
        try:
            # Validate configuration
            if not config.name:
                logger.error("Content source name is required")
                return None

            if not config.source_type:
                logger.error("Content source type is required")
                return None
            if config.source_type == ContentSourceType.FILE:
                if config.watch_directory:
                    return DirectoryWatcherSource(config)
                else:
                    return FileContentSource(config)

            elif config.source_type == ContentSourceType.API:
                return APIContentSource(config)

            elif config.source_type == ContentSourceType.WEBHOOK:
                return WebhookContentSource(config)

            elif config.source_type == ContentSourceType.RSS:
                return RSSContentSource(config)

            elif config.source_type == ContentSourceType.DATABASE:
                # Ensure we have a DatabaseSourceConfig
                if not isinstance(config, DatabaseSourceConfig):
                    logger.error(
                        f"Expected DatabaseSourceConfig for database source, got {type(config)}"
                    )
                    return None

                # Validate database configuration
                if not config.connection_string:
                    logger.error("Database connection string is required")
                    return None

                # Determine database type from connection string
                connection_string = config.connection_string.lower()
                if "mongodb" in connection_string:
                    return MongoDBContentSource(config)
                elif "redis" in connection_string:
                    return RedisContentSource(config)
                else:
                    return SQLContentSource(config)

            elif config.source_type == ContentSourceType.WEB_SCRAPING:
                # Choose scraping method based on configuration
                if config.metadata.get("use_selenium", False):
                    return SeleniumScrapingSource(config)
                else:
                    return BeautifulSoupScrapingSource(config)

            elif config.source_type == ContentSourceType.S3:
                # Ensure we have an S3SourceConfig
                if not isinstance(config, S3SourceConfig):
                    logger.error(
                        f"Expected S3SourceConfig for S3 source, got {type(config)}"
                    )
                    return None
                return S3ContentSource(config)

            else:
                logger.error(f"Unsupported content source type: {config.source_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to create content source {config.name}: {e}")
            return None

    @staticmethod
    def create_uploaded_file_source(
        name: str, upload_directory: str = "uploads"
    ) -> UploadedFileSource:
        """Create a special uploaded file source."""
        config = FileSourceConfig(
            name=name,
            source_type=ContentSourceType.FILE,
            file_paths=[upload_directory],
            supported_formats=[".txt", ".md", ".json", ".yaml", ".yml", ".html"],
            watch_directory=True,
        )

        return UploadedFileSource(config, upload_directory)


class ContentSourceManager(BaseContentSourceManager):
    """Enhanced content source manager with additional features."""

    def __init__(self):
        super().__init__()
        self.factory = ContentSourceFactory()
        self.content_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_ttl: int = 300  # 5 minutes
        self.last_cache_update: Dict[str, datetime] = {}

    async def add_source_from_config(self, config: SourceConfig) -> bool:
        """Add a content source from configuration."""
        logger.info(f"Creating source '{config.name}' from configuration...")
        source = self.factory.create_source(config)
        if source:
            logger.info(f"  Source object created, initializing...")
            result = await self.add_source(source)
            if result:
                logger.info(f"  ✓ Source '{config.name}' added successfully")
            else:
                logger.warning(f"  ✗ Failed to add source '{config.name}'")
            return result
        else:
            logger.error(f"  ✗ Failed to create source object for '{config.name}'")
        return False

    async def add_multiple_sources(
        self, configs: List[SourceConfig]
    ) -> Dict[str, bool]:
        """Add multiple content sources from configurations."""
        results = {}
        for config in configs:
            results[config.name] = await self.add_source_from_config(config)
        return results

    async def fetch_content_with_cache(
        self, limit_per_source: Optional[int] = None, use_cache: bool = True
    ) -> List[ContentSourceResult]:
        """Fetch content with optional caching."""
        if not use_cache:
            return await self.fetch_all_content(limit_per_source)

        # Check cache validity
        now = datetime.now()
        cache_valid = True

        for source_name, last_update in self.last_cache_update.items():
            if (now - last_update).seconds > self.cache_ttl:
                cache_valid = False
                break

        if cache_valid and self.content_cache:
            # Return cached content
            results = []
            for source_name, cached_items in self.content_cache.items():
                results.append(
                    ContentSourceResult(
                        source_name=source_name,
                        content_items=cached_items,
                        total_count=len(cached_items),
                        success=True,
                        metadata={"from_cache": True},
                    )
                )
            return results

        # Fetch fresh content
        results = await self.fetch_all_content(limit_per_source)

        # Update cache
        self.content_cache.clear()
        self.last_cache_update.clear()

        for result in results:
            if result.success:
                self.content_cache[result.source_name] = result.content_items
                self.last_cache_update[result.source_name] = now

        return results

    async def fetch_content_as_models(
        self, limit_per_source: Optional[int] = None
    ) -> List[ContentContext]:
        """Fetch content and convert to ContentContext models."""
        results = await self.fetch_all_content(limit_per_source)

        content_models = []
        for result in results:
            if result.success and result.content_items:
                for content_item in result.content_items:
                    try:
                        # Convert dictionary to ContentContext model
                        content_model = convert_dict_to_content_context(content_item)
                        content_models.append(content_model)
                    except Exception as e:
                        logger.warning(f"Failed to convert content item to model: {e}")
                        continue

        return content_models

    async def get_content_by_type(
        self, content_type: str, limit_per_source: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get content filtered by type."""
        results = await self.fetch_all_content(limit_per_source)

        filtered_content = []
        for result in results:
            if result.success:
                for item in result.content_items:
                    # Check if content matches the requested type
                    if self._content_matches_type(item, content_type):
                        filtered_content.append(item)

        return filtered_content

    async def get_content_models_by_type(
        self, content_type: str, limit_per_source: Optional[int] = None
    ) -> List[ContentContext]:
        """Get content models filtered by type."""
        content_models = await self.fetch_content_as_models(limit_per_source)

        filtered_models = []
        for content_model in content_models:
            # Check if content model matches the requested type
            if self._model_matches_type(content_model, content_type):
                filtered_models.append(content_model)

        return filtered_models

    def _model_matches_type(
        self, content_model: ContentContext, content_type: str
    ) -> bool:
        """Check if content model matches the specified type."""
        model_type = content_model.__class__.__name__.lower()

        if content_type.lower() == "transcript":
            return "transcript" in model_type
        elif content_type.lower() == "blog_post":
            return "blog" in model_type
        elif content_type.lower() == "release_notes":
            return "release" in model_type
        elif content_type.lower() == "email":
            return "email" in model_type

        return False

    def _content_matches_type(
        self, content_item: Dict[str, Any], content_type: str
    ) -> bool:
        """Check if content item matches the specified type."""
        # Check metadata for content type
        metadata = content_item.get("metadata", {})
        item_type = metadata.get("content_type", "").lower()

        if content_type.lower() in item_type:
            return True

        # Check for type-specific fields
        if content_type.lower() == "transcript":
            return "speakers" in content_item or "transcript_type" in content_item
        elif content_type.lower() == "blog_post":
            return "author" in content_item or "tags" in content_item
        elif content_type.lower() == "release_notes":
            return "version" in content_item or "changes" in content_item

        return False

    async def search_content(
        self, query: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search content across all sources."""
        results = await self.fetch_all_content()

        matching_content = []
        query_lower = query.lower()

        for result in results:
            if result.success:
                for item in result.content_items:
                    # Search in title, content, and snippet
                    searchable_text = f"{item.get('title', '')} {item.get('content', '')} {item.get('snippet', '')}"
                    if query_lower in searchable_text.lower():
                        matching_content.append(item)

                        if limit and len(matching_content) >= limit:
                            break

        return matching_content

    async def search_content_models(
        self, query: str, limit: Optional[int] = None
    ) -> List[ContentContext]:
        """Search content models across all sources."""
        content_models = await self.fetch_content_as_models()

        matching_models = []
        query_lower = query.lower()

        for content_model in content_models:
            # Search in title, content, and snippet
            searchable_text = (
                f"{content_model.title} {content_model.content} {content_model.snippet}"
            )
            if query_lower in searchable_text.lower():
                matching_models.append(content_model)

                if limit and len(matching_models) >= limit:
                    break

        return matching_models

    async def get_source_statistics(self) -> Dict[str, Any]:
        """Get statistics about all sources."""
        stats = {
            "total_sources": len(self.sources),
            "active_sources": 0,
            "error_sources": 0,
            "total_content_items": 0,
            "sources": {},
        }

        for name, source in self.sources.items():
            source_stats = source.get_status()
            stats["sources"][name] = source_stats

            if source_stats["status"] == "active":
                stats["active_sources"] += 1
            elif source_stats["status"] == "error":
                stats["error_sources"] += 1

        # Get content counts
        results = await self.fetch_all_content()
        for result in results:
            if result.success:
                stats["total_content_items"] += result.total_count

        return stats

    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all sources."""
        health_status = {}

        for name, source in self.sources.items():
            try:
                health_status[name] = await source.health_check()
            except Exception as e:
                logger.warning(f"Health check failed for source {name}: {e}")
                health_status[name] = False

        return health_status

    async def restart_failed_sources(self) -> Dict[str, bool]:
        """Restart sources that are in error state."""
        restart_results = {}

        for name, source in self.sources.items():
            if source.status.value == "error":
                try:
                    # Remove and re-add the source
                    await self.remove_source(name)

                    # Recreate from original config (this would need to be stored)
                    # For now, just mark as attempted
                    restart_results[name] = False
                    logger.info(f"Attempted to restart failed source: {name}")

                except Exception as e:
                    logger.error(f"Failed to restart source {name}: {e}")
                    restart_results[name] = False

        return restart_results

    def clear_cache(self) -> None:
        """Clear the content cache."""
        self.content_cache.clear()
        self.last_cache_update.clear()

    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Set cache time-to-live in seconds."""
        self.cache_ttl = ttl_seconds

    async def list_sources(self) -> List[Dict[str, Any]]:
        """
        List all content sources with their status and item counts.

        Returns:
            List of dictionaries containing source information:
            - name: Source name
            - active: Whether the source is active
            - item_count: Number of content items from this source
            - type: Source type
            - status: Source status
        """
        sources_list = []

        # Get content from all sources to count items
        results = await self.fetch_all_content()

        # Create a mapping of source name to item count
        source_item_counts: Dict[str, int] = {}
        for result in results:
            if result.success:
                source_item_counts[result.source_name] = result.total_count

        # Build the list of source information
        for name, source in self.sources.items():
            status = source.get_status()
            # Compare against enum value for safety
            is_active = source.status == ContentSourceStatus.ACTIVE
            item_count = source_item_counts.get(name, 0)

            sources_list.append(
                {
                    "name": name,
                    "active": is_active,
                    "item_count": item_count,
                    "type": status.get("type", "unknown"),
                    "status": status.get("status", "unknown"),
                }
            )

        return sources_list

    async def cleanup(self) -> None:
        """Cleanup all resources."""
        self.clear_cache()
        await super().cleanup()


# Convenience function for creating a manager with common sources
async def create_default_content_manager() -> ContentSourceManager:
    """Create a content manager with some default sources."""
    manager = ContentSourceManager()

    # Add a file source for local content
    file_config = FileSourceConfig(
        name="local_files",
        source_type=ContentSourceType.FILE,
        file_paths=["content/"],
        file_patterns=["content/**/*.md", "content/**/*.json"],
        watch_directory=True,
    )

    await manager.add_source_from_config(file_config)

    return manager
