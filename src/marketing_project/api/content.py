"""
Content source management API endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from marketing_project.models import (
    ContentFetchResponse,
    ContentSourceListResponse,
    ContentSourceResponse,
)
from marketing_project.services.content_source_config_loader import (
    ContentSourceConfigLoader,
)
from marketing_project.services.content_source_factory import ContentSourceManager

logger = logging.getLogger("marketing_project.api.content")

# Create router
router = APIRouter()

# Content source manager - will be initialized in lifespan
_content_manager: Optional[ContentSourceManager] = None
config_loader = ContentSourceConfigLoader()


def get_source_display_name(source_name: str) -> str:
    """
    Get the display name for a content source.

    Maps internal source names to user-friendly display names.

    Args:
        source_name: Internal source name (e.g., "s3_content")

    Returns:
        Display name for the frontend (e.g., "S3 Database")
    """
    display_name_mapping = {
        "s3_content": "S3",
        "content_api": "API",
        "content_database": "Database",
        "web_content": "Web Scraping",
        "local_content": "Local Files",
        "rss_content": "RSS",
    }
    return display_name_mapping.get(source_name, source_name)


def get_content_manager() -> ContentSourceManager:
    """Get the initialized content manager instance."""
    if _content_manager is None:
        raise RuntimeError("Content manager not initialized. This should not happen!")
    return _content_manager


def set_content_manager(manager: ContentSourceManager):
    """Set the content manager instance (called during startup)."""
    global _content_manager
    _content_manager = manager


async def initialize_content_sources():
    """
    Initialize content sources from configuration.

    This should be called during application startup to load all configured
    content sources from the pipeline.yml configuration file.
    """
    try:
        logger.info("=" * 80)
        logger.info("INITIALIZING CONTENT SOURCES")
        logger.info("=" * 80)

        # Create a NEW content manager instance for this application lifecycle
        manager = ContentSourceManager()
        set_content_manager(manager)
        logger.info(f"Created ContentManager instance ID: {id(manager)}")

        # Load source configurations from pipeline.yml
        source_configs = config_loader.create_source_configs()

        if not source_configs:
            logger.warning("⚠️  No content sources found in configuration")
            logger.warning(
                "   Check your pipeline.yml file and ensure content_sources.enabled=true"
            )
            return

        logger.info(
            f"Loaded {len(source_configs)} source configurations from pipeline.yml"
        )

        # Add all sources to the manager
        logger.info("Adding sources to content manager...")
        results = await manager.add_multiple_sources(source_configs)

        # Log results
        successful = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]

        logger.info("-" * 80)
        if successful:
            logger.info(
                f"✓ Successfully initialized {len(successful)} content source(s):"
            )
            for name in successful:
                logger.info(f"  ✓ {name}")

        if failed:
            logger.warning(f"✗ Failed to initialize {len(failed)} content source(s):")
            for name in failed:
                logger.warning(f"  ✗ {name}")

        # Log available sources
        available_sources = manager.get_all_sources()
        logger.info(f"ContentManager instance ID after init: {id(manager)}")
        logger.info(f"Total available content sources: {len(available_sources)}")
        for source in available_sources:
            logger.info(f"  - {source.config.name} ({source.config.source_type.value})")

        logger.info("=" * 80)
        logger.info("CONTENT SOURCES INITIALIZATION COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(
            f"✗ FATAL: Failed to initialize content sources: {e}", exc_info=True
        )
        logger.error("=" * 80)


@router.get("/content-sources", response_model=ContentSourceListResponse)
async def list_content_sources():
    """
    List all configured content sources.

    Returns a list of all available content sources with their status and configuration.
    """
    try:
        logger.info("Listing content sources")
        content_manager = get_content_manager()
        logger.info(f"ContentManager instance ID: {id(content_manager)}")

        # Get all content sources
        sources = content_manager.get_all_sources()
        logger.info(f"Found {len(sources)} sources in content_manager")

        source_list = []
        for source in sources:
            try:
                # Get source status
                is_healthy = await source.health_check()
                source_info = {
                    "name": source.config.name,
                    "display_name": get_source_display_name(source.config.name),
                    "type": source.config.source_type.value,
                    "status": "healthy" if is_healthy else "unhealthy",
                    "healthy": is_healthy,
                    "last_check": None,
                    "metadata": {
                        "enabled": getattr(source.config, "enabled", True),
                        "priority": getattr(source.config, "priority", 0),
                        "path": getattr(source.config, "metadata", {}).get(
                            "path", "/test/path"
                        ),
                    },
                }
                source_list.append(source_info)
            except Exception as e:
                logger.warning(
                    f"Failed to get status for source {source.config.name}: {e}"
                )
                source_info = {
                    "name": source.config.name,
                    "display_name": get_source_display_name(source.config.name),
                    "type": source.config.source_type.value,
                    "status": "error",
                    "healthy": False,
                    "last_check": None,
                    "metadata": {
                        "enabled": getattr(source.config, "enabled", True),
                        "priority": getattr(source.config, "priority", 0),
                        "path": getattr(source.config, "metadata", {}).get(
                            "path", "/test/path"
                        ),
                        "error": str(e),
                    },
                }
                source_list.append(source_info)

        return ContentSourceListResponse(
            success=True,
            message=f"Found {len(source_list)} content sources",
            sources=source_list,
        )

    except Exception as e:
        logger.error(f"Failed to list content sources: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list content sources: {str(e)}"
        )


@router.get(
    "/content-sources/{source_name}/status", response_model=ContentSourceResponse
)
async def get_source_status(source_name: str):
    """
    Get the status of a specific content source.

    Args:
        source_name: Name of the content source to check
    """
    try:
        logger.info(f"Getting status for content source: {source_name}")

        # Get the content manager instance
        content_manager = get_content_manager()

        # Find the source
        source = content_manager.get_source(source_name)
        if not source:
            raise HTTPException(
                status_code=404, detail=f"Content source '{source_name}' not found"
            )

        # Check source health
        is_healthy = await source.health_check()

        # Get additional source information
        source_info = {
            "name": source.config.name,
            "display_name": get_source_display_name(source.config.name),
            "type": source.config.source_type.value,
            "status": "healthy" if is_healthy else "unhealthy",
            "last_checked": None,  # Could be implemented with timestamps
            "config": {
                "enabled": getattr(source.config, "enabled", True),
                "priority": getattr(source.config, "priority", 0),
                "metadata": getattr(source.config, "metadata", {}),
            },
        }

        return ContentSourceResponse(
            success=True,
            message=f"Content source '{source_name}' status retrieved",
            source=source_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for source {source_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get source status: {str(e)}"
        )


@router.post(
    "/content-sources/{source_name}/fetch", response_model=ContentFetchResponse
)
async def fetch_from_source(
    source_name: str, limit: int = 10, include_cached: bool = True
):
    """
    Fetch content from a specific content source, or from all sources if source_name is "all".

    Args:
        source_name: Name of the content source to fetch from, or "all" to fetch from all sources
        limit: Maximum number of content items to fetch (default: 10)
        include_cached: Include cached/previously fetched items (default: True)
    """
    try:
        logger.info(
            f"Fetching content from source: {source_name} (limit: {limit}, include_cached: {include_cached})"
        )

        # Get the content manager instance
        content_manager = get_content_manager()

        # Debug: Log available sources
        available_sources = content_manager.get_all_sources()
        logger.info(f"ContentManager instance ID: {id(content_manager)}")
        logger.info(
            f"Available sources in content_manager: {[s.config.name for s in available_sources]}"
        )
        logger.info(f"Total sources: {len(available_sources)}")

        # Handle special case: "all" means fetch from all sources
        if source_name.lower() == "all":
            logger.info("Fetching content from ALL sources")
            results = await content_manager.fetch_all_content(limit_per_source=limit)

            # Combine all content items from all sources
            all_content_items = []
            total_count = 0
            success = True
            error_messages = []

            for result in results:
                all_content_items.extend(result.content_items)
                total_count += result.total_count
                if not result.success:
                    success = False
                    if result.error_message:
                        error_messages.append(
                            f"{result.source_name}: {result.error_message}"
                        )

            return ContentFetchResponse(
                success=success,
                message=f"Content fetch from all sources completed (fetched from {len(results)} sources)",
                content_items=all_content_items,
                total_count=total_count,
                source_name="all",
                error_message="; ".join(error_messages) if error_messages else None,
            )

        # Find the specific source
        source = content_manager.get_source(source_name)
        if not source:
            logger.error(f"Source '{source_name}' not found!")
            logger.error(f"Requested: '{source_name}'")
            logger.error(f"Available: {[s.config.name for s in available_sources]}")
            raise HTTPException(
                status_code=404, detail=f"Content source '{source_name}' not found"
            )

        # Fetch content from the source
        # Pass include_cached parameter if the source supports it (file sources)
        try:
            result = await source.fetch_content(
                limit=limit, include_cached=include_cached
            )
        except TypeError:
            # Source doesn't support include_cached parameter, use default
            result = await source.fetch_content(limit=limit)

        return ContentFetchResponse(
            success=result.success,
            message=f"Content fetch from '{source_name}' completed",
            content_items=result.content_items,
            total_count=result.total_count,
            source_name=result.source_name,
            error_message=result.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch content from source {source_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch content: {str(e)}"
        )
