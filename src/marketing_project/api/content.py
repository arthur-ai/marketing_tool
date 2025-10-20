"""
Content source management API endpoints.
"""

import logging

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

# Initialize content source manager
content_manager = ContentSourceManager()
config_loader = ContentSourceConfigLoader()


async def initialize_content_sources():
    """
    Initialize content sources from configuration.

    This should be called during application startup to load all configured
    content sources from the pipeline.yml configuration file.
    """
    try:
        logger.info("Initializing content sources...")

        # Load source configurations from pipeline.yml
        source_configs = config_loader.create_source_configs()

        if not source_configs:
            logger.warning("No content sources found in configuration")
            return

        # Add all sources to the manager
        results = await content_manager.add_multiple_sources(source_configs)

        # Log results
        successful = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]

        logger.info(
            f"Successfully initialized {len(successful)} content sources: {successful}"
        )
        if failed:
            logger.warning(
                f"Failed to initialize {len(failed)} content sources: {failed}"
            )

    except Exception as e:
        logger.error(f"Failed to initialize content sources: {e}", exc_info=True)


@router.get("/content-sources", response_model=ContentSourceListResponse)
async def list_content_sources():
    """
    List all configured content sources.

    Returns a list of all available content sources with their status and configuration.
    """
    try:
        logger.info("Listing content sources")

        # Get all content sources
        sources = content_manager.get_all_sources()

        source_list = []
        for source in sources:
            try:
                # Get source status
                is_healthy = await source.health_check()
                source_info = {
                    "name": source.config.name,
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
async def fetch_from_source(source_name: str, limit: int = 10):
    """
    Fetch content from a specific content source.

    Args:
        source_name: Name of the content source to fetch from
        limit: Maximum number of content items to fetch (default: 10)
    """
    try:
        logger.info(f"Fetching content from source: {source_name} (limit: {limit})")

        # Find the source
        source = content_manager.get_source(source_name)
        if not source:
            raise HTTPException(
                status_code=404, detail=f"Content source '{source_name}' not found"
            )

        # Fetch content from the source
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
