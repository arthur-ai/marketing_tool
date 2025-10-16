"""
Content source management API endpoints.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from marketing_project.models import (
    ContentSourceListResponse, ContentSourceResponse, ContentFetchResponse
)
from marketing_project.middleware.auth import get_current_user
from marketing_project.services.content_source_factory import ContentSourceManager
from marketing_project.services.content_source_config_loader import ContentSourceConfigLoader

logger = logging.getLogger("marketing_project.api.content")

# Create router
router = APIRouter()

# Initialize content source manager
content_manager = ContentSourceManager()
config_loader = ContentSourceConfigLoader()


@router.get("/content-sources", response_model=ContentSourceListResponse)
async def list_content_sources(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
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
                    "config": {
                        "enabled": getattr(source.config, 'enabled', True),
                        "priority": getattr(source.config, 'priority', 0),
                        "metadata": getattr(source.config, 'metadata', {})
                    }
                }
                source_list.append(source_info)
            except Exception as e:
                logger.warning(f"Failed to get status for source {source.config.name}: {e}")
                source_info = {
                    "name": source.config.name,
                    "type": source.config.source_type.value,
                    "status": "error",
                    "error": str(e),
                    "config": {
                        "enabled": getattr(source.config, 'enabled', True),
                        "priority": getattr(source.config, 'priority', 0),
                        "metadata": getattr(source.config, 'metadata', {})
                    }
                }
                source_list.append(source_info)
        
        return ContentSourceListResponse(
            success=True,
            message=f"Found {len(source_list)} content sources",
            sources=source_list,
            total_count=len(source_list)
        )
        
    except Exception as e:
        logger.error(f"Failed to list content sources: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list content sources: {str(e)}"
        )


@router.get("/content-sources/{source_name}/status", response_model=ContentSourceResponse)
async def get_source_status(
    source_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
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
                status_code=404,
                detail=f"Content source '{source_name}' not found"
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
                "enabled": getattr(source.config, 'enabled', True),
                "priority": getattr(source.config, 'priority', 0),
                "metadata": getattr(source.config, 'metadata', {})
            }
        }
        
        return ContentSourceResponse(
            success=True,
            message=f"Content source '{source_name}' status retrieved",
            source=source_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for source {source_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get source status: {str(e)}"
        )


@router.post("/content-sources/{source_name}/fetch", response_model=ContentFetchResponse)
async def fetch_from_source(
    source_name: str,
    limit: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
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
                status_code=404,
                detail=f"Content source '{source_name}' not found"
            )
        
        # Fetch content from the source
        result = await source.fetch_content(limit=limit)
        
        return ContentFetchResponse(
            success=result.success,
            message=f"Content fetch from '{source_name}' completed",
            content_items=result.content_items,
            total_count=result.total_count,
            source_name=result.source_name,
            error_message=result.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch content from source {source_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch content: {str(e)}"
        )
