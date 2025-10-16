"""
API endpoints for FastAPI.

This module defines the main API endpoints for the marketing project.
"""

import logging
import os
import time
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, Request, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse

from marketing_project.models import (
    AnalyzeRequest, PipelineRequest, ContentAnalysisResponse, PipelineResponse,
    ContentSourceListResponse, ContentSourceResponse, ContentFetchResponse,
    APIResponse, ErrorResponse
)
from marketing_project.security import validate_content_input, security_auditor
from marketing_project.security.input_validation import SecurityValidationError
from marketing_project.middleware.auth import get_current_user, get_current_api_key
from marketing_project.plugins.content_analysis import analyze_content_for_pipeline
from marketing_project.runner import run_marketing_project_pipeline
from marketing_project.services.content_source_factory import ContentSourceManager
from marketing_project.services.content_source_config_loader import ContentSourceConfigLoader

logger = logging.getLogger("marketing_project.api_endpoints")

# Create API router
api_router = APIRouter(prefix="/api/v1", tags=["Marketing API"])


@api_router.post("/analyze", response_model=ContentAnalysisResponse)
async def analyze_content(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Analyze content for marketing pipeline processing.
    
    This endpoint performs comprehensive content analysis including:
    - Content quality assessment
    - SEO potential analysis
    - Marketing value evaluation
    - Processing recommendations
    """
    try:
        logger.info(f"Content analysis request for content ID: {request.content.id}")
        
        # Validate content input for security
        try:
            validate_content_input(request.content.dict())
        except SecurityValidationError as e:
            # Log security violation
            security_auditor.audit_logger.log_xss_attempt(
                source_ip=request.client.host if request.client else "unknown",
                user_id=current_user.get('role', 'unknown'),
                content=str(request.content.dict())[:100],
                request_id=getattr(request.state, 'request_id', 'unknown')
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content validation failed: {str(e)}"
            )
        
        # Perform analysis
        analysis_result = analyze_content_for_pipeline(request.content.dict())
        
        if not analysis_result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Content analysis failed: {analysis_result.get('error', 'Unknown error')}"
            )
        
        return ContentAnalysisResponse(
            success=True,
            message="Content analysis completed successfully",
            data=analysis_result.get("data", {}),
            metadata={
                "content_id": request.content.id,
                "content_type": request.content.content_type,
                "analysis_timestamp": time.time(),
                "user_role": current_user.get("role", "unknown")
            }
        )
        
    except Exception as e:
        logger.error(f"Content analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content analysis failed: {str(e)}"
        )


@api_router.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Run the complete marketing pipeline on content.
    
    This endpoint processes content through the entire marketing pipeline:
    - Content analysis
    - SEO keyword extraction
    - Marketing brief generation
    - Article structure generation
    - SEO optimization
    - Content formatting
    """
    try:
        start_time = time.time()
        logger.info(f"Pipeline execution request for content ID: {request.content.id}")
        
        # Validate content input for security
        try:
            validate_content_input(request.content.dict())
        except SecurityValidationError as e:
            # Log security violation
            security_auditor.audit_logger.log_xss_attempt(
                source_ip=request.client.host if request.client else "unknown",
                user_id=current_user.get('role', 'unknown'),
                content=str(request.content.dict())[:100],
                request_id=getattr(request.state, 'request_id', 'unknown')
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content validation failed: {str(e)}"
            )
        
        # Get prompts directory
        base = os.path.dirname(__file__)
        prompts_dir = os.path.abspath(os.path.join(base, "prompts", os.getenv("TEMPLATE_VERSION", "v1")))
        
        # Run pipeline asynchronously
        background_tasks.add_task(
            run_marketing_project_pipeline,
            prompts_dir,
            "en"
        )
        
        execution_time = time.time() - start_time
        
        return PipelineResponse(
            success=True,
            message="Pipeline execution started successfully",
            data={
                "content_id": request.content.id,
                "content_type": request.content.content_type,
                "pipeline_status": "started",
                "estimated_completion_time": "5-10 minutes"
            },
            metadata={
                "execution_time": execution_time,
                "user_role": current_user.get("role", "unknown"),
                "pipeline_options": request.options
            },
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(e)}"
        )


@api_router.get("/content-sources", response_model=ContentSourceListResponse)
async def list_content_sources(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    List all configured content sources.
    
    Returns information about all available content sources including
    their status, type, and health information.
    """
    try:
        logger.info("Content sources list request")
        
        # Initialize content manager
        content_manager = ContentSourceManager()
        config_loader = ContentSourceConfigLoader()
        source_configs = config_loader.create_source_configs()
        
        # Add sources to manager
        for config in source_configs:
            await content_manager.add_source_from_config(config)
        
        # Get source information
        sources = []
        for name, source in content_manager.sources.items():
            status_info = source.get_status()
            is_healthy = await source.health_check()
            
            sources.append(ContentSourceResponse(
                name=name,
                type=status_info.get("type", "unknown"),
                status=status_info.get("status", "unknown"),
                healthy=is_healthy,
                last_check=None,  # Could be added to source status
                metadata=status_info.get("metadata", {})
            ))
        
        await content_manager.cleanup()
        
        return ContentSourceListResponse(
            success=True,
            message=f"Found {len(sources)} content sources",
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Failed to list content sources: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list content sources: {str(e)}"
        )


@api_router.get("/content-sources/{source_name}/status", response_model=ContentSourceResponse)
async def get_source_status(
    source_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get status of a specific content source.
    
    Returns detailed status information for the specified content source.
    """
    try:
        logger.info(f"Source status request for: {source_name}")
        
        # Initialize content manager
        content_manager = ContentSourceManager()
        config_loader = ContentSourceConfigLoader()
        source_configs = config_loader.create_source_configs()
        
        # Add sources to manager
        for config in source_configs:
            await content_manager.add_source_from_config(config)
        
        # Check if source exists
        if source_name not in content_manager.sources:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source '{source_name}' not found"
            )
        
        # Get source status
        source = content_manager.sources[source_name]
        status_info = source.get_status()
        is_healthy = await source.health_check()
        
        await content_manager.cleanup()
        
        return ContentSourceResponse(
            name=source_name,
            type=status_info.get("type", "unknown"),
            status=status_info.get("status", "unknown"),
            healthy=is_healthy,
            last_check=None,
            metadata=status_info.get("metadata", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get source status for {source_name}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get source status: {str(e)}"
        )


@api_router.post("/content-sources/{source_name}/fetch", response_model=ContentFetchResponse)
async def fetch_source_content(
    source_name: str,
    limit: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Fetch content from a specific source.
    
    Retrieves content from the specified source with optional limit.
    """
    try:
        logger.info(f"Content fetch request for source: {source_name}, limit: {limit}")
        
        # Validate limit
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Limit must be between 1 and 100"
            )
        
        # Initialize content manager
        content_manager = ContentSourceManager()
        config_loader = ContentSourceConfigLoader()
        source_configs = config_loader.create_source_configs()
        
        # Add sources to manager
        for config in source_configs:
            await content_manager.add_source_from_config(config)
        
        # Check if source exists
        if source_name not in content_manager.sources:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source '{source_name}' not found"
            )
        
        # Fetch content
        source = content_manager.sources[source_name]
        result = await source.fetch_content(limit)
        
        await content_manager.cleanup()
        
        return ContentFetchResponse(
            success=result.success,
            message="Content fetch completed" if result.success else "Content fetch failed",
            source_name=result.source_name,
            total_count=result.total_count,
            items=[],  # Could be populated with actual content items
            error_message=result.error_message if not result.success else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch content from {source_name}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch content: {str(e)}"
        )


@api_router.get("/health")
async def health_check():
    """
    Health check endpoint for Kubernetes probes.
    
    Returns 200 OK if the service is healthy.
    """
    try:
        from marketing_project.server import PIPELINE_SPEC, PROMPTS_DIR
        
        health_status = {
            "status": "healthy",
            "service": "marketing-project",
            "version": "1.0.0",
            "checks": {
                "config_loaded": PIPELINE_SPEC is not None,
                "prompts_dir_exists": os.path.exists(PROMPTS_DIR),
            },
        }
        
        # Check if any critical checks failed
        if not all(health_status["checks"].values()):
            health_status["status"] = "unhealthy"
            return JSONResponse(status_code=503, content=health_status)
        
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503, 
            content={"status": "unhealthy", "error": str(e)}
        )


@api_router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes probes.
    
    Returns 200 OK if the service is ready to accept traffic.
    """
    try:
        from marketing_project.server import PIPELINE_SPEC, PROMPTS_DIR
        
        ready_status = {
            "status": "ready",
            "service": "marketing-project",
            "checks": {
                "config_loaded": PIPELINE_SPEC is not None,
                "prompts_dir_exists": os.path.exists(PROMPTS_DIR),
            },
        }
        
        if not all(ready_status["checks"].values()):
            ready_status["status"] = "not_ready"
            return JSONResponse(status_code=503, content=ready_status)
        
        return ready_status
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503, 
            content={"status": "not_ready", "error": str(e)}
        )
