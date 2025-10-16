"""
Core API endpoints for content analysis and pipeline processing.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException

from marketing_project.models import (
    AnalyzeRequest, PipelineRequest, ContentAnalysisResponse, PipelineResponse
)
from marketing_project.security import validate_content_input, security_auditor
from marketing_project.security.input_validation import SecurityValidationError
from marketing_project.middleware.auth import get_current_user
from marketing_project.plugins.content_analysis import analyze_content_for_pipeline
from marketing_project.runner import run_marketing_project_pipeline

logger = logging.getLogger("marketing_project.api.core")

# Create router
router = APIRouter()


@router.post("/analyze", response_model=ContentAnalysisResponse)
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
        
        # Security validation
        try:
            validate_content_input(request.content.dict())
        except SecurityValidationError as e:
            logger.warning(f"Security validation failed for content {request.content.id}: {e}")
            await security_auditor.audit_logger.log_security_event(
                event_type="input_validation_failure",
                details={
                    "content_id": request.content.id,
                    "error": str(e),
                    "user_id": current_user.get("role"),
                    "ip_address": "unknown"
                }
            )
            raise HTTPException(
                status_code=400,
                detail=f"Content validation failed: {str(e)}"
            )
        
        # Perform content analysis
        analysis_result = await analyze_content_for_pipeline(request.content)
        
        # Log successful analysis
        await security_auditor.audit_logger.log_security_event(
            event_type="content_analysis_success",
            details={
                "content_id": request.content.id,
                "content_type": request.content.type,
                "user_id": current_user.get("role")
            }
        )
        
        return ContentAnalysisResponse(
            success=True,
            message="Content analysis completed successfully",
            analysis=analysis_result,
            content_id=request.content.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content analysis failed for {request.content.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Content analysis failed: {str(e)}"
        )


@router.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Run the complete marketing pipeline on content.
    
    This endpoint executes the full 7-step marketing pipeline:
    1. AnalyzeContent - Initial content analysis
    2. ExtractSEOKeywords - SEO keyword extraction
    3. GenerateMarketingBrief - Marketing brief generation
    4. GenerateArticle - Article generation
    5. OptimizeSEO - SEO optimization
    6. SuggestInternalDocs - Internal document suggestions
    7. FormatContent - Final content formatting
    """
    try:
        logger.info(f"Pipeline request for content ID: {request.content.id}")
        
        # Security validation
        try:
            validate_content_input(request.content.dict())
        except SecurityValidationError as e:
            logger.warning(f"Security validation failed for pipeline content {request.content.id}: {e}")
            await security_auditor.audit_logger.log_security_event(
                event_type="input_validation_failure",
                details={
                    "content_id": request.content.id,
                    "error": str(e),
                    "user_id": current_user.get("role"),
                    "ip_address": "unknown"
                }
            )
            raise HTTPException(
                status_code=400,
                detail=f"Content validation failed: {str(e)}"
            )
        
        # Run the marketing pipeline
        pipeline_result = await run_marketing_project_pipeline(
            content=request.content,
            background_tasks=background_tasks
        )
        
        # Log successful pipeline execution
        await security_auditor.audit_logger.log_security_event(
            event_type="pipeline_execution_success",
            details={
                "content_id": request.content.id,
                "content_type": request.content.type,
                "user_id": current_user.get("role"),
                "pipeline_steps": len(pipeline_result.get("steps", []))
            }
        )
        
        return PipelineResponse(
            success=True,
            message="Marketing pipeline completed successfully",
            result=pipeline_result,
            content_id=request.content.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline execution failed for {request.content.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}"
        )
