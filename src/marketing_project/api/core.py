"""
Core API endpoints for content analysis and pipeline processing.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from marketing_project.models import (
    AnalyzeRequest, PipelineRequest, ContentAnalysisResponse, PipelineResponse
)
from marketing_project.plugins.content_analysis import analyze_content_for_pipeline
from marketing_project.runner import run_marketing_project_pipeline

logger = logging.getLogger("marketing_project.api.core")

# Create router
router = APIRouter()


@router.post("/analyze", response_model=ContentAnalysisResponse)
async def analyze_content(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
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
        
        # Perform content analysis
        analysis_result = analyze_content_for_pipeline(request.content)
        
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
    background_tasks: BackgroundTasks
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
        
        # Run the marketing pipeline
        from marketing_project.server import PROMPTS_DIR
        pipeline_result = await run_marketing_project_pipeline(
            prompts_dir=PROMPTS_DIR,
            lang="en"
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
