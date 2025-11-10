"""
Core API endpoints for content analysis and pipeline processing.
"""

import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from marketing_project.config.settings import PROMPTS_DIR
from marketing_project.models import (
    AnalyzeRequest,
    BlogPostContext,
    ContentAnalysisResponse,
    PipelineRequest,
    PipelineResponse,
    ReleaseNotesContext,
    TranscriptContext,
)
from marketing_project.plugins.content_analysis import analyze_content_for_pipeline
from marketing_project.processors import (
    process_blog_post,
    process_release_notes,
    process_transcript,
)

logger = logging.getLogger("marketing_project.api.core")

# Create router
router = APIRouter()


@router.post("/analyze", response_model=ContentAnalysisResponse)
async def analyze_content(request: AnalyzeRequest, background_tasks: BackgroundTasks):
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
            content_id=request.content.id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content analysis failed for {request.content.id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Content analysis failed: {str(e)}"
        )


@router.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Run the complete marketing pipeline on content with auto-routing.

    This endpoint automatically routes content to the appropriate deterministic processor
    based on content type, then executes the full 8-step marketing pipeline:

    Processor Steps (Deterministic):
    1. Input validation
    2. Type analysis
    3. Structure validation
    4. Metadata extraction

    Agent Pipeline Steps (Non-Deterministic):
    5. SEO Keywords - Extract and analyze SEO keywords
    6. Marketing Brief - Generate marketing strategy
    7. Article Generation - Create high-quality content
    8. SEO Optimization - Apply comprehensive SEO
    9. Internal Docs - Suggest document references
    10. Content Formatting - Format and structure
    11. Design Kit - Apply design templates
    12. Final Assembly - Compile results
    """
    try:
        logger.info(f"Pipeline request for content ID: {request.content.id}")

        # Route to appropriate deterministic processor based on content type
        content_json = request.content.model_dump_json()

        # Get output_content_type from request, defaulting to "blog_post" if not provided
        output_content_type = request.output_content_type or "blog_post"

        if isinstance(request.content, BlogPostContext):
            logger.info(
                f"Routing to blog processor for content ID: {request.content.id} with output_content_type: {output_content_type}"
            )
            result_json = await process_blog_post(
                content_json, output_content_type=output_content_type
            )
            content_type = "blog_post"

        elif isinstance(request.content, ReleaseNotesContext):
            logger.info(
                f"Routing to release notes processor for content ID: {request.content.id} with output_content_type: {output_content_type}"
            )
            result_json = await process_release_notes(
                content_json, output_content_type=output_content_type
            )
            content_type = "release_notes"

        elif isinstance(request.content, TranscriptContext):
            logger.info(
                f"Routing to transcript processor for content ID: {request.content.id} with output_content_type: {output_content_type}"
            )
            result_json = await process_transcript(
                content_json, output_content_type=output_content_type
            )
            content_type = "transcript"

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {type(request.content).__name__}",
            )

        # Parse processor result
        try:
            result_dict = json.loads(result_json)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse processor result: {result_json}")
            raise HTTPException(
                status_code=500, detail="Processor returned invalid JSON"
            )

        # Check for errors in result
        if result_dict.get("status") == "error":
            error_msg = result_dict.get("message", "Unknown error")
            logger.error(
                f"Pipeline processing failed for {request.content.id}: {error_msg}"
            )
            raise HTTPException(status_code=400, detail=error_msg)

        # Build response
        serializable_result = {
            "content_type": content_type,
            "metadata": result_dict.get("metadata"),
            "pipeline_result": result_dict.get("pipeline_result"),
            "validation": result_dict.get("validation"),
            "processing_steps_completed": result_dict.get("processing_steps_completed"),
        }

        return PipelineResponse(
            success=True,
            message=result_dict.get(
                "message", "Marketing pipeline completed successfully"
            ),
            result=serializable_result,
            content_id=request.content.id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline execution failed for {request.content.id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Pipeline execution failed: {str(e)}"
        )
