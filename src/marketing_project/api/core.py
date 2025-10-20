"""
Core API endpoints for content analysis and pipeline processing.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from marketing_project.agents.blog_agent import get_blog_agent
from marketing_project.agents.marketing_agent import get_marketing_orchestrator_agent
from marketing_project.agents.releasenotes_agent import get_releasenotes_agent
from marketing_project.agents.transcripts_agent import get_transcripts_agent
from marketing_project.config.settings import PROMPTS_DIR
from marketing_project.models import (
    AnalyzeRequest,
    ContentAnalysisResponse,
    PipelineRequest,
    PipelineResponse,
)
from marketing_project.plugins.content_analysis import analyze_content_for_pipeline

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

        # Set up specialized agents and orchestrator
        transcripts_agent = await get_transcripts_agent(PROMPTS_DIR, "en")
        blog_agent = await get_blog_agent(PROMPTS_DIR, "en")
        releasenotes_agent = await get_releasenotes_agent(PROMPTS_DIR, "en")
        orchestrator_agent = await get_marketing_orchestrator_agent(
            PROMPTS_DIR, "en", transcripts_agent, blog_agent, releasenotes_agent
        )

        # Process the content from the request through the orchestrator
        # Convert Pydantic model to JSON string for LangChain compatibility
        content_dict = request.content.model_dump(mode="json")
        content_type = request.content.__class__.__name__.replace("Context", "").lower()

        # Format as a clear prompt string for the agent
        prompt = f"""Process the following {content_type} content:

Content ID: {request.content.id}
Title: {request.content.title or 'N/A'}
Content Type: {content_type}

Content:
{request.content.content or 'No content provided'}

Additional Context:
{content_dict}
"""

        # Run through orchestrator
        logger.info(
            f"Processing content through orchestrator for content ID: {request.content.id}"
        )
        processed = await orchestrator_agent.run_async(prompt)

        # Extract only serializable data from the result
        serializable_result = {
            "processed_content": processed,
            "stats": {
                "content_type": content_type,
            },
        }

        return PipelineResponse(
            success=True,
            message="Marketing pipeline completed successfully",
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
