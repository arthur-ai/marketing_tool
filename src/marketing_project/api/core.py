"""
Core API endpoints for content analysis and pipeline processing.
"""

import json
import logging
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException

from marketing_project.config.settings import PROMPTS_DIR
from marketing_project.models import (
    AnalyzeRequest,
    BlogPostContext,
    ContentAnalysisResponse,
    PipelineRequest,
    PipelineResponse,
    ReleaseNotesContext,
    StepExecutionRequest,
    StepExecutionResponse,
    StepInfo,
    StepListResponse,
    StepRequirementsResponse,
    TranscriptContext,
)
from marketing_project.plugins.content_analysis import analyze_content_for_pipeline
from marketing_project.plugins.registry import get_plugin_registry
from marketing_project.processors import (
    process_blog_post,
    process_release_notes,
    process_transcript,
)
from marketing_project.services.function_pipeline import FunctionPipeline
from marketing_project.services.job_manager import JobStatus, get_job_manager

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


@router.get("/pipeline/steps", response_model=StepListResponse)
async def list_pipeline_steps():
    """
    List all available pipeline steps.

    Returns a list of all pipeline steps with their names, numbers, and descriptions.
    """
    try:
        registry = get_plugin_registry()
        plugins = registry.get_plugins_in_order()

        steps = []
        for plugin in plugins:
            steps.append(
                StepInfo(
                    step_name=plugin.step_name,
                    step_number=plugin.step_number,
                    description=f"Step {plugin.step_number}: {plugin.step_name.replace('_', ' ').title()}",
                )
            )

        return StepListResponse(steps=steps)

    except Exception as e:
        logger.error(f"Failed to list pipeline steps: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list pipeline steps: {str(e)}"
        )


@router.get(
    "/pipeline/steps/{step_name}/requirements", response_model=StepRequirementsResponse
)
async def get_step_requirements(step_name: str):
    """
    Get requirements for a specific pipeline step.

    Returns the required context keys and their descriptions for the step.
    """
    try:
        registry = get_plugin_registry()
        plugin = registry.get_plugin(step_name)

        if not plugin:
            available_steps = ", ".join(registry.get_all_plugins().keys())
            raise HTTPException(
                status_code=404,
                detail=f"Step '{step_name}' not found. Available steps: {available_steps}",
            )

        required_keys = plugin.get_required_context_keys()

        # Generate descriptions for each required key
        descriptions = {}
        for key in required_keys:
            if key == "input_content":
                descriptions[key] = (
                    "The input content to process (e.g., blog post, article)"
                )
            elif key == "content_type":
                descriptions[key] = (
                    "Type of content (e.g., 'blog_post', 'release_notes', 'transcript')"
                )
            elif key == "output_content_type":
                descriptions[key] = (
                    "Desired output content type (e.g., 'blog_post', 'press_release', 'case_study')"
                )
            elif key in registry.get_all_plugins():
                # This is a step output
                step_plugin = registry.get_plugin(key)
                descriptions[key] = (
                    f"Output from {key.replace('_', ' ')} step (step {step_plugin.step_number if step_plugin else 'N/A'})"
                )
            else:
                descriptions[key] = f"Required context key: {key}"

        return StepRequirementsResponse(
            step_name=step_name,
            step_number=plugin.step_number,
            required_context_keys=required_keys,
            descriptions=descriptions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get requirements for step {step_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get step requirements: {str(e)}",
        )


@router.post(
    "/pipeline/steps/{step_name}/execute", response_model=StepExecutionResponse
)
async def execute_pipeline_step(step_name: str, request: StepExecutionRequest):
    """
    Execute a single pipeline step independently.

    This endpoint allows executing individual pipeline steps with user-provided
    inputs, separate from the full pipeline execution. The step is executed
    asynchronously and results are saved to the database.

    Args:
        step_name: Name of the step to execute (e.g., "seo_keywords")
        request: StepExecutionRequest with content and context

    Returns:
        StepExecutionResponse with job_id for tracking
    """
    try:
        # Validate step exists
        registry = get_plugin_registry()
        plugin = registry.get_plugin(step_name)

        if not plugin:
            available_steps = ", ".join(registry.get_all_plugins().keys())
            raise HTTPException(
                status_code=404,
                detail=f"Step '{step_name}' not found. Available steps: {available_steps}",
            )

        # Validate required context keys
        required_keys = plugin.get_required_context_keys()
        missing_keys = [key for key in required_keys if key not in request.context]

        if missing_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required context keys: {missing_keys}. Required keys: {required_keys}",
            )

        # Get job manager
        job_manager = get_job_manager()

        # Create content ID from content if available
        content_id = request.content.get("id", f"step_{step_name}_{int(time.time())}")

        # Prepare job metadata
        job_metadata = {
            "step_name": step_name,
            "step_number": plugin.step_number,
            "content": request.content,
            "context_keys": list(request.context.keys()),
        }

        # Store pipeline_config in metadata if provided
        if request.pipeline_config:
            from marketing_project.models.pipeline_steps import PipelineConfig

            if isinstance(request.pipeline_config, dict):
                job_metadata["pipeline_config"] = request.pipeline_config
            else:
                job_metadata["pipeline_config"] = request.pipeline_config.model_dump()

        # Create job
        job = await job_manager.create_job(
            job_type=f"step_{step_name}",
            content_id=content_id,
            metadata=job_metadata,
        )

        # Convert content to JSON string
        content_json = json.dumps(request.content)

        # Submit job to ARQ for background processing
        arq_job_id = await job_manager.submit_to_arq(
            job.id,
            "execute_single_step_job",  # ARQ function name
            step_name,
            content_json,
            request.context,
            job.id,
        )

        logger.info(
            f"Step execution job {job.id} submitted to ARQ (arq_id: {arq_job_id}) for step {step_name}"
        )

        return StepExecutionResponse(
            step_name=step_name,
            job_id=job.id,
            status=JobStatus.QUEUED.value,
            message=f"Step '{step_name}' submitted for execution",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute step {step_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute step: {str(e)}")
