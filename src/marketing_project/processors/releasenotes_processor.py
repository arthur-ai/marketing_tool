"""
Simplified release notes processor for Marketing Project.

This processor handles release notes content with minimal pre-processing,
letting the AI function pipeline handle all analysis and extraction.

Uses the function-based pipeline with OpenAI structured outputs for:
- Guaranteed JSON output
- Faster execution
- Predictable costs
- Full type safety
"""

import json
import logging
import uuid
from datetime import date, datetime
from typing import Any, Optional

from marketing_project.models.content_models import ReleaseNotesContext
from marketing_project.services.function_pipeline import FunctionPipeline

logger = logging.getLogger("marketing_project.processors")


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for datetime and other non-serializable objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


async def process_release_notes(
    content_data: str,
    job_id: Optional[str] = None,
    output_content_type: Optional[str] = None,
) -> str:
    """
    Process release notes content through the AI function pipeline.

    Simplified workflow:
    1. Parse and validate input (Pydantic models)
    2. Run AI pipeline (handles all analysis and extraction)
    3. Return results

    Args:
        content_data: JSON string containing release notes data
        job_id: Optional job ID for tracking (generated if not provided)
        output_content_type: Optional output content type (blog_post, press_release, case_study)

    Returns:
        JSON string with processing results
    """
    # Generate job ID if not provided
    if not job_id:
        job_id = str(uuid.uuid4())

    try:
        # Step 1: Parse input
        logger.info("Release Notes Processor: Parsing input data")
        data = (
            json.loads(content_data) if isinstance(content_data, str) else content_data
        )

        logger.info(
            f"Release Notes Processor: Processing release notes '{data.get('title', 'Untitled')}'"
        )

        # Convert to Pydantic model (validates required fields automatically)
        try:
            release_model = ReleaseNotesContext(**data)
        except Exception as e:
            logger.error(f"Release Notes Processor: Invalid input: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "error": "invalid_input",
                    "message": f"Failed to parse release notes data: {e}",
                }
            )

        # Get output_content_type from parameter, job metadata, or default
        if output_content_type is None:
            output_content_type = "blog_post"
            try:
                from marketing_project.services.job_manager import get_job_manager

                job_manager = get_job_manager()
                job = await job_manager.get_job(job_id)
                if job and job.metadata.get("output_content_type"):
                    output_content_type = job.metadata.get("output_content_type")
                    logger.info(
                        f"Release Notes Processor: Using output_content_type={output_content_type} from job metadata"
                    )
            except Exception as e:
                logger.warning(
                    f"Release Notes Processor: Could not get output_content_type from job metadata: {e}"
                )
        else:
            logger.info(
                f"Release Notes Processor: Using output_content_type={output_content_type} from parameter"
            )

        # Step 2: Run through AI function pipeline
        logger.info("Release Notes Processor: Running function-based pipeline")
        try:
            pipeline = FunctionPipeline(model="gpt-4o-mini", temperature=0.7)

            pipeline_result = await pipeline.execute_pipeline(
                content_json=release_model.model_dump_json(),
                job_id=job_id,
                content_type="release_notes",
                output_content_type=output_content_type,
            )

            logger.info(
                "Release Notes Processor: Function pipeline completed successfully"
            )

        except ValueError as e:
            # Approval rejected by user
            logger.error(
                f"Release Notes Processor: Content rejected during approval: {e}"
            )
            return json.dumps(
                {"status": "error", "error": "approval_rejected", "message": str(e)}
            )
        except Exception as e:
            logger.error(f"Release Notes Processor: Pipeline execution failed: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "error": "pipeline_failed",
                    "message": f"Content pipeline execution failed: {e}",
                }
            )

        # Step 3: Return results
        logger.info("Release Notes Processor: Processing complete")
        result = {
            "status": "success",
            "content_type": "release_notes",
            "pipeline_result": pipeline_result,
            "message": f"Release notes '{release_model.title or 'Untitled'}' processed successfully",
        }

        return json.dumps(result, indent=2, default=_json_serializer)

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON input: {e}"
        logger.error(f"Release Notes Processor: {error_msg}")
        return json.dumps(
            {"status": "error", "error": "invalid_json", "message": error_msg}
        )
    except Exception as e:
        error_msg = f"Unexpected error during release notes processing: {e}"
        logger.error(f"Release Notes Processor: {error_msg}", exc_info=True)
        return json.dumps(
            {"status": "error", "error": "processing_exception", "message": error_msg}
        )
