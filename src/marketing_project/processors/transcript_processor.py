"""
Simplified transcript processor for Marketing Project.

This processor handles transcript content with minimal pre-processing,
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

from marketing_project.models.content_models import TranscriptContext
from marketing_project.services.function_pipeline import FunctionPipeline

logger = logging.getLogger("marketing_project.processors")


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for datetime and other non-serializable objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def _parse_duration_to_seconds(duration: Any) -> Optional[int]:
    """
    Parse duration string to seconds (integer).

    Supports formats:
    - "MM:SS" (e.g., "10:00" = 600 seconds)
    - "HH:MM:SS" (e.g., "1:10:00" = 4200 seconds)
    - Integer (already in seconds)

    Args:
        duration: Duration as string or integer

    Returns:
        Duration in seconds as integer, or None if parsing fails
    """
    if duration is None:
        return None

    # If already an integer, return as-is
    if isinstance(duration, int):
        return duration

    # If not a string, try to convert
    if not isinstance(duration, str):
        try:
            return int(duration)
        except (ValueError, TypeError):
            return None

    # Parse string format (MM:SS or HH:MM:SS)
    try:
        parts = duration.split(":")
        if len(parts) == 2:
            # MM:SS format
            minutes, seconds = int(parts[0]), int(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:
            # HH:MM:SS format
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        else:
            # Try to parse as integer string
            return int(duration)
    except (ValueError, TypeError):
        logger.warning(f"Failed to parse duration '{duration}', using None")
        return None


async def process_transcript(
    content_data: str,
    job_id: Optional[str] = None,
    output_content_type: Optional[str] = None,
) -> str:
    """
    Process transcript content through the AI function pipeline.

    Simplified workflow:
    1. Parse and validate input (Pydantic models)
    2. Run AI pipeline (handles all analysis and extraction)
    3. Return results

    Args:
        content_data: JSON string containing transcript data
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
        logger.info("Transcript Processor: Parsing input data")
        data = (
            json.loads(content_data) if isinstance(content_data, str) else content_data
        )

        logger.info(
            f"Transcript Processor: Processing transcript '{data.get('title', 'Untitled')}'"
        )

        # Parse duration if it's a string
        if "duration" in data and isinstance(data["duration"], str):
            data["duration"] = _parse_duration_to_seconds(data["duration"])
            if data["duration"] is None:
                logger.warning("Failed to parse duration, removing from data")
                data.pop("duration", None)

        # Convert to Pydantic model (validates required fields automatically)
        try:
            transcript_model = TranscriptContext(**data)
        except Exception as e:
            logger.error(f"Transcript Processor: Invalid input: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "error": "invalid_input",
                    "message": f"Failed to parse transcript data: {e}",
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
                        f"Transcript Processor: Using output_content_type={output_content_type} from job metadata"
                    )
            except Exception as e:
                logger.warning(
                    f"Transcript Processor: Could not get output_content_type from job metadata: {e}"
                )
        else:
            logger.info(
                f"Transcript Processor: Using output_content_type={output_content_type} from parameter"
            )

        # Step 2: Run through AI function pipeline
        logger.info(
            f"Transcript Processor: Running function-based pipeline with output_content_type={output_content_type}"
        )
        try:
            pipeline = FunctionPipeline(model="gpt-5.1", temperature=0.7)

            pipeline_result = await pipeline.execute_pipeline(
                content_json=transcript_model.model_dump_json(),
                job_id=job_id,
                content_type="transcript",
                output_content_type=output_content_type,
            )

            logger.info(
                "Transcript Processor: Function pipeline completed successfully"
            )

        except ValueError as e:
            # Approval rejected by user
            logger.error(f"Transcript Processor: Content rejected during approval: {e}")
            return json.dumps(
                {"status": "error", "error": "approval_rejected", "message": str(e)}
            )
        except Exception as e:
            logger.error(f"Transcript Processor: Pipeline execution failed: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "error": "pipeline_failed",
                    "message": f"Content pipeline execution failed: {e}",
                }
            )

        # Step 3: Return results
        logger.info("Transcript Processor: Processing complete")
        result = {
            "status": "success",
            "content_type": "transcript",
            "pipeline_result": pipeline_result,
            "message": f"Transcript '{transcript_model.title or 'Untitled'}' processed successfully",
        }

        return json.dumps(result, indent=2, default=_json_serializer)

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON input: {e}"
        logger.error(f"Transcript Processor: {error_msg}")
        return json.dumps(
            {"status": "error", "error": "invalid_json", "message": error_msg}
        )
    except Exception as e:
        error_msg = f"Unexpected error during transcript processing: {e}"
        logger.error(f"Transcript Processor: {error_msg}", exc_info=True)
        return json.dumps(
            {"status": "error", "error": "processing_exception", "message": error_msg}
        )
