"""
Simplified blog post processor for Marketing Project.

This processor handles blog post content with minimal pre-processing,
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

from marketing_project.models.content_models import BlogPostContext
from marketing_project.services.function_pipeline import FunctionPipeline

logger = logging.getLogger("marketing_project.processors")


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for datetime and other non-serializable objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


async def process_blog_post(
    content_data: str,
    job_id: Optional[str] = None,
    output_content_type: Optional[str] = None,
) -> str:
    """
    Process blog post content through the AI function pipeline.

    Simplified workflow:
    1. Parse and validate input (Pydantic models)
    2. Run AI pipeline (handles all analysis and extraction)
    3. Return results

    Args:
        content_data: JSON string containing blog post data
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
        logger.info("Blog Processor: Parsing input data")
        data = (
            json.loads(content_data) if isinstance(content_data, str) else content_data
        )

        logger.info(
            f"Blog Processor: Processing blog post '{data.get('title', 'Untitled')}'"
        )

        # Convert to Pydantic model (validates required fields automatically)
        try:
            blog_post_model = BlogPostContext(**data)
        except Exception as e:
            logger.error(f"Blog Processor: Invalid input: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "error": "invalid_input",
                    "message": f"Failed to parse blog post data: {e}",
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
                        f"Blog Processor: Using output_content_type={output_content_type} from job metadata"
                    )
            except Exception as e:
                logger.warning(
                    f"Blog Processor: Could not get output_content_type from job metadata: {e}"
                )
        else:
            logger.info(
                f"Blog Processor: Using output_content_type={output_content_type} from parameter"
            )

        # Step 2: Route to appropriate pipeline
        logger.info("Blog Processor: Routing to appropriate pipeline")
        try:
            # Check if this is a social media post request
            if output_content_type == "social_media_post":
                # Get social media platform and email type from job metadata
                social_media_platform = "linkedin"
                email_type = None
                try:
                    from marketing_project.services.job_manager import get_job_manager

                    job_manager = get_job_manager()
                    job = await job_manager.get_job(job_id)
                    if job:
                        social_media_platform = job.metadata.get(
                            "social_media_platform", "linkedin"
                        )
                        email_type = job.metadata.get("email_type")
                        logger.info(
                            f"Blog Processor: Using social_media_platform={social_media_platform}, email_type={email_type}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Blog Processor: Could not get social media parameters from job metadata: {e}"
                    )

                # Route to social media pipeline
                logger.info(
                    f"Blog Processor: Running social media pipeline for platform {social_media_platform}"
                )
                pipeline = SocialMediaPipeline(model="gpt-4o-mini", temperature=0.7)

                pipeline_result = await pipeline.execute_pipeline(
                    content_json=blog_post_model.model_dump_json(),
                    job_id=job_id,
                    content_type="blog_post",
                    social_media_platform=social_media_platform,
                    email_type=email_type,
                )

                logger.info(
                    "Blog Processor: Social media pipeline completed successfully"
                )
            else:
                # Route to regular function pipeline
                logger.info("Blog Processor: Running function-based pipeline")
                pipeline = FunctionPipeline(model="gpt-4o-mini", temperature=0.7)

                pipeline_result = await pipeline.execute_pipeline(
                    content_json=blog_post_model.model_dump_json(),
                    job_id=job_id,
                    content_type="blog_post",
                    output_content_type=output_content_type,
                )

                logger.info("Blog Processor: Function pipeline completed successfully")

        except ValueError as e:
            # Approval rejected by user
            logger.error(f"Blog Processor: Content rejected during approval: {e}")
            return json.dumps(
                {"status": "error", "error": "approval_rejected", "message": str(e)}
            )
        except Exception as e:
            logger.error(f"Blog Processor: Pipeline execution failed: {e}")
            return json.dumps(
                {
                    "status": "error",
                    "error": "pipeline_failed",
                    "message": f"Content pipeline execution failed: {e}",
                }
            )

        # Step 3: Return results
        logger.info("Blog Processor: Processing complete")
        result = {
            "status": "success",
            "content_type": "blog_post",
            "pipeline_result": pipeline_result,
            "message": f"Blog post '{blog_post_model.title or 'Untitled'}' processed successfully",
        }

        return json.dumps(result, indent=2, default=_json_serializer)

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON input: {e}"
        logger.error(f"Blog Processor: {error_msg}")
        return json.dumps(
            {"status": "error", "error": "invalid_json", "message": error_msg}
        )
    except Exception as e:
        error_msg = f"Unexpected error during blog post processing: {e}"
        logger.error(f"Blog Processor: {error_msg}", exc_info=True)
        return json.dumps(
            {"status": "error", "error": "processing_exception", "message": error_msg}
        )
