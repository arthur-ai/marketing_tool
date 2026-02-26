"""
ARQ Worker Configuration.

This module defines the ARQ worker settings and background job functions
for processing marketing content asynchronously.

To run the worker:
    arq marketing_project.worker.WorkerSettings

Or with custom Redis:
    ARQ_REDIS_HOST=redis arq marketing_project.worker.WorkerSettings
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from arq import create_pool
from arq.connections import RedisSettings
from arq.cron import cron

from marketing_project.processors import (
    process_blog_post,
    process_release_notes,
    process_transcript,
)
from marketing_project.services.function_pipeline import FunctionPipeline
from marketing_project.services.function_pipeline.tracing import (
    add_job_metadata_to_span,
    close_span,
    create_job_root_span,
    create_span,
    is_tracing_available,
    record_span_exception,
    set_job_output,
    set_span_attribute,
    set_span_status,
)
from marketing_project.services.internal_docs_scanner import get_internal_docs_scanner
from marketing_project.services.job_manager import (
    JobMetadataKeys,
    JobStatus,
    get_job_manager,
)
from marketing_project.services.social_media_pipeline import SocialMediaPipeline

# Import Status for tracing
try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None

logger = logging.getLogger(__name__)


# ARQ Job Functions
async def process_blog_job(ctx, content_json: str, job_id: str, **kwargs) -> Dict:
    """
    Background job for processing blog posts.

    Args:
        ctx: ARQ context
        content_json: JSON string of blog content
        job_id: Job ID for tracking
        **kwargs: Additional ARQ-specific parameters (e.g., metadata, _timeout)

    Returns:
        Processing result dictionary
    """
    job_manager = get_job_manager()
    # Single get_job call shared by telemetry + metadata storage (Design 7)
    job = await job_manager.get_job(job_id, _skip_arq_poll=True)

    # Store input content in job metadata immediately (before processing starts)
    try:
        content_dict = json.loads(content_json)
        if job:
            job.metadata[JobMetadataKeys.INPUT_CONTENT] = content_dict
            if "title" in content_dict:
                job.metadata[JobMetadataKeys.TITLE] = content_dict["title"]
            await job_manager._save_job(job)
    except Exception as e:
        logger.warning(f"Failed to store input content for job {job_id}: {e}")

    # Create root job span for this job execution
    job_span = None
    if is_tracing_available():
        job_span = create_job_root_span(
            job_id=job_id, job_type="blog", input_value=content_json, job=job
        )

    try:
        logger.info(f"ARQ Worker: Processing blog job {job_id}")
        await job_manager.update_job_progress(job_id, 10, "Starting blog processing")

        # Process the blog post
        result_json = await process_blog_post(content_json, job_id=job_id)
        result_dict = json.loads(result_json)

        if result_dict.get("status") == "error":
            raise Exception(result_dict.get("message", "Processing failed"))

        if result_dict.get("status") == "waiting_for_approval":
            # Explicitly set status so get_job() doesn't need sentinel detection
            await job_manager.update_job_status(job_id, JobStatus.WAITING_FOR_APPROVAL)
            logger.info(f"ARQ Worker: Blog job {job_id} is waiting for approval")
        else:
            await job_manager.update_job_progress(job_id, 100, "Completed")
            logger.info(f"ARQ Worker: Blog job {job_id} completed successfully")

        if job_span:
            # Refresh job to get updated metadata
            updated_job = await job_manager.get_job(job_id)
            if updated_job:
                add_job_metadata_to_span(job_span, updated_job, job_id, "blog")
            set_job_output(job_span, result_dict)
            set_span_status(job_span, StatusCode.OK if StatusCode else None)

        return result_dict

    except Exception as e:
        logger.error(f"ARQ Worker: Blog job {job_id} failed: {e}")
        if job_span:
            set_span_attribute(job_span, "job.status", "failed")
            record_span_exception(job_span, e)
            set_span_status(job_span, StatusCode.ERROR if StatusCode else None, str(e))
        raise
    finally:
        close_span(job_span)


async def process_release_notes_job(
    ctx, content_json: str, job_id: str, **kwargs
) -> Dict:
    """
    Background job for processing release notes.

    Args:
        ctx: ARQ context
        content_json: JSON string of release notes content
        job_id: Job ID for tracking
        **kwargs: Additional ARQ-specific parameters (e.g., metadata, _timeout)

    Returns:
        Processing result dictionary
    """
    job_manager = get_job_manager()
    # Single get_job call shared by telemetry + metadata storage (Design 7)
    job = await job_manager.get_job(job_id, _skip_arq_poll=True)

    # Store input content in job metadata immediately (before processing starts)
    try:
        content_dict = json.loads(content_json)
        if job:
            job.metadata[JobMetadataKeys.INPUT_CONTENT] = content_dict
            if "title" in content_dict:
                job.metadata[JobMetadataKeys.TITLE] = content_dict["title"]
            await job_manager._save_job(job)
    except Exception as e:
        logger.warning(f"Failed to store input content for job {job_id}: {e}")

    # Create root job span for this job execution
    job_span = None
    if is_tracing_available():
        job_span = create_job_root_span(
            job_id=job_id, job_type="release_notes", input_value=content_json, job=job
        )

    try:
        logger.info(f"ARQ Worker: Processing release notes job {job_id}")
        await job_manager.update_job_progress(
            job_id, 10, "Starting release notes processing"
        )

        # Process the release notes
        result_json = await process_release_notes(content_json, job_id=job_id)
        result_dict = json.loads(result_json)

        if result_dict.get("status") == "error":
            raise Exception(result_dict.get("message", "Processing failed"))

        if result_dict.get("status") == "waiting_for_approval":
            await job_manager.update_job_status(job_id, JobStatus.WAITING_FOR_APPROVAL)
            logger.info(
                f"ARQ Worker: Release notes job {job_id} is waiting for approval"
            )
        else:
            await job_manager.update_job_progress(job_id, 100, "Completed")
            logger.info(
                f"ARQ Worker: Release notes job {job_id} completed successfully"
            )

        if job_span:
            # Refresh job to get updated metadata
            updated_job = await job_manager.get_job(job_id)
            if updated_job:
                add_job_metadata_to_span(job_span, updated_job, job_id, "release_notes")
            set_job_output(job_span, result_dict)
            set_span_status(job_span, StatusCode.OK if StatusCode else None)

        return result_dict

    except Exception as e:
        logger.error(f"ARQ Worker: Release notes job {job_id} failed: {e}")
        if job_span:
            set_span_attribute(job_span, "job.status", "failed")
            record_span_exception(job_span, e)
            set_span_status(job_span, StatusCode.ERROR if StatusCode else None, str(e))
        raise
    finally:
        close_span(job_span)


async def process_transcript_job(ctx, content_json: str, job_id: str, **kwargs) -> Dict:
    """
    Background job for processing transcripts.

    Args:
        ctx: ARQ context
        content_json: JSON string of transcript content
        job_id: Job ID for tracking
        **kwargs: Additional ARQ-specific parameters (e.g., metadata, _timeout)

    Returns:
        Processing result dictionary
    """
    job_manager = get_job_manager()
    # Single get_job call shared by telemetry + metadata storage + param extraction (Design 7)
    job = await job_manager.get_job(job_id, _skip_arq_poll=True)

    # Store input content in job metadata immediately (before processing starts)
    try:
        content_dict = json.loads(content_json)
        if job:
            job.metadata[JobMetadataKeys.INPUT_CONTENT] = content_dict
            if "title" in content_dict:
                job.metadata[JobMetadataKeys.TITLE] = content_dict["title"]
            await job_manager._save_job(job)
    except Exception as e:
        logger.warning(f"Failed to store input content for job {job_id}: {e}")

    # Create root job span for this job execution
    job_span = None
    if is_tracing_available():
        job_span = create_job_root_span(
            job_id=job_id, job_type="transcript", input_value=content_json, job=job
        )

    try:
        logger.info(f"ARQ Worker: Processing transcript job {job_id}")
        await job_manager.update_job_progress(
            job_id, 10, "Starting transcript processing"
        )

        # Extract output_content_type from kwargs (ARQ metadata) or job metadata
        # Use the already-loaded job object — no extra get_job call needed
        output_content_type = None
        try:
            if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
                output_content_type = kwargs["metadata"].get("output_content_type")
                if output_content_type:
                    logger.info(
                        f"ARQ Worker: Using output_content_type={output_content_type} from ARQ metadata"
                    )

            if not output_content_type and job:
                output_content_type = job.metadata.get("output_content_type")
                if output_content_type:
                    logger.info(
                        f"ARQ Worker: Using output_content_type={output_content_type} from job metadata"
                    )
        except Exception as e:
            logger.warning(
                f"ARQ Worker: Could not get output_content_type from metadata: {e}"
            )

        # Process the transcript (pass output_content_type explicitly)
        result_json = await process_transcript(
            content_json, job_id=job_id, output_content_type=output_content_type
        )
        result_dict = json.loads(result_json)

        if result_dict.get("status") == "error":
            raise Exception(result_dict.get("message", "Processing failed"))

        if result_dict.get("status") == "waiting_for_approval":
            await job_manager.update_job_status(job_id, JobStatus.WAITING_FOR_APPROVAL)
            logger.info(f"ARQ Worker: Transcript job {job_id} is waiting for approval")
        else:
            await job_manager.update_job_progress(job_id, 100, "Completed")
            logger.info(f"ARQ Worker: Transcript job {job_id} completed successfully")

        if job_span:
            # Refresh job to get updated metadata
            updated_job = await job_manager.get_job(job_id)
            if updated_job:
                add_job_metadata_to_span(job_span, updated_job, job_id, "transcript")
            set_job_output(job_span, result_dict)
            set_span_status(job_span, StatusCode.OK if StatusCode else None)

        return result_dict

    except Exception as e:
        logger.error(f"ARQ Worker: Transcript job {job_id} failed: {e}")
        if job_span:
            set_span_attribute(job_span, "job.status", "failed")
            record_span_exception(job_span, e)
            set_span_status(job_span, StatusCode.ERROR if StatusCode else None, str(e))
        raise
    finally:
        close_span(job_span)


async def process_social_media_job(
    ctx, content_json: str, job_id: str, **kwargs
) -> Dict:
    """
    Background job for processing social media posts from blog content.

    Args:
        ctx: ARQ context
        content_json: JSON string of blog content
        job_id: Job ID for tracking
        **kwargs: Additional ARQ-specific parameters (e.g., metadata, _timeout)

    Returns:
        Processing result dictionary
    """
    # Create root job span for this job execution
    job_span = None
    if is_tracing_available():
        # Get job to extract metadata
        job_manager = get_job_manager()
        job = await job_manager.get_job(job_id)

        job_span = create_job_root_span(
            job_id=job_id, job_type="social_media", input_value=content_json, job=job
        )

    try:
        logger.info(f"ARQ Worker: Processing social media job {job_id}")

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 10, "Starting social media pipeline"
        )

        # Store input content in job metadata
        try:
            content_dict = json.loads(content_json)
            job = await job_manager.get_job(job_id)
            if job:
                job.metadata[JobMetadataKeys.INPUT_CONTENT] = content_dict
                # Also extract and store title for easier access
                if "title" in content_dict:
                    job.metadata[JobMetadataKeys.TITLE] = content_dict["title"]
                await job_manager._save_job(job)
        except Exception as e:
            logger.warning(f"Failed to store input content for job {job_id}: {e}")

        # Get social media platform, email type, and variations count from job metadata
        social_media_platform = "linkedin"
        email_type = None
        variations_count = 1
        try:
            job = await job_manager.get_job(job_id)
            if job:
                social_media_platform = job.metadata.get(
                    "social_media_platform", "linkedin"
                )
                email_type = job.metadata.get("email_type")
                variations_count = job.metadata.get("variations_count", 1)
                logger.info(
                    f"ARQ Worker: Using social_media_platform={social_media_platform}, email_type={email_type}, variations_count={variations_count}"
                )
        except Exception as e:
            logger.warning(
                f"ARQ Worker: Could not get social media parameters from job metadata: {e}"
            )

        # Execute social media pipeline
        await job_manager.update_job_progress(
            job_id, 20, f"Running social media pipeline for {social_media_platform}"
        )

        # Get pipeline config from job metadata if available
        pipeline_config = None
        try:
            job = await job_manager.get_job(job_id)
            if job and job.metadata.get("pipeline_config"):
                from marketing_project.models.pipeline_steps import PipelineConfig

                pipeline_config = PipelineConfig(**job.metadata["pipeline_config"])
        except Exception as e:
            logger.warning(f"Could not load pipeline config from job metadata: {e}")

        # Create pipeline with config (no defaults - must be configured in settings)
        if pipeline_config:
            pipeline = SocialMediaPipeline(pipeline_config=pipeline_config)
        else:
            # Fallback: create with default config (should not happen if settings are configured)
            logger.warning("No pipeline config found in job metadata, using defaults")
            from marketing_project.models.pipeline_steps import PipelineConfig

            pipeline = SocialMediaPipeline(
                pipeline_config=PipelineConfig(
                    default_model="gpt-5.1",
                    default_temperature=0.7,
                    default_max_retries=2,
                    step_configs={},
                )
            )

        pipeline_result = await pipeline.execute_pipeline(
            content_json=content_json,
            job_id=job_id,
            content_type="blog_post",
            social_media_platform=social_media_platform,
            email_type=email_type,
            generate_variations=variations_count,
            pipeline_config=pipeline_config,
        )

        # Check if pipeline is waiting for approval
        if pipeline_result.get("pipeline_status") == "waiting_for_approval":
            logger.info(
                f"ARQ Worker: Social media job {job_id} waiting for approval at step {pipeline_result.get('metadata', {}).get('stopped_at_step')}"
            )
            # Return the result - job status is already updated by the pipeline
            return pipeline_result

        # Check if pipeline failed
        if pipeline_result.get("pipeline_status") == "failed":
            error_msg = pipeline_result.get("metadata", {}).get(
                "error", "Pipeline failed"
            )
            raise Exception(error_msg)

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(
            f"ARQ Worker: Social media job {job_id} completed successfully for platform {social_media_platform}"
        )

        if job_span:
            # Refresh job to get updated metadata
            updated_job = await job_manager.get_job(job_id)
            if updated_job:
                add_job_metadata_to_span(job_span, updated_job, job_id, "social_media")
            set_job_output(job_span, pipeline_result)
            set_span_status(job_span, StatusCode.OK if StatusCode else None)

        return pipeline_result

    except Exception as e:
        logger.error(f"ARQ Worker: Social media job {job_id} failed: {e}")
        if job_span:
            set_span_attribute(job_span, "job.status", "failed")
            record_span_exception(job_span, e)
            set_span_status(job_span, StatusCode.ERROR if StatusCode else None, str(e))
        raise
    finally:
        close_span(job_span)


async def process_multi_platform_social_media_job(
    ctx, content_json: str, job_id: str, **kwargs
) -> Dict:
    """
    Background job for processing social media posts for multiple platforms from blog content.

    Args:
        ctx: ARQ context
        content_json: JSON string of blog content
        job_id: Job ID for tracking
        **kwargs: Additional ARQ-specific parameters (e.g., metadata, _timeout)

    Returns:
        Processing result dictionary with results for each platform
    """
    # Create root job span for this job execution
    job_span = None
    if is_tracing_available():
        # Get job to extract metadata
        job_manager = get_job_manager()
        job = await job_manager.get_job(job_id)

        job_span = create_job_root_span(
            job_id=job_id,
            job_type="multi_platform_social_media",
            input_value=content_json,
            job=job,
        )

    try:
        logger.info(f"ARQ Worker: Processing multi-platform social media job {job_id}")

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 10, "Starting multi-platform social media pipeline"
        )

        # Store input content in job metadata
        try:
            content_dict = json.loads(content_json)
            job = await job_manager.get_job(job_id)
            if job:
                job.metadata[JobMetadataKeys.INPUT_CONTENT] = content_dict
                # Also extract and store title for easier access
                if "title" in content_dict:
                    job.metadata[JobMetadataKeys.TITLE] = content_dict["title"]
                await job_manager._save_job(job)
        except Exception as e:
            logger.warning(f"Failed to store input content for job {job_id}: {e}")

        # Get platforms, email type from job metadata
        platforms = ["linkedin"]
        email_type = None
        try:
            job = await job_manager.get_job(job_id)
            if job:
                # Check for social_media_platforms (list) first, fallback to single platform
                if "social_media_platforms" in job.metadata:
                    platforms = job.metadata["social_media_platforms"]
                elif "social_media_platform" in job.metadata:
                    platforms = [job.metadata["social_media_platform"]]
                email_type = job.metadata.get("email_type")
                logger.info(
                    f"ARQ Worker: Using platforms={platforms}, email_type={email_type}"
                )
        except Exception as e:
            logger.warning(
                f"ARQ Worker: Could not get social media parameters from job metadata: {e}"
            )

        # Validate platforms
        if not platforms or len(platforms) == 0:
            raise ValueError("At least one platform must be specified")
        if len(platforms) == 1:
            logger.warning(
                f"ARQ Worker: Only one platform specified ({platforms[0]}), consider using process_social_media_job instead"
            )

        # Execute multi-platform social media pipeline
        await job_manager.update_job_progress(
            job_id, 20, f"Running multi-platform pipeline for {', '.join(platforms)}"
        )

        # Get pipeline config from job metadata if available
        pipeline_config = None
        try:
            job = await job_manager.get_job(job_id)
            if job and job.metadata.get("pipeline_config"):
                from marketing_project.models.pipeline_steps import PipelineConfig

                pipeline_config = PipelineConfig(**job.metadata["pipeline_config"])
        except Exception as e:
            logger.warning(f"Could not load pipeline config from job metadata: {e}")

        # Create pipeline with config (no defaults - must be configured in settings)
        if pipeline_config:
            pipeline = SocialMediaPipeline(pipeline_config=pipeline_config)
        else:
            # Fallback: create with default config (should not happen if settings are configured)
            logger.warning("No pipeline config found in job metadata, using defaults")
            from marketing_project.models.pipeline_steps import PipelineConfig

            pipeline = SocialMediaPipeline(
                pipeline_config=PipelineConfig(
                    default_model="gpt-5.1",
                    default_temperature=0.7,
                    default_max_retries=2,
                    step_configs={},
                )
            )

        pipeline_result = await pipeline.execute_multi_platform_pipeline(
            content_json=content_json,
            platforms=platforms,
            job_id=job_id,
            content_type="blog_post",
            email_type=email_type,
        )

        # Check if pipeline failed
        if pipeline_result.get("pipeline_status") == "failed":
            error_msg = pipeline_result.get("metadata", {}).get(
                "error", "Pipeline failed"
            )
            raise Exception(error_msg)

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(
            f"ARQ Worker: Multi-platform social media job {job_id} completed successfully for platforms {', '.join(platforms)}"
        )

        if job_span:
            # Refresh job to get updated metadata
            updated_job = await job_manager.get_job(job_id)
            if updated_job:
                add_job_metadata_to_span(
                    job_span, updated_job, job_id, "multi_platform_social_media"
                )
            set_job_output(job_span, pipeline_result)
            set_span_status(job_span, StatusCode.OK if StatusCode else None)

        return pipeline_result

    except Exception as e:
        logger.error(
            f"ARQ Worker: Multi-platform social media job {job_id} failed: {e}"
        )
        if job_span:
            set_span_attribute(job_span, "job.status", "failed")
            record_span_exception(job_span, e)
            set_span_status(job_span, StatusCode.ERROR if StatusCode else None, str(e))
        raise
    finally:
        close_span(job_span)


async def analyze_brand_kit_batch_job(
    ctx,
    content_batch: List[Dict[str, Any]],
    batch_index: int,
    parent_job_id: str,
    batch_job_id: str,
    **kwargs,
) -> Dict:
    """
    Analyze a batch of content pieces for brand kit patterns.

    Args:
        ctx: ARQ context
        content_batch: List of content documents to analyze
        batch_index: Index of this batch (for logging)
        parent_job_id: Parent job ID
        batch_job_id: This batch job's ID

    Returns:
        Dictionary with batch analysis results
    """
    try:
        logger.info(
            f"ARQ Worker: Analyzing brand kit batch {batch_index + 1} (job {batch_job_id}, parent {parent_job_id})"
        )

        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            batch_job_id,
            0,
            f"Analyzing batch {batch_index + 1} ({len(content_batch)} pieces)",
        )

        import os

        from marketing_project.plugins.brand_kit.tasks import BrandKitPlugin
        from marketing_project.services.function_pipeline import FunctionPipeline

        pipeline = FunctionPipeline(
            model=os.getenv("OPENAI_MODEL", "gpt-5.1"), temperature=0.7
        )

        plugin = BrandKitPlugin()
        analyses = []

        for idx, content_doc in enumerate(content_batch):
            progress = int((idx / len(content_batch)) * 90)  # 0-90%
            await job_manager.update_job_progress(
                batch_job_id,
                progress,
                f"Batch {batch_index + 1}: Analyzing {idx + 1}/{len(content_batch)}",
            )

            analysis = await plugin._analyze_content(
                pipeline, content_doc, idx, len(content_batch)
            )
            if analysis:
                analyses.append(analysis)

        await job_manager.update_job_progress(
            batch_job_id, 100, f"Batch {batch_index + 1} completed"
        )

        result = {
            "status": "success",
            "batch_index": batch_index,
            "analyses": analyses,
            "count": len(analyses),
        }

        logger.info(
            f"ARQ Worker: Batch {batch_index + 1} completed with {len(analyses)} analyses"
        )
        return result

    except Exception as e:
        logger.error(f"ARQ Worker: Batch {batch_index + 1} failed: {e}")
        raise


async def synthesize_brand_kit_job(
    ctx,
    all_analyses: List[Dict[str, Any]],
    parent_job_id: str,
    synthesis_job_id: str,
    **kwargs,
) -> Dict:
    """
    Synthesize all content analyses into a final brand kit config.

    Args:
        ctx: ARQ context
        all_analyses: List of all analysis results from batches
        parent_job_id: Parent job ID
        synthesis_job_id: This synthesis job's ID

    Returns:
        Dictionary with synthesized brand kit config
    """
    try:
        logger.info(
            f"ARQ Worker: Synthesizing brand kit config (job {synthesis_job_id}, parent {parent_job_id})"
        )

        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            synthesis_job_id, 0, f"Synthesizing {len(all_analyses)} analyses"
        )

        import os

        from marketing_project.plugins.brand_kit.tasks import BrandKitPlugin
        from marketing_project.services.function_pipeline import FunctionPipeline

        pipeline = FunctionPipeline(
            model=os.getenv("OPENAI_MODEL", "gpt-5.1"), temperature=0.7
        )

        plugin = BrandKitPlugin()

        # Use the plugin's synthesis logic
        await job_manager.update_job_progress(
            synthesis_job_id, 50, "Calling AI to synthesize config"
        )

        # Flatten all analyses into a single list
        flat_analyses = []
        for batch_result in all_analyses:
            if isinstance(batch_result, dict) and "analyses" in batch_result:
                flat_analyses.extend(batch_result["analyses"])
            elif isinstance(batch_result, list):
                flat_analyses.extend(batch_result)
            else:
                flat_analyses.append(batch_result)

        # Generate config from analyses
        generated_config = await plugin._synthesize_config(
            pipeline, flat_analyses, synthesis_job_id
        )

        await job_manager.update_job_progress(
            synthesis_job_id, 100, "Synthesis completed"
        )

        result = {
            "status": "success",
            "config": generated_config.model_dump(mode="json"),
            "version": generated_config.version,
        }

        logger.info(f"ARQ Worker: Synthesis completed successfully")
        return result

    except Exception as e:
        logger.error(f"ARQ Worker: Synthesis failed: {e}")
        raise


async def refresh_brand_kit_job(
    ctx, use_internal_docs: bool, job_id: str, **kwargs
) -> Dict:
    """
    Background job for refreshing brand kit configuration using AI.

    This job orchestrates the process:
    1. Fetches all content from internal_docs
    2. Splits into batches of 20
    3. Creates parallel sub-jobs to analyze each batch
    4. Waits for all batches to complete
    5. Creates a synthesis job to combine all analyses
    6. Saves the final config

    Args:
        ctx: ARQ context
        use_internal_docs: Whether to enrich with internal docs configuration
        job_id: Job ID for tracking

    Returns:
        Dictionary with the generated config
    """
    try:
        logger.info(f"ARQ Worker: Refreshing brand kit configuration (job {job_id})")

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 5, "Starting brand kit generation"
        )

        # Import here to avoid circular imports
        import uuid

        from marketing_project.plugins.brand_kit.tasks import BrandKitPlugin
        from marketing_project.services.brand_kit_manager import get_brand_kit_manager
        from marketing_project.services.scanned_document_db import (
            get_scanned_document_db,
        )

        manager = await get_brand_kit_manager()

        content_batches = []
        batch_job_ids = []

        if use_internal_docs:
            await job_manager.update_job_progress(
                job_id, 10, "Fetching content from internal_docs"
            )
            logger.info(f"Design kit job {job_id}: Fetching content from internal_docs")

            try:
                db = get_scanned_document_db()
                all_docs = db.get_all_active_documents()

                logger.info(
                    f"Found {len(all_docs)} total active documents in internal_docs"
                )

                # Log detailed info about all documents for debugging
                if all_docs:
                    logger.info(f"Sample of first 3 documents structure:")
                    for i, doc in enumerate(all_docs[:3]):
                        logger.info(
                            f"  Doc {i+1}: title='{doc.title[:60]}', "
                            f"url='{doc.url[:60]}', "
                            f"has_metadata={doc.metadata is not None}, "
                            f"content_text_len={len(doc.metadata.content_text) if doc.metadata and doc.metadata.content_text else 0}, "
                            f"content_summary_len={len(doc.metadata.content_summary) if doc.metadata and doc.metadata.content_summary else 0}, "
                            f"headings_count={len(doc.metadata.headings) if doc.metadata and doc.metadata.headings else 0}, "
                            f"word_count={doc.metadata.word_count if doc.metadata else None}"
                        )

                # Filter to only content that has substantial text content
                # Accept documents with content_text OR content_summary (fallback)
                # Lowered threshold to 50 chars to be more inclusive
                content_docs = []
                filtered_reasons = {
                    "no_metadata": 0,
                    "no_content": 0,
                    "too_short": 0,
                    "accepted": 0,
                }

                for doc in all_docs:
                    # Check if metadata exists
                    if not doc.metadata:
                        filtered_reasons["no_metadata"] += 1
                        continue

                    # Check primary content_text field (lowered threshold to 50)
                    if (
                        doc.metadata.content_text
                        and len(doc.metadata.content_text.strip()) >= 50
                    ):
                        content_docs.append(doc)
                        filtered_reasons["accepted"] += 1
                        continue

                    # Fallback: use content_summary if content_text is missing/short
                    if (
                        doc.metadata.content_summary
                        and len(doc.metadata.content_summary.strip()) >= 50
                    ):
                        # Copy summary to content_text for analysis
                        doc.metadata.content_text = doc.metadata.content_summary
                        content_docs.append(doc)
                        filtered_reasons["accepted"] += 1
                        continue

                    # Fallback: use title + headings if available
                    if (
                        doc.title
                        and doc.metadata.headings
                        and len(doc.metadata.headings) > 0
                    ):
                        # Create minimal content from title and headings
                        minimal_content = f"{doc.title}\n\n" + "\n".join(
                            doc.metadata.headings[:10]
                        )
                        if len(minimal_content.strip()) >= 50:
                            doc.metadata.content_text = minimal_content
                            content_docs.append(doc)
                            filtered_reasons["accepted"] += 1
                            continue

                    # Fallback: use title alone if it's substantial
                    if doc.title and len(doc.title.strip()) >= 50:
                        doc.metadata.content_text = doc.title
                        content_docs.append(doc)
                        filtered_reasons["accepted"] += 1
                        continue

                    # Document was filtered out
                    if (
                        not doc.metadata.content_text
                        and not doc.metadata.content_summary
                    ):
                        filtered_reasons["no_content"] += 1
                    else:
                        filtered_reasons["too_short"] += 1

                # Log filtering statistics
                logger.info(
                    f"Content filtering results: {filtered_reasons['accepted']} accepted, "
                    f"{filtered_reasons['no_metadata']} no metadata, "
                    f"{filtered_reasons['no_content']} no content, "
                    f"{filtered_reasons['too_short']} too short"
                )

                if not content_docs and all_docs:
                    # Log detailed info about why documents were filtered out
                    logger.warning(
                        f"All {len(all_docs)} documents filtered out. Filtering breakdown: {filtered_reasons}"
                    )
                    # Show detailed info about first few documents
                    for i, doc in enumerate(all_docs[:5]):
                        logger.warning(
                            f"  Filtered doc {i+1}: title='{doc.title[:80]}', "
                            f"has_metadata={doc.metadata is not None}, "
                            f"content_text={bool(doc.metadata.content_text) if doc.metadata else False} "
                            f"(len={len(doc.metadata.content_text.strip()) if doc.metadata and doc.metadata.content_text else 0}), "
                            f"content_summary={bool(doc.metadata.content_summary) if doc.metadata else False} "
                            f"(len={len(doc.metadata.content_summary.strip()) if doc.metadata and doc.metadata.content_summary else 0}), "
                            f"headings={len(doc.metadata.headings) if doc.metadata and doc.metadata.headings else 0}"
                        )

                if content_docs:
                    # Split into batches of 20
                    BATCH_SIZE = 20
                    for i in range(0, len(content_docs), BATCH_SIZE):
                        batch = content_docs[i : i + BATCH_SIZE]
                        content_batches.append(
                            [doc.model_dump(mode="json") for doc in batch]
                        )

                    logger.info(
                        f"Split {len(content_docs)} content pieces into {len(content_batches)} batches "
                        f"of up to {BATCH_SIZE} pieces each"
                    )
                else:
                    logger.warning(
                        f"No content found in internal_docs (checked {len(all_docs)} documents)"
                    )
            except Exception as e:
                logger.warning(f"Error fetching content from internal_docs: {e}")

        if content_batches:
            # Create and submit batch analysis jobs
            await job_manager.update_job_progress(
                job_id, 15, f"Creating {len(content_batches)} analysis batches"
            )

            for batch_idx, batch in enumerate(content_batches):
                batch_job_id = str(uuid.uuid4())
                batch_job_ids.append(batch_job_id)

                # Create job for this batch
                batch_job = await job_manager.create_job(
                    job_type="brand_kit_batch_analysis",
                    content_id=f"batch_{batch_idx}",
                    metadata={
                        "parent_job_id": job_id,
                        "batch_index": batch_idx,
                        "batch_size": len(batch),
                    },
                    job_id=batch_job_id,
                )

                # Submit batch analysis job via JobManager to properly track arq_job_id
                arq_batch_job_id = await job_manager.submit_to_arq(
                    batch_job_id,
                    "analyze_brand_kit_batch_job",
                    batch,
                    batch_idx,
                    job_id,
                    batch_job_id,
                    _timeout=600,  # 10 minutes per batch
                )

                logger.info(
                    f"Submitted batch {batch_idx + 1}/{len(content_batches)} for analysis (job {batch_job_id})"
                )

            # Wait for all batch jobs to complete
            await job_manager.update_job_progress(
                job_id, 20, f"Waiting for {len(batch_job_ids)} batches to complete"
            )
            logger.info(f"Waiting for {len(batch_job_ids)} batch jobs to complete...")

            all_analyses = []
            completed_batches = 0

            # Poll for batch completion
            import asyncio

            max_wait_time = 600 * len(batch_job_ids)  # 10 min per batch
            start_time = asyncio.get_event_loop().time()

            while completed_batches < len(batch_job_ids):
                if asyncio.get_event_loop().time() - start_time > max_wait_time:
                    raise TimeoutError(f"Batch jobs timed out after {max_wait_time}s")

                for batch_job_id in batch_job_ids:
                    batch_job = await job_manager.get_job(batch_job_id)
                    if batch_job:
                        if batch_job.status.value == "completed":
                            if (
                                batch_job.result
                                and batch_job.result.get("status") == "success"
                            ):
                                all_analyses.append(batch_job.result)
                                completed_batches += 1
                                batch_job_ids.remove(batch_job_id)

                                progress = 20 + int(
                                    (completed_batches / len(content_batches)) * 50
                                )  # 20-70%
                                await job_manager.update_job_progress(
                                    job_id,
                                    progress,
                                    f"Completed {completed_batches}/{len(content_batches)} batches",
                                )
                                logger.info(
                                    f"Batch {completed_batches}/{len(content_batches)} completed"
                                )
                        elif batch_job.status.value == "failed":
                            logger.error(
                                f"Batch job {batch_job_id} failed: {batch_job.error}"
                            )
                            # Continue with other batches
                            batch_job_ids.remove(batch_job_id)
                            completed_batches += 1

                if completed_batches < len(content_batches):
                    await asyncio.sleep(2)  # Poll every 2 seconds

            logger.info(
                f"All {len(content_batches)} batches completed. Total analyses: {sum(len(a.get('analyses', [])) for a in all_analyses)}"
            )

            # Create synthesis job
            await job_manager.update_job_progress(
                job_id, 75, "Synthesizing all analyses"
            )
            synthesis_job_id = str(uuid.uuid4())

            synthesis_job = await job_manager.create_job(
                job_type="brand_kit_synthesis",
                content_id="synthesis",
                metadata={"parent_job_id": job_id, "analysis_count": len(all_analyses)},
                job_id=synthesis_job_id,
            )

            # Submit synthesis job via JobManager to properly track arq_job_id
            arq_synthesis_job_id = await job_manager.submit_to_arq(
                synthesis_job_id,
                "synthesize_brand_kit_job",
                all_analyses,
                job_id,
                synthesis_job_id,
                _timeout=600,  # 10 minutes for synthesis
            )

            logger.info(f"Submitted synthesis job {synthesis_job_id}")

            # Wait for synthesis to complete
            await job_manager.update_job_progress(
                job_id, 80, "Waiting for synthesis to complete"
            )

            synthesis_complete = False
            max_synthesis_wait = 600  # 10 minutes
            synthesis_start = asyncio.get_event_loop().time()

            while not synthesis_complete:
                if (
                    asyncio.get_event_loop().time() - synthesis_start
                    > max_synthesis_wait
                ):
                    raise TimeoutError("Synthesis job timed out")

                synth_job = await job_manager.get_job(synthesis_job_id)
                if synth_job:
                    if synth_job.status.value == "completed":
                        if (
                            synth_job.result
                            and synth_job.result.get("status") == "success"
                        ):
                            generated_config_dict = synth_job.result.get("config")
                            if generated_config_dict:
                                from marketing_project.models.brand_kit_config import (
                                    BrandKitConfig,
                                )

                                generated_config = BrandKitConfig(
                                    **generated_config_dict
                                )
                                synthesis_complete = True
                            else:
                                raise Exception(
                                    "Synthesis job completed but no config in result"
                                )
                        else:
                            raise Exception(f"Synthesis job failed: {synth_job.result}")
                    elif synth_job.status.value == "failed":
                        raise Exception(f"Synthesis job failed: {synth_job.error}")

                await asyncio.sleep(2)

            logger.info("Synthesis completed successfully")
        else:
            # No content batches - generate generic config
            await job_manager.update_job_progress(
                job_id, 30, "Generating generic config (no content found)"
            )
            logger.info("No content batches, generating generic config")
            generated_config = await manager.generate_config_with_ai(
                use_internal_docs=False, job_id=job_id
            )

        # Set metadata and enrich
        generated_config.version = "1.0.0"
        generated_config.created_at = datetime.now(timezone.utc)
        generated_config.updated_at = datetime.now(timezone.utc)
        generated_config.is_active = True

        if use_internal_docs:
            await job_manager.update_job_progress(
                job_id, 90, "Enriching with internal docs"
            )
            await manager._enrich_with_internal_docs(generated_config)

        await job_manager.update_job_progress(job_id, 95, "Saving configuration")

        # Save the generated config
        success = await manager.save_config(generated_config, set_active=True)

        if not success:
            raise Exception("Failed to save generated brand kit configuration")

        await job_manager.update_job_progress(job_id, 100, "Completed")

        # Return the config as dict
        result = {
            "status": "success",
            "config": generated_config.model_dump(mode="json"),
            "version": generated_config.version,
        }

        logger.info(
            f"ARQ Worker: Brand kit refresh job {job_id} completed successfully"
        )
        return result

    except Exception as e:
        logger.error(
            f"ARQ Worker: Brand kit refresh job {job_id} failed: {e}", exc_info=True
        )
        raise


async def resume_pipeline_job(
    ctx, original_job_id: str, context_data: Dict, job_id: str
) -> Dict:
    """
    Background job for resuming a pipeline after approval.

    Args:
        ctx: ARQ context
        original_job_id: ID of the original job that was waiting for approval
        context_data: Saved pipeline context from approval_manager
        job_id: New job ID for the resume job

    Returns:
        Processing result dictionary
    """
    # Create root job span for this job execution
    job_span = None
    if is_tracing_available():
        # Get job to extract metadata
        job_manager = get_job_manager()
        job = await job_manager.get_job(job_id)

        # Add original_job_id to metadata if not already present
        if job and job.metadata and "original_job_id" not in job.metadata:
            job.metadata[JobMetadataKeys.ORIGINAL_JOB_ID] = original_job_id

        job_span = create_job_root_span(
            job_id=job_id, job_type="resume_pipeline", input_value=context_data, job=job
        )

    try:
        logger.info(
            f"ARQ Worker: Resuming pipeline for original job {original_job_id}, new job {job_id}"
        )

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 10, "Resuming pipeline after approval"
        )

        # Store input content in job metadata if available
        input_content = context_data.get("input_content") or context_data.get(
            "original_content"
        )
        if input_content:
            job = await job_manager.get_job(job_id)
            if job:
                job.metadata[JobMetadataKeys.INPUT_CONTENT] = input_content
                # Also extract and store title for easier access
                if isinstance(input_content, dict) and "title" in input_content:
                    job.metadata[JobMetadataKeys.TITLE] = input_content["title"]
                await job_manager._save_job(job)

        # Import pipeline
        from marketing_project.services.function_pipeline import FunctionPipeline

        # Create pipeline and resume
        pipeline = FunctionPipeline(
            model=os.getenv("OPENAI_MODEL", "gpt-5.1"), temperature=0.7
        )

        # Determine content type from context
        content_type = context_data.get("content_type", "blog_post")

        # Resume pipeline
        result = await pipeline.resume_pipeline(
            context_data=context_data, job_id=job_id, content_type=content_type
        )

        pipeline_status = result.get("pipeline_status")

        if pipeline_status == "failed":
            raise Exception(result.get("error", "Resume pipeline failed"))

        if pipeline_status == "waiting_for_approval":
            # Pipeline stopped for approval - job is already marked as WAITING_FOR_APPROVAL
            # by resume_pipeline, just return the result
            logger.info(
                f"ARQ Worker: Resume job {job_id} stopped for approval at step "
                f"{result.get('metadata', {}).get('approval_step_name', 'unknown')}"
            )
            return {
                "status": "waiting_for_approval",
                "original_job_id": original_job_id,
                "resume_job_id": job_id,
                "result": result,
                "message": f"Pipeline waiting for approval at step {result.get('metadata', {}).get('approval_step_name', 'unknown')}",
            }

        # Pipeline completed successfully
        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(f"ARQ Worker: Resume job {job_id} completed successfully")

        if job_span:
            # Refresh job to get updated metadata
            updated_job = await job_manager.get_job(job_id)
            if updated_job:
                add_job_metadata_to_span(
                    job_span, updated_job, job_id, "resume_pipeline"
                )
            set_job_output(
                job_span,
                {
                    "status": "success",
                    "original_job_id": original_job_id,
                    "resume_job_id": job_id,
                    "result": result,
                },
            )
            set_span_status(job_span, StatusCode.OK if StatusCode else None)

        return {
            "status": "success",
            "original_job_id": original_job_id,
            "resume_job_id": job_id,
            "result": result,
            "message": "Pipeline resumed successfully",
        }

    except Exception as e:
        logger.error(f"ARQ Worker: Resume job {job_id} failed: {e}")
        if job_span:
            set_span_attribute(job_span, "job.status", "failed")
            record_span_exception(job_span, e)
            set_span_status(job_span, StatusCode.ERROR if StatusCode else None, str(e))
        raise
    finally:
        close_span(job_span)


async def retry_step_job(
    ctx,
    step_name: str,
    input_data: Dict,
    context: Dict,
    job_id: str,
    approval_id: str,
    user_guidance: Optional[str] = None,
) -> Dict:
    """
    Background job for retrying a single pipeline step.

    Args:
        ctx: ARQ context
        step_name: Name of the step to retry (e.g., "seo_keywords", "marketing_brief")
        input_data: Input data for the step
        context: Context from previous pipeline steps
        job_id: Job ID for tracking
        approval_id: Approval ID that triggered the retry
        user_guidance: Optional user guidance/feedback for regeneration

    Returns:
        Step execution result dictionary
    """
    # Create root job span for this job execution
    job_span = None
    if is_tracing_available():
        # Get job to extract metadata
        job_manager = get_job_manager()
        job = await job_manager.get_job(job_id)

        # Add retry-specific metadata if not already present
        if job and job.metadata:
            if "approval_id" not in job.metadata:
                job.metadata["approval_id"] = approval_id
            if "step_name" not in job.metadata:
                job.metadata["step_name"] = step_name
            if "has_user_guidance" not in job.metadata:
                job.metadata["has_user_guidance"] = bool(user_guidance)
            if user_guidance and "user_guidance_length" not in job.metadata:
                job.metadata["user_guidance_length"] = len(user_guidance)
            if context:
                content_type = context.get("content_type")
                if content_type and "content_type" not in job.metadata:
                    job.metadata["content_type"] = content_type

        # Prepare input value
        input_value = {
            "input_data": input_data,
            "context": context,
            "user_guidance": user_guidance,
        }

        job_span = create_job_root_span(
            job_id=job_id,
            job_type=f"retry_step_{step_name}",
            input_value=input_value,
            job=job,
        )

    try:
        logger.info(
            f"ARQ Worker: Retrying step '{step_name}' for job {job_id} (approval: {approval_id})"
        )
        if user_guidance:
            logger.info(
                f"ARQ Worker: Using user guidance for retry: {user_guidance[:100]}..."
            )

        # Import retry service
        from marketing_project.services.step_retry_service import get_retry_service

        # Update job manager - mark as processing and update progress
        job_manager = get_job_manager()
        await job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        await job_manager.update_job_progress(job_id, 10, f"Retrying step: {step_name}")

        # Get retry service and execute the step
        retry_service = get_retry_service()
        result = await retry_service.retry_step(
            step_name=step_name,
            input_data=input_data,
            context=context,
            job_id=job_id,
            user_guidance=user_guidance,
        )

        if job_span:
            set_span_attribute(
                job_span, "step_execution_status", result.get("status", "unknown")
            )
            if result.get("execution_time"):
                set_span_attribute(
                    job_span, "step_execution_time", result.get("execution_time")
                )

        if result["status"] == "error":
            error_msg = result.get("error_message", "Step retry failed")
            if job_span:
                set_span_status(
                    job_span, StatusCode.ERROR if StatusCode else None, error_msg
                )
                set_span_attribute(job_span, "error.type", "StepExecutionError")
            raise Exception(error_msg)

        # Get the original approval to retrieve job_id and other metadata
        from marketing_project.services.approval_manager import get_approval_manager
        from marketing_project.services.step_retry_service import STEP_NUMBER_MAP

        approval_manager = await get_approval_manager(reload_from_db=True)
        original_approval = await approval_manager.get_approval(approval_id)

        if not original_approval:
            logger.error(
                f"ARQ Worker: Original approval {approval_id} not found, cannot create new approval"
            )
            raise Exception(f"Original approval {approval_id} not found")

        # Get step number for this step
        step_number = STEP_NUMBER_MAP.get(step_name, 1)

        # Load existing pipeline context
        context_data = await approval_manager.load_pipeline_context(
            original_approval.job_id
        )
        if not context_data:
            logger.warning(
                f"ARQ Worker: No pipeline context found for job {original_approval.job_id}, "
                f"creating new context for rerun"
            )
            context_data = {
                "context": context,
                "last_step": step_name,
                "last_step_number": step_number,
                "step_result": result.get("result", {}),
                "original_content": input_data.get("content"),
                "content_type": context.get("content_type", "blog_post"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "job_id": original_approval.job_id,
            }
        else:
            # Update context with the new rerun result
            updated_context = context_data.get("context", {}).copy()
            updated_context[step_name] = result.get("result", {})

            # Update context_data with new result
            context_data["context"] = updated_context
            context_data["last_step"] = step_name
            context_data["last_step_number"] = step_number
            context_data["step_result"] = result.get("result", {})
            context_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Save updated pipeline context so pipeline can resume after approval
        await approval_manager.save_pipeline_context(
            job_id=original_approval.job_id,
            context=context_data["context"],
            step_name=step_name,
            step_number=step_number,
            step_result=result.get("result", {}),
            original_content=context_data.get("original_content"),
        )

        # Create a new approval request for the rerun result
        # Use the original job_id so it's associated with the same pipeline
        new_approval = await approval_manager.create_approval_request(
            job_id=original_approval.job_id,
            agent_name=step_name,
            step_name=step_name,
            input_data=input_data,
            output_data=result.get("result", {}),
            pipeline_step=step_name,
        )

        logger.info(
            f"ARQ Worker: Created new approval {new_approval.id} for rerun of step '{step_name}' "
            f"(original approval: {approval_id}, job: {original_approval.job_id}). "
            f"Pipeline context updated and saved for resume after approval."
        )

        await job_manager.update_job_progress(
            job_id, 100, f"Step '{step_name}' retry completed"
        )
        await job_manager.update_job_status(job_id, JobStatus.COMPLETED)
        logger.info(
            f"ARQ Worker: Step '{step_name}' retry completed successfully "
            f"for job {job_id} (approval: {approval_id}, new approval: {new_approval.id})"
        )

        if job_span:
            # Refresh job to get updated metadata
            updated_job = await job_manager.get_job(job_id)
            if updated_job:
                # Add new approval ID to metadata
                if updated_job.metadata:
                    updated_job.metadata["new_approval_id"] = new_approval.id
                add_job_metadata_to_span(
                    job_span, updated_job, job_id, f"retry_step_{step_name}"
                )
            set_span_status(
                job_span,
                StatusCode.OK if StatusCode else None,
                "Rerun completed successfully",
            )
            set_job_output(
                job_span,
                {
                    "status": "success",
                    "approval_id": approval_id,
                    "new_approval_id": new_approval.id,
                    "step_name": step_name,
                    "result": result,
                },
            )

        return {
            "status": "success",
            "approval_id": approval_id,
            "new_approval_id": new_approval.id,
            "step_name": step_name,
            "result": result,
            "message": f"Step '{step_name}' retried successfully",
        }

    except Exception as e:
        if job_span:
            record_span_exception(job_span, e)
            set_span_status(job_span, StatusCode.ERROR if StatusCode else None, str(e))
            set_span_attribute(job_span, "job.status", "failed")
            set_span_attribute(job_span, "error.type", type(e).__name__)
        logger.error(
            f"ARQ Worker: Step '{step_name}' retry failed for job {job_id}: {e}",
            exc_info=True,
        )
        job_manager = get_job_manager()
        await job_manager.update_job_status(job_id, JobStatus.FAILED)
        await job_manager.update_job_progress(job_id, 0, f"Step retry failed: {str(e)}")
        raise
    finally:
        close_span(job_span)


async def execute_single_step_job(
    ctx,
    step_name: str,
    content_json: str,
    context: Dict[str, Any],
    job_id: str,
    **kwargs,
) -> Dict:
    """
    Background job for executing a single pipeline step independently.

    Args:
        ctx: ARQ context
        step_name: Name of the step to execute (e.g., "seo_keywords")
        content_json: JSON string of input content
        context: Dictionary containing all required context keys for the step
        job_id: Job ID for tracking
        **kwargs: Additional ARQ-specific parameters

    Returns:
        Step execution result dictionary
    """
    # Create root job span for this job execution
    job_span = None
    if is_tracing_available():
        # Get job to extract metadata
        job_manager = get_job_manager()
        job = await job_manager.get_job(job_id)

        # Prepare input value (content_json + context)
        input_data = {"content": content_json, "context": context}

        job_span = create_job_root_span(
            job_id=job_id, job_type=f"step_{step_name}", input_value=input_data, job=job
        )

    try:
        logger.info(f"ARQ Worker: Executing single step '{step_name}' for job {job_id}")

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        await job_manager.update_job_progress(
            job_id, 10, f"Executing step: {step_name}"
        )

        # Create pipeline instance
        pipeline = FunctionPipeline()

        # Execute the step
        result = await pipeline.execute_single_step(
            step_name=step_name,
            content_json=content_json,
            context=context,
            job_id=job_id,
        )

        # Update job with result
        await job_manager.update_job_progress(
            job_id, 90, f"Step '{step_name}' completed"
        )
        await job_manager.update_job_status(job_id, JobStatus.COMPLETED)

        # Store result in job
        job = await job_manager.get_job(job_id)
        if job:
            job.result = result
            job.current_step = step_name
            await job_manager._save_job(job)

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(
            f"ARQ Worker: Single step '{step_name}' execution completed successfully for job {job_id}"
        )

        if job_span:
            # Refresh job to get updated metadata
            updated_job = await job_manager.get_job(job_id)
            if updated_job:
                add_job_metadata_to_span(
                    job_span, updated_job, job_id, f"step_{step_name}"
                )
            set_job_output(
                job_span,
                {
                    "status": "success",
                    "step_name": step_name,
                    "result": result,
                },
            )
            set_span_status(job_span, StatusCode.OK if StatusCode else None)

        return {
            "status": "success",
            "step_name": step_name,
            "result": result,
            "message": f"Step '{step_name}' executed successfully",
        }

    except Exception as e:
        logger.error(
            f"ARQ Worker: Single step '{step_name}' execution failed for job {job_id}: {e}"
        )
        if job_span:
            set_span_attribute(job_span, "job.status", "failed")
            record_span_exception(job_span, e)
            set_span_status(job_span, StatusCode.ERROR if StatusCode else None, str(e))
        job_manager = get_job_manager()
        await job_manager.update_job_status(job_id, JobStatus.FAILED)
        await job_manager.update_job_progress(
            job_id, 0, f"Step execution failed: {str(e)}"
        )
        job = await job_manager.get_job(job_id)
        if job:
            job.error = str(e)
            await job_manager._save_job(job)
        raise
    finally:
        close_span(job_span)


async def bulk_rescan_documents_job(ctx, urls: List[str], job_id: str) -> Dict:
    """
    Background job for bulk re-scanning documents.

    Args:
        ctx: ARQ context
        urls: List of URLs to re-scan
        job_id: Job ID for tracking

    Returns:
        Processing result dictionary
    """
    try:
        logger.info(
            f"ARQ Worker: Bulk re-scanning {len(urls)} documents for job {job_id}"
        )

        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 0, f"Starting bulk re-scan of {len(urls)} documents"
        )

        scanner = await get_internal_docs_scanner()
        scanned_count = 0
        failed_count = 0
        errors = []

        for idx, url in enumerate(urls):
            try:
                await scanner._scan_single_url(url, save_to_db=True)
                scanned_count += 1
                progress = int((idx + 1) / len(urls) * 100)
                await job_manager.update_job_progress(
                    job_id, progress, f"Scanned {idx + 1}/{len(urls)} documents"
                )
            except Exception as e:
                failed_count += 1
                errors.append(f"{url}: {str(e)}")
                logger.warning(f"Failed to scan {url}: {e}")

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(
            f"ARQ Worker: Bulk re-scan job {job_id} completed: {scanned_count} succeeded, {failed_count} failed"
        )

        return {
            "status": "success",
            "scanned_count": scanned_count,
            "failed_count": failed_count,
            "errors": errors[:20],  # Limit errors
        }

    except Exception as e:
        logger.error(f"ARQ Worker: Bulk re-scan job {job_id} failed: {e}")
        raise


async def scan_from_url_job(
    ctx,
    base_url: str,
    max_depth: int,
    follow_external: bool,
    max_pages: int,
    merge_with_existing: bool,
    job_id: str,
) -> Dict:
    """
    Background job for scanning documents from a base URL.

    Args:
        ctx: ARQ context
        base_url: Base URL to start crawling from
        max_depth: Maximum crawl depth
        follow_external: Whether to follow external links
        max_pages: Maximum number of pages to scan
        merge_with_existing: Whether to merge with existing config
        job_id: Job ID for tracking

    Returns:
        Processing result dictionary
    """
    try:
        logger.info(f"ARQ Worker: Scanning from base URL {base_url} for job {job_id}")

        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 0, f"Starting scan from {base_url}"
        )

        scanner = await get_internal_docs_scanner()
        scanned_docs = await scanner.scan_from_base_url(
            base_url=base_url,
            max_depth=max_depth,
            follow_external=follow_external,
            max_pages=max_pages,
        )

        await job_manager.update_job_progress(
            job_id, 50, f"Scanned {len(scanned_docs)} documents, merging with config..."
        )

        if merge_with_existing:
            from datetime import datetime

            from marketing_project.models.internal_docs_config import InternalDocsConfig
            from marketing_project.services.internal_docs_manager import (
                get_internal_docs_manager,
            )

            try:
                manager = await get_internal_docs_manager()
                config = await manager.get_active_config()

                if config:
                    # Update existing config
                    logger.info(
                        f"Updating existing config with {len(scanned_docs)} scanned documents"
                    )
                    config = await manager.merge_scan_results(config, scanned_docs)
                    success = await manager.save_config(config, set_active=True)
                    if not success:
                        logger.error(f"Failed to save updated config for job {job_id}")
                        raise Exception("Failed to save updated config")
                    logger.info(
                        f"Successfully updated config with {len(config.scanned_documents)} documents"
                    )
                else:
                    # Create new config if none exists (save_config will handle versioning)
                    logger.info(
                        f"Creating new config with {len(scanned_docs)} scanned documents"
                    )
                    config = InternalDocsConfig(
                        scanned_documents=scanned_docs,
                        version="1.0.0",
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    success = await manager.save_config(config, set_active=True)
                    if not success:
                        logger.error(f"Failed to save new config for job {job_id}")
                        raise Exception("Failed to save new config")
                    logger.info(
                        f"Successfully created new config with {len(scanned_docs)} documents"
                    )
            except Exception as e:
                logger.error(
                    f"Error saving config for job {job_id}: {e}", exc_info=True
                )
                raise
        else:
            logger.info(
                f"merge_with_existing=False, skipping config save for job {job_id}"
            )

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(
            f"ARQ Worker: Scan from URL job {job_id} completed: {len(scanned_docs)} documents scanned"
        )

        return {
            "status": "success",
            "message": f"Scanned {len(scanned_docs)} documents from {base_url}",
            "scanned_count": len(scanned_docs),
            "scanned_documents": [doc.model_dump(mode="json") for doc in scanned_docs],
        }

    except Exception as e:
        logger.error(f"ARQ Worker: Scan from URL job {job_id} failed: {e}")
        raise


async def scan_from_list_job(
    ctx, urls: List[str], merge_with_existing: bool, job_id: str
) -> Dict:
    """
    Background job for scanning documents from a list of URLs.

    Args:
        ctx: ARQ context
        urls: List of URLs to scan
        merge_with_existing: Whether to merge with existing config
        job_id: Job ID for tracking

    Returns:
        Processing result dictionary
    """
    try:
        logger.info(
            f"ARQ Worker: Scanning from URL list ({len(urls)} URLs) for job {job_id}"
        )

        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 0, f"Starting scan from {len(urls)} URLs"
        )

        scanner = await get_internal_docs_scanner()
        scanned_docs = await scanner.scan_from_url_list(urls=urls)

        await job_manager.update_job_progress(
            job_id, 50, f"Scanned {len(scanned_docs)} documents, merging with config..."
        )

        if merge_with_existing:
            from datetime import datetime

            from marketing_project.models.internal_docs_config import InternalDocsConfig
            from marketing_project.services.internal_docs_manager import (
                get_internal_docs_manager,
            )

            try:
                manager = await get_internal_docs_manager()
                config = await manager.get_active_config()

                if config:
                    # Update existing config
                    logger.info(
                        f"Updating existing config with {len(scanned_docs)} scanned documents"
                    )
                    config = await manager.merge_scan_results(config, scanned_docs)
                    success = await manager.save_config(config, set_active=True)
                    if not success:
                        logger.error(f"Failed to save updated config for job {job_id}")
                        raise Exception("Failed to save updated config")
                    logger.info(
                        f"Successfully updated config with {len(config.scanned_documents)} documents"
                    )
                else:
                    # Create new config if none exists (save_config will handle versioning)
                    logger.info(
                        f"Creating new config with {len(scanned_docs)} scanned documents"
                    )
                    config = InternalDocsConfig(
                        scanned_documents=scanned_docs,
                        version="1.0.0",
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    success = await manager.save_config(config, set_active=True)
                    if not success:
                        logger.error(f"Failed to save new config for job {job_id}")
                        raise Exception("Failed to save new config")
                    logger.info(
                        f"Successfully created new config with {len(scanned_docs)} documents"
                    )
            except Exception as e:
                logger.error(
                    f"Error saving config for job {job_id}: {e}", exc_info=True
                )
                raise
        else:
            logger.info(
                f"merge_with_existing=False, skipping config save for job {job_id}"
            )

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(
            f"ARQ Worker: Scan from list job {job_id} completed: {len(scanned_docs)} documents scanned"
        )

        return {
            "status": "success",
            "message": f"Scanned {len(scanned_docs)} documents from {len(urls)} URLs",
            "scanned_count": len(scanned_docs),
            "scanned_documents": [doc.model_dump(mode="json") for doc in scanned_docs],
        }

    except Exception as e:
        logger.error(f"ARQ Worker: Scan from list job {job_id} failed: {e}")
        raise


# ARQ Worker startup/shutdown hooks
async def startup(ctx):
    """Called when worker starts."""
    logger.info("ARQ Worker starting up...")
    logger.info(f"Redis connection: {ctx['redis']}")

    # Initialize database connection
    try:
        # Import models to register them with SQLAlchemy Base
        from marketing_project.models import db_models  # noqa: F401
        from marketing_project.services.database import get_database_manager

        db_manager = get_database_manager()
        if await db_manager.initialize():
            # Create tables if they don't exist
            await db_manager.create_tables()
            logger.info("✓ Database connection initialized and tables created")
        else:
            logger.warning(
                "⚠ Database not configured (DATABASE_URL or POSTGRES_URL not set). Configuration persistence will be disabled."
            )
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize database connection: {e}")

    # Initialize telemetry
    try:
        import os as os_module
        import socket

        from marketing_project.services.telemetry import setup_tracing

        # Generate unique worker instance identifier
        # Use worker name + hostname + process ID to ensure uniqueness across all worker instances
        worker_name = WorkerSettings.name  # "marketing-worker"
        hostname = socket.gethostname()
        pid = os_module.getpid()
        worker_instance_id = f"{worker_name}-{hostname}-{pid}"

        if setup_tracing(service_instance_id=worker_instance_id):
            logger.info(
                f"✓ Telemetry initialized successfully (instance: {worker_instance_id})"
            )
        else:
            logger.info(
                "⚠ Telemetry not configured (missing ARTHUR_API_KEY or ARTHUR_TASK_ID)"
            )
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize telemetry: {e}")


async def shutdown(ctx):
    """Called when worker shuts down."""
    logger.info("ARQ Worker shutting down...")

    # Clean up database connection
    try:
        from marketing_project.services.database import get_database_manager

        db_manager = get_database_manager()
        if db_manager.is_initialized:
            await db_manager.cleanup()
            logger.info("✓ Database connection cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up database connection: {e}")


async def expire_stale_approvals(ctx):
    """
    Auto-cancel waiting_for_approval jobs older than 7 days.

    For each stale job:
    - PostgreSQL: sets job status to 'cancelled'
    - Approvals: rejects pending approvals AND creates retry jobs (same as user-initiated rejection)
    - Redis: deletes saved pipeline contexts
    """
    from datetime import timedelta, timezone

    from sqlalchemy import select, update

    from marketing_project.models.approval_models import ApprovalDecisionRequest
    from marketing_project.models.db_models import JobModel
    from marketing_project.services.approval_manager import get_approval_manager
    from marketing_project.services.database import get_database_manager

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    db_manager = get_database_manager()
    if not db_manager.is_initialized:
        logger.warning("expire_stale_approvals: database not initialized, skipping")
        return

    # Step 1: find stale job IDs before updating their status
    stale_job_ids: list[str] = []
    async with db_manager.get_session() as session:
        select_stmt = select(JobModel.job_id).where(
            JobModel.status == "waiting_for_approval",
            JobModel.created_at < cutoff,
        )
        result = await session.execute(select_stmt)
        stale_job_ids = [row[0] for row in result.fetchall()]

        if stale_job_ids:
            update_stmt = (
                update(JobModel)
                .where(JobModel.status == "waiting_for_approval")
                .where(JobModel.created_at < cutoff)
                .values(status="cancelled", completed_at=datetime.now(timezone.utc))
            )
            update_result = await session.execute(update_stmt)
            logger.info(
                f"Auto-cancelled {update_result.rowcount} stale waiting_for_approval jobs"
            )

    if not stale_job_ids:
        return

    # Step 2: reject pending approvals (with retry job creation) + delete pipeline contexts
    try:
        approval_manager = await get_approval_manager()
        job_manager = get_job_manager()
        rejected = 0
        retried = 0
        contexts_deleted = 0

        for job_id in stale_job_ids:
            # Reject pending approvals, creating retry jobs (consistent with user rejection)
            try:
                pending = await approval_manager.list_approvals(
                    job_id=job_id, status="pending"
                )
                for approval in pending:
                    try:
                        # First mark the approval as rejected in PostgreSQL/Redis
                        await approval_manager.decide_approval(
                            approval.id,
                            ApprovalDecisionRequest(
                                decision="reject",
                                comment="Job expired (older than 7 days)",
                                reviewed_by="system",
                            ),
                        )
                        rejected += 1
                        # Then create a retry job so the pipeline can be re-attempted
                        retry_job_id = (
                            await approval_manager.execute_rejection_with_retry(
                                approval=approval,
                                job_manager=job_manager,
                                user_comment="Auto-expired after 7 days — retrying",
                                reviewed_by="system",
                            )
                        )
                        if retry_job_id:
                            retried += 1
                    except Exception as e:
                        logger.warning(
                            f"expire_stale_approvals: could not reject/retry approval "
                            f"{approval.id} for job {job_id}: {e}"
                        )
            except Exception as e:
                logger.warning(
                    f"expire_stale_approvals: could not list approvals for {job_id}: {e}"
                )

            # Delete saved pipeline context to free Redis memory
            try:
                import redis.asyncio as _redis

                from marketing_project.services.redis_manager import get_redis_manager

                redis_mgr = get_redis_manager()

                async def _del_context(client: _redis.Redis):
                    await client.delete(f"pipeline:context:{job_id}")

                await redis_mgr.execute(_del_context)
                contexts_deleted += 1
            except Exception as e:
                logger.warning(
                    f"expire_stale_approvals: could not delete pipeline context "
                    f"for {job_id}: {e}"
                )

        logger.info(
            f"expire_stale_approvals: rejected {rejected} approvals, "
            f"created {retried} retry jobs, "
            f"deleted {contexts_deleted} pipeline contexts for "
            f"{len(stale_job_ids)} stale jobs"
        )
    except Exception as e:
        logger.error(f"expire_stale_approvals: cleanup failed: {e}")


# ARQ Worker Settings
class WorkerSettings:
    """
    ARQ Worker configuration.

    NOTE: ARQ manages its own Redis connections and cannot share the RedisManager
    connection pool. ARQ requires RedisSettings for its internal connection management.
    However, we use the same environment variables to ensure consistency.

    Environment Variables:
        REDIS_HOST: Redis host (default: localhost)
        REDIS_PORT: Redis port (default: 6379)
        REDIS_DATABASE: Redis database number (default: 0)
        REDIS_PASSWORD: Redis password (optional)
        ARQ_MAX_JOBS: Maximum concurrent jobs (default: 10)
        ARQ_JOB_TIMEOUT: Job timeout in seconds (default: 1800 = 30 minutes)
    """

    # Redis connection settings
    # ARQ uses its own connection management, but we use the same env vars for consistency
    # SSL is not supported
    redis_settings = RedisSettings(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        database=int(os.getenv("REDIS_DATABASE", "0")),
        password=os.getenv("REDIS_PASSWORD"),
    )

    # Job functions to register
    # ARQ will use the function names automatically
    functions = [
        process_blog_job,
        process_release_notes_job,
        process_transcript_job,
        process_social_media_job,
        process_multi_platform_social_media_job,
        resume_pipeline_job,
        retry_step_job,
        execute_single_step_job,
        bulk_rescan_documents_job,
        scan_from_url_job,
        scan_from_list_job,
        refresh_brand_kit_job,
        analyze_brand_kit_batch_job,
        synthesize_brand_kit_job,
    ]

    # Worker settings
    max_jobs = int(os.getenv("ARQ_MAX_JOBS", "10"))  # Max concurrent jobs
    # Default timeout: 30 minutes (1800s) to support long-running brand kit jobs
    # Brand kit jobs can take 20-30 minutes due to multiple batch analysis jobs + synthesis
    job_timeout = int(os.getenv("ARQ_JOB_TIMEOUT", "1800"))  # 30 minutes default
    keep_result = 3600  # Keep results for 1 hour
    keep_result_forever = False

    # Cron jobs
    cron_jobs = [
        cron(expire_stale_approvals, hour={0}, minute={0}),  # daily at midnight UTC
    ]

    # Startup and shutdown hooks
    on_startup = startup
    on_shutdown = shutdown

    # Worker name
    name = "marketing-worker"

    # Health check interval (seconds)
    health_check_interval = 60

    # Log all jobs
    log_results = True


# Helper function to get ARQ pool (for API to enqueue jobs)
async def get_arq_pool():
    """
    Get ARQ Redis pool for enqueueing jobs.

    NOTE: This creates a separate ARQ-managed connection pool. ARQ cannot use
    the RedisManager connection pool because ARQ requires its own connection
    management for job queuing, result storage, and worker coordination.

    Both ARQ and RedisManager use the same environment variables for consistency.
    """
    redis_settings = WorkerSettings.redis_settings
    return await create_pool(redis_settings)
