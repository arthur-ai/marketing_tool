"""
Direct deterministic processor endpoints for content processing.

These endpoints bypass the orchestrator agent and directly call deterministic
processors for known content types. This provides:
1. Faster processing (no LLM routing overhead)
2. Predictable behavior (fixed workflow)
3. Better testability (deterministic paths)
4. Lower costs (fewer LLM calls)

Use these endpoints when you know the content type in advance.
For auto-routing based on content analysis, use the /pipeline endpoint instead.

NOTE: These endpoints now return job IDs immediately for async processing.
Poll GET /api/v1/jobs/{job_id}/status to check progress.
"""

import json
import logging
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..models import (
    BlogProcessorRequest,
    ReleaseNotesProcessorRequest,
    TranscriptProcessorRequest,
)
from ..processors import process_blog_post, process_release_notes, process_transcript
from ..services.job_manager import get_job_manager

logger = logging.getLogger("marketing_project.api.processors")

# Create router
router = APIRouter()


class JobSubmissionResponse(BaseModel):
    """Response model for job submission."""

    success: bool = True
    message: str
    job_id: str
    content_id: str
    status_url: str


async def _process_blog_post_job(content_json: str, job_id: str) -> Dict:
    """Background task for processing blog posts."""
    try:
        result_json = await process_blog_post(content_json, job_id=job_id)
        result_dict = json.loads(result_json)

        if result_dict.get("status") == "error":
            raise Exception(result_dict.get("message", "Processing failed"))

        return result_dict
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse processor result: {e}")


@router.post("/process/blog", response_model=JobSubmissionResponse)
async def process_blog_endpoint(request: BlogProcessorRequest):
    """
    Submit blog post for processing (returns job ID immediately).

    This endpoint submits the blog post for background processing and returns
    a job ID immediately. The processing follows a deterministic workflow:
    1. Parse and validate input
    2. Analyze blog post type (tutorial, article, guide, etc.)
    3. Validate structure (title, content, metadata)
    4. Extract metadata (author, category, tags, word count)
    5. Run through 8-step content pipeline
    6. Return processed results

    Poll GET /api/v1/jobs/{job_id}/status to check progress and get results.
    """
    try:
        logger.info(f"Blog processor request for content ID: {request.content.id}")

        # Validate social media parameters if output_content_type is social_media_post
        output_content_type = request.output_content_type or "blog_post"
        if output_content_type == "social_media_post":
            if not request.social_media_platform:
                raise HTTPException(
                    status_code=400,
                    detail="social_media_platform is required when output_content_type is 'social_media_post'",
                )
            # Validate platform value
            valid_platforms = ["linkedin", "hackernews", "email"]
            if request.social_media_platform not in valid_platforms:
                raise HTTPException(
                    status_code=400,
                    detail=f"social_media_platform must be one of {valid_platforms}, got '{request.social_media_platform}'",
                )
            # Validate email_type if platform is email
            if request.social_media_platform == "email":
                if request.email_type:
                    valid_email_types = ["newsletter", "promotional"]
                    if request.email_type not in valid_email_types:
                        raise HTTPException(
                            status_code=400,
                            detail=f"email_type must be one of {valid_email_types}, got '{request.email_type}'",
                        )

        # Get job manager
        job_manager = get_job_manager()

        # Create job metadata
        job_metadata = {
            "content_type": "blog_post",
            "output_content_type": output_content_type,
        }

        # Add social media parameters if applicable
        if output_content_type == "social_media_post":
            job_metadata["social_media_platform"] = request.social_media_platform
            if request.email_type:
                job_metadata["email_type"] = request.email_type

        job = await job_manager.create_job(
            job_type="blog_post",
            content_id=request.content.id,
            metadata=job_metadata,
        )

        # Convert Pydantic model to JSON string for processor
        content_json = request.content.model_dump_json()

        # Determine which ARQ job function to use
        if output_content_type == "social_media_post":
            arq_function_name = "process_social_media_job"
        else:
            arq_function_name = "process_blog_job"

        # Submit job to ARQ for background processing
        arq_job_id = await job_manager.submit_to_arq(
            job.id,
            arq_function_name,  # ARQ function name
            content_json,
            job.id,
            metadata=job_metadata,
        )

        logger.info(
            f"Blog processor job {job.id} submitted to ARQ (arq_id: {arq_job_id}) for content {request.content.id}"
        )

        return JobSubmissionResponse(
            message="Blog post submitted for processing",
            job_id=job.id,
            content_id=request.content.id,
            status_url=f"/api/v1/jobs/{job.id}/status",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to submit blog processor job for {request.content.id}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


async def _process_release_notes_job(content_json: str, job_id: str) -> Dict:
    """Background task for processing release notes."""
    try:
        result_json = await process_release_notes(content_json, job_id=job_id)
        result_dict = json.loads(result_json)

        if result_dict.get("status") == "error":
            raise Exception(result_dict.get("message", "Processing failed"))

        return result_dict
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse processor result: {e}")


@router.post("/process/release-notes", response_model=JobSubmissionResponse)
async def process_release_notes_endpoint(request: ReleaseNotesProcessorRequest):
    """
    Submit release notes for processing (returns job ID immediately).

    This endpoint submits release notes for background processing and returns
    a job ID immediately. Poll GET /api/v1/jobs/{job_id}/status to check progress.
    """
    try:
        logger.info(
            f"Release notes processor request for content ID: {request.content.id}"
        )

        # Get job manager
        job_manager = get_job_manager()

        # Create job
        output_content_type = request.output_content_type or "blog_post"
        job = await job_manager.create_job(
            job_type="release_notes",
            content_id=request.content.id,
            metadata={
                "content_type": "release_notes",
                "output_content_type": output_content_type,
            },
        )

        # Convert Pydantic model to JSON string for processor
        content_json = request.content.model_dump_json()

        # Submit job to ARQ for background processing (pass output_content_type in metadata)
        arq_job_id = await job_manager.submit_to_arq(
            job.id,
            "process_release_notes_job",  # ARQ function name
            content_json,
            job.id,
            metadata={"output_content_type": output_content_type},
        )

        logger.info(
            f"Release notes processor job {job.id} submitted to ARQ (arq_id: {arq_job_id}) for content {request.content.id}"
        )

        return JobSubmissionResponse(
            message="Release notes submitted for processing",
            job_id=job.id,
            content_id=request.content.id,
            status_url=f"/api/v1/jobs/{job.id}/status",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to submit release notes processor job for {request.content.id}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


async def _process_transcript_job(content_json: str, job_id: str) -> Dict:
    """Background task for processing transcripts."""
    try:
        result_json = await process_transcript(content_json, job_id=job_id)
        result_dict = json.loads(result_json)

        if result_dict.get("status") == "error":
            raise Exception(result_dict.get("message", "Processing failed"))

        return result_dict
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse processor result: {e}")


@router.post("/process/transcript", response_model=JobSubmissionResponse)
async def process_transcript_endpoint(request: TranscriptProcessorRequest):
    """
    Submit transcript for processing (returns job ID immediately).

    This endpoint submits the transcript for background processing and returns
    a job ID immediately. Poll GET /api/v1/jobs/{job_id}/status to check progress.
    """
    try:
        logger.info(
            f"Transcript processor request for content ID: {request.content.id}"
        )

        # Get job manager
        job_manager = get_job_manager()

        # Create job
        output_content_type = request.output_content_type or "blog_post"
        job = await job_manager.create_job(
            job_type="transcript",
            content_id=request.content.id,
            metadata={
                "content_type": "transcript",
                "output_content_type": output_content_type,
            },
        )

        # Convert Pydantic model to JSON string for processor
        content_json = request.content.model_dump_json()

        # Submit job to ARQ for background processing (pass output_content_type in metadata)
        arq_job_id = await job_manager.submit_to_arq(
            job.id,
            "process_transcript_job",  # ARQ function name
            content_json,
            job.id,
            metadata={"output_content_type": output_content_type},
        )

        logger.info(
            f"Transcript processor job {job.id} submitted to ARQ (arq_id: {arq_job_id}) for content {request.content.id}"
        )

        return JobSubmissionResponse(
            message="Transcript submitted for processing",
            job_id=job.id,
            content_id=request.content.id,
            status_url=f"/api/v1/jobs/{job.id}/status",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to submit transcript processor job for {request.content.id}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")
