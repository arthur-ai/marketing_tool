"""
Job Management API Endpoints.

Endpoints for managing and tracking background job execution.
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..models.approval_models import ApprovalDecisionRequest
from ..services.approval_manager import get_approval_manager
from ..services.job_manager import Job, JobManager, JobStatus, get_job_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class JobResponse(BaseModel):
    """Response model for job details."""

    success: bool = True
    message: str
    job: Job


class JobListResponse(BaseModel):
    """Response model for job list."""

    success: bool = True
    message: str
    jobs: List[Job]
    total: int


class JobStatusResponse(BaseModel):
    """Response model for job status check."""

    success: bool = True
    message: str
    job_id: str
    status: JobStatus
    progress: int = Field(..., ge=0, le=100)
    current_step: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


@router.get("", response_model=JobListResponse)
async def list_jobs(
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    limit: int = Query(
        50, ge=1, le=100, description="Maximum number of jobs to return"
    ),
    include_subjob_status: bool = Query(
        False, description="Include subjob status for parent jobs"
    ),
):
    """
    List all jobs with optional filters.

    Returns a list of jobs with their current status and metadata.
    If include_subjob_status is True, parent jobs will include subjob status counts.
    """
    try:
        manager = get_job_manager()
        jobs = await manager.list_jobs(job_type=job_type, status=status, limit=limit)

        # Enhance jobs with subjob status if requested
        if include_subjob_status:
            enhanced_jobs = []
            for job in jobs:
                # Only check subjob status for root jobs (not subjobs themselves)
                if not job.metadata.get("original_job_id"):
                    job_with_status = await manager.get_job_with_subjob_status(job.id)
                    if job_with_status and job_with_status["subjob_status"]:
                        # Add subjob status to metadata
                        job.metadata["subjob_status"] = job_with_status["subjob_status"]
                        job.metadata["chain_status"] = job_with_status["chain_status"]
                enhanced_jobs.append(job)
            jobs = enhanced_jobs

        return JobListResponse(
            message=f"Found {len(jobs)} jobs", jobs=jobs, total=len(jobs)
        )

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Get detailed information about a specific job.

    Returns the complete job object including results (if completed) and error messages (if failed).
    Includes subjob information and performance metrics if available.
    """
    try:
        manager = get_job_manager()
        job = await manager.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        # Enrich job metadata with subjob information if not already present
        if not job.metadata.get("subjobs_info"):
            subjob_info = {}

            # Check if this job has subjobs
            if job.metadata.get("resume_job_id"):
                resume_job_id = job.metadata.get("resume_job_id")
                resume_job = await manager.get_job(resume_job_id)
                if resume_job:
                    subjob_info["resume_job_id"] = resume_job_id
                    subjob_info["resume_job_status"] = (
                        resume_job.status.value
                        if hasattr(resume_job.status, "value")
                        else str(resume_job.status)
                    )

            # Check if this job is a subjob
            if job.metadata.get("original_job_id"):
                original_job_id = job.metadata.get("original_job_id")
                original_job = await manager.get_job(original_job_id)
                if original_job:
                    subjob_info["parent_job_id"] = original_job_id
                    subjob_info["parent_job_status"] = (
                        original_job.status.value
                        if hasattr(original_job.status, "value")
                        else str(original_job.status)
                    )

            # Also check job.result for original_job_id (for resume jobs)
            if job.result and isinstance(job.result, dict):
                if job.result.get("original_job_id"):
                    original_job_id = job.result.get("original_job_id")
                    if not subjob_info.get("parent_job_id"):
                        original_job = await manager.get_job(original_job_id)
                        if original_job:
                            subjob_info["parent_job_id"] = original_job_id
                            subjob_info["parent_job_status"] = (
                                original_job.status.value
                                if hasattr(original_job.status, "value")
                                else str(original_job.status)
                            )

            if subjob_info:
                job.metadata["subjobs_info"] = subjob_info

        return JobResponse(message=f"Job {job_id} retrieved successfully", job=job)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/chain")
async def get_job_chain(job_id: str):
    """
    Get complete job chain hierarchy for a given job.

    Returns the root job, all subjobs in order, chain metadata, and status of each job.

    Args:
        job_id: Job identifier (can be root or any job in chain)

    Returns:
        Dictionary with complete chain structure including all jobs and their statuses
    """
    try:
        manager = get_job_manager()
        chain_data = await manager.get_job_chain(job_id)

        if not chain_data["root_job_id"]:
            raise HTTPException(
                status_code=404, detail=f"Job {job_id} not found or has no chain"
            )

        # Convert jobs to dict format for response
        jobs_data = []
        for job in chain_data["jobs"]:
            jobs_data.append(
                {
                    "id": job.id,
                    "type": job.type,
                    "status": (
                        job.status.value
                        if hasattr(job.status, "value")
                        else str(job.status)
                    ),
                    "content_id": job.content_id,
                    "created_at": (
                        job.created_at.isoformat() if job.created_at else None
                    ),
                    "started_at": (
                        job.started_at.isoformat() if job.started_at else None
                    ),
                    "completed_at": (
                        job.completed_at.isoformat() if job.completed_at else None
                    ),
                    "progress": job.progress,
                    "current_step": job.current_step,
                    "metadata": job.metadata,
                }
            )

        return {
            "success": True,
            "root_job_id": chain_data["root_job_id"],
            "chain_length": chain_data["chain_length"],
            "chain_order": chain_data["chain_order"],
            "all_job_ids": chain_data["all_job_ids"],
            "chain_status": chain_data["chain_status"],
            "jobs": jobs_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job chain for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the current status of a job.

    This is a lightweight endpoint optimized for polling.
    Returns only the essential status information without full results.
    """
    try:
        manager = get_job_manager()
        job = await manager.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return JobStatusResponse(
            message=f"Job {job_id} status: {job.status}",
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            current_step=job.current_step,
            result=job.result if job.status == JobStatus.COMPLETED else None,
            error=job.error if job.status == JobStatus.FAILED else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get the result of a completed job.

    Returns the full result data. Only works for completed jobs.
    """
    try:
        manager = get_job_manager()
        job = await manager.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if (
            job.status == JobStatus.PENDING
            or job.status == JobStatus.QUEUED
            or job.status == JobStatus.PROCESSING
        ):
            raise HTTPException(
                status_code=202,  # Accepted but not ready
                detail=f"Job {job_id} is still {job.status}. Progress: {job.progress}%",
            )

        if job.status == JobStatus.FAILED:
            raise HTTPException(
                status_code=500, detail=f"Job {job_id} failed: {job.error}"
            )

        if job.status == JobStatus.CANCELLED:
            raise HTTPException(
                status_code=410, detail=f"Job {job_id} was cancelled"  # Gone
            )

        # Ensure result includes input_content and final_content
        result = job.result or {}
        if isinstance(result, dict):
            # Add input_content from metadata if not in result
            if "input_content" not in result and job.metadata.get("input_content"):
                result["input_content"] = job.metadata.get("input_content")
            # Ensure final_content is accessible
            if "final_content" not in result and result.get("result", {}).get(
                "final_content"
            ):
                result["final_content"] = result.get("result", {}).get("final_content")

        return {
            "success": True,
            "message": f"Job {job_id} completed successfully",
            "job_id": job.id,
            "content_id": job.content_id,
            "result": result,
            "completed_at": job.completed_at,
            "input_content": job.metadata.get(
                "input_content"
            ),  # Also include in top level
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job result {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/all", summary="Delete all jobs")
async def delete_all_jobs():
    """
    Delete all jobs from the system.

    This will:
    - Delete all jobs from Redis
    - Clear the jobs index
    - Clear in-memory job storage
    - Clear ARQ jobs

    WARNING: This is a destructive operation and cannot be undone.
    """
    try:
        manager = get_job_manager()
        deleted_count = await manager.delete_all_jobs()

        return {
            "success": True,
            "message": f"Deleted {deleted_count} jobs",
            "deleted_count": deleted_count,
        }

    except Exception as e:
        logger.error(f"Failed to delete all jobs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete all jobs: {str(e)}"
        )


@router.delete("/clear-arq", summary="Clear all ARQ jobs")
async def clear_all_arq_jobs():
    """
    Clear all jobs from the ARQ queue.

    This endpoint will:
    - Abort all queued jobs in ARQ
    - Delete all ARQ job keys from Redis
    - Clear completed/failed job results from ARQ

    Note: This does NOT delete jobs tracked in JobManager (those are separate).
    This only clears the ARQ queue itself. Use with caution!

    Returns:
        Number of ARQ jobs/keys cleared
    """
    try:
        manager = get_job_manager()
        deleted_count = await manager.clear_all_arq_jobs()

        return {
            "success": True,
            "message": f"Cleared {deleted_count} ARQ jobs/keys from queue",
            "deleted_count": deleted_count,
        }

    except Exception as e:
        logger.error(f"Failed to clear ARQ jobs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to clear ARQ jobs: {str(e)}"
        )


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a running or queued job.

    Jobs that are already completed, failed, or cancelled cannot be cancelled.
    For jobs in WAITING_FOR_APPROVAL status, we also check if there are pending approvals
    and mark them appropriately.
    """
    try:
        manager = get_job_manager()
        approval_manager = await get_approval_manager()

        job = await manager.get_job(job_id)

        # If job not found, check if it's waiting for approval (might have expired from Redis)
        if not job:
            # Check if there are pending approvals for this job
            approvals = await approval_manager.list_approvals(
                job_id=job_id, status="pending"
            )
            if approvals:
                # Job exists in approval system but not in job manager
                # Create a minimal job record to mark as cancelled
                from datetime import datetime

                job = Job(
                    id=job_id,
                    type="unknown",  # We don't know the type if job expired
                    content_id="unknown",
                    status=JobStatus.WAITING_FOR_APPROVAL,
                    created_at=datetime.utcnow(),
                    metadata={},
                )
                # Mark all pending approvals as cancelled/rejected
                for approval in approvals:
                    try:
                        await approval_manager.decide_approval(
                            approval.id,
                            ApprovalDecisionRequest(
                                decision="reject",
                                comment="Job cancelled by user",
                                reviewed_by="system",
                            ),
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to reject approval {approval.id} during job cancellation: {e}"
                        )

                # Mark job as cancelled
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                await manager._save_job_to_redis(job)

                logger.info(
                    f"Job {job_id} cancelled (was waiting for approval, job record was expired)"
                )
                return {
                    "success": True,
                    "message": f"Job {job_id} cancelled successfully (was waiting for approval)",
                }
            else:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        success = await manager.cancel_job(job_id)

        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} cannot be cancelled (already completed/failed/cancelled)",
            )

        # If job was waiting for approval, also reject any pending approvals
        if job.status == JobStatus.WAITING_FOR_APPROVAL:
            approvals = await approval_manager.list_approvals(
                job_id=job_id, status="pending"
            )
            for approval in approvals:
                try:
                    await approval_manager.decide_approval(
                        approval.id,
                        ApprovalDecisionRequest(
                            decision="reject",
                            comment="Job cancelled by user",
                            reviewed_by="system",
                        ),
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to reject approval {approval.id} during job cancellation: {e}"
                    )

        return {"success": True, "message": f"Job {job_id} cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{job_id}/resume")
async def resume_job(job_id: str):
    """
    Resume a pipeline that was stopped for approval.

    Loads the saved pipeline context from Redis and creates a new job to continue
    the pipeline from where it left off.

    Args:
        job_id: ID of the job that was waiting for approval

    Returns:
        Information about the new resume job
    """
    try:
        job_manager = get_job_manager()
        approval_manager = await get_approval_manager()

        # Check original job exists and is in WAITING_FOR_APPROVAL status
        original_job = await job_manager.get_job(job_id)
        if not original_job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if original_job.status != JobStatus.WAITING_FOR_APPROVAL:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not waiting for approval (status: {original_job.status})",
            )

        # Load pipeline context from Redis
        context_data = await approval_manager.load_pipeline_context(job_id)
        if not context_data:
            raise HTTPException(
                status_code=404,
                detail=f"No pipeline context found for job {job_id}. Cannot resume.",
            )

        # Find the root job (original job that started the chain)
        root_job_id = job_id
        current_job = original_job
        while current_job.metadata.get("original_job_id"):
            root_job_id = current_job.metadata.get("original_job_id")
            current_job = await job_manager.get_job(root_job_id)
            if not current_job:
                break

        # Get existing chain metadata from root job
        root_job = await job_manager.get_job(root_job_id)
        existing_chain = root_job.metadata.get("job_chain", {}) if root_job else {}
        existing_chain_order = existing_chain.get("chain_order", [root_job_id])
        existing_all_job_ids = existing_chain.get("all_job_ids", [root_job_id])

        # Create new job for resume
        resume_job_id = str(uuid.uuid4())
        new_chain_order = existing_chain_order + [resume_job_id]
        new_all_job_ids = existing_all_job_ids + [resume_job_id]

        resume_job = await job_manager.create_job(
            job_id=resume_job_id,
            job_type="resume_pipeline",
            content_id=original_job.content_id,
            metadata={
                "original_job_id": root_job_id,  # Use root job, not direct parent
                "resumed_from_step": context_data.get("last_step_number"),
                "resumed_from_step_name": context_data.get("last_step"),
                "original_content_type": original_job.type,
                "job_chain": {
                    "root_job_id": root_job_id,
                    "chain_length": len(new_chain_order),
                    "chain_order": new_chain_order,
                    "current_position": len(new_chain_order),
                    "all_job_ids": new_all_job_ids,
                    "chain_status": "in_progress",
                },
            },
        )

        # Update chain metadata for all jobs in chain
        await job_manager.update_job_chain_metadata(root_job_id)

        # Submit to ARQ with context
        await job_manager.submit_to_arq(
            resume_job_id,
            "resume_pipeline_job",
            job_id,  # original_job_id
            context_data,  # context_data
            resume_job_id,  # job_id (new resume job ID)
        )

        logger.info(
            f"Created resume job {resume_job_id} for original job {job_id}, "
            f"resuming from step {context_data.get('last_step_number')}"
        )

        return {
            "success": True,
            "message": f"Pipeline resume job created",
            "original_job_id": job_id,
            "resume_job_id": resume_job_id,
            "resuming_from_step": context_data.get("last_step_number"),
            "resuming_from_step_name": context_data.get("last_step"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume job {job_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to resume pipeline: {str(e)}"
        )
