"""
API endpoints for managing and retrieving pipeline step results.

Provides endpoints to:
- List all jobs with results
- Get detailed results for a specific job
- Retrieve individual step result files
- Download step result files
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.middleware.rbac import verify_job_ownership
from marketing_project.models.user_context import UserContext
from marketing_project.services.approval_manager import get_approval_manager
from marketing_project.services.job_manager import get_job_manager
from marketing_project.services.step_result_manager import get_step_result_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/results", tags=["Step Results"])


# Response Models
class PendingApprovalSummary(BaseModel):
    """Pending approval embedded in job results — contains all data the review panel needs."""

    id: str
    job_id: str
    pipeline_step: str
    step_name: str
    status: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: Optional[float] = None
    suggestions: Optional[List[str]] = None
    created_at: str


class StepInfo(BaseModel):
    """Information about a single pipeline step result."""

    filename: str = Field(..., description="Step result filename")
    step_number: int = Field(..., description="Step sequence number")
    step_name: str = Field(..., description="Human-readable step name")
    timestamp: str = Field(..., description="When the step completed")
    has_result: bool = Field(..., description="Whether result data exists")
    file_size: int = Field(..., description="File size in bytes")
    job_id: Optional[str] = Field(
        None, description="Job ID this step belongs to (for subjobs)"
    )
    root_job_id: Optional[str] = Field(
        None, description="Root job ID this step belongs to"
    )
    execution_context_id: Optional[str] = Field(
        None, description="Execution context ID (0=initial, 1+=resume cycles)"
    )
    execution_time: Optional[float] = Field(
        None, description="Step execution time in seconds"
    )
    tokens_used: Optional[int] = Field(None, description="Tokens consumed by this step")
    status: Optional[str] = Field(
        None, description="Step status (success/failed/skipped)"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if step failed"
    )


class JobResultsSummary(BaseModel):
    """Summary of results for a specific job."""

    job_id: str = Field(..., description="Job identifier")
    metadata: dict = Field(..., description="Job metadata")
    steps: List[StepInfo] = Field(..., description="List of step results")
    total_steps: int = Field(..., description="Total number of steps")
    subjobs: Optional[List[str]] = Field(None, description="List of subjob IDs")
    parent_job_id: Optional[str] = Field(
        None, description="Parent job ID if this is a subjob"
    )
    performance_metrics: Optional[dict] = Field(
        None, description="Performance metrics (execution time, tokens, etc.)"
    )
    quality_warnings: Optional[List[str]] = Field(
        None, description="Quality warnings from pipeline execution"
    )
    pending_approvals: Optional[List[PendingApprovalSummary]] = Field(
        None,
        description="Pending approvals for this job, embedded to avoid extra round-trips",
    )


class JobListItem(BaseModel):
    """Summary information for a job in the list."""

    job_id: str = Field(..., description="Job identifier")
    content_type: Optional[str] = Field(None, description="Type of content processed")
    content_id: Optional[str] = Field(None, description="Content identifier")
    started_at: Optional[str] = Field(None, description="Job start timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    step_count: int = Field(..., description="Number of step results")
    created_at: Optional[str] = Field(None, description="Result creation timestamp")
    status: Optional[str] = Field(
        None, description="Job status (completed/failed/processing/etc.)"
    )
    pending_approval_count: Optional[int] = Field(
        None,
        description="Number of pending approvals (set when status is waiting_for_approval)",
    )


class JobListResponse(BaseModel):
    """Response for listing all jobs."""

    jobs: List[JobListItem] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")


# Endpoints
@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    limit: int = 50,
    date_from: Optional[str] = Query(
        None,
        description="Filter jobs from this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    ),
    date_to: Optional[str] = Query(
        None,
        description="Filter jobs until this date (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
    ),
    user: UserContext = Depends(get_current_user),
):
    """
    List all jobs with step results.

    Args:
        limit: Maximum number of jobs to return (default: 50)
        date_from: Filter jobs created/started from this date onwards (ISO format)
        date_to: Filter jobs created/started until this date (ISO format)

    Returns:
        List of jobs with their summaries (filtered by date if provided)
    """
    try:
        step_manager = get_step_result_manager()
        jobs = await step_manager.list_all_jobs(
            limit=None
        )  # Get all jobs first for filtering

        # Scope to current user's jobs unless admin; enrich with status either way
        job_manager = get_job_manager()
        if not user.has_role("admin"):
            user_jobs = await job_manager.list_jobs(user_id=user.user_id, limit=10000)
            user_job_ids = {j.id for j in user_jobs}
            job_status_lookup = {
                j.id: (j.status.value if hasattr(j.status, "value") else str(j.status))
                for j in user_jobs
            }
            if user_job_ids:
                # Primary path: filter to jobs tracked in DB/Redis for this user
                jobs = [j for j in jobs if j.get("job_id") in user_job_ids]
            else:
                # Fallback: no DB/Redis records for this user (pre-migration jobs or
                # transient DB unavailability). Try filtering by user_id stored in
                # filesystem metadata.json (written by newer deployments). If none have
                # it, show all filesystem jobs — content access is still gated by
                # verify_job_ownership so no data is exposed.
                fs_user_jobs = [j for j in jobs if j.get("user_id") == user.user_id]
                if fs_user_jobs:
                    jobs = fs_user_jobs
                # else: keep all filesystem jobs as last-resort fallback

        else:
            # Admin sees all jobs; fetch status for all of them
            all_jobs = await job_manager.list_jobs(limit=100000)
            job_status_lookup = {
                j.id: (j.status.value if hasattr(j.status, "value") else str(j.status))
                for j in all_jobs
            }

        # Enrich jobs with status from job_manager (DB already provides status,
        # but this covers jobs from the filesystem fallback path)
        for job in jobs:
            job_id_val = job.get("job_id")
            if job_id_val and job_id_val in job_status_lookup:
                job["status"] = job_status_lookup[job_id_val]

        # Apply date filtering if provided
        if date_from or date_to:
            filtered_jobs = []

            # Parse date_from
            date_from_dt = None
            if date_from:
                try:
                    # Try parsing with time first, then date only
                    try:
                        date_from_dt = datetime.fromisoformat(
                            date_from.replace("Z", "+00:00")
                        )
                    except ValueError:
                        # Try date only format
                        date_from_dt = datetime.fromisoformat(date_from)
                        # Set to start of day
                        date_from_dt = date_from_dt.replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )
                except ValueError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid date_from format: {date_from}. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                    )

            # Parse date_to
            date_to_dt = None
            if date_to:
                try:
                    # Try parsing with time first, then date only
                    try:
                        date_to_dt = datetime.fromisoformat(
                            date_to.replace("Z", "+00:00")
                        )
                    except ValueError:
                        # Try date only format
                        date_to_dt = datetime.fromisoformat(date_to)
                        # Set to end of day
                        date_to_dt = date_to_dt.replace(
                            hour=23, minute=59, second=59, microsecond=999999
                        )
                except ValueError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid date_to format: {date_to}. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                    )

            # Filter jobs by date
            for job in jobs:
                # Use created_at as primary, fallback to started_at
                job_date_str = job.get("created_at") or job.get("started_at")
                if not job_date_str:
                    continue  # Skip jobs without dates

                try:
                    # Parse job date
                    job_date = datetime.fromisoformat(
                        job_date_str.replace("Z", "+00:00")
                    )

                    # Check if job date is within range
                    if date_from_dt and job_date < date_from_dt:
                        continue
                    if date_to_dt and job_date > date_to_dt:
                        continue

                    filtered_jobs.append(job)
                except (ValueError, AttributeError):
                    # Skip jobs with invalid date formats
                    continue

            jobs = filtered_jobs

        # Apply limit after filtering
        if limit:
            jobs = jobs[:limit]

        # Enrich waiting_for_approval jobs with pending approval counts
        waiting_job_ids = [
            j.get("job_id") for j in jobs if j.get("status") == "waiting_for_approval"
        ]
        if waiting_job_ids:
            try:
                approval_manager = await get_approval_manager()
                pending_approvals = await approval_manager.list_approvals(
                    status="pending"
                )
                pending_counts: dict = {}
                for a in pending_approvals:
                    if a.job_id in waiting_job_ids:
                        pending_counts[a.job_id] = pending_counts.get(a.job_id, 0) + 1
                for job in jobs:
                    job_id_val = job.get("job_id")
                    if job_id_val in pending_counts:
                        job["pending_approval_count"] = pending_counts[job_id_val]
            except Exception as e:
                logger.warning(f"Failed to enrich job list with approval counts: {e}")

        return JobListResponse(jobs=jobs, total=len(jobs))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.get("/jobs/{job_id}", response_model=JobResultsSummary)
async def get_job_results(
    job_id: str,
    filter_by_job: Optional[str] = Query(
        None, description="Filter steps by specific job_id"
    ),
    filter_by_status: Optional[str] = Query(
        None, description="Filter steps by status (success/failed/pending)"
    ),
    group_by_job: bool = Query(False, description="Group steps by job_id"),
    search: Optional[str] = Query(None, description="Search within step names/results"),
    user: UserContext = Depends(get_current_user),
):
    """
    Get detailed results for a specific job.

    Args:
        job_id: Job identifier
        filter_by_job: Filter steps by specific job_id
        filter_by_status: Filter steps by status (success/failed/pending)
        group_by_job: Group steps by job_id
        search: Search within step names/results

    Returns:
        Job results with all step information (filtered and grouped as requested)
    """
    try:
        job_manager = get_job_manager()
        await verify_job_ownership(job_id, user, job_manager)

        step_manager = get_step_result_manager()
        results = await step_manager.get_job_results(job_id)

        # Apply filters
        steps = results.get("steps", [])

        # Filter by job_id
        if filter_by_job:
            steps = [s for s in steps if s.get("job_id") == filter_by_job]

        # Filter by status
        if filter_by_status:
            status_map = {
                "success": "completed",
                "failed": "failed",
                "pending": "pending",
            }
            target_status = status_map.get(
                filter_by_status.lower(), filter_by_status.lower()
            )
            steps = [s for s in steps if s.get("status", "").lower() == target_status]

        # Search in step names
        if search:
            search_lower = search.lower()
            steps = [
                s
                for s in steps
                if search_lower in s.get("step_name", "").lower()
                or search_lower in s.get("filename", "").lower()
            ]

        # Group by execution_context_id if requested (new grouping option)
        # Also support grouping by job_id for backward compatibility
        if group_by_job:
            grouped_steps = {}
            for step in steps:
                # Group by execution_context_id if available, otherwise by job_id
                group_key = (
                    step.get("execution_context_id") or step.get("job_id") or job_id
                )
                if group_key not in grouped_steps:
                    grouped_steps[group_key] = []
                grouped_steps[group_key].append(step)

            # Update results with grouped structure
            results["steps"] = steps  # Keep original list
            results["steps_grouped"] = grouped_steps  # Add grouped version
        else:
            results["steps"] = steps

        results["total_steps"] = len(steps)

        # Convert steps to StepInfo objects (Pydantic will handle this automatically)
        # Ensure all steps have the required fields
        formatted_steps = []
        for step in steps:
            formatted_step = {
                "filename": step.get("filename", ""),
                "step_number": step.get("step_number", 0),
                "step_name": step.get("step_name", ""),
                "timestamp": step.get("timestamp", ""),
                "has_result": step.get("has_result", False),
                "file_size": step.get("file_size", 0),
                "job_id": step.get("job_id"),
                "root_job_id": step.get("root_job_id"),
                "execution_context_id": step.get("execution_context_id"),
                "execution_time": step.get("execution_time"),
                "tokens_used": step.get("tokens_used"),
                "status": step.get("status"),
                "error_message": step.get("error_message"),
            }
            formatted_steps.append(StepInfo(**formatted_step))

        results["steps"] = formatted_steps

        # Embed pending approvals so the frontend can render the review panel immediately
        try:
            approval_manager = await get_approval_manager()
            raw_pending = await approval_manager.list_approvals(
                job_id=job_id, status="pending"
            )
            if raw_pending:
                results["pending_approvals"] = [
                    PendingApprovalSummary(
                        id=a.id,
                        job_id=a.job_id,
                        pipeline_step=a.pipeline_step,
                        step_name=a.step_name,
                        status=a.status,
                        input_data=a.input_data,
                        output_data=a.output_data,
                        confidence_score=a.confidence_score,
                        suggestions=a.suggestions,
                        created_at=a.created_at.isoformat(),
                    )
                    for a in raw_pending
                ]
        except Exception as e:
            logger.warning(f"Failed to embed pending approvals for job {job_id}: {e}")

        return JobResultsSummary(**results)

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"No results found for job {job_id}"
        )
    except Exception as e:
        logger.error(f"Failed to get job results for {job_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get job results: {str(e)}"
        )


@router.get("/jobs/{job_id}/timeline")
async def get_job_timeline(job_id: str, user: UserContext = Depends(get_current_user)):
    """
    Get chronological timeline of all events for a job and its subjobs.

    Returns a list of all events (steps, approvals, job boundaries) in chronological order
    with timestamps, durations, and event types.

    Args:
        job_id: Job identifier (can be root or any job in chain)

    Returns:
        Timeline with all events in chronological order
    """
    try:
        from marketing_project.services.approval_manager import get_approval_manager

        job_manager = get_job_manager()
        await verify_job_ownership(job_id, user, job_manager)

        step_manager = get_step_result_manager()

        # Get job chain to find all related jobs
        chain_data = await job_manager.get_job_chain(job_id)
        if not chain_data["root_job_id"]:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        timeline_events = []
        approval_manager = await get_approval_manager()

        # Process each job in the chain
        for i, job_id_in_chain in enumerate(chain_data["chain_order"]):
            job = await job_manager.get_job(job_id_in_chain)
            if not job:
                continue

            # Add job boundary event
            is_root = i == 0
            is_subjob = i > 0

            timeline_events.append(
                {
                    "event_type": "job_boundary",
                    "job_id": job_id_in_chain,
                    "timestamp": job.created_at.isoformat() if job.created_at else None,
                    "job_type": job.type,
                    "is_root": is_root,
                    "is_subjob": is_subjob,
                    "position": i + 1,
                    "total_jobs": len(chain_data["chain_order"]),
                    "status": (
                        job.status.value
                        if hasattr(job.status, "value")
                        else str(job.status)
                    ),
                    "duration": None,
                }
            )

            # Get steps for this job (without aggregation from subjobs)
            try:
                # First try to get steps from step files or job.result using aggregate_steps_from_jobs
                steps = await step_manager.aggregate_steps_from_jobs([job_id_in_chain])

                # If no steps found, try extracting directly from job.result metadata
                if not steps:
                    steps = await step_manager.extract_step_info_from_job_result(
                        job_id_in_chain
                    )
                    # Ensure job_id is set for extracted steps
                    for step in steps:
                        if not step.get("job_id"):
                            step["job_id"] = job_id_in_chain

                # If still no steps and this is a subjob, the steps might be in the final subjob's result
                # The final subjob's result contains all steps executed in the chain
                if not steps and is_subjob and chain_data["chain_length"] > 1:
                    # Get the final subjob (last in chain)
                    final_subjob_id = chain_data["chain_order"][-1]
                    if final_subjob_id != job_id_in_chain:
                        # Extract steps from final subjob's result
                        final_steps = (
                            await step_manager.extract_step_info_from_job_result(
                                final_subjob_id
                            )
                        )
                        # For now, assign steps to jobs based on their position in the chain
                        # Steps executed after a job boundary belong to that job
                        # This is a heuristic - ideally steps should have job_id saved when executed
                        job_index = chain_data["chain_order"].index(job_id_in_chain)
                        # Approximate: if this is the first subjob, it likely has steps 1-2, etc.
                        # For now, just get all steps and we'll filter by timestamp relative to job boundaries
                        steps = final_steps
                        for step in steps:
                            if not step.get("job_id"):
                                step["job_id"] = job_id_in_chain

                for step in steps:
                    # Include steps that belong to this specific job
                    step_job_id = step.get("job_id")
                    execution_context_id = step.get("execution_context_id")

                    # Match by job_id or by execution_context_id (for context-based grouping)
                    if step_job_id == job_id_in_chain or (
                        execution_context_id
                        and str(chain_data["chain_order"].index(job_id_in_chain))
                        == execution_context_id
                    ):
                        step_timestamp = step.get("timestamp")
                        execution_time = step.get("execution_time")

                        timeline_events.append(
                            {
                                "event_type": "step",
                                "job_id": job_id_in_chain,
                                "root_job_id": step.get("root_job_id"),
                                "execution_context_id": execution_context_id,
                                "step_name": step.get("step_name"),
                                "step_number": step.get("step_number"),
                                "timestamp": step_timestamp,
                                "status": step.get("status", "completed"),
                                "execution_time": execution_time,
                                "tokens_used": step.get("tokens_used"),
                                "error_message": step.get("error_message"),
                                "duration": execution_time,
                            }
                        )
            except Exception as e:
                logger.warning(f"Failed to get steps for job {job_id_in_chain}: {e}")

            # Get approvals for this job
            try:
                approvals = await approval_manager.list_approvals(
                    job_id=job_id_in_chain
                )
                for approval in approvals:
                    timeline_events.append(
                        {
                            "event_type": "approval",
                            "job_id": job_id_in_chain,
                            "approval_id": approval.id,
                            "step_name": approval.step_name,
                            "agent_name": approval.agent_name,
                            "timestamp": (
                                approval.created_at.isoformat()
                                if approval.created_at
                                else None
                            ),
                            "status": approval.status,
                            "reviewed_at": (
                                approval.reviewed_at.isoformat()
                                if approval.reviewed_at
                                else None
                            ),
                            "reviewed_by": approval.reviewed_by,
                            "duration": None,
                        }
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to get approvals for job {job_id_in_chain}: {e}"
                )

            # Add job completion event if completed
            if job.completed_at:
                timeline_events.append(
                    {
                        "event_type": "job_completion",
                        "job_id": job_id_in_chain,
                        "timestamp": job.completed_at.isoformat(),
                        "status": (
                            job.status.value
                            if hasattr(job.status, "value")
                            else str(job.status)
                        ),
                        "duration": None,
                    }
                )

        # Sort all events by timestamp
        timeline_events.sort(key=lambda x: x.get("timestamp") or "", reverse=False)

        # Calculate durations between events
        for i in range(1, len(timeline_events)):
            prev_event = timeline_events[i - 1]
            curr_event = timeline_events[i]

            prev_time = prev_event.get("timestamp")
            curr_time = curr_event.get("timestamp")

            if prev_time and curr_time:
                try:
                    from datetime import datetime

                    prev_dt = datetime.fromisoformat(prev_time.replace("Z", "+00:00"))
                    curr_dt = datetime.fromisoformat(curr_time.replace("Z", "+00:00"))
                    duration_seconds = (curr_dt - prev_dt).total_seconds()
                    curr_event["time_since_previous"] = duration_seconds
                except Exception:
                    pass

        return {
            "success": True,
            "job_id": job_id,
            "root_job_id": chain_data["root_job_id"],
            "chain_length": chain_data["chain_length"],
            "total_events": len(timeline_events),
            "events": timeline_events,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get timeline for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/steps/{step_filename}")
async def get_step_result(
    job_id: str,
    step_filename: str,
    execution_context_id: Optional[str] = Query(
        None, description="Optional execution context ID to search in specific context"
    ),
    user: UserContext = Depends(get_current_user),
):
    """
    Get the content of a specific step result.

    Args:
        job_id: Job identifier (may be subjob, will find root)
        step_filename: Step result filename (e.g., "01_seo_keywords.json")
        execution_context_id: Optional execution context ID to search in specific context

    Returns:
        Step result data as JSON
    """
    try:
        job_manager = get_job_manager()
        await verify_job_ownership(job_id, user, job_manager)

        step_manager = get_step_result_manager()
        result = await step_manager.get_step_result(
            job_id, step_filename, execution_context_id
        )

        return result

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Step result not found: {step_filename} for job {job_id}",
        )
    except Exception as e:
        logger.error(f"Failed to get step result {step_filename} for job {job_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get step result: {str(e)}"
        )


@router.get("/jobs/{job_id}/steps/by-name/{step_name}")
async def get_step_result_by_name(
    job_id: str,
    step_name: str,
    execution_context_id: Optional[str] = Query(
        None, description="Optional execution context ID to search in specific context"
    ),
    user: UserContext = Depends(get_current_user),
):
    """
    Get the content of a specific step result by step name.

    Args:
        job_id: Job identifier (may be subjob, will find root)
        step_name: Step name (e.g., "seo_keywords")
        execution_context_id: Optional execution context ID to search in specific context

    Returns:
        Step result data as JSON
    """
    try:
        job_manager = get_job_manager()
        await verify_job_ownership(job_id, user, job_manager)

        step_manager = get_step_result_manager()
        result = await step_manager.get_step_result_by_name(
            job_id, step_name, execution_context_id
        )

        return result

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Step result not found: {step_name} for job {job_id}",
        )
    except Exception as e:
        logger.error(f"Failed to get step result {step_name} for job {job_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get step result: {str(e)}"
        )


@router.get("/jobs/{job_id}/steps/{step_filename}/download")
async def download_step_result(
    job_id: str,
    step_filename: str,
    execution_context_id: Optional[str] = Query(
        None, description="Optional execution context ID to search in specific context"
    ),
    user: UserContext = Depends(get_current_user),
):
    """
    Download a step result file.

    Args:
        job_id: Job identifier (may be subjob, will find root)
        step_filename: Step result filename
        execution_context_id: Optional execution context ID to search in specific context

    Returns:
        File download response
    """
    try:
        job_manager = get_job_manager()
        await verify_job_ownership(job_id, user, job_manager)

        step_manager = get_step_result_manager()

        # Check if using S3
        if step_manager._use_s3 and step_manager.s3_storage:
            try:
                # Get step result data
                step_data = await step_manager.get_step_result(
                    job_id, step_filename, execution_context_id
                )

                # Convert to JSON string
                import json

                from fastapi.responses import Response

                json_content = json.dumps(step_data, indent=2, ensure_ascii=False)
                download_filename = f"{job_id}_{step_filename}"

                return Response(
                    content=json_content.encode("utf-8"),
                    media_type="application/json",
                    headers={
                        "Content-Disposition": f'attachment; filename="{download_filename}"'
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to download from S3, trying local: {e}")

        # Fallback to local file
        file_path = await step_manager.get_step_file_path(
            job_id, step_filename, execution_context_id
        )

        # Generate a friendly download filename
        download_filename = f"{job_id}_{step_filename}"

        return FileResponse(
            path=str(file_path),
            media_type="application/json",
            filename=download_filename,
        )

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Step result not found: {step_filename} for job {job_id}",
        )
    except Exception as e:
        logger.error(
            f"Failed to download step result {step_filename} for job {job_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to download step result: {str(e)}"
        )


@router.get("/jobs/{job_id}/pipeline-flow")
async def get_pipeline_flow(job_id: str, user: UserContext = Depends(get_current_user)):
    """
    Get complete pipeline flow visualization data.

    Returns structured data showing:
    - Original input content
    - Each step with its input snapshot and output
    - Final output
    - Execution summary

    Args:
        job_id: Job identifier

    Returns:
        PipelineFlowResponse with complete flow data
    """
    try:
        from marketing_project.models.pipeline_steps import PipelineFlowResponse

        job_manager = get_job_manager()
        await verify_job_ownership(job_id, user, job_manager)

        step_manager = get_step_result_manager()
        flow_data = await step_manager.get_pipeline_flow(job_id)

        # Convert to Pydantic model for validation
        return PipelineFlowResponse(**flow_data)

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"No results found for job {job_id}"
        )
    except Exception as e:
        logger.error(f"Failed to get pipeline flow for {job_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get pipeline flow: {str(e)}"
        )


@router.delete("/jobs/{job_id}")
async def delete_job_results(
    job_id: str, user: UserContext = Depends(get_current_user)
):
    """
    Delete all results for a specific job.

    Args:
        job_id: Job identifier

    Returns:
        Success status
    """
    try:
        job_manager = get_job_manager()
        await verify_job_ownership(job_id, user, job_manager)

        step_manager = get_step_result_manager()
        success = await step_manager.cleanup_job(job_id)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"No results found for job {job_id}"
            )

        return {"success": True, "message": f"Results deleted for job {job_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete job results for {job_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete job results: {str(e)}"
        )
