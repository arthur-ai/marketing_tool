"""
Approval API Endpoints.

Endpoints for managing human-in-the-loop approvals of non-deterministic agent outputs.
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from ..models.approval_models import (
    ApprovalDecisionRequest,
    ApprovalListItem,
    ApprovalRequest,
    ApprovalSettings,
    ApprovalStats,
    PendingApprovalsResponse,
    RetryStepRequest,
)
from ..services.approval_manager import (
    get_approval_manager,
    get_approval_manager_sync,
    set_approval_settings,
    set_approval_settings_sync,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/pending", response_model=PendingApprovalsResponse)
async def get_pending_approvals(
    job_id: Optional[str] = Query(None, description="Filter by job ID")
):
    """
    Get all pending approval requests.

    This endpoint returns approval requests that are waiting for user review.
    """
    try:
        manager = await get_approval_manager(reload_from_db=True)

        # Force reload from Redis to ensure we have latest approvals
        await manager._load_all_approvals_from_redis()

        # Get pending approvals
        approvals = await manager.list_approvals(job_id=job_id, status="pending")

        # Filter out approvals for cancelled/failed/completed jobs
        from ..services.job_manager import JobStatus, get_job_manager

        job_manager = get_job_manager()
        filtered_approvals = []

        for approval in approvals:
            # Check if job is still valid for approval
            job = await job_manager.get_job(approval.job_id)
            if job:
                # Only include if job is waiting for approval
                if job.status == JobStatus.WAITING_FOR_APPROVAL:
                    filtered_approvals.append(approval)
                else:
                    # Job is cancelled/failed/completed, mark approval as rejected if still pending
                    # BUT: Skip auto-rejection if:
                    # 1. Job was completed due to approval (has resume_job_id metadata)
                    # 2. seo_keywords step doesn't support rejection (only keyword selection)
                    should_auto_reject = True
                    if job.status == JobStatus.COMPLETED:
                        # Check if job was completed due to approval (has resume_job_id)
                        if (
                            job.metadata.get("resume_job_id")
                            or job.metadata.get("status") == "approved_and_resumed"
                        ):
                            # Job was completed because approval was made - don't auto-reject
                            should_auto_reject = False
                            logger.debug(
                                f"Skipping auto-rejection for approval {approval.id} - "
                                f"job {approval.job_id} was completed after approval"
                            )

                    # Check if this step doesn't support rejection
                    if approval.pipeline_step == "seo_keywords":
                        should_auto_reject = False
                        logger.debug(
                            f"Skipping auto-rejection for approval {approval.id} - "
                            f"seo_keywords step does not support rejection"
                        )

                    if should_auto_reject and approval.status == "pending":
                        try:
                            await manager.decide_approval(
                                approval.id,
                                ApprovalDecisionRequest(
                                    decision="reject",
                                    comment=f"Job {approval.job_id} was {job.status.value}",
                                    reviewed_by="system",
                                ),
                            )
                            logger.info(
                                f"Auto-rejected approval {approval.id} because job {approval.job_id} is {job.status.value}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to auto-reject approval {approval.id}: {e}"
                            )
            else:
                # Job not found - might have expired or been deleted
                # Exclude from pending since job doesn't exist
                # If approval is still pending, mark it as rejected
                # BUT: Skip auto-rejection for seo_keywords step (doesn't support rejection)
                if (
                    approval.status == "pending"
                    and approval.pipeline_step != "seo_keywords"
                ):
                    try:
                        await manager.decide_approval(
                            approval.id,
                            ApprovalDecisionRequest(
                                decision="reject",
                                comment=f"Job {approval.job_id} no longer exists",
                                reviewed_by="system",
                            ),
                        )
                        logger.info(
                            f"Auto-rejected approval {approval.id} because job {approval.job_id} no longer exists"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to auto-reject approval {approval.id}: {e}"
                        )
                elif approval.pipeline_step == "seo_keywords":
                    logger.debug(
                        f"Skipping auto-rejection for approval {approval.id} (job not found) - "
                        f"seo_keywords step does not support rejection"
                    )
                # Don't include in filtered list

        approvals = filtered_approvals

        # If no pending approvals but job_id provided, check if job is waiting and recreate approval
        if len(approvals) == 0 and job_id:
            job = await job_manager.get_job(job_id)
            if job and job.status == JobStatus.WAITING_FOR_APPROVAL:
                context_data = await manager.load_pipeline_context(job_id)
                if context_data:
                    logger.info(
                        f"Recreating missing approval for job {job_id} from pipeline context"
                    )
                    last_step = context_data.get("last_step", "")
                    step_result = context_data.get("step_result", {})
                    pipeline_step = (
                        last_step.split(":")[-1].strip()
                        if ":" in last_step
                        else last_step
                    )

                    try:
                        await manager.create_approval_request(
                            job_id=job_id,
                            agent_name=pipeline_step,
                            step_name=last_step,
                            input_data=context_data.get("context", {}).get(
                                "input_content", {}
                            ),
                            output_data=step_result,
                            pipeline_step=pipeline_step,
                        )
                        # Reload after creating
                        approvals = await manager.list_approvals(
                            job_id=job_id, status="pending"
                        )
                        # Re-filter to ensure we only include valid approvals
                        approvals = [a for a in approvals if a.status == "pending"]
                    except Exception as e:
                        logger.warning(f"Failed to recreate approval: {e}")

        all_approvals = await manager.list_approvals(job_id=job_id)

        # Convert to list items
        items = [
            ApprovalListItem(
                id=a.id,
                job_id=a.job_id,
                agent_name=a.agent_name,
                step_name=a.step_name,
                status=a.status,
                created_at=a.created_at,
                reviewed_at=a.reviewed_at,
            )
            for a in approvals
        ]

        return PendingApprovalsResponse(
            approvals=items,
            total=len(all_approvals),
            pending=len(approvals),
        )

    except Exception as e:
        logger.error(f"Failed to get pending approvals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/all")
async def delete_all_approvals():
    """
    Delete all approvals from the system.

    This will:
    - Delete all approvals from Redis
    - Clear the approval list
    - Clear job approval mappings
    - Clear pipeline contexts
    - Clear in-memory approval storage

    WARNING: This is a destructive operation and cannot be undone.
    """
    try:
        manager = await get_approval_manager()
        deleted_count = await manager.delete_all_approvals()

        return {
            "success": True,
            "message": f"Deleted {deleted_count} approvals",
            "deleted_count": deleted_count,
        }

    except Exception as e:
        logger.error(f"Failed to delete all approvals: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete all approvals: {str(e)}"
        )


@router.get("/analytics")
async def get_approval_analytics():
    """
    Get comprehensive approval analytics including metrics, trends, and performance.

    Returns:
    - Approval statistics (total, pending, approved, rejected, modified)
    - Average approval time
    - Most common modifications
    - Approval rate by step type
    - Reviewer performance
    - Trends over time
    """
    try:
        manager = await get_approval_manager()

        # Get all approvals
        all_approvals = await manager.list_approvals()

        # Basic stats
        total = len(all_approvals)
        pending = len([a for a in all_approvals if a.status == "pending"])
        approved = len([a for a in all_approvals if a.status == "approved"])
        rejected = len([a for a in all_approvals if a.status == "rejected"])
        modified = len([a for a in all_approvals if a.status == "modified"])

        # Calculate average review time
        reviewed = [a for a in all_approvals if a.reviewed_at and a.created_at]
        avg_review_time = None
        if reviewed:
            review_times = [
                (a.reviewed_at - a.created_at).total_seconds() for a in reviewed
            ]
            avg_review_time = sum(review_times) / len(review_times)

        # Approval rate
        decided = approved + rejected + modified
        approval_rate = (approved + modified) / decided if decided > 0 else 0.0

        # Most common modifications (by step type)
        modifications_by_step = {}
        for a in all_approvals:
            if a.status == "modified":
                step = a.step_name
                modifications_by_step[step] = modifications_by_step.get(step, 0) + 1

        most_common_modifications = sorted(
            modifications_by_step.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Approval rate by step type
        approval_rate_by_step = {}
        for a in all_approvals:
            step = a.step_name
            if step not in approval_rate_by_step:
                approval_rate_by_step[step] = {
                    "total": 0,
                    "approved": 0,
                    "rejected": 0,
                    "modified": 0,
                }

            approval_rate_by_step[step]["total"] += 1
            if a.status == "approved":
                approval_rate_by_step[step]["approved"] += 1
            elif a.status == "rejected":
                approval_rate_by_step[step]["rejected"] += 1
            elif a.status == "modified":
                approval_rate_by_step[step]["modified"] += 1

        # Calculate rates
        for step, counts in approval_rate_by_step.items():
            total_decided = counts["approved"] + counts["rejected"] + counts["modified"]
            if total_decided > 0:
                counts["approval_rate"] = (
                    counts["approved"] + counts["modified"]
                ) / total_decided
            else:
                counts["approval_rate"] = 0.0

        # Reviewer performance
        reviewer_stats = {}
        for a in reviewed:
            reviewer = a.reviewed_by or "unknown"
            if reviewer not in reviewer_stats:
                reviewer_stats[reviewer] = {
                    "total_reviews": 0,
                    "approved": 0,
                    "rejected": 0,
                    "modified": 0,
                    "avg_review_time": 0,
                    "review_times": [],
                }

            reviewer_stats[reviewer]["total_reviews"] += 1
            if a.status == "approved":
                reviewer_stats[reviewer]["approved"] += 1
            elif a.status == "rejected":
                reviewer_stats[reviewer]["rejected"] += 1
            elif a.status == "modified":
                reviewer_stats[reviewer]["modified"] += 1

            review_time = (a.reviewed_at - a.created_at).total_seconds()
            reviewer_stats[reviewer]["review_times"].append(review_time)

        # Calculate average review time per reviewer
        for reviewer, stats in reviewer_stats.items():
            if stats["review_times"]:
                stats["avg_review_time"] = sum(stats["review_times"]) / len(
                    stats["review_times"]
                )
            del stats["review_times"]  # Remove raw times from response

        # Trends over time (last 30 days)
        from datetime import timedelta

        now = datetime.utcnow()
        thirty_days_ago = now - timedelta(days=30)

        recent_approvals = [
            a for a in all_approvals if a.created_at and a.created_at >= thirty_days_ago
        ]

        # Group by day
        daily_trends = {}
        for a in recent_approvals:
            day = a.created_at.date().isoformat()
            if day not in daily_trends:
                daily_trends[day] = {
                    "total": 0,
                    "approved": 0,
                    "rejected": 0,
                    "modified": 0,
                    "pending": 0,
                }
            daily_trends[day]["total"] += 1
            daily_trends[day][a.status] = daily_trends[day].get(a.status, 0) + 1

        # Sort by date
        sorted_trends = sorted(daily_trends.items())

        return {
            "success": True,
            "statistics": {
                "total_requests": total,
                "pending": pending,
                "approved": approved,
                "rejected": rejected,
                "modified": modified,
                "avg_review_time_seconds": avg_review_time,
                "approval_rate": approval_rate,
            },
            "most_common_modifications": [
                {"step_name": step, "count": count}
                for step, count in most_common_modifications
            ],
            "approval_rate_by_step": approval_rate_by_step,
            "reviewer_performance": reviewer_stats,
            "trends": {
                "period_days": 30,
                "daily_trends": [
                    {"date": date, **stats} for date, stats in sorted_trends
                ],
            },
        }

    except Exception as e:
        logger.error(f"Failed to get approval analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=ApprovalStats)
async def get_approval_stats():
    """
    Get approval statistics.

    Returns metrics about approval requests including approval rate,
    average review time, and status distribution.
    """
    try:
        manager = await get_approval_manager()
        stats = await manager.get_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get approval stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings", response_model=ApprovalSettings)
async def get_approval_settings():
    """
    Get current approval settings.
    """
    try:
        manager = await get_approval_manager()
        return manager.settings

    except Exception as e:
        logger.error(f"Failed to get approval settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/settings", response_model=ApprovalSettings)
async def update_approval_settings(settings: ApprovalSettings):
    """
    Update approval settings.

    Configure which agents require approval, auto-approval thresholds, and timeouts.
    """
    try:
        await set_approval_settings(settings)
        logger.info(f"Approval settings updated: {settings.model_dump()}")
        return settings

    except Exception as e:
        logger.error(f"Failed to update approval settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{approval_id}/impact")
async def get_approval_impact(approval_id: str):
    """
    Get approval impact analysis showing what changed after approval.

    Returns:
    - Original output before approval
    - Modified/final output after approval
    - Comparison view
    - Impact on downstream steps
    """
    try:
        manager = await get_approval_manager()
        approval = await manager.get_approval(approval_id)

        if not approval:
            raise HTTPException(
                status_code=404, detail=f"Approval {approval_id} not found"
            )

        # Original output is stored in approval.output_data
        original_output = approval.output_data

        # Final output is modified_output if exists, otherwise original
        final_output = (
            approval.modified_output
            if approval.modified_output
            else approval.output_data
        )

        # Calculate differences (simple comparison)
        has_changes = approval.modified_output is not None
        changes_summary = None

        if has_changes:
            # Simple comparison - check if keys/values differ
            if isinstance(original_output, dict) and isinstance(final_output, dict):
                added_keys = set(final_output.keys()) - set(original_output.keys())
                removed_keys = set(original_output.keys()) - set(final_output.keys())
                changed_keys = []

                for key in set(original_output.keys()) & set(final_output.keys()):
                    if original_output[key] != final_output[key]:
                        changed_keys.append(key)

                changes_summary = {
                    "added_keys": list(added_keys),
                    "removed_keys": list(removed_keys),
                    "changed_keys": changed_keys,
                    "has_changes": len(added_keys) > 0
                    or len(removed_keys) > 0
                    or len(changed_keys) > 0,
                }
            else:
                changes_summary = {
                    "has_changes": original_output != final_output,
                    "type": "non_dict_comparison",
                }

        # Check impact on downstream steps (if subjob was created)
        downstream_impact = None
        from ..services.job_manager import get_job_manager

        job_manager = get_job_manager()

        job = await job_manager.get_job(approval.job_id)
        if job:
            resume_job_id = job.metadata.get("resume_job_id")
            if resume_job_id:
                resume_job = await job_manager.get_job(resume_job_id)
                if resume_job:
                    downstream_impact = {
                        "subjob_created": True,
                        "subjob_id": resume_job_id,
                        "subjob_status": (
                            resume_job.status.value
                            if hasattr(resume_job.status, "value")
                            else str(resume_job.status)
                        ),
                        "subjob_completed": (
                            resume_job.status.value == "completed"
                            if hasattr(resume_job.status, "value")
                            else False
                        ),
                    }

        return {
            "success": True,
            "approval_id": approval_id,
            "step_name": approval.step_name,
            "original_output": original_output,
            "final_output": final_output,
            "has_modifications": has_changes,
            "changes_summary": changes_summary,
            "downstream_impact": downstream_impact,
            "decision": approval.status,
            "reviewed_at": (
                approval.reviewed_at.isoformat() if approval.reviewed_at else None
            ),
            "reviewed_by": approval.reviewed_by,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get approval impact for {approval_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{approval_id}")
async def get_approval(approval_id: str):
    """
    Get details of a specific approval request with full context.

    Returns the full approval request including:
    - Input/output data
    - Modified output (if applicable)
    - Reviewer comments
    - Approval decision details
    - Link to related subjob (if created)
    - Timestamps
    """
    try:
        manager = await get_approval_manager()
        approval = await manager.get_approval(approval_id)

        if not approval:
            raise HTTPException(
                status_code=404, detail=f"Approval {approval_id} not found"
            )

        # Enhance approval with additional context
        from ..services.job_manager import get_job_manager

        job_manager = get_job_manager()

        resume_job_id = None
        resume_job = None

        # Find related subjob if this approval created one
        job = await job_manager.get_job(approval.job_id)
        if job:
            resume_job_id = job.metadata.get("resume_job_id")
            if resume_job_id:
                # Check if this approval triggered the resume
                resume_job = await job_manager.get_job(resume_job_id)
                if (
                    resume_job
                    and resume_job.metadata.get("created_by_approval_id") != approval_id
                ):
                    resume_job_id = None
                    resume_job = None

        # Add decision details if reviewed
        decision_details = None
        if approval.reviewed_at:
            decision_details = {
                "decision": approval.status,
                "reviewed_at": (
                    approval.reviewed_at.isoformat() if approval.reviewed_at else None
                ),
                "reviewed_by": approval.reviewed_by,
                "comment": approval.user_comment,
                "has_modifications": approval.modified_output is not None,
                "original_output": approval.output_data,
                "final_output": (
                    approval.modified_output
                    if approval.modified_output
                    else approval.output_data
                ),
            }

        # Create enhanced response dict
        approval_dict = approval.model_dump()
        if decision_details:
            approval_dict["decision_details"] = decision_details
        if resume_job_id and resume_job:
            approval_dict["related_subjob_id"] = resume_job_id
            approval_dict["related_subjob_status"] = (
                resume_job.status.value
                if hasattr(resume_job.status, "value")
                else str(resume_job.status)
            )

        return approval_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get approval {approval_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{approval_id}/decide", response_model=ApprovalRequest)
async def decide_approval(approval_id: str, decision: ApprovalDecisionRequest):
    """
    Make a decision on an approval request.

    Approve, reject, or modify the output from a non-deterministic agent.
    Once decided, the pipeline can continue with the approved/modified output.
    """
    try:
        manager = await get_approval_manager()

        # Get approval first to check job_id
        approval = await manager.get_approval(approval_id)
        if not approval:
            raise HTTPException(
                status_code=404, detail=f"Approval {approval_id} not found"
            )

        # Make decision
        approval = await manager.decide_approval(approval_id, decision)

        logger.info(
            f"Approval {approval_id} decided: {decision.decision} "
            f"by {decision.reviewed_by or 'unknown'}"
        )

        # Import job manager utilities for resume/retry logic
        import uuid

        from ..services.job_manager import JobStatus, get_job_manager

        job_manager = get_job_manager()

        # Handle approval/modify - automatically resume pipeline
        if decision.decision in ["approve", "modify"]:

            # Check if job is waiting for approval
            job = await job_manager.get_job(approval.job_id)
            if job and job.status == JobStatus.WAITING_FOR_APPROVAL:
                # Load pipeline context from Redis
                context_data = await manager.load_pipeline_context(approval.job_id)
                if context_data:
                    # Update context with approval decision (for keyword selection or modified output)
                    # Normalize step name (handle "Step 2: marketing_brief" format)
                    normalized_step = approval.pipeline_step
                    if ":" in normalized_step:
                        normalized_step = normalized_step.split(":")[-1].strip()

                    # Ensure context dictionary exists
                    if "context" not in context_data:
                        context_data["context"] = {}

                    # Update both step_result and context[step_name] so it's available in results when loading
                    if approval.modified_output:
                        context_data["step_result"] = approval.modified_output
                        context_data["context"][
                            normalized_step
                        ] = approval.modified_output
                        logger.debug(
                            f"Updated context with modified_output for step {normalized_step}"
                        )
                    elif approval.output_data and approval.status == "approved":
                        # For direct approvals (not modified), use output_data
                        context_data["step_result"] = approval.output_data
                        context_data["context"][normalized_step] = approval.output_data
                        logger.debug(
                            f"Updated context with output_data for step {normalized_step}"
                        )

                    # Find the root job (original job that started the chain)
                    root_job_id = approval.job_id
                    current_job = job
                    while current_job.metadata.get("original_job_id"):
                        root_job_id = current_job.metadata.get("original_job_id")
                        current_job = await job_manager.get_job(root_job_id)
                        if not current_job:
                            break

                    # Get existing chain metadata from root job
                    root_job = await job_manager.get_job(root_job_id)
                    existing_chain = (
                        root_job.metadata.get("job_chain", {}) if root_job else {}
                    )
                    existing_chain_order = existing_chain.get(
                        "chain_order", [root_job_id]
                    )
                    existing_all_job_ids = existing_chain.get(
                        "all_job_ids", [root_job_id]
                    )

                    # Create new job for resume
                    resume_job_id = str(uuid.uuid4())
                    new_chain_order = existing_chain_order + [resume_job_id]
                    new_all_job_ids = existing_all_job_ids + [resume_job_id]

                    resume_job = await job_manager.create_job(
                        job_id=resume_job_id,
                        job_type="resume_pipeline",
                        content_id=job.content_id,
                        metadata={
                            "original_job_id": root_job_id,  # Use root job, not direct parent
                            "resumed_from_step": context_data.get("last_step_number"),
                            "resumed_from_step_name": context_data.get("last_step"),
                            "original_content_type": job.type,
                            "created_by_approval_id": approval.id,
                            "created_by_approval_step": approval.step_name,
                            "created_by_user": decision.reviewed_by,
                            "approval_decision": decision.decision,
                            "approval_comment": decision.comment,
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
                        approval.job_id,  # original_job_id
                        context_data,  # context_data
                        resume_job_id,  # job_id (new resume job ID)
                    )

                    # Update old job status - mark as completed and add metadata about resume job
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.utcnow()
                    job.current_step = "Approved - Pipeline resumed"
                    job.metadata["resume_job_id"] = resume_job_id
                    job.metadata["approved_at"] = datetime.utcnow().isoformat()
                    job.metadata["status"] = "approved_and_resumed"
                    await job_manager._save_job(job)

                    logger.info(
                        f"Auto-resumed pipeline for approval {approval_id} "
                        f"(resume job: {resume_job_id}, resuming from step {context_data.get('last_step_number')}). "
                        f"Original job {approval.job_id} marked as completed."
                    )
                else:
                    logger.warning(
                        f"Cannot auto-resume pipeline for approval {approval_id}: "
                        f"No pipeline context found for job {approval.job_id}"
                    )

        # Handle rejection - auto-regenerate if enabled, otherwise mark as failed
        if decision.decision == "reject":

            # Auto-regenerate if enabled (default: True)
            if decision.auto_retry:
                # Trigger retry automatically using user comment as guidance
                pipeline_step = approval.pipeline_step
                input_data = approval.input_data

                # Get context from pipeline context
                context_data = await manager.load_pipeline_context(approval.job_id)
                context = context_data.get("context", {}) if context_data else {}

                # Create retry job
                retry_job_id = str(uuid.uuid4())
                retry_job = await job_manager.create_job(
                    job_id=retry_job_id,
                    job_type=f"retry_step_{pipeline_step}",
                    content_id=approval.job_id,
                    metadata={
                        "original_job_id": approval.job_id,
                        "approval_id": approval_id,
                        "step_name": pipeline_step,
                        "retry_attempt": approval.retry_count + 1,
                    },
                )

                # Submit to ARQ worker with user guidance
                await job_manager.submit_to_arq(
                    retry_job_id,
                    "retry_step_job",
                    step_name=pipeline_step,
                    input_data=input_data,
                    context=context,
                    job_id=retry_job_id,
                    approval_id=approval_id,
                    user_guidance=decision.comment,  # Use comment as guidance
                )

                # Update approval with retry info
                approval.retry_job_id = retry_job_id
                approval.retry_count += 1
                await manager._save_approval_to_redis(approval)

                logger.info(
                    f"Auto-regenerated step '{pipeline_step}' for approval {approval_id} "
                    f"(retry job: {retry_job_id}, attempt: {approval.retry_count})"
                )
            else:
                # Mark job as failed if auto_retry is False
                job = await job_manager.get_job(approval.job_id)
                if job and job.status == JobStatus.WAITING_FOR_APPROVAL:
                    await job_manager.update_job_status(
                        approval.job_id, JobStatus.FAILED
                    )
                    job.error = (
                        f"Approval rejected: {decision.comment or 'No reason provided'}"
                    )
                    await job_manager._save_job(job)
                    logger.info(
                        f"Job {approval.job_id} marked as FAILED due to approval rejection (auto_retry=False)"
                    )

        return approval

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to decide approval {approval_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/all")
async def get_all_approvals_for_job(job_id: str):
    """
    Get all approvals across the entire job chain.

    Returns approvals grouped by job, with pending approvals highlighted.
    """
    try:
        from ..services.job_manager import get_job_manager

        manager = await get_approval_manager()
        job_manager = get_job_manager()

        # Get job chain to find all related jobs
        chain_data = await job_manager.get_job_chain(job_id)
        if not chain_data["root_job_id"]:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        all_approvals_by_job = {}
        pending_approvals = []

        # Get approvals for each job in chain
        for job_id_in_chain in chain_data["chain_order"]:
            approvals = await manager.list_approvals(job_id=job_id_in_chain)
            all_approvals_by_job[job_id_in_chain] = [
                {
                    "id": a.id,
                    "step_name": a.step_name,
                    "agent_name": a.agent_name,
                    "status": a.status,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                    "reviewed_at": a.reviewed_at.isoformat() if a.reviewed_at else None,
                    "reviewed_by": a.reviewed_by,
                    "comment": a.user_comment,
                }
                for a in approvals
            ]

            # Collect pending approvals
            for a in approvals:
                if a.status == "pending":
                    pending_approvals.append(
                        {
                            "id": a.id,
                            "job_id": job_id_in_chain,
                            "step_name": a.step_name,
                            "agent_name": a.agent_name,
                            "created_at": (
                                a.created_at.isoformat() if a.created_at else None
                            ),
                        }
                    )

        return {
            "success": True,
            "root_job_id": chain_data["root_job_id"],
            "approvals_by_job": all_approvals_by_job,
            "pending_approvals": pending_approvals,
            "total_pending": len(pending_approvals),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get all approvals for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-approve")
async def bulk_approve(
    approval_ids: List[str] = Body(..., description="List of approval IDs to approve"),
    decision: ApprovalDecisionRequest = Body(..., description="Approval decision"),
):
    """
    Approve multiple similar approvals at once.

    Args:
        approval_ids: List of approval IDs to approve
        decision: Approval decision (should be "approve" for bulk operations)

    Returns:
        List of approved approvals
    """
    try:
        if decision.decision != "approve":
            raise HTTPException(
                status_code=400,
                detail="Bulk approve endpoint only supports 'approve' decision",
            )

        manager = await get_approval_manager()
        approved_approvals = []
        errors = []

        for approval_id in approval_ids:
            try:
                approval = await manager.get_approval(approval_id)
                if not approval:
                    errors.append({"approval_id": approval_id, "error": "Not found"})
                    continue

                if approval.status != "pending":
                    errors.append(
                        {
                            "approval_id": approval_id,
                            "error": f"Already {approval.status}",
                        }
                    )
                    continue

                # Approve the approval
                approved = await manager.decide_approval(approval_id, decision)
                approved_approvals.append(
                    {
                        "id": approved.id,
                        "job_id": approved.job_id,
                        "step_name": approved.step_name,
                        "status": approved.status,
                    }
                )
            except Exception as e:
                errors.append({"approval_id": approval_id, "error": str(e)})

        return {
            "success": True,
            "approved_count": len(approved_approvals),
            "approved_approvals": approved_approvals,
            "errors": errors,
            "total_requested": len(approval_ids),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to bulk approve: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job/{job_id}", response_model=PendingApprovalsResponse)
async def get_job_approvals(
    job_id: str, status: Optional[str] = Query(None, description="Filter by status")
):
    """
    Get all approvals for a specific job.

    Useful for tracking approval progress in a multi-step pipeline.
    """
    try:
        manager = await get_approval_manager(reload_from_db=True)

        # Force reload from Redis
        await manager._load_all_approvals_from_redis()

        # If no approvals found but job is waiting, try to recreate from pipeline context
        approvals = await manager.list_approvals(job_id=job_id, status=status)
        if len(approvals) == 0:
            # Check if job is waiting for approval and has pipeline context
            from ..services.job_manager import JobStatus, get_job_manager

            job_manager = get_job_manager()
            job = await job_manager.get_job(job_id)
            if job and job.status == JobStatus.WAITING_FOR_APPROVAL:
                # Try to load pipeline context which has the approval step info
                context_data = await manager.load_pipeline_context(job_id)
                if context_data:
                    logger.info(
                        f"Job {job_id} is waiting for approval but no approvals found. Attempting to recreate from pipeline context."
                    )
                    # Recreate approval from pipeline context
                    last_step = context_data.get("last_step", "")
                    step_result = context_data.get("step_result", {})
                    step_number = context_data.get("last_step_number", 0)

                    # Determine pipeline step name from last_step (e.g., "Step 1: seo_keywords" -> "seo_keywords")
                    pipeline_step = last_step
                    if ":" in last_step:
                        pipeline_step = last_step.split(":")[-1].strip()

                    # Create a synthetic approval from the context
                    try:
                        recreated_approval = await manager.create_approval_request(
                            job_id=job_id,
                            agent_name=pipeline_step,
                            step_name=last_step,
                            input_data=context_data.get("context", {}).get(
                                "input_content", {}
                            ),
                            output_data=step_result,
                            confidence_score=None,
                            suggestions=None,
                            pipeline_step=pipeline_step,
                        )
                        logger.info(
                            f"Recreated approval {recreated_approval.id} for job {job_id} from pipeline context"
                        )
                        # Reload approvals now that we've created one
                        approvals = await manager.list_approvals(
                            job_id=job_id, status=status
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to recreate approval from pipeline context: {e}"
                        )

        all_approvals = await manager.list_approvals(job_id=job_id)
        pending = await manager.list_approvals(job_id=job_id, status="pending")

        # Convert to list items
        items = [
            ApprovalListItem(
                id=a.id,
                job_id=a.job_id,
                agent_name=a.agent_name,
                step_name=a.step_name,
                status=a.status,
                created_at=a.created_at,
                reviewed_at=a.reviewed_at,
            )
            for a in approvals
        ]

        return PendingApprovalsResponse(
            approvals=items,
            total=len(all_approvals),
            pending=len(pending),
        )

    except Exception as e:
        logger.error(f"Failed to get approvals for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{approval_id}/retry")
async def retry_rejected_step(
    approval_id: str, retry_request: Optional[RetryStepRequest] = None
):
    """
    Retry a single rejected pipeline step.

    This endpoint creates a new job to re-execute the specific step that was rejected,
    without re-running the entire pipeline.

    Args:
        approval_id: ID of the rejected approval request

    Returns:
        Dictionary with new job details:
            - job_id: ID of the new retry job
            - step_name: Name of the step being retried
            - status: Initial job status
            - message: Success message
    """
    try:
        manager = await get_approval_manager()
        approval = await manager.get_approval(approval_id)

        if not approval:
            raise HTTPException(
                status_code=404, detail=f"Approval {approval_id} not found"
            )

        # Validate approval is rejected
        if approval.status != "rejected":
            raise HTTPException(
                status_code=400,
                detail=f"Can only retry rejected approvals. Current status: {approval.status}",
            )

        # Extract step context for retry
        pipeline_step = approval.pipeline_step
        input_data = approval.input_data
        job_id = approval.job_id

        # Get context from pipeline context
        context_data = await manager.load_pipeline_context(job_id)
        context = context_data.get("context", {}) if context_data else {}

        # Get user guidance from request if provided
        user_guidance = retry_request.guidance if retry_request else None

        # Create a new job for the retry
        import uuid

        from marketing_project.services.job_manager import get_job_manager

        job_manager = get_job_manager()
        retry_job_id = str(uuid.uuid4())

        retry_job = await job_manager.create_job(
            job_id=retry_job_id,
            job_type=f"retry_step_{pipeline_step}",
            content_id=job_id,  # Link to original job
            metadata={
                "original_job_id": job_id,
                "approval_id": approval_id,
                "step_name": pipeline_step,
                "retry_attempt": approval.retry_count + 1,
            },
        )

        # Submit to ARQ worker with user guidance
        await job_manager.submit_to_arq(
            retry_job_id,
            "retry_step_job",
            step_name=pipeline_step,
            input_data=input_data,
            context=context,
            job_id=retry_job_id,
            approval_id=approval_id,
            user_guidance=user_guidance,
        )

        # Update approval with retry info
        approval.retry_job_id = retry_job_id
        approval.retry_count += 1

        logger.info(
            f"Created retry job {retry_job_id} for approval {approval_id}, "
            f"step '{pipeline_step}' (attempt {approval.retry_count})"
        )

        return {
            "job_id": retry_job_id,
            "step_name": pipeline_step,
            "status": "queued",
            "message": f"Step '{pipeline_step}' retry initiated successfully",
            "retry_attempt": approval.retry_count,
            "approval_id": approval_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry step for approval {approval_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initiate retry: {str(e)}"
        )
