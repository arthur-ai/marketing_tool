"""
Approval integration for human-in-the-loop review.
"""

import copy
import json
import logging
import time
from datetime import date, datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel

logger = logging.getLogger("marketing_project.services.function_pipeline.approval")


async def check_step_approval(
    parsed_result: BaseModel,
    step_name: str,
    step_number: int,
    job_id: str,
    prompt: str,
    system_instruction: str,
    context: Optional[Dict[str, Any]],
    start_time: float,
    step_info_list: list,
):
    """
    Check if step requires approval and handle approval workflow.

    Args:
        parsed_result: The parsed result from LLM
        step_name: Name of the step
        step_number: Step number
        job_id: Job ID
        prompt: User prompt
        system_instruction: System instruction
        context: Pipeline context
        start_time: Start time for execution
        step_info_list: List to append step info to

    Returns:
        ApprovalResult indicating whether approval is required. If requires_approval
        is True, the pipeline should stop and wait for approval.
    """
    if not job_id:
        from marketing_project.processors.approval_helper import (
            ApprovalResult,
            ApprovalStatus,
        )

        return ApprovalResult(status=ApprovalStatus.NOT_REQUIRED)

    try:
        from marketing_project.processors.approval_helper import (
            ApprovalCheckFailedException,
            ApprovalResult,
            check_and_create_approval_request,
        )
        from marketing_project.services.job_manager import JobStatus, get_job_manager

        logger.info(
            f"[APPROVAL] Checking approval for step {step_number} ({step_name}) in job {job_id}. "
            f"Content type: {context.get('content_type', 'unknown') if context else 'unknown'}"
        )

        # Convert result to dict for approval system — use mode="json" so
        # datetime/UUID fields are serialized; fall back to a JSON round-trip.
        def _to_jsonb_safe(obj: Any) -> Any:
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        try:
            result_dict = parsed_result.model_dump(mode="json")
        except (TypeError, ValueError):
            raw = parsed_result.model_dump()
            try:
                result_dict = json.loads(json.dumps(raw, default=_to_jsonb_safe))
            except Exception:
                result_dict = raw

        # Extract confidence score if available
        confidence = result_dict.get("confidence_score")

        # Prepare input data for approval context
        pipeline_content = context.get("input_content") if context else None
        content_for_approval = (
            pipeline_content if pipeline_content else {"title": "N/A", "content": "N/A"}
        )
        input_data = {
            "prompt": prompt[:500],  # Truncate for readability
            "system_instruction": system_instruction[:200],
            "context_keys": list(context.keys()) if context else [],
            "original_content": pipeline_content or content_for_approval,
        }

        # Check if approval is needed (returns ApprovalResult)
        approval_result = await check_and_create_approval_request(
            job_id=job_id,
            agent_name=step_name,
            step_name=f"Step {step_number}: {step_name}",
            step_number=step_number,
            input_data=input_data,
            output_data=result_dict,
            context=context or {},
            confidence_score=confidence,
            suggestions=[
                f"Review {step_name} output quality",
                "Check alignment with content goals",
                "Verify accuracy and appropriateness",
            ],
        )

        # Handle approval required case
        if approval_result.requires_approval:
            # Approval required - pipeline should stop
            logger.info(
                f"[APPROVAL] Step {step_number} ({step_name}) requires approval. "
                f"Pipeline stopping. Approval ID: {approval_result.approval_id}"
            )

            # Update job status to WAITING_FOR_APPROVAL
            job_manager = get_job_manager()
            await job_manager.update_job_status(job_id, JobStatus.WAITING_FOR_APPROVAL)
            await job_manager.update_job_progress(
                job_id,
                90,
                f"Waiting for approval at step {step_number}: {step_name}",
            )

            # Track step info
            from marketing_project.models.pipeline_steps import PipelineStepInfo

            execution_time = time.time() - start_time
            step_info = PipelineStepInfo(
                step_name=step_name,
                step_number=step_number,
                status="waiting_for_approval",
                execution_time=execution_time,
            )
            step_info_list.append(step_info)

            # Save step result to disk if job_id is provided (even if waiting for approval)
            if job_id:
                try:
                    from marketing_project.services.step_result_manager import (
                        get_step_result_manager,
                    )

                    step_manager = get_step_result_manager()
                    # Use model_dump(mode='json') to ensure datetime objects are serialized to strings
                    if hasattr(parsed_result, "model_dump"):
                        try:
                            result_data = parsed_result.model_dump(mode="json")
                        except (TypeError, ValueError):
                            raw = parsed_result.model_dump()
                            try:
                                result_data = json.loads(
                                    json.dumps(raw, default=_to_jsonb_safe)
                                )
                            except Exception:
                                result_data = raw
                    else:
                        result_data = parsed_result

                    # Capture input snapshot and context keys used
                    input_snapshot = None
                    context_keys_used = []
                    relative_step_number = None
                    if context:
                        # Create a snapshot of the context state before execution
                        input_snapshot = copy.deepcopy(context)
                        # Get context keys that were available
                        context_keys_used = list(context.keys())
                        # Extract relative_step_number from context if present
                        relative_step_number = context.get("_relative_step_number")
                        if relative_step_number is None and step_number:
                            # For initial execution, relative equals absolute
                            relative_step_number = step_number

                    await step_manager.save_step_result(
                        job_id=job_id,
                        step_number=step_number,
                        step_name=step_name,
                        result_data=result_data,
                        metadata={
                            "execution_time": execution_time,
                            "status": "waiting_for_approval",
                        },
                        execution_context_id=None,  # Will be auto-determined
                        root_job_id=None,  # Will be auto-determined
                        input_snapshot=input_snapshot,
                        context_keys_used=context_keys_used,
                        relative_step_number=relative_step_number,
                    )
                except Exception as e2:
                    logger.warning(
                        f"Failed to save step result to disk for step {step_number}: {e2}"
                    )

            # Return result indicating approval is required
            return approval_result

        # Approval not needed or auto-approved, continue pipeline
        logger.info(
            f"[APPROVAL] Approval check completed for step {step_number} ({step_name}). "
            f"Status: {approval_result.status.value}. Continuing pipeline."
        )

        return approval_result

    except Exception as e:
        # Approval system error - CRITICAL: Do not silently skip approvals
        # If approval check fails, we must fail the pipeline to ensure approvals are never skipped
        logger.error(
            f"[APPROVAL ERROR] Step {step_number} ({step_name}): Approval check failed. "
            f"This is a critical error - pipeline will fail to prevent skipping required approvals. "
            f"Error: {e}",
            exc_info=True,
        )

        # Re-raise the exception to fail the pipeline
        # This ensures that if approval is required but the check fails,
        # we don't silently skip the approval
        raise ApprovalCheckFailedException(
            step_name=step_name,
            step_number=step_number,
            original_error=e,
        )
