"""
Approval integration for human-in-the-loop review.
"""

import copy
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel

from marketing_project.services.function_pipeline.tracing import (
    close_span,
    create_span,
    is_tracing_available,
    record_span_exception,
    set_span_attribute,
    set_span_input,
    set_span_kind,
    set_span_output,
    set_span_status,
)

logger = logging.getLogger("marketing_project.services.function_pipeline.approval")

# Import Status for tracing
try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None


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
) -> None:
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

    Raises:
        ApprovalRequiredException: If approval is required
    """
    if not job_id:
        return

    # Create approval check span
    approval_span = None
    if is_tracing_available():
        approval_span = create_span(
            f"pipeline.approval_check.{step_name}",
            attributes={
                "step_name": step_name,
                "step_number": step_number,
                "job_id": job_id,
            },
        )
        if approval_span:
            # Set OpenInference span kind
            set_span_kind(approval_span, "GUARDRAIL")

            if context:
                content_type = context.get("content_type")
                if content_type:
                    set_span_attribute(approval_span, "content_type", content_type)

    try:
        from marketing_project.processors.approval_helper import (
            ApprovalCheckFailedException,
            ApprovalRequiredException,
            check_and_create_approval_request,
        )
        from marketing_project.services.job_manager import JobStatus, get_job_manager

        logger.info(
            f"[APPROVAL] Checking approval for step {step_number} ({step_name}) in job {job_id}. "
            f"Content type: {context.get('content_type', 'unknown') if context else 'unknown'}"
        )

        # Convert result to dict for approval system
        try:
            result_dict = parsed_result.model_dump(mode="json")
        except (TypeError, ValueError):
            result_dict = parsed_result.model_dump()

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

        # Set input attributes on span
        if approval_span:
            set_span_input(approval_span, input_data)

        # Check if approval is needed (raises ApprovalRequiredException if required)
        await check_and_create_approval_request(
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
        # If no exception raised, approval not needed or auto-approved, continue
        logger.info(
            f"[APPROVAL] No approval required for step {step_number} ({step_name}), continuing pipeline"
        )

        # Set output attributes (no approval needed)
        if approval_span:
            output_data = {
                "approval_required": False,
                "status": "auto_approved_or_not_needed",
                "step_name": step_name,
                "step_number": step_number,
            }
            set_span_output(approval_span, output_data)
            if Status and StatusCode:
                set_span_status(approval_span, StatusCode.OK)

    except ApprovalRequiredException as e:
        # Approval required - pipeline should stop
        logger.info(
            f"[APPROVAL] Step {step_number} ({step_name}) requires approval. "
            f"Pipeline stopping. Approval ID: {e.approval_id}"
        )

        # Set output attributes (approval required)
        if approval_span:
            output_data = {
                "approval_required": True,
                "status": "waiting_for_approval",
                "approval_id": e.approval_id,
                "step_name": step_name,
                "step_number": step_number,
            }
            set_span_output(approval_span, output_data)
            set_span_attribute(approval_span, "approval_id", e.approval_id)
            if Status and StatusCode:
                set_span_status(
                    approval_span, StatusCode.OK
                )  # Not an error, just needs approval

        # Update job status to WAITING_FOR_APPROVAL
        job_manager = get_job_manager()
        await job_manager.update_job_status(job_id, JobStatus.WAITING_FOR_APPROVAL)
        await job_manager.update_job_progress(
            job_id,
            90,
            f"Waiting for approval at step {step_number}: {step_name}",
        )

        # Track step info
        import time

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
                        # Fallback to regular model_dump if mode='json' fails
                        result_data = parsed_result.model_dump()
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

        # Re-raise to signal pipeline should stop
        raise

    except Exception as e:
        # Approval system error - CRITICAL: Do not silently skip approvals
        # If approval check fails, we must fail the pipeline to ensure approvals are never skipped
        if approval_span:
            record_span_exception(approval_span, e)
            if Status and StatusCode:
                set_span_status(approval_span, StatusCode.ERROR, str(e))
            set_span_attribute(approval_span, "error.type", type(e).__name__)

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
    finally:
        close_span(approval_span)
