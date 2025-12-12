"""
Step result saving and tracking utilities.
"""

import copy
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel

logger = logging.getLogger("marketing_project.services.function_pipeline.step_results")


async def save_step_result(
    parsed_result: Optional[BaseModel],
    step_name: str,
    step_number: int,
    job_id: str,
    context: Optional[Dict[str, Any]],
    execution_time: float,
    response_usage: Optional[Any] = None,
    status: str = "success",
    error_message: Optional[str] = None,
) -> None:
    """
    Save step result to disk using step_result_manager.

    Args:
        parsed_result: The parsed result from LLM (None for failures)
        step_name: Name of the step
        step_number: Step number
        job_id: Job ID
        context: Pipeline context
        execution_time: Execution time in seconds
        response_usage: Optional response usage object for token tracking
        status: Status of the step ("success" or "failed")
        error_message: Optional error message for failed steps
    """
    try:
        from marketing_project.services.step_result_manager import (
            get_step_result_manager,
        )

        step_manager = get_step_result_manager()

        # Prepare result data
        if status == "success" and parsed_result:
            # Use model_dump(mode='json') to ensure datetime objects are serialized to strings
            if hasattr(parsed_result, "model_dump"):
                try:
                    result_data = parsed_result.model_dump(mode="json")
                except (TypeError, ValueError):
                    # Fallback to regular model_dump if mode='json' fails
                    result_data = parsed_result.model_dump()
            else:
                result_data = parsed_result
        else:
            result_data = {}

        # Capture input snapshot and context keys used
        input_snapshot = None
        context_keys_used = []
        if context:
            # Create a snapshot of the context state before execution
            input_snapshot = copy.deepcopy(context)
            # Get context keys that were available
            context_keys_used = list(context.keys())

        # Build metadata
        metadata = {
            "execution_time": execution_time,
            "status": status,
        }
        if response_usage and response_usage.total_tokens:
            metadata["tokens_used"] = response_usage.total_tokens
        if error_message:
            metadata["error_message"] = error_message

        await step_manager.save_step_result(
            job_id=job_id,
            step_number=step_number,
            step_name=step_name,
            result_data=result_data,
            metadata=metadata,
            execution_context_id=None,  # Will be auto-determined
            root_job_id=None,  # Will be auto-determined
            input_snapshot=input_snapshot,
            context_keys_used=context_keys_used,
        )
    except Exception as e:
        logger.warning(
            f"Failed to save step result to disk for step {step_number}: {e}"
        )
