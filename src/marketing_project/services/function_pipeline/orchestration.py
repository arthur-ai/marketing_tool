"""
Pipeline orchestration utilities for shared logic between execute_pipeline and resume_pipeline.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from marketing_project.services.function_pipeline.tracing import (
    close_span,
    create_span,
    is_tracing_available,
    set_span_attribute,
    set_span_status,
)

logger = logging.getLogger("marketing_project.services.function_pipeline.orchestration")

# Import Status for tracing
try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None


async def load_pipeline_configs() -> Dict[str, Any]:
    """
    Load internal docs and design kit configurations.

    Returns:
        Dictionary with 'internal_docs_config' and 'design_kit_config' keys
    """
    configs = {
        "internal_docs_config": None,
        "design_kit_config": None,
    }

    # Load internal docs configuration (if available)
    try:
        from marketing_project.services.internal_docs_manager import (
            get_internal_docs_manager,
        )

        internal_docs_manager = await get_internal_docs_manager()
        internal_docs_config = await internal_docs_manager.get_active_config()
        if internal_docs_config:
            logger.info("Loaded internal docs configuration")
            configs["internal_docs_config"] = internal_docs_config
        else:
            logger.warning(
                "No internal docs configuration found - pipeline will run without it"
            )
    except Exception as e:
        logger.warning(
            f"Failed to load internal docs configuration: {e} - pipeline will run without it"
        )

    # Load design kit configuration (if available)
    try:
        from marketing_project.services.design_kit_manager import get_design_kit_manager

        design_kit_manager = await get_design_kit_manager()
        design_kit_config = await design_kit_manager.get_active_config()
        if design_kit_config:
            logger.info("Loaded design kit configuration")
            configs["design_kit_config"] = design_kit_config
        else:
            logger.warning(
                "No design kit configuration found - pipeline will run without it (allowing pipeline to continue)"
            )
    except Exception as e:
        logger.warning(
            f"Failed to load design kit configuration: {e} - pipeline will run without it (allowing pipeline to continue)"
        )

    return configs


def build_initial_context(
    content: Dict[str, Any],
    content_type: str,
    output_content_type: str,
    internal_docs_config: Optional[Any] = None,
    design_kit_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Build initial pipeline context.

    Args:
        content: Input content
        content_type: Content type
        output_content_type: Output content type
        internal_docs_config: Optional internal docs config
        design_kit_config: Optional design kit config

    Returns:
        Initial pipeline context dictionary
    """
    return {
        "input_content": content,
        "content_type": content_type,
        "output_content_type": output_content_type,
        "internal_docs_config": (
            internal_docs_config.model_dump(mode="json")
            if internal_docs_config
            else None
        ),
        "design_kit_config": (
            design_kit_config.model_dump(mode="json") if design_kit_config else None
        ),
    }


def filter_active_plugins(plugins: List[Any], content_type: str) -> List[Any]:
    """
    Filter plugins based on content type.

    Args:
        plugins: List of plugin objects
        content_type: Content type to filter by

    Returns:
        Filtered list of active plugins
    """
    active_plugins = []
    for plugin in plugins:
        # Skip transcript_preprocessing_approval for non-transcript content
        if (
            plugin.step_name == "transcript_preprocessing_approval"
            and content_type != "transcript"
        ):
            logger.info(f"Skipping step {plugin.step_name} (not transcript content)")
            continue
        # Skip blog_post_preprocessing_approval for non-blog_post content
        if (
            plugin.step_name == "blog_post_preprocessing_approval"
            and content_type != "blog_post"
        ):
            logger.info(f"Skipping step {plugin.step_name} (not blog_post content)")
            continue
        active_plugins.append(plugin)
    return active_plugins


async def update_job_progress(
    job_id: str,
    execution_index: int,
    total_steps: int,
    plugin: Any,
    pipeline_start_time: float,
) -> None:
    """
    Update job progress with ETA calculation.

    Args:
        job_id: Job ID
        execution_index: Current step index (1-based)
        total_steps: Total number of steps
        plugin: Current plugin
        pipeline_start_time: Pipeline start time
    """
    try:
        from marketing_project.services.job_manager import get_job_manager

        job_manager = get_job_manager()
        progress_percent = int((execution_index - 1) / total_steps * 100)

        # Calculate ETA based on average step time so far
        if execution_index > 1:
            elapsed_time = time.time() - pipeline_start_time
            avg_step_time = elapsed_time / (execution_index - 1)
            remaining_steps = total_steps - (execution_index - 1)
            estimated_remaining = avg_step_time * remaining_steps
            eta_seconds = int(estimated_remaining)
            eta_message = f" (ETA: ~{eta_seconds}s)"
        else:
            eta_message = ""

        step_description = plugin.step_name.replace("_", " ").title()
        progress_message = (
            f"Step {execution_index}/{total_steps}: {step_description}{eta_message}"
        )
        await job_manager.update_job_progress(
            job_id, progress_percent, progress_message
        )
    except Exception as e:
        logger.warning(f"Failed to update progress: {e}")


async def register_step_output(
    job_id: str,
    step_name: str,
    step_number: int,
    output_data: Dict[str, Any],
    pipeline_context: Dict[str, Any],
    required_context_keys: List[str],
) -> None:
    """
    Register step output in context registry.

    Args:
        job_id: Job ID
        step_name: Step name
        step_number: Step number
        output_data: Step output data
        pipeline_context: Pipeline context
        required_context_keys: Required context keys for the step
    """
    try:
        from marketing_project.services.context_registry import get_context_registry

        context_registry = get_context_registry()
        # Capture input snapshot before adding result
        input_snapshot = copy.deepcopy(pipeline_context)

        await context_registry.register_step_output(
            job_id=job_id,
            step_name=step_name,
            step_number=step_number,
            output_data=output_data,
            input_snapshot=input_snapshot,
            context_keys_used=required_context_keys,
        )
    except Exception as e:
        logger.warning(f"Failed to register step output in context registry: {e}")


def compile_pipeline_result(
    results: Dict[str, Any],
    content: Dict[str, Any],
    content_type: str,
    execution_time: float,
    total_tokens: int,
    model: str,
    step_info: List[Any],
    failed_steps: Optional[List[Dict[str, Any]]] = None,
    quality_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compile final pipeline result dictionary.

    Args:
        results: Step results dictionary
        content: Input content
        content_type: Content type
        execution_time: Total execution time
        total_tokens: Total tokens used
        model: Model used
        step_info: List of step info objects
        failed_steps: Optional list of failed steps
        quality_warnings: Optional list of quality warnings

    Returns:
        Compiled result dictionary
    """
    from datetime import datetime

    # Get final content from formatting result if available
    final_content = None
    if "content_formatting" in results:
        try:
            from marketing_project.models.pipeline_steps import ContentFormattingResult

            formatting = ContentFormattingResult(**results["content_formatting"])
            final_content = formatting.formatted_html
        except Exception as e:
            logger.warning(
                f"Failed to parse content_formatting result: {e}. "
                "Skipping final_content extraction."
            )
            # Try to extract formatted_html directly if it's a dict
            if isinstance(results["content_formatting"], dict):
                final_content = results["content_formatting"].get("formatted_html")

    # Build metadata
    metadata = {
        "content_id": content.get("id"),
        "content_type": content_type,
        "title": content.get("title"),
        "steps_completed": len(results),
        "execution_time_seconds": execution_time,
        "total_tokens_used": total_tokens,
        "model": model,
        "completed_at": datetime.utcnow().isoformat(),
        "step_info": [
            (
                step.model_dump(mode="json")
                if hasattr(step, "model_dump")
                else (step.model_dump() if hasattr(step, "model_dump") else step)
            )
            for step in step_info
        ],
    }

    # Add failed steps info if any
    if failed_steps:
        metadata["failed_steps"] = failed_steps
        metadata["partial_success"] = True

    return {
        "pipeline_status": (
            "completed_with_warnings" if quality_warnings else "completed"
        ),
        "step_results": results,
        "quality_warnings": quality_warnings or [],
        "final_content": final_content,
        "input_content": content,  # Include input content in result
        "metadata": metadata,
    }
