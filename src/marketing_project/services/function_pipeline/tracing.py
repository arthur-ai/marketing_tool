"""
OpenTelemetry tracing utilities for the function pipeline.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("marketing_project.services.function_pipeline.tracing")

# OpenTelemetry imports for tracing
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    _tracing_available = True
except ImportError:
    _tracing_available = False
    logger.debug("OpenTelemetry not available, tracing disabled")


def is_tracing_available() -> bool:
    """Check if OpenTelemetry tracing is available."""
    return _tracing_available


def get_tracer(name: str = __name__):
    """Get an OpenTelemetry tracer."""
    if not _tracing_available:
        return None
    return trace.get_tracer(name)


def create_span(
    name: str,
    kind: Optional[trace.SpanKind] = None,
    attributes: Optional[dict] = None,
):
    """
    Create and enter an OpenTelemetry span.

    Args:
        name: Span name
        kind: Span kind (default: INTERNAL)
        attributes: Initial span attributes

    Returns:
        Span object or None if tracing unavailable
    """
    if not _tracing_available:
        return None

    try:
        tracer = trace.get_tracer(__name__)
        span = tracer.start_as_current_span(name, kind=kind or trace.SpanKind.INTERNAL)
        span.__enter__()

        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception:
                    pass

        return span
    except Exception as e:
        logger.debug(f"Failed to create span {name}: {e}")
        return None


def set_span_status(span, status_code: StatusCode, description: Optional[str] = None):
    """Set span status safely."""
    if not span or not _tracing_available:
        return
    try:
        span.set_status(Status(status_code, description))
    except Exception:
        pass


def set_span_attribute(span, key: str, value):
    """Set span attribute safely."""
    if not span or not _tracing_available:
        return
    try:
        span.set_attribute(key, value)
    except Exception:
        pass


def set_span_input(span, input_value: Any, input_mime_type: str = "application/json"):
    """
    Set input attributes on a span in OpenInference format.

    Args:
        span: The span to set input attributes on
        input_value: Input data (will be JSON serialized if not a string)
        input_mime_type: MIME type for input (default: "application/json")
    """
    if not span or not _tracing_available:
        return

    try:
        import json

        # Serialize input value to JSON string if it's not already a string
        if isinstance(input_value, str):
            input_json = input_value
        else:
            input_json = json.dumps(input_value, default=str)

        span.set_attribute("input.value", input_json)
        span.set_attribute("input.mime_type", input_mime_type)
    except Exception as e:
        logger.debug(f"Failed to set input attributes: {e}")


def set_span_output(
    span, output_value: Any, output_mime_type: str = "application/json"
):
    """
    Set output attributes on a span in OpenInference format.

    Args:
        span: The span to set output attributes on
        output_value: Output data (will be JSON serialized if not a string)
        output_mime_type: MIME type for output (default: "application/json")
    """
    if not span or not _tracing_available:
        return

    try:
        import json

        # Serialize output value to JSON string if it's not already a string
        if isinstance(output_value, str):
            output_json = output_value
        else:
            output_json = json.dumps(output_value, default=str)

        span.set_attribute("output.value", output_json)
        span.set_attribute("output.mime_type", output_mime_type)
    except Exception as e:
        logger.debug(f"Failed to set output attributes: {e}")


def set_job_output(span, output_value: Any, output_mime_type: str = "application/json"):
    """
    Set output attributes on a job span in OpenInference format.

    This is a convenience alias for set_span_output() for backward compatibility.

    Args:
        span: The span to set output attributes on
        output_value: Output data (will be JSON serialized)
        output_mime_type: MIME type for output (default: "application/json")
    """
    set_span_output(span, output_value, output_mime_type)


def set_span_kind(span, kind_value: str):
    """
    Set OpenInference span kind attribute.

    Args:
        span: The span to set kind on
        kind_value: OpenInference span kind (e.g., "LLM", "CHAIN", "TOOL", "AGENT", "GUARDRAIL", "RETRIEVER", "EMBEDDING")
    """
    if not span or not _tracing_available:
        return

    try:
        span.set_attribute("openinference.span.kind", kind_value)
    except Exception as e:
        logger.debug(f"Failed to set span kind: {e}")


def set_llm_messages(span, input_messages: Any, output_messages: Any = None):
    """
    Set LLM input and output messages on a span in OpenInference format.

    Args:
        span: The span to set messages on
        input_messages: Messages sent to LLM (list of message dicts, not a JSON string)
        output_messages: Optional messages received from LLM (list of message dicts, not a JSON string)
    """
    if not span or not _tracing_available:
        return

    try:
        import json

        # Serialize input messages
        if input_messages is not None:
            # If input_messages is a string, parse it as JSON first to convert to dict/list
            if isinstance(input_messages, str):
                try:
                    input_messages = json.loads(input_messages)
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, log and use the string as-is (fallback)
                    logger.warning(
                        f"input_messages is a string but not valid JSON, using as-is: {input_messages[:100]}"
                    )
                    input_messages_json = input_messages
                else:
                    # Successfully parsed, now serialize the dict/list
                    input_messages_json = json.dumps(input_messages, default=str)
            else:
                # Already a dict/list, serialize it
                input_messages_json = json.dumps(input_messages, default=str)
            span.set_attribute("llm.input_messages", input_messages_json)

        # Serialize output messages if provided
        if output_messages is not None:
            # If output_messages is a string, parse it as JSON first to convert to dict/list
            if isinstance(output_messages, str):
                try:
                    output_messages = json.loads(output_messages)
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, log and use the string as-is (fallback)
                    logger.warning(
                        f"output_messages is a string but not valid JSON, using as-is: {output_messages[:100]}"
                    )
                    output_messages_json = output_messages
                else:
                    # Successfully parsed, now serialize the dict/list
                    output_messages_json = json.dumps(output_messages, default=str)
            else:
                # Already a dict/list, serialize it
                output_messages_json = json.dumps(output_messages, default=str)
            span.set_attribute("llm.output_messages", output_messages_json)
    except Exception as e:
        logger.debug(f"Failed to set LLM messages: {e}")


def set_llm_token_counts(
    span,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
):
    """
    Set LLM token counts on a span in OpenInference format.

    Args:
        span: The span to set token counts on
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total number of tokens (will be calculated if not provided but prompt/completion are)
    """
    if not span or not _tracing_available:
        return

    try:
        # Set prompt tokens
        if prompt_tokens is not None:
            span.set_attribute("llm.token_count.prompt", prompt_tokens)

        # Set completion tokens
        if completion_tokens is not None:
            span.set_attribute("llm.token_count.completion", completion_tokens)

        # Set total tokens (use provided value or calculate from prompt + completion)
        if total_tokens is not None:
            span.set_attribute("llm.token_count.total", total_tokens)
        elif prompt_tokens is not None and completion_tokens is not None:
            calculated_total = prompt_tokens + completion_tokens
            span.set_attribute("llm.token_count.total", calculated_total)
    except Exception as e:
        logger.debug(f"Failed to set LLM token counts: {e}")


def set_prompt_template_variables(span, variables: Any):
    """
    Set prompt template variables on a span in OpenInference format.

    Args:
        span: The span to set template variables on
        variables: Template variables (dict, list, or JSON-serializable object)
    """
    if not span or not _tracing_available:
        return

    try:
        import json

        # Serialize variables
        if isinstance(variables, str):
            variables_json = variables
        else:
            variables_json = json.dumps(variables, default=str)

        span.set_attribute("llm.prompt_template.variables", variables_json)
    except Exception as e:
        logger.debug(f"Failed to set prompt template variables: {e}")


def add_job_metadata_to_span(span, job, job_id: str, job_type: str):
    """
    Add all important job metadata to a span in OpenInference format.

    This function extracts metadata from the job object and adds it to the span
    following OpenInference conventions with nested attribute names.

    Args:
        span: The span to add metadata to
        job: Job object (can be None)
        job_id: Job identifier
        job_type: Type of job
    """
    if not span or not _tracing_available:
        return

    try:
        import json

        # Basic job information
        set_span_attribute(span, "job.id", job_id)
        set_span_attribute(span, "job.type", job_type)

        if job:
            # Job status and timing
            set_span_attribute(
                span,
                "job.status",
                job.status.value if hasattr(job.status, "value") else str(job.status),
            )
            set_span_attribute(span, "job.progress", job.progress)
            if job.current_step:
                set_span_attribute(span, "job.current_step", job.current_step)
            if job.content_id:
                set_span_attribute(span, "job.content_id", job.content_id)

            # Timestamps
            if job.created_at:
                set_span_attribute(
                    span,
                    "job.created_at",
                    (
                        job.created_at.isoformat()
                        if hasattr(job.created_at, "isoformat")
                        else str(job.created_at)
                    ),
                )
            if job.started_at:
                set_span_attribute(
                    span,
                    "job.started_at",
                    (
                        job.started_at.isoformat()
                        if hasattr(job.started_at, "isoformat")
                        else str(job.started_at)
                    ),
                )
            if job.completed_at:
                set_span_attribute(
                    span,
                    "job.completed_at",
                    (
                        job.completed_at.isoformat()
                        if hasattr(job.completed_at, "isoformat")
                        else str(job.completed_at)
                    ),
                )

            # Metadata fields (following OpenInference format)
            if job.metadata:
                metadata = job.metadata

                # User and session information
                if "user_id" in metadata:
                    set_span_attribute(span, "user.id", metadata["user_id"])
                if "session_id" in metadata:
                    set_span_attribute(span, "session.id", metadata["session_id"])

                # Content information
                if "content_type" in metadata:
                    set_span_attribute(
                        span, "metadata.content_type", metadata["content_type"]
                    )
                if "output_content_type" in metadata:
                    set_span_attribute(
                        span,
                        "metadata.output_content_type",
                        metadata["output_content_type"],
                    )
                if "title" in metadata:
                    set_span_attribute(span, "metadata.title", metadata["title"])

                # Pipeline configuration
                if "pipeline_config" in metadata:
                    try:
                        pipeline_config_json = json.dumps(
                            metadata["pipeline_config"], default=str
                        )
                        set_span_attribute(
                            span, "metadata.pipeline_config", pipeline_config_json
                        )
                    except Exception:
                        pass

                # Step information
                if "step_name" in metadata:
                    set_span_attribute(
                        span, "metadata.step_name", metadata["step_name"]
                    )
                if "step_number" in metadata:
                    set_span_attribute(
                        span, "metadata.step_number", metadata["step_number"]
                    )
                if "context_keys" in metadata:
                    try:
                        context_keys_json = json.dumps(
                            metadata["context_keys"], default=str
                        )
                        set_span_attribute(
                            span, "metadata.context_keys", context_keys_json
                        )
                    except Exception:
                        pass

                # Job relationships
                if "original_job_id" in metadata:
                    set_span_attribute(
                        span, "metadata.original_job_id", metadata["original_job_id"]
                    )
                if "parent_job_id" in metadata:
                    set_span_attribute(
                        span, "metadata.parent_job_id", metadata["parent_job_id"]
                    )
                if "resume_job_id" in metadata:
                    set_span_attribute(
                        span, "metadata.resume_job_id", metadata["resume_job_id"]
                    )

                # Social media specific
                if "social_media_platform" in metadata:
                    set_span_attribute(
                        span,
                        "metadata.social_media_platform",
                        metadata["social_media_platform"],
                    )
                if "social_media_platforms" in metadata:
                    try:
                        platforms_json = json.dumps(
                            metadata["social_media_platforms"], default=str
                        )
                        set_span_attribute(
                            span, "metadata.social_media_platforms", platforms_json
                        )
                    except Exception:
                        pass
                if "email_type" in metadata:
                    set_span_attribute(
                        span, "metadata.email_type", metadata["email_type"]
                    )
                if "variations_count" in metadata:
                    set_span_attribute(
                        span, "metadata.variations_count", metadata["variations_count"]
                    )

                # Approval information
                if "approval_id" in metadata:
                    set_span_attribute(
                        span, "metadata.approval_id", metadata["approval_id"]
                    )

                # Additional metadata (store as JSON for complex structures)
                important_keys = {
                    "runId",
                    "threadId",
                    "resourceId",
                    "run_id",
                    "thread_id",
                    "resource_id",
                    "subjob_ids",
                    "subjob_status",
                    "chain_status",
                    "step_count",
                }
                for key in important_keys:
                    if key in metadata:
                        try:
                            value = metadata[key]
                            if isinstance(value, (dict, list)):
                                value_json = json.dumps(value, default=str)
                                set_span_attribute(span, f"metadata.{key}", value_json)
                            else:
                                set_span_attribute(span, f"metadata.{key}", value)
                        except Exception:
                            pass

            # Extract LLM token counts and model information from job result
            # Following OpenInference format: llm.provider, llm.model_name, llm.token_count.total
            if job.result and isinstance(job.result, dict):
                result_metadata = job.result.get("metadata", {})

                # LLM Provider (we use OpenAI, format matches OpenInference: "openai.responses")
                set_span_attribute(span, "llm.provider", "openai.responses")

                # Aggregate token counts and models from step_info
                step_info = result_metadata.get("step_info", [])
                total_tokens = 0
                models_used = set()
                model_token_counts = {}
                invocation_params = {}

                if step_info:
                    for step in step_info:
                        if isinstance(step, dict):
                            step_tokens = step.get("tokens_used")
                            step_model = step.get("model")
                            step_temperature = step.get("temperature")

                            if step_tokens:
                                total_tokens += step_tokens

                            if step_model:
                                models_used.add(step_model)
                                if step_tokens:
                                    model_token_counts[step_model] = (
                                        model_token_counts.get(step_model, 0)
                                        + step_tokens
                                    )

                            # Capture invocation parameters (temperature) if available
                            if step_temperature is not None:
                                if "temperature" not in invocation_params:
                                    invocation_params["temperature"] = step_temperature

                # If no step_info, try to get from metadata directly
                if not step_info:
                    total_tokens = result_metadata.get("total_tokens_used", 0)
                    model = result_metadata.get("model")
                    if model:
                        models_used.add(model)

                # Set token count (OpenInference format: llm.token_count.total)
                if total_tokens:
                    set_span_attribute(span, "llm.token_count.total", total_tokens)

                # Set model name (use primary model if single, or first if multiple)
                if models_used:
                    models_list = list(models_used)
                    # Use the model from metadata if available, otherwise use first from step_info
                    primary_model = result_metadata.get("model") or (
                        models_list[0] if models_list else None
                    )
                    if primary_model:
                        set_span_attribute(span, "llm.model_name", primary_model)

                    # If multiple models, store all models used
                    if len(models_list) > 1:
                        set_span_attribute(
                            span, "llm.models_used", json.dumps(models_list)
                        )

                # Set invocation parameters (temperature, etc.)
                if invocation_params:
                    set_span_attribute(
                        span, "llm.invocation_parameters", json.dumps(invocation_params)
                    )

                # Store step info summary for detailed breakdown
                if step_info:
                    step_summary = [
                        {
                            "step_name": step.get("step_name"),
                            "step_number": step.get("step_number"),
                            "tokens_used": step.get("tokens_used"),
                            "model": step.get("model"),
                            "temperature": step.get("temperature"),
                        }
                        for step in step_info
                        if isinstance(step, dict)
                    ]
                    if step_summary:
                        set_span_attribute(
                            span, "metadata.step_info", json.dumps(step_summary)
                        )

                # Execution time
                execution_time = result_metadata.get("execution_time_seconds")
                if execution_time:
                    set_span_attribute(
                        span, "metadata.execution_time_seconds", execution_time
                    )

                # Pipeline status
                pipeline_status = job.result.get("pipeline_status")
                if pipeline_status:
                    set_span_attribute(
                        span, "metadata.pipeline_status", pipeline_status
                    )
    except Exception as e:
        logger.debug(f"Failed to add job metadata to span: {e}")


def record_span_exception(span, exception: Exception):
    """Record exception on span safely."""
    if not span or not _tracing_available:
        return
    try:
        span.record_exception(exception)
    except Exception:
        pass


def close_span(span, exc_type=None, exc_val=None, exc_tb=None):
    """Close span safely."""
    if not span or not _tracing_available:
        return
    try:
        span.__exit__(exc_type, exc_val, exc_tb)
    except Exception:
        pass


def create_job_root_span(
    job_id: str,
    job_type: str,
    input_value: Optional[Any] = None,
    input_mime_type: str = "application/json",
    job: Optional[Any] = None,
    attributes: Optional[dict] = None,
):
    """
    Create a root span for a job execution.

    This span serves as the parent for all spans created during job execution,
    including agent runs, LLM calls, and tool invocations.

    The span is created as a root span (no parent) and uses OpenTelemetry's
    context propagation to ensure all child spans are properly nested.

    Args:
        job_id: Job identifier
        job_type: Type of job (e.g., "blog", "pipeline", "step_seo_keywords")
        input_value: Optional input data to record (will be JSON serialized)
        input_mime_type: MIME type for input (default: "application/json")
        job: Optional job object to extract metadata from
        attributes: Optional additional span attributes

    Returns:
        Span context manager or None if tracing unavailable
    """
    if not _tracing_available:
        return None

    try:
        import json

        tracer = trace.get_tracer(__name__)
        span_name = f"job.{job_type}"

        # Create root span (no parent context)
        span = tracer.start_as_current_span(span_name, kind=trace.SpanKind.INTERNAL)
        span.__enter__()

        # Set standard job attributes
        span.set_attribute("job.id", job_id)
        span.set_attribute("job.type", job_type)

        # Set input attributes in OpenInference format
        if input_value is not None:
            try:
                # Serialize input value to JSON string if it's not already a string
                if isinstance(input_value, str):
                    input_json = input_value
                else:
                    input_json = json.dumps(input_value, default=str)

                span.set_attribute("input.value", input_json)
                span.set_attribute("input.mime_type", input_mime_type)
            except Exception as e:
                logger.debug(f"Failed to set input attributes: {e}")

        # Add all job metadata to the span
        if job:
            add_job_metadata_to_span(span, job, job_id, job_type)

        # Set additional attributes if provided (these override metadata from job)
        if attributes:
            for key, value in attributes.items():
                try:
                    # Handle nested attributes (e.g., "user.id", "session.id")
                    if "." in key:
                        # OpenTelemetry doesn't support nested attributes directly,
                        # so we'll use dot notation as the key name
                        span.set_attribute(key, value)
                    else:
                        span.set_attribute(key, value)
                except Exception:
                    pass

        return span
    except Exception as e:
        logger.debug(f"Failed to create job root span for {job_id}: {e}")
        return None
