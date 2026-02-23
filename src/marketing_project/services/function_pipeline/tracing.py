"""
OpenTelemetry tracing utilities for the function pipeline.
"""

import logging
import time
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
    span_type: Optional[str] = None,
):
    """
    Create and enter an OpenTelemetry span with minimum required metadata.

    Args:
        name: Span name
        kind: Span kind (default: INTERNAL)
        attributes: Initial span attributes
        span_type: Type of span for metadata inference (e.g., "step_execution", "llm_call")

    Returns:
        Span object or None if tracing unavailable
    """
    if not _tracing_available:
        return None

    try:
        tracer = trace.get_tracer(__name__)
        span = tracer.start_as_current_span(name, kind=kind or trace.SpanKind.INTERNAL)
        span.__enter__()

        # Set initial attributes
        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception:
                    pass

        # Ensure minimum metadata is present
        ensure_span_has_minimum_metadata(span, name, span_type)

        # Ensure input is always set (will be overridden if set_span_input is called later)
        # But we set a default to ensure span is never blank
        try:
            set_span_input(span, {})
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
    Always sets input.value and input.mime_type - never leaves blank.

    Args:
        span: The span to set input attributes on
        input_value: Input data (will be JSON serialized if not a string). If None, uses empty dict.
        input_mime_type: MIME type for input (default: "application/json")
    """
    if not span or not _tracing_available:
        return

    try:
        import json

        # Ensure we always have a value - use empty dict if None
        if input_value is None:
            input_value = {}

        # Serialize input value to JSON string if it's not already a string
        if isinstance(input_value, str):
            input_json = input_value if input_value else "{}"
        else:
            input_json = json.dumps(input_value, default=str)

        # Always set input.value and input.mime_type
        span.set_attribute("input.value", input_json)
        span.set_attribute("input.mime_type", input_mime_type)

        # Add content size metrics
        try:
            size_bytes = len(input_json.encode("utf-8"))
            span.set_attribute("content.input_size_bytes", size_bytes)
            # Rough token estimate: ~4 chars per token
            span.set_attribute("content.input_token_estimate", size_bytes // 4)
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Failed to set input attributes: {e}")
        # Even on error, try to set minimal input
        try:
            span.set_attribute("input.value", "{}")
            span.set_attribute("input.mime_type", "application/json")
        except Exception:
            pass


def set_span_output(
    span, output_value: Any, output_mime_type: str = "application/json"
):
    """
    Set output attributes on a span in OpenInference format.
    Always sets output.value and output.mime_type - never leaves blank.

    Args:
        span: The span to set output attributes on
        output_value: Output data (will be JSON serialized if not a string). If None, uses empty dict.
        output_mime_type: MIME type for output (default: "application/json")
    """
    if not span or not _tracing_available:
        return

    try:
        import json

        # Ensure we always have a value - use empty dict if None
        if output_value is None:
            output_value = {}

        # Serialize output value to JSON string if it's not already a string
        if isinstance(output_value, str):
            output_json = output_value if output_value else "{}"
        else:
            output_json = json.dumps(output_value, default=str)

        # Always set output.value and output.mime_type
        span.set_attribute("output.value", output_json)
        span.set_attribute("output.mime_type", output_mime_type)

        # Add content size metrics
        try:
            size_bytes = len(output_json.encode("utf-8"))
            span.set_attribute("content.output_size_bytes", size_bytes)
            # Rough token estimate: ~4 chars per token
            span.set_attribute("content.output_token_estimate", size_bytes // 4)
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Failed to set output attributes: {e}")
        # Even on error, try to set minimal output
        try:
            span.set_attribute("output.value", "{}")
            span.set_attribute("output.mime_type", "application/json")
        except Exception:
            pass


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

    Uses OpenInference flattened attribute convention:
    - llm.input_messages.<index>.message.role
    - llm.input_messages.<index>.message.content
    - llm.output_messages.<index>.message.role
    - llm.output_messages.<index>.message.content

    Args:
        span: The span to set messages on
        input_messages: Messages sent to LLM (list of message dicts, or JSON string to parse)
        output_messages: Optional messages received from LLM (list of message dicts, or JSON string to parse)
    """
    if not span or not _tracing_available:
        return

    try:
        import json

        def _parse_messages(messages: Any) -> list:
            """Parse messages from various formats into a list of message dicts."""
            if messages is None:
                return []

            # If it's a string, try to parse as JSON
            if isinstance(messages, str):
                try:
                    parsed = json.loads(messages)
                    # If parsed result is a list, return it
                    if isinstance(parsed, list):
                        return parsed
                    # If it's a dict, wrap it in a list
                    elif isinstance(parsed, dict):
                        return [parsed]
                    else:
                        logger.warning(
                            f"Parsed messages is not a list or dict: {type(parsed)}"
                        )
                        return []
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse messages JSON string: {e}. "
                        f"First 100 chars: {messages[:100]}"
                    )
                    return []

            # If it's already a list, return it
            if isinstance(messages, list):
                return messages

            # If it's a dict, wrap it in a list
            if isinstance(messages, dict):
                return [messages]

            # Unknown type
            logger.warning(
                f"Messages is of unexpected type: {type(messages)}. "
                f"Expected list, dict, or JSON string."
            )
            return []

        def _set_message_attributes(span, messages: list, prefix: str):
            """Set flattened OpenInference message attributes."""
            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    logger.debug(
                        f"Message at index {index} is not a dict, skipping: {type(message)}"
                    )
                    continue

                # Set role attribute
                role = message.get("role")
                if role is not None:
                    span.set_attribute(f"{prefix}.{index}.message.role", str(role))

                # Set content attribute
                content = message.get("content")
                if content is not None:
                    # Convert content to string if it's not already
                    if not isinstance(content, str):
                        content = json.dumps(content, default=str)
                    span.set_attribute(f"{prefix}.{index}.message.content", content)

                # Set any additional message properties (e.g., name, function_call, etc.)
                for key, value in message.items():
                    if key not in ("role", "content"):
                        attr_key = f"{prefix}.{index}.message.{key}"
                        # Convert value to string if needed
                        if not isinstance(value, (str, int, float, bool)):
                            value = json.dumps(value, default=str)
                        span.set_attribute(attr_key, value)

        # Process input messages
        if input_messages is not None:
            parsed_input = _parse_messages(input_messages)
            if parsed_input:
                _set_message_attributes(span, parsed_input, "llm.input_messages")

        # Process output messages
        if output_messages is not None:
            parsed_output = _parse_messages(output_messages)
            if parsed_output:
                _set_message_attributes(span, parsed_output, "llm.output_messages")
    except Exception as e:
        logger.debug(f"Failed to set LLM messages: {e}", exc_info=True)


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


def set_prompt_template(span, template: str, version: Optional[str] = None):
    """
    Set prompt template content and version on a span in OpenInference format.

    Args:
        span: The span to set template on
        template: Template content (e.g., Jinja2 template string)
        version: Optional template version (e.g., "v1.0")
    """
    if not span or not _tracing_available:
        return

    try:
        span.set_attribute("llm.prompt_template.template", template)
        if version:
            span.set_attribute("llm.prompt_template.version", version)
    except Exception as e:
        logger.debug(f"Failed to set prompt template: {e}")


def set_llm_response_format(span, response_format: Any):
    """
    Set LLM response format on a span in OpenInference format.

    Args:
        span: The span to set response format on
        response_format: Response format (can be Pydantic model, dict, or string)
    """
    if not span or not _tracing_available:
        return

    try:
        import json

        # If it's a Pydantic model, get its name
        if hasattr(response_format, "__name__"):
            format_name = response_format.__name__
            # Try to get schema if it's a Pydantic model
            if hasattr(response_format, "model_json_schema"):
                try:
                    schema = response_format.model_json_schema()
                    span.set_attribute("llm.response_format", format_name)
                    span.set_attribute(
                        "llm.response_format.schema", json.dumps(schema, default=str)
                    )
                except Exception:
                    span.set_attribute("llm.response_format", format_name)
            else:
                span.set_attribute("llm.response_format", format_name)
        elif isinstance(response_format, dict):
            span.set_attribute(
                "llm.response_format", json.dumps(response_format, default=str)
            )
        elif isinstance(response_format, str):
            span.set_attribute("llm.response_format", response_format)
        else:
            span.set_attribute("llm.response_format", str(response_format))
    except Exception as e:
        logger.debug(f"Failed to set LLM response format: {e}")


def set_llm_system_and_provider(span, system: str = "openai", provider: str = "openai"):
    """
    Set LLM system and provider attributes on a span in OpenInference format.

    Args:
        span: The span to set attributes on
        system: AI product/vendor (e.g., "openai", "anthropic", "cohere") - defaults to "openai"
        provider: Hosting provider (e.g., "openai", "azure", "google", "aws") - defaults to "openai"
    """
    if not span or not _tracing_available:
        return

    try:
        span.set_attribute("llm.system", system)
        span.set_attribute("llm.provider", provider)
    except Exception as e:
        logger.debug(f"Failed to set LLM system and provider: {e}")


def set_llm_invocation_parameters(span, parameters: dict):
    """
    Set LLM invocation parameters on a span in OpenInference format.

    Args:
        span: The span to set parameters on
        parameters: Dictionary of invocation parameters (e.g., {"temperature": 0.7, "max_tokens": 1000})
    """
    if not span or not _tracing_available:
        return

    try:
        import json

        # Serialize parameters as JSON string
        parameters_json = json.dumps(parameters, default=str)
        span.set_attribute("llm.invocation_parameters", parameters_json)
    except Exception as e:
        logger.debug(f"Failed to set LLM invocation parameters: {e}")


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

            # User information (from job.user_id or metadata)
            if job.user_id:
                set_span_attribute(span, "user.id", job.user_id)
            elif job.metadata and "triggered_by_user_id" in job.metadata:
                set_span_attribute(
                    span, "user.id", job.metadata["triggered_by_user_id"]
                )

            # Metadata fields (following OpenInference format)
            if job.metadata:
                metadata = job.metadata

                # User and session information (check metadata as fallback)
                if "user_id" in metadata and not job.user_id:
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

                # LLM System and Provider (we use OpenAI, per OpenInference spec)
                set_llm_system_and_provider(span, system="openai", provider="openai")

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


def set_span_duration(span, start_time: float):
    """
    Set span duration from start time.

    Args:
        span: The span to set duration on
        start_time: Start time from time.time()
    """
    if not span or not _tracing_available:
        return
    try:
        duration = time.time() - start_time
        span.set_attribute("duration_ms", duration * 1000)
        span.set_attribute("duration_seconds", duration)
    except Exception:
        pass


def extract_quality_metrics(span, result: Any):
    """
    Extract quality metrics from result and add to span.

    Args:
        span: The span to set quality metrics on
        result: Result object (Pydantic model, dict, or other)
    """
    if not span or not result:
        return

    try:
        # Extract result as dict
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        elif hasattr(result, "dict"):
            result_dict = result.dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            return

        # Extract quality metrics
        if "confidence_score" in result_dict:
            span.set_attribute(
                "quality.confidence_score", result_dict["confidence_score"]
            )

        # Extract other quality scores based on step type
        quality_attrs = [
            "relevance_score",
            "readability_score",
            "seo_score",
            "strategy_alignment_score",
            "keyword_density_score",
        ]
        for attr in quality_attrs:
            if attr in result_dict:
                span.set_attribute(f"quality.{attr}", result_dict[attr])
    except Exception:
        pass


def categorize_error(error: Exception) -> dict:
    """
    Categorize error for better observability.

    Args:
        error: Exception to categorize

    Returns:
        Dictionary with category, is_retryable, and recovery_action
    """
    error_type = type(error).__name__
    error_str = str(error).lower()

    category = "unknown"
    is_retryable = False
    recovery_action = "fail"
    user_visible = True
    requires_manual_intervention = False

    # Network errors
    if any(
        term in error_str for term in ["timeout", "connection", "network", "socket"]
    ):
        category = "network"
        is_retryable = True
        recovery_action = "retry"
        user_visible = False

    # Rate limiting
    elif "rate limit" in error_str or "429" in error_str or "quota" in error_str:
        category = "rate_limit"
        is_retryable = True
        recovery_action = "retry_with_backoff"
        user_visible = False

    # Validation errors
    elif any(
        term in error_str for term in ["validation", "invalid", "required", "missing"]
    ):
        category = "validation"
        is_retryable = False
        recovery_action = "fail"
        user_visible = True
        requires_manual_intervention = True

    # API errors
    elif "api" in error_str or error_type in [
        "APIError",
        "OpenAIError",
        "APIConnectionError",
    ]:
        category = "api_error"
        is_retryable = True
        recovery_action = "retry"
        user_visible = False

    # Authentication/Authorization errors
    elif any(
        term in error_str
        for term in ["auth", "unauthorized", "forbidden", "401", "403"]
    ):
        category = "authentication"
        is_retryable = False
        recovery_action = "fail"
        user_visible = True
        requires_manual_intervention = True

    # Timeout errors
    elif "timeout" in error_str or error_type in [
        "TimeoutError",
        "asyncio.TimeoutError",
    ]:
        category = "timeout"
        is_retryable = True
        recovery_action = "retry"
        user_visible = False

    # Value/Type errors
    elif error_type in ["ValueError", "TypeError", "KeyError", "AttributeError"]:
        category = "validation"
        is_retryable = False
        recovery_action = "fail"
        user_visible = True

    return {
        "category": category,
        "is_retryable": is_retryable,
        "recovery_action": recovery_action,
        "user_visible": user_visible,
        "requires_manual_intervention": requires_manual_intervention,
    }


def set_span_error(span, error: Exception, context: Optional[dict] = None):
    """
    Set comprehensive error attributes on a span.

    Args:
        span: The span to set error attributes on
        error: Exception that occurred
        context: Optional context dictionary to add as error.context.* attributes
    """
    if not span or not _tracing_available:
        return

    try:
        span.set_attribute("error", True)
        span.set_attribute("error.type", type(error).__name__)
        span.set_attribute("error.message", str(error))

        # Categorize error
        error_cat = categorize_error(error)
        span.set_attribute("error.category", error_cat["category"])
        span.set_attribute("error.is_retryable", error_cat["is_retryable"])
        span.set_attribute("error.recovery_action", error_cat["recovery_action"])
        span.set_attribute("error.user_visible", error_cat["user_visible"])
        span.set_attribute(
            "error.requires_manual_intervention",
            error_cat["requires_manual_intervention"],
        )

        # Add context if available
        if context:
            for key, value in context.items():
                try:
                    span.set_attribute(f"error.context.{key}", str(value))
                except Exception:
                    pass

        # Record exception (includes stacktrace)
        record_span_exception(span, error)
    except Exception as e:
        logger.debug(f"Failed to set span error: {e}")


def add_span_event(span, name: str, attributes: Optional[dict] = None):
    """
    Add an event to a span.

    Args:
        span: The span to add event to
        name: Event name
        attributes: Optional event attributes
    """
    if not span or not _tracing_available:
        return

    try:
        span.add_event(name, attributes=attributes or {})
    except Exception:
        pass


def link_spans(span, linked_span_context, attributes: Optional[dict] = None):
    """
    Link related spans.

    Args:
        span: The span to add link to
        linked_span_context: SpanContext of the linked span
        attributes: Optional attributes for the link
    """
    if not span or not _tracing_available or not linked_span_context:
        return

    try:
        from opentelemetry.trace import Link

        link = Link(linked_span_context, attributes=attributes or {})
        if not hasattr(span, "links"):
            span.links = []
        span.links.append(link)
    except Exception as e:
        logger.debug(f"Failed to link spans: {e}")


def extract_content_characteristics(content: Any) -> dict:
    """
    Extract content characteristics for observability.

    Args:
        content: Content object (dict, string, or other)

    Returns:
        Dictionary with content characteristics (0/None for missing values)
    """
    characteristics = {
        "word_count": 0,
        "character_count": 0,
        "has_images": False,
        "image_count": 0,
        "language": "unknown",
        "readability_score": None,
        "complexity": "unknown",
    }

    try:
        if isinstance(content, dict):
            # Extract text content
            text = None
            if "content" in content:
                text = str(content["content"])
            elif "text" in content:
                text = str(content["text"])
            elif "body" in content:
                text = str(content["body"])

            if text:
                words = text.split()
                characteristics["word_count"] = len(words)
                characteristics["character_count"] = len(text)

                # Determine complexity based on word count
                if characteristics["word_count"] < 100:
                    characteristics["complexity"] = "simple"
                elif characteristics["word_count"] < 500:
                    characteristics["complexity"] = "medium"
                else:
                    characteristics["complexity"] = "complex"

            # Image count
            if "images" in content:
                images = content["images"]
                if isinstance(images, list):
                    characteristics["image_count"] = len(images)
                    characteristics["has_images"] = len(images) > 0
                elif images:
                    characteristics["has_images"] = True
                    characteristics["image_count"] = 1

            # Content type specific
            if "category" in content:
                characteristics["blog_category"] = content["category"]
            if "tags" in content:
                characteristics["blog_tags_count"] = (
                    len(content["tags"]) if isinstance(content["tags"], list) else 1
                )
            if "duration" in content:
                characteristics["transcript_duration_seconds"] = content["duration"]

        elif isinstance(content, str):
            # Handle string content
            words = content.split()
            characteristics["word_count"] = len(words)
            characteristics["character_count"] = len(content)
            if len(words) < 100:
                characteristics["complexity"] = "simple"
            elif len(words) < 500:
                characteristics["complexity"] = "medium"
            else:
                characteristics["complexity"] = "complex"

    except Exception as e:
        logger.debug(f"Failed to extract content characteristics: {e}")

    return characteristics


def extract_step_business_metrics(step_name: str, result: Any) -> dict:
    """
    Extract step-specific business metrics from result.

    Args:
        step_name: Name of the step
        result: Result object (Pydantic model, dict, or other)

    Returns:
        Dictionary with step-specific business metrics
    """
    metrics = {}

    try:
        # Extract result as dict
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        elif hasattr(result, "dict"):
            result_dict = result.dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            return metrics

        # SEO Keywords Step
        if step_name == "seo_keywords" or "seo" in step_name.lower():
            if "main_keyword" in result_dict:
                metrics["seo.main_keyword"] = str(result_dict["main_keyword"])[
                    :100
                ]  # Truncate
            if "primary_keywords" in result_dict:
                primary = result_dict["primary_keywords"]
                metrics["seo.primary_keywords_count"] = (
                    len(primary) if isinstance(primary, list) else 1
                )
            if "secondary_keywords" in result_dict:
                secondary = result_dict["secondary_keywords"]
                metrics["seo.secondary_keywords_count"] = (
                    len(secondary) if isinstance(secondary, list) else 0
                )
            if "lsi_keywords" in result_dict:
                lsi = result_dict["lsi_keywords"]
                metrics["seo.lsi_keywords_count"] = (
                    len(lsi) if isinstance(lsi, list) else 0
                )
            if "search_intent" in result_dict:
                metrics["seo.search_intent"] = str(result_dict["search_intent"])
            if "keyword_clusters" in result_dict:
                clusters = result_dict["keyword_clusters"]
                metrics["seo.keyword_clusters_count"] = (
                    len(clusters) if isinstance(clusters, list) else 0
                )

        # Marketing Brief Step
        elif step_name == "marketing_brief" or "marketing" in step_name.lower():
            if "target_audience" in result_dict:
                audience = result_dict["target_audience"]
                if isinstance(audience, list):
                    metrics["marketing.target_audience_count"] = len(audience)
                else:
                    metrics["marketing.target_audience_count"] = 1
            if "key_messages" in result_dict:
                messages = result_dict["key_messages"]
                if isinstance(messages, list):
                    metrics["marketing.key_messages_count"] = len(messages)
                else:
                    metrics["marketing.key_messages_count"] = 1
            if "competitive_angle" in result_dict:
                metrics["marketing.has_competitive_angle"] = bool(
                    result_dict["competitive_angle"]
                )
            if "kpis" in result_dict:
                kpis = result_dict["kpis"]
                metrics["marketing.kpis_count"] = (
                    len(kpis) if isinstance(kpis, list) else 0
                )

        # Article Generation Step
        elif "article" in step_name.lower() or "blog" in step_name.lower():
            if "word_count" in result_dict:
                metrics["article.word_count_actual"] = result_dict["word_count"]
            if "target_word_count" in result_dict:
                metrics["article.word_count_target"] = result_dict["target_word_count"]
            if "sections" in result_dict:
                sections = result_dict["sections"]
                metrics["article.sections_count"] = (
                    len(sections) if isinstance(sections, list) else 0
                )
            if "cta" in result_dict:
                metrics["article.has_cta"] = bool(result_dict["cta"])
            if "internal_links" in result_dict:
                links = result_dict["internal_links"]
                metrics["article.internal_links_count"] = (
                    len(links) if isinstance(links, list) else 0
                )

    except Exception as e:
        logger.debug(f"Failed to extract step business metrics for {step_name}: {e}")

    return metrics


def extract_model_info(model_name: str) -> dict:
    """
    Extract model family and version from model name.

    Args:
        model_name: Model name (e.g., "gpt-4-turbo-preview", "gpt-3.5-turbo")

    Returns:
        Dictionary with model.family and model.version
    """
    info = {
        "model.family": "unknown",
        "model.version": "unknown",
    }

    try:
        if not model_name:
            return info

        model_lower = model_name.lower()

        # Extract family
        if "gpt-4" in model_lower:
            info["model.family"] = "gpt-4"
        elif "gpt-3.5" in model_lower or "gpt-35" in model_lower:
            info["model.family"] = "gpt-3.5"
        elif "gpt-3" in model_lower:
            info["model.family"] = "gpt-3"
        elif "gpt" in model_lower:
            info["model.family"] = "gpt"
        else:
            # Try to extract first part as family
            parts = model_name.split("-")
            if parts:
                info["model.family"] = parts[0]

        # Extract version
        if "turbo" in model_lower:
            info["model.version"] = "turbo"
        elif "base" in model_lower:
            info["model.version"] = "base"
        elif "preview" in model_lower:
            info["model.version"] = "preview"
        elif "mini" in model_lower:
            info["model.version"] = "mini"
        else:
            # Try to extract version from model name
            parts = model_name.split("-")
            if len(parts) > 1:
                info["model.version"] = "-".join(parts[1:])

    except Exception as e:
        logger.debug(f"Failed to extract model info from {model_name}: {e}")

    return info


def calculate_schema_metrics(schema: dict) -> dict:
    """
    Calculate schema complexity and metrics.

    Args:
        schema: JSON schema dictionary

    Returns:
        Dictionary with schema metrics
    """
    metrics = {
        "schema.complexity_score": 0,
        "schema.field_count": 0,
        "schema.nested_depth": 0,
        "schema.required_fields_count": 0,
        "schema.optional_fields_count": 0,
    }

    try:
        if not isinstance(schema, dict):
            return metrics

        def count_fields(obj: dict, depth: int = 0) -> tuple:
            """Recursively count fields and calculate depth."""
            field_count = 0
            max_depth = depth
            required_count = 0
            optional_count = 0

            if "properties" in obj:
                props = obj["properties"]
                if isinstance(props, dict):
                    field_count = len(props)
                    required = obj.get("required", [])
                    required_count = len(required)
                    optional_count = field_count - required_count

                    for prop_name, prop_value in props.items():
                        if isinstance(prop_value, dict):
                            sub_fields, sub_depth, sub_req, sub_opt = count_fields(
                                prop_value, depth + 1
                            )
                            max_depth = max(max_depth, sub_depth)

            return field_count, max_depth, required_count, optional_count

        field_count, nested_depth, required_count, optional_count = count_fields(schema)

        # Calculate complexity score (rough estimate)
        complexity_score = field_count * (nested_depth + 1) * 10

        metrics["schema.field_count"] = field_count
        metrics["schema.nested_depth"] = nested_depth
        metrics["schema.required_fields_count"] = required_count
        metrics["schema.optional_fields_count"] = optional_count
        metrics["schema.complexity_score"] = complexity_score

    except Exception as e:
        logger.debug(f"Failed to calculate schema metrics: {e}")

    return metrics


def ensure_span_has_minimum_metadata(
    span, span_name: str, span_type: str = None, job_id: Optional[str] = None
):
    """
    Ensure span has minimum required metadata - never leave blank.

    Args:
        span: The span to validate
        span_name: Name of the span
        span_type: Type of span (e.g., "step_execution", "llm_call") - inferred from name if not provided
        job_id: Optional job ID to help retrieve session_id if needed
    """
    if not span or not _tracing_available:
        return

    try:
        # Infer span type from name if not provided
        if not span_type:
            if "step_execution" in span_name:
                span_type = "step_execution"
            elif "llm_call" in span_name or "function_pipeline" in span_name:
                span_type = "llm_call"
            elif "prompt_preparation" in span_name:
                span_type = "prompt_preparation"
            elif "context_building" in span_name:
                span_type = "context_building"
            elif "schema_generation" in span_name:
                span_type = "schema_generation"
            elif "result_parsing" in span_name:
                span_type = "result_parsing"
            elif "approval_check" in span_name:
                span_type = "approval_check"
            elif "pipeline.execute" in span_name or span_name == "pipeline.execute":
                span_type = "pipeline_execute"
            elif "step_retry" in span_name:
                span_type = "step_retry"
            elif "rerun" in span_name:
                span_type = "rerun"
            else:
                span_type = "unknown"

        # Always set span kind if not set
        try:
            # Check if span kind is already set by checking attributes
            # Note: We can't easily check existing attributes, so we'll set it if span_type is known
            kind_map = {
                "step_execution": "AGENT",
                "llm_call": "LLM",
                "function_pipeline": "LLM",
                "prompt_preparation": "TOOL",
                "context_building": "TOOL",
                "schema_generation": "TOOL",
                "result_parsing": "TOOL",
                "approval_check": "GUARDRAIL",
                "pipeline_execute": "CHAIN",
                "step_retry": "AGENT",
                "rerun": "AGENT",
            }
            if span_type in kind_map:
                set_span_kind(span, kind_map[span_type])
        except Exception:
            pass

        # session.id is resolved at span-creation time via create_job_root_span (which accepts
        # an explicit session_id parameter) or by the caller setting it directly.
        # Async callers that need to traverse the parent chain should call
        # get_session_id_for_job(job_id) and pass the result to create_job_root_span.

    except Exception as e:
        logger.debug(f"Failed to ensure span minimum metadata: {e}")


def record_span_exception(span, exception: Exception):
    """Record exception on span safely."""
    if not span or not _tracing_available:
        return
    try:
        span.record_exception(exception)
    except Exception:
        pass


def close_span(
    span, exc_type=None, exc_val=None, exc_tb=None, start_time: Optional[float] = None
):
    """
    Close span safely, ensuring all required attributes are set.

    Args:
        span: The span to close
        exc_type: Exception type (if any)
        exc_val: Exception value (if any)
        exc_tb: Exception traceback (if any)
        start_time: Optional start time to calculate duration
    """
    if not span or not _tracing_available:
        return
    try:
        # Set duration if start_time provided
        if start_time:
            set_span_duration(span, start_time)

        # Ensure status is set if there was an exception
        if exc_type is not None:
            try:
                from opentelemetry.trace import Status, StatusCode

                if Status and StatusCode:
                    span.set_status(
                        Status(StatusCode.ERROR, str(exc_val) if exc_val else "")
                    )
            except Exception:
                pass

        span.__exit__(exc_type, exc_val, exc_tb)
    except Exception:
        pass


async def get_session_id_for_job(job_id: str) -> Optional[str]:
    """
    Get session_id for a job, checking parent job chain if needed.

    Traverses up the job chain (following original_job_id) to find the root job's
    session_id, ensuring all jobs in a chain share the same session_id.

    Args:
        job_id: Job ID to get session_id for

    Returns:
        session_id if found, None otherwise
    """
    if not job_id or not _tracing_available:
        return None

    try:
        from marketing_project.services.job_manager import get_job_manager

        job_manager = get_job_manager()
        current_job_id = job_id

        # Traverse up the job chain to find session_id
        # Limit to prevent infinite loops (max 10 levels)
        max_depth = 10
        depth = 0

        while current_job_id and depth < max_depth:
            job = await job_manager.get_job(current_job_id)
            if not job:
                break

            # Check if current job has session_id
            if job.metadata and "session_id" in job.metadata:
                return job.metadata["session_id"]

            # Check if this job has a parent (original_job_id)
            original_job_id = (
                job.metadata.get("original_job_id") if job.metadata else None
            )
            if not original_job_id:
                # No parent, we've reached the root
                break

            # Move up to parent job
            current_job_id = original_job_id
            depth += 1

        return None
    except Exception as e:
        logger.debug(f"Failed to get session_id for job {job_id}: {e}")
        return None


def create_job_root_span(
    job_id: str,
    job_type: str,
    input_value: Optional[Any] = None,
    input_mime_type: str = "application/json",
    job: Optional[Any] = None,
    attributes: Optional[dict] = None,
    session_id: Optional[str] = None,
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

        # Set input attributes in OpenInference format - always set (never blank)
        # Use input_value if provided, otherwise use empty dict
        if input_value is None:
            input_value = {}

        try:
            # Serialize input value to JSON string if it's not already a string
            if isinstance(input_value, str):
                input_json = input_value if input_value else "{}"
            else:
                input_json = json.dumps(input_value, default=str)

            span.set_attribute("input.value", input_json)
            span.set_attribute("input.mime_type", input_mime_type)
        except Exception as e:
            logger.debug(f"Failed to set input attributes: {e}")
            # Even on error, set minimal input
            try:
                span.set_attribute("input.value", "{}")
                span.set_attribute("input.mime_type", "application/json")
            except Exception:
                pass

        # Add all job metadata to the span
        if job:
            add_job_metadata_to_span(span, job, job_id, job_type)

            # Ensure user.id is set from job.user_id if not already set
            if job.user_id:
                set_span_attribute(span, "user.id", job.user_id)

        # Resolve session.id: explicit param > job metadata.
        # Callers in async contexts should pass session_id=await get_session_id_for_job(job_id)
        # when the job is a subjob whose session_id may live in an ancestor's metadata.
        resolved_session_id = session_id or (
            job.metadata.get("session_id") if job and job.metadata else None
        )
        if resolved_session_id:
            set_span_attribute(span, "session.id", resolved_session_id)

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
