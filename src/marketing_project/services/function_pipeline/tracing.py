"""
OpenTelemetry tracing utilities for the function pipeline.

Provides job-level root spans, pipeline step spans, and helpers.
LLM call spans are auto-instrumented by the arthur_observability_sdk
via instrument_openai() and instrument_litellm() — no custom LLM spans needed.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger("marketing_project.services.function_pipeline.tracing")

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


def create_span(name: str, attributes: Optional[dict] = None, **_kwargs):
    """Create and enter an OpenTelemetry span."""
    if not _tracing_available:
        logger.debug(f"Tracing unavailable, skipping span: {name}")
        return None
    try:
        tracer = trace.get_tracer(__name__)
        provider_type = type(trace.get_tracer_provider()).__name__
        span = tracer.start_as_current_span(name, kind=trace.SpanKind.INTERNAL)
        span.__enter__()
        # Warn if we got a no-op span (provider not properly configured)
        is_noop = type(span).__name__ in ("NonRecordingSpan", "DefaultSpan")
        if is_noop:
            logger.warning(
                f"Span '{name}' is a no-op (provider={provider_type}) — spans will NOT be exported"
            )
        else:
            logger.debug(f"Span created: {name} (provider={provider_type})")
        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception:
                    pass
        return span
    except Exception as e:
        logger.warning(f"Failed to create span {name}: {e}")
        return None


def close_span(span, exc_type=None, exc_val=None, exc_tb=None, **_kwargs):
    """Close span safely."""
    if not span or not _tracing_available:
        return
    try:
        if exc_type is not None:
            span.set_status(Status(StatusCode.ERROR, str(exc_val) if exc_val else ""))
        span.__exit__(exc_type, exc_val, exc_tb)
    except Exception:
        pass


def set_span_status(span, status_code, description: Optional[str] = None):
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


def record_span_exception(span, exception: Exception):
    """Record exception on span safely."""
    if not span or not _tracing_available:
        return
    try:
        span.record_exception(exception)
    except Exception:
        pass


def set_span_output(
    span, output_value: Any, output_mime_type: str = "application/json"
):
    """Set output.value on a span."""
    if not span or not _tracing_available:
        return
    try:
        if isinstance(output_value, str):
            output_json = output_value or "{}"
        else:
            output_json = json.dumps(output_value, default=str)
        span.set_attribute("output.value", output_json)
        span.set_attribute("output.mime_type", output_mime_type)
    except Exception:
        pass


def set_job_output(span, output_value: Any, output_mime_type: str = "application/json"):
    """Set output on a job span."""
    set_span_output(span, output_value, output_mime_type)


def create_step_span(step_name: str, step_number: int, job_id: str):
    """
    Create a span for a single pipeline step.

    LLM calls executed within this step are automatically captured as child spans
    by the arthur_observability_sdk instrumentation.
    """
    return create_span(
        f"pipeline.step.{step_name}",
        attributes={
            "step.name": step_name,
            "step.number": step_number,
            "pipeline.job_id": job_id,
        },
    )


def add_job_metadata_to_span(span, job, job_id: str, job_type: str):
    """Add essential job metadata to a span."""
    if not span or not _tracing_available:
        return
    set_span_attribute(span, "job.id", job_id)
    set_span_attribute(span, "job.type", job_type)
    if not job:
        return
    if job.user_id:
        set_span_attribute(span, "user.id", job.user_id)
    elif job.metadata and "triggered_by_user_id" in job.metadata:
        set_span_attribute(span, "user.id", job.metadata["triggered_by_user_id"])
    if job.metadata:
        if "session_id" in job.metadata:
            set_span_attribute(span, "session.id", job.metadata["session_id"])
        if "parent_job_id" in job.metadata:
            set_span_attribute(span, "parent_job_id", job.metadata["parent_job_id"])


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

    Serves as the parent for all pipeline step spans and auto-instrumented LLM spans.
    """
    if not _tracing_available:
        return None
    try:
        tracer = trace.get_tracer(__name__)
        span = tracer.start_as_current_span(
            f"job.{job_type}", kind=trace.SpanKind.INTERNAL
        )
        span.__enter__()

        span.set_attribute("job.id", job_id)
        span.set_attribute("job.type", job_type)

        # Record job input
        try:
            if input_value is None:
                input_value = {}
            input_json = (
                input_value
                if isinstance(input_value, str)
                else json.dumps(input_value, default=str)
            )
            span.set_attribute("input.value", input_json or "{}")
            span.set_attribute("input.mime_type", input_mime_type)
        except Exception:
            pass

        # User and session attribution
        if job:
            if job.user_id:
                span.set_attribute("user.id", job.user_id)
            elif job.metadata and "triggered_by_user_id" in job.metadata:
                span.set_attribute("user.id", job.metadata["triggered_by_user_id"])

            resolved_session_id = session_id or (
                job.metadata.get("session_id") if job.metadata else None
            )
            if resolved_session_id:
                span.set_attribute("session.id", resolved_session_id)

            if job.metadata and "parent_job_id" in job.metadata:
                span.set_attribute("parent_job_id", job.metadata["parent_job_id"])

        # Extra attributes override
        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception:
                    pass

        return span
    except Exception as e:
        logger.debug(f"Failed to create job root span for {job_id}: {e}")
        return None
