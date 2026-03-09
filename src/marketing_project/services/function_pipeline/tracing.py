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
    from opentelemetry import context as context_api
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _tracing_available = True
except ImportError:
    _tracing_available = False
    logger.debug("OpenTelemetry not available, tracing disabled")


class _SpanHolder:
    """
    Bundles an OTel Span with the context token needed to detach it.

    Using start_span() + context_api.attach() instead of the
    start_as_current_span().__enter__() pattern ensures:
      - The real Span object is retained (not the context-manager wrapper).
      - The context token is always paired with this span for correct
        detach/restore when close_span() is called.
      - Parent–child relationships are determined explicitly at span-creation
        time, surviving async await boundaries correctly.
    """

    __slots__ = ("span", "token")

    def __init__(self, span, token):
        self.span = span
        self.token = token

    # Proxy the most-used Span methods so all call sites need no changes.
    def set_attribute(self, key, value):
        self.span.set_attribute(key, value)

    def record_exception(self, exc):
        self.span.record_exception(exc)

    def set_status(self, *args, **kwargs):
        self.span.set_status(*args, **kwargs)

    def get_span_context(self):
        return self.span.get_span_context()


def is_tracing_available() -> bool:
    """Check if OpenTelemetry tracing is available."""
    return _tracing_available


def create_span(
    name: str, attributes: Optional[dict] = None, **_kwargs
) -> Optional[_SpanHolder]:
    """
    Create an OTel span parented to the current context and set it as current.

    Uses start_span() + context_api.attach() so the parent is captured
    explicitly at call time rather than relying on the implicit context
    stack surviving every async suspension point.
    """
    if not _tracing_available:
        logger.debug(f"Tracing unavailable, skipping span: {name}")
        return None
    try:
        tracer = trace.get_tracer(__name__)
        # Snapshot caller's context so the new span is correctly parented.
        current_ctx = context_api.get_current()
        span = tracer.start_span(
            name, context=current_ctx, kind=trace.SpanKind.INTERNAL
        )
        # Set the new span as current; store token so we can restore parent on close.
        token = context_api.attach(trace.set_span_in_context(span))

        is_noop = type(span).__name__ in ("NonRecordingSpan", "DefaultSpan")
        provider_type = type(trace.get_tracer_provider()).__name__
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

        return _SpanHolder(span, token)
    except Exception as e:
        logger.warning(f"Failed to create span {name}: {e}")
        return None


def close_span(holder, exc_type=None, exc_val=None, exc_tb=None, **_kwargs):
    """
    Detach span from context and end it.

    Detach is called before end() so the parent span is restored as the
    current span while child spans can still be correctly closed.
    """
    if not holder or not _tracing_available:
        return
    try:
        if exc_type is not None:
            holder.span.set_status(
                Status(StatusCode.ERROR, str(exc_val) if exc_val else "")
            )
        # Restore parent context before ending this span.
        context_api.detach(holder.token)
        holder.span.end()
    except Exception:
        pass


def set_span_status(holder, status_code, description: Optional[str] = None):
    """Set span status safely."""
    if not holder or not _tracing_available:
        return
    try:
        holder.span.set_status(Status(status_code, description))
    except Exception:
        pass


def set_span_attribute(holder, key: str, value):
    """Set span attribute safely."""
    if not holder or not _tracing_available:
        return
    try:
        holder.span.set_attribute(key, value)
    except Exception:
        pass


def record_span_exception(holder, exception: Exception):
    """Record exception on span safely."""
    if not holder or not _tracing_available:
        return
    try:
        holder.span.record_exception(exception)
    except Exception:
        pass


def set_span_output(
    holder, output_value: Any, output_mime_type: str = "application/json"
):
    """Set output.value on a span."""
    if not holder or not _tracing_available:
        return
    try:
        if isinstance(output_value, str):
            output_json = output_value or "{}"
        else:
            output_json = json.dumps(output_value, default=str)
        holder.span.set_attribute("output.value", output_json)
        holder.span.set_attribute("output.mime_type", output_mime_type)
    except Exception:
        pass


def set_job_output(
    holder, output_value: Any, output_mime_type: str = "application/json"
):
    """Set output on a job span."""
    set_span_output(holder, output_value, output_mime_type)


def create_step_span(
    step_name: str, step_number: int, job_id: str
) -> Optional[_SpanHolder]:
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


def add_job_metadata_to_span(holder, job, job_id: str, job_type: str):
    """Add essential job metadata to a span."""
    if not holder or not _tracing_available:
        return
    set_span_attribute(holder, "job.id", job_id)
    set_span_attribute(holder, "job.type", job_type)
    if not job:
        return
    if job.user_id:
        set_span_attribute(holder, "user.id", job.user_id)
    elif job.metadata and "triggered_by_user_id" in job.metadata:
        set_span_attribute(holder, "user.id", job.metadata["triggered_by_user_id"])
    if job.metadata:
        if "session_id" in job.metadata:
            set_span_attribute(holder, "session.id", job.metadata["session_id"])
        if "parent_job_id" in job.metadata:
            set_span_attribute(holder, "parent_job_id", job.metadata["parent_job_id"])


def get_current_traceparent() -> Optional[str]:
    """
    Return the W3C traceparent string for the currently active span.

    Format: '00-{trace_id_hex}-{span_id_hex}-{trace_flags}'

    Used to persist the current trace context into context_data before an
    approval gate so that resume_pipeline_job can link its root span back to
    the original job's trace, producing one coherent trace across the
    approval boundary.

    Returns None when tracing is unavailable or no span is active.
    """
    if not _tracing_available:
        return None
    try:
        carrier: dict = {}
        TraceContextTextMapPropagator().inject(
            carrier, context=context_api.get_current()
        )
        return carrier.get("traceparent")
    except Exception:
        return None


def create_job_root_span(
    job_id: str,
    job_type: str,
    input_value: Optional[Any] = None,
    input_mime_type: str = "application/json",
    job: Optional[Any] = None,
    attributes: Optional[dict] = None,
    session_id: Optional[str] = None,
    parent_context: Optional[Any] = None,
) -> Optional[_SpanHolder]:
    """
    Create a root span for a job execution.

    Serves as the parent for all pipeline step spans and auto-instrumented LLM spans.

    Args:
        parent_context: Optional OTel context extracted from a serialized traceparent.
                        When provided (e.g. for resume_pipeline), the new span becomes
                        a child of the original job's span, keeping the full pipeline
                        execution in one coherent trace across the approval gate.
    """
    if not _tracing_available:
        return None
    try:
        tracer = trace.get_tracer(__name__)
        # Use provided parent context (resume case) or current context (normal case).
        ctx_to_use = (
            parent_context if parent_context is not None else context_api.get_current()
        )
        span = tracer.start_span(
            f"job.{job_type}", context=ctx_to_use, kind=trace.SpanKind.INTERNAL
        )
        token = context_api.attach(trace.set_span_in_context(span))

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

        # Extra attributes
        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception:
                    pass

        return _SpanHolder(span, token)
    except Exception as e:
        logger.debug(f"Failed to create job root span for {job_id}: {e}")
        return None
