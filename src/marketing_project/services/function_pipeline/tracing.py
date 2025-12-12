"""
OpenTelemetry tracing utilities for the function pipeline.
"""

import logging
from typing import Optional

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
