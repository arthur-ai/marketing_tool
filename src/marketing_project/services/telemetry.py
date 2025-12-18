"""
Telemetry setup for OpenInference with OpenTelemetry.

This module configures OpenTelemetry tracing to send traces to Arthur
for monitoring and observability of all LLM/Agent calls.

This implementation follows OpenInference specifications:
https://github.com/Arize-ai/openinference/tree/main/python/instrumentation
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Global tracer provider for cleanup
_tracer_provider: Optional[object] = None


class FilteringSpanProcessor:
    """
    Span processor that filters out noisy Redis operations like ZRANGEBYSCORE.

    ARQ uses ZRANGEBYSCORE frequently to poll for jobs, which creates excessive
    telemetry noise. This processor wraps another span processor and filters
    out spans for specific Redis commands before they reach the exporter.
    """

    def __init__(self, wrapped_processor, filtered_commands=None):
        """
        Initialize the filtering span processor.

        Args:
            wrapped_processor: The underlying span processor to forward spans to
            filtered_commands: List of Redis command names to filter out.
                              Default: ['ZRANGEBYSCORE', 'ZPOPMIN', 'ZPOPMAX']
        """
        self.wrapped_processor = wrapped_processor
        # Commands that are called frequently by ARQ for job polling
        # These create too much noise in telemetry
        self.filtered_commands = filtered_commands or [
            "ZRANGEBYSCORE",
            "ZPOPMIN",
            "ZPOPMAX",
        ]

    def on_start(self, span, parent_context=None):
        """Called when a span starts - forward to wrapped processor."""
        self.wrapped_processor.on_start(span, parent_context)

    def on_end(self, span):
        """Filter spans before forwarding to wrapped processor."""
        # Check if this is a Redis span with a filtered command
        span_name = getattr(span, "name", "")

        # Check span attributes for Redis command
        # Redis instrumentation sets db.operation or db.statement attributes
        should_filter = False

        # Try to get attributes from span (different span types store them differently)
        attrs = {}
        if hasattr(span, "attributes"):
            attrs = span.attributes if isinstance(span.attributes, dict) else {}
        elif hasattr(span, "_attributes"):
            # Some span implementations use _attributes
            attrs = span._attributes if isinstance(span._attributes, dict) else {}
        elif hasattr(span, "resource") and hasattr(span.resource, "attributes"):
            # Check resource attributes as fallback
            attrs = (
                span.resource.attributes
                if isinstance(span.resource.attributes, dict)
                else {}
            )

        # Check db.operation attribute (most common for Redis instrumentation)
        if attrs:
            db_operation = attrs.get("db.operation", "")
            if db_operation:
                # db.operation might be a string or AttributeValue, convert to string
                db_op_str = str(db_operation).upper()
                if db_op_str in [cmd.upper() for cmd in self.filtered_commands]:
                    should_filter = True

            # Also check db.statement (some instrumentation uses this)
            if not should_filter:
                db_statement = attrs.get("db.statement", "")
                if db_statement:
                    # db.statement might be the full command, check if it starts with filtered command
                    statement_str = str(db_statement).upper()
                    for cmd in self.filtered_commands:
                        if statement_str.startswith(cmd.upper()):
                            should_filter = True
                            break

        # Also check span name (some Redis instrumentation uses command name as span name)
        if not should_filter and span_name:
            span_name_upper = span_name.upper()
            for cmd in self.filtered_commands:
                if cmd.upper() in span_name_upper:
                    should_filter = True
                    break

        # If this span should be filtered, don't forward it to the exporter
        if should_filter:
            return

        # Not a filtered span, forward to wrapped processor
        self.wrapped_processor.on_end(span)

    def shutdown(self):
        """Shutdown the wrapped processor."""
        self.wrapped_processor.shutdown()

    def force_flush(self, timeout_millis=30000):
        """Force flush the wrapped processor."""
        return self.wrapped_processor.force_flush(timeout_millis)


def setup_tracing(service_instance_id: Optional[str] = None) -> bool:
    """
    Set up OpenInference tracing with Arthur endpoint and/or console export.

    This function:
    1. Loads configuration from environment variables
    2. Creates OpenTelemetry TracerProvider with OpenInference-compliant resource attributes
    3. Configures exporter(s): OTLP exporter for Arthur and/or Console exporter for local development
    4. Instruments OpenAI SDK (if used)
    5. Instruments LangChain (if used)

    Args:
        service_instance_id: Optional unique identifier for this service instance (e.g., worker-1, worker-2).
                            This helps differentiate telemetry from multiple instances of the same service.
                            If not provided, will use OTEL_SERVICE_INSTANCE_ID env var or generate from hostname+pid.

    Returns:
        True if tracing was successfully set up, False otherwise

    Environment Variables:
        ARTHUR_BASE_URL: Base URL for Arthur API (default: http://localhost:3030)
        ARTHUR_API_KEY: API key for Arthur authentication (optional if OTEL_EXPORT_CONSOLE is enabled)
        ARTHUR_TASK_ID: Task ID for Arthur (optional if OTEL_EXPORT_CONSOLE is enabled)
        OTEL_EXPORT_CONSOLE: Enable console export for local development (default: "false")
                              Set to "true" to export spans to stdout/stderr (visible in Docker logs)
        OTEL_SERVICE_NAME: Service name for tracing (default: "marketing-tool")
        OTEL_SERVICE_INSTANCE_ID: Unique instance identifier (default: auto-generated from hostname+pid)
        OTEL_DEPLOYMENT_ENVIRONMENT: Deployment environment (default: "production")
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: Capture message content (default: "true")
    """
    global _tracer_provider

    try:
        # Load configuration from environment
        arthur_base_url = os.getenv("ARTHUR_BASE_URL", "http://localhost:3030")
        arthur_api_key = os.getenv("ARTHUR_API_KEY")
        arthur_task_id = os.getenv("ARTHUR_TASK_ID")

        # Check if console export is enabled (for local development)
        export_console = os.getenv("OTEL_EXPORT_CONSOLE", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        # Determine if we should use Arthur export
        use_arthur = bool(arthur_api_key and arthur_task_id)

        # If neither console nor Arthur is configured, warn and return False
        if not export_console and not use_arthur:
            logger.warning(
                "No telemetry export configured. "
                "Set OTEL_EXPORT_CONSOLE=true for local development, "
                "or set ARTHUR_API_KEY and ARTHUR_TASK_ID for Arthur export."
            )
            return False

        # Import OpenTelemetry components
        from opentelemetry import trace as trace_api
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )

        # Import exporters conditionally
        if use_arthur:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

        if export_console:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        # Import OpenInference instrumentations
        # OpenAI instrumentation
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            openai_available = True
        except ImportError:
            logger.debug(
                "OpenAI instrumentation not available (openinference-instrumentation-openai not installed)"
            )
            openai_available = False

        # LangChain instrumentation
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor

            langchain_available = True
        except ImportError:
            logger.debug(
                "LangChain instrumentation not available (LangChain may not be used)"
            )
            langchain_available = False

        # Redis instrumentation
        try:
            from opentelemetry.instrumentation.redis import RedisInstrumentor

            redis_available = True
        except ImportError:
            logger.debug(
                "Redis instrumentation not available (opentelemetry-instrumentation-redis not installed)"
            )
            redis_available = False

        # Create tracer provider with OpenInference-compliant resource attributes
        # Following OpenInference specs: https://github.com/Arize-ai/openinference/tree/main/python/instrumentation
        service_name = os.getenv("OTEL_SERVICE_NAME") or os.getenv(
            "SERVICE_NAME", "marketing-tool"
        )
        deployment_env = os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "production")

        # Determine service instance ID
        # This is critical for distinguishing multiple worker instances (worker-1, worker-2, etc.)
        if service_instance_id:
            instance_id = service_instance_id
        else:
            instance_id = os.getenv("OTEL_SERVICE_INSTANCE_ID")
            if not instance_id:
                # Auto-generate from hostname + process ID for uniqueness
                import os as os_module
                import socket

                hostname = socket.gethostname()
                pid = os_module.getpid()
                instance_id = f"{hostname}-{pid}"

        # Build resource attributes
        resource_attrs = {
            # OpenInference standard attributes
            "service.name": service_name,
            "service.instance.id": instance_id,  # Unique identifier for this instance
            "deployment.environment": deployment_env,
        }
        # Add Arthur-specific metadata only if using Arthur
        if use_arthur and arthur_task_id:
            resource_attrs["arthur.task"] = arthur_task_id

        _tracer_provider = trace_sdk.TracerProvider(
            resource=Resource.create(resource_attrs)
        )
        trace_api.set_tracer_provider(_tracer_provider)

        # Configure exporters and add span processors
        exporters_configured = []

        # Add Arthur OTLP exporter if configured
        if use_arthur:
            endpoint = f"{arthur_base_url}/v1/traces"
            arthur_exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers={"Authorization": f"Bearer {arthur_api_key}"},
            )
            # Create span processor and wrap it with filtering to exclude noisy Redis operations
            base_processor = SimpleSpanProcessor(arthur_exporter)
            filtered_processor = FilteringSpanProcessor(base_processor)
            _tracer_provider.add_span_processor(filtered_processor)
            exporters_configured.append(f"Arthur ({endpoint})")
            logger.info(
                f"Arthur OTLP exporter configured: {endpoint} (filtering ZRANGEBYSCORE and other polling commands)"
            )

        # Add console exporter if configured (for local development/Docker logs)
        if export_console:
            console_exporter = ConsoleSpanExporter()
            # Create span processor and wrap it with filtering to exclude noisy Redis operations
            base_processor = SimpleSpanProcessor(console_exporter)
            filtered_processor = FilteringSpanProcessor(base_processor)
            _tracer_provider.add_span_processor(filtered_processor)
            exporters_configured.append("Console (stdout/stderr)")
            logger.info(
                "Console span exporter enabled - spans will appear in Docker logs (filtering ZRANGEBYSCORE and other polling commands)"
            )

        # Instrument OpenAI SDK if available
        # This must be done before any OpenAI client is instantiated
        # The instrumentor automatically:
        # - Creates spans with proper OpenInference span names (e.g., "openai.ChatCompletion.create")
        # - Adds OpenInference semantic attributes (span.kind=LLM, input.value, output.value, etc.)
        # - Captures model info, token counts, invocation parameters
        # - Respects OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT for content capture
        if openai_available:
            try:
                OpenAIInstrumentor().instrument(tracer_provider=_tracer_provider)
                logger.info(
                    "OpenAI instrumentation enabled (OpenInference-compliant spans)"
                )
            except Exception as e:
                logger.warning(f"Failed to instrument OpenAI: {e}")

        # Instrument LangChain if available
        # The instrumentor automatically:
        # - Creates spans with proper OpenInference span names for chains, LLMs, retrievers, etc.
        # - Adds OpenInference semantic attributes (span.kind=CHAIN/LLM/RETRIEVER/etc.)
        # - Captures input/output values, model info, token usage
        # - Respects OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT for content capture
        if langchain_available:
            try:
                LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
                logger.info(
                    "LangChain instrumentation enabled (OpenInference-compliant spans)"
                )
            except Exception as e:
                logger.warning(f"Failed to instrument LangChain: {e}")

        # Instrument Redis if available
        # This traces Redis operations used by ARQ job queue, caching, and analytics
        # Helps identify Redis bottlenecks affecting LLM pipeline performance
        if redis_available:
            try:
                RedisInstrumentor().instrument(tracer_provider=_tracer_provider)
                logger.info("Redis instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument Redis: {e}")

        logger.info(
            f"Telemetry initialized successfully: "
            f"exporters={', '.join(exporters_configured)}, "
            f"service={service_name}, instance={instance_id}, environment={deployment_env}"
        )
        return True

    except ImportError as e:
        logger.warning(
            f"OpenTelemetry dependencies not available: {e}. "
            "Install openinference-instrumentation-openai, openinference-instrumentation-langchain, "
            "opentelemetry-instrumentation-redis and opentelemetry-exporter-otlp to enable tracing."
        )
        return False
    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}", exc_info=True)
        return False


def cleanup_tracing():
    """
    Clean up telemetry resources.

    This should be called during application shutdown to gracefully
    flush and close telemetry connections.
    """
    global _tracer_provider

    if _tracer_provider is None:
        return

    try:
        from opentelemetry.sdk import trace as trace_sdk

        if isinstance(_tracer_provider, trace_sdk.TracerProvider):
            # Force flush any pending spans
            _tracer_provider.force_flush()
            # Shutdown the tracer provider
            _tracer_provider.shutdown()
            logger.info("Telemetry cleaned up successfully")
    except Exception as e:
        logger.warning(f"Error during telemetry cleanup: {e}")
    finally:
        _tracer_provider = None
