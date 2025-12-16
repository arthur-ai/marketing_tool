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


def setup_tracing() -> bool:
    """
    Set up OpenInference tracing with Arthur endpoint and/or console export.

    This function:
    1. Loads configuration from environment variables
    2. Creates OpenTelemetry TracerProvider with OpenInference-compliant resource attributes
    3. Configures exporter(s): OTLP exporter for Arthur and/or Console exporter for local development
    4. Instruments OpenAI SDK (if used)
    5. Instruments LangChain (if used)

    Returns:
        True if tracing was successfully set up, False otherwise

    Environment Variables:
        ARTHUR_BASE_URL: Base URL for Arthur API (default: http://localhost:3030)
        ARTHUR_API_KEY: API key for Arthur authentication (optional if OTEL_EXPORT_CONSOLE is enabled)
        ARTHUR_TASK_ID: Task ID for Arthur (optional if OTEL_EXPORT_CONSOLE is enabled)
        OTEL_EXPORT_CONSOLE: Enable console export for local development (default: "false")
                              Set to "true" to export spans to stdout/stderr (visible in Docker logs)
        OTEL_SERVICE_NAME: Service name for tracing (default: "marketing-tool")
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
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

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

        # Build resource attributes
        resource_attrs = {
            # OpenInference standard attributes
            "service.name": service_name,
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
            _tracer_provider.add_span_processor(SimpleSpanProcessor(arthur_exporter))
            exporters_configured.append(f"Arthur ({endpoint})")
            logger.info(f"Arthur OTLP exporter configured: {endpoint}")

        # Add console exporter if configured (for local development/Docker logs)
        if export_console:
            console_exporter = ConsoleSpanExporter()
            _tracer_provider.add_span_processor(SimpleSpanProcessor(console_exporter))
            exporters_configured.append("Console (stdout/stderr)")
            logger.info(
                "Console span exporter enabled - spans will appear in Docker logs"
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
            f"service={service_name}, environment={deployment_env}"
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
