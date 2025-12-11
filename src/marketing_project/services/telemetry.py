"""
Telemetry setup for OpenInference with OpenTelemetry.

This module configures OpenTelemetry tracing to send traces to Arthur
for monitoring and observability of all LLM/Agent calls.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Global tracer provider for cleanup
_tracer_provider: Optional[object] = None


def setup_tracing() -> bool:
    """
    Set up OpenInference tracing with Arthur endpoint.

    This function:
    1. Loads Arthur configuration from environment variables
    2. Creates OpenTelemetry TracerProvider with Arthur task metadata
    3. Configures OTLP exporter pointing to Arthur endpoint
    4. Instruments LangChain (if used)

    Returns:
        True if tracing was successfully set up, False otherwise

    Environment Variables:
        ARTHUR_BASE_URL: Base URL for Arthur API (default: http://localhost:3030)
        ARTHUR_API_KEY: API key for Arthur authentication (required)
        ARTHUR_TASK_ID: Task ID for Arthur (required, must have is_agentic=True)
    """
    global _tracer_provider

    try:
        # Load configuration from environment
        arthur_base_url = os.getenv("ARTHUR_BASE_URL", "http://localhost:3030")
        arthur_api_key = os.getenv("ARTHUR_API_KEY")
        arthur_task_id = os.getenv("ARTHUR_TASK_ID")

        # Check if required configuration is present
        if not arthur_api_key:
            logger.warning(
                "ARTHUR_API_KEY not set. Telemetry will not be initialized. "
                "Set ARTHUR_API_KEY to enable tracing."
            )
            return False

        if not arthur_task_id:
            logger.warning(
                "ARTHUR_TASK_ID not set. Telemetry will not be initialized. "
                "Set ARTHUR_TASK_ID to enable tracing."
            )
            return False

        # Import OpenTelemetry components
        from opentelemetry import trace as trace_api
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        # Import OpenInference LangChain instrumentation
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor

            langchain_available = True
        except ImportError:
            logger.debug(
                "LangChain instrumentation not available (LangChain may not be used)"
            )
            langchain_available = False

        # Create tracer provider with Arthur task metadata
        service_name = os.getenv("SERVICE_NAME", "marketing-tool")
        _tracer_provider = trace_sdk.TracerProvider(
            resource=Resource.create(
                {
                    "arthur.task": arthur_task_id,
                    "service.name": service_name,
                }
            )
        )
        trace_api.set_tracer_provider(_tracer_provider)

        # Configure OTLP exporter and add span processor
        endpoint = f"{arthur_base_url}/v1/traces"
        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={"Authorization": f"Bearer {arthur_api_key}"},
        )

        _tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

        # Instrument LangChain if available
        if langchain_available:
            try:
                LangChainInstrumentor().instrument()
                logger.info("LangChain instrumentation enabled")
            except Exception as e:
                logger.warning(f"Failed to instrument LangChain: {e}")

        logger.info(
            f"Telemetry initialized successfully: "
            f"endpoint={endpoint}, task_id={arthur_task_id}, service={service_name}"
        )
        return True

    except ImportError as e:
        logger.warning(
            f"OpenTelemetry dependencies not available: {e}. "
            "Install openinference-instrumentation-langchain and opentelemetry-exporter-otlp to enable tracing."
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
