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
    Set up OpenInference tracing with Arthur endpoint.

    This function:
    1. Loads Arthur configuration from environment variables
    2. Creates OpenTelemetry TracerProvider with OpenInference-compliant resource attributes
    3. Configures OTLP exporter pointing to Arthur endpoint
    4. Instruments OpenAI SDK (if used)
    5. Instruments LangChain (if used)

    Returns:
        True if tracing was successfully set up, False otherwise

    Environment Variables:
        ARTHUR_BASE_URL: Base URL for Arthur API (default: http://localhost:3030)
        ARTHUR_API_KEY: API key for Arthur authentication (required)
        ARTHUR_TASK_ID: Task ID for Arthur (required, must have is_agentic=True)
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

        _tracer_provider = trace_sdk.TracerProvider(
            resource=Resource.create(
                {
                    # OpenInference standard attributes
                    "service.name": service_name,
                    "deployment.environment": deployment_env,
                    # Arthur-specific metadata
                    "arthur.task": arthur_task_id,
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
            f"endpoint={endpoint}, task_id={arthur_task_id}, service={service_name}, "
            f"environment={deployment_env}"
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
