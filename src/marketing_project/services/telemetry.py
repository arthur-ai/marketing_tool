"""
Telemetry setup using the Arthur Observability SDK.

Instruments OpenAI calls and exports traces to Arthur via OTLP.
Custom job/pipeline spans are created via services/function_pipeline/tracing.py,
which uses the global OpenTelemetry tracer set up here.

Install the SDK:
    pip install vendor/arthur_observability_sdk-1.0.0-py3-none-any.whl
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Holds the Arthur instance for lifetime management
_arthur_client: Optional[object] = None


def setup_tracing(service_instance_id: Optional[str] = None) -> bool:
    """
    Initialise tracing with the Arthur Observability SDK.

    Reads configuration from environment variables:
        ARTHUR_API_KEY          – required; Arthur authentication key
        ARTHUR_TASK_ID          – optional; Arthur task identifier (created if not provided)
        ARTHUR_BASE_URL         – optional; defaults to https://app.arthur.ai
        OTEL_SERVICE_NAME       – optional; defaults to "marketing-tool"
        OTEL_SERVICE_INSTANCE_ID – optional; auto-generated from hostname+pid
        OTEL_DEPLOYMENT_ENVIRONMENT – optional; defaults to "production"

    Returns True if tracing was successfully initialised, False otherwise.
    """
    global _arthur_client

    try:
        from arthur_observability_sdk import Arthur
    except ImportError:
        logger.warning(
            "Arthur Observability SDK not installed. "
            "Run: pip install vendor/arthur_observability_sdk-1.0.0-py3-none-any.whl"
        )
        return False

    try:
        arthur_api_key = os.getenv("ARTHUR_API_KEY")
        arthur_task_id = os.getenv("ARTHUR_TASK_ID")
        arthur_base_url = os.getenv("ARTHUR_BASE_URL", "https://app.arthur.ai")

        logger.info(
            f"Telemetry config: ARTHUR_API_KEY={'set (' + str(len(arthur_api_key)) + ' chars)' if arthur_api_key else 'MISSING'}, "
            f"ARTHUR_BASE_URL={arthur_base_url}, "
            f"ARTHUR_TASK_ID={'set' if arthur_task_id else 'not set (will auto-create)'}"
        )

        if not arthur_api_key:
            logger.warning("⚠ Telemetry not configured (missing ARTHUR_API_KEY)")
            return False

        service_name = os.getenv("OTEL_SERVICE_NAME") or os.getenv(
            "SERVICE_NAME", "marketing-tool"
        )
        deployment_env = os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "production")

        if not service_instance_id:
            service_instance_id = os.getenv("OTEL_SERVICE_INSTANCE_ID")
            if not service_instance_id:
                import socket

                service_instance_id = f"{socket.gethostname()}-{os.getpid()}"

        otlp_endpoint = f"{arthur_base_url.rstrip('/')}/api/v1/traces"
        logger.info(
            f"Telemetry: service={service_name}, instance={service_instance_id}, "
            f"environment={deployment_env}, otlp_endpoint={otlp_endpoint}, "
            f"task_id={arthur_task_id or '(auto)'}"
        )

        arthur_kwargs = dict(
            api_key=arthur_api_key,
            base_url=arthur_base_url,
            service_name=service_name,
        )
        if arthur_task_id:
            arthur_kwargs["task_id"] = arthur_task_id
            logger.info(f"Telemetry: passing task_id={arthur_task_id!r} to Arthur")
        else:
            logger.warning(
                "Telemetry: ARTHUR_TASK_ID is not set — traces may not be routed to a task"
            )

        logger.info(
            f"Telemetry: creating Arthur client with kwargs keys={list(arthur_kwargs.keys())}"
        )
        _arthur_client = Arthur(**arthur_kwargs)
        logger.info("Telemetry: Arthur() constructor completed successfully")

        # Log exactly what the Arthur client resolved internally
        logger.info(
            f"Arthur client created: "
            f"telemetry_active={_arthur_client.telemetry_active}, "
            f"resolved_task_id={_arthur_client.task_id!r}, "
            f"otlp_endpoint={getattr(_arthur_client, '_otlp_endpoint', 'unknown')}, "
            f"base_url={getattr(_arthur_client, '_base_url', 'unknown')}"
        )

        # Verify the global TracerProvider was set (not a no-op proxy)
        try:
            from opentelemetry import trace as otel_trace

            provider = otel_trace.get_tracer_provider()
            provider_type = type(provider).__name__
            logger.info(f"Global TracerProvider: {provider_type}")
            if "Proxy" in provider_type or "NoOp" in provider_type:
                logger.warning(
                    f"TracerProvider is a no-op ({provider_type}) — spans will not be exported!"
                )
        except Exception as e:
            logger.warning(f"Could not inspect TracerProvider: {e}")

        # Instrument LLM frameworks – spans flow into the Arthur-managed tracer provider
        try:
            _arthur_client.instrument_openai()
            logger.info("OpenAI instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument OpenAI: {e}")

        try:
            _arthur_client.instrument_litellm()
            logger.info("LiteLLM instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument LiteLLM: {e}")

        logger.info("Telemetry: instrumentation phase complete — emitting startup span")

        # Emit a test span and force-flush to verify the export pipeline is working
        try:
            from opentelemetry import trace as otel_trace

            tracer = otel_trace.get_tracer("marketing_project.telemetry")
            with tracer.start_as_current_span("telemetry.startup_check") as span:
                span.set_attribute("service.name", service_name)
                span.set_attribute("deployment.environment", deployment_env)
            logger.info("Startup telemetry check span emitted — flushing...")

            # Force-flush so we can detect export failures at startup
            if (
                _arthur_client
                and hasattr(_arthur_client, "_tracer_provider")
                and _arthur_client._tracer_provider
            ):
                flush_result = _arthur_client._tracer_provider.force_flush(
                    timeout_millis=10000
                )
                logger.info(
                    f"Telemetry flush result: {flush_result} (True=success, False=timeout/error)"
                )
            else:
                logger.warning("No tracer provider on Arthur client — cannot flush")
        except Exception as e:
            logger.warning(
                f"Failed to emit/flush startup check span: {e}", exc_info=True
            )

        logger.info("✓ Telemetry initialised successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialise telemetry: {e}", exc_info=True)
        return False


def cleanup_tracing():
    """Flush pending spans and shut down the Arthur client on service shutdown."""
    global _arthur_client

    if _arthur_client is None:
        return

    try:
        _arthur_client.shutdown()
        logger.info("Telemetry cleaned up successfully")
    except Exception as e:
        logger.warning(f"Error during telemetry cleanup: {e}")
    finally:
        _arthur_client = None
