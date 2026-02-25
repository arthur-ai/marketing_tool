"""
Telemetry setup using the Arthur Observability SDK.

Instruments OpenAI and LangChain calls and exports traces to Arthur via OTLP.
Custom job/pipeline spans are created via services/function_pipeline/tracing.py,
which uses the global OpenTelemetry tracer set up here.

Install the SDK (local repo):
    pip install -e /path/to/arthur-observability-sdk[openai,langchain]
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Holds the ArthurClient instance for lifetime management
_arthur_client: Optional[object] = None


def setup_tracing(service_instance_id: Optional[str] = None) -> bool:
    """
    Initialise tracing with the Arthur Observability SDK.

    Reads configuration from environment variables:
        ARTHUR_API_KEY          – required; Arthur authentication key
        ARTHUR_TASK_ID          – required; Arthur task identifier
        ARTHUR_BASE_URL         – optional; defaults to https://app.arthur.ai
        OTEL_SERVICE_NAME       – optional; defaults to "marketing-tool"
        OTEL_SERVICE_INSTANCE_ID – optional; auto-generated from hostname+pid
        OTEL_DEPLOYMENT_ENVIRONMENT – optional; defaults to "production"

    Returns True if tracing was successfully initialised, False otherwise.
    """
    global _arthur_client

    try:
        from arthur_obs_sdk import ArthurClient, instrument_langchain, instrument_openai
    except ImportError:
        logger.warning(
            "Arthur Observability SDK not installed. "
            "Run: pip install -e /path/to/arthur-observability-sdk[openai,langchain]"
        )
        return False

    try:
        arthur_api_key = os.getenv("ARTHUR_API_KEY")
        arthur_task_id = os.getenv("ARTHUR_TASK_ID")
        arthur_base_url = os.getenv("ARTHUR_BASE_URL", "https://app.arthur.ai")

        if not arthur_api_key or not arthur_task_id:
            logger.info(
                "⚠ Telemetry not configured (missing ARTHUR_API_KEY or ARTHUR_TASK_ID)"
            )
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

        _arthur_client = ArthurClient(
            task_id=arthur_task_id,
            api_key=arthur_api_key,
            base_url=arthur_base_url,
            service_name=service_name,
            resource_attributes={
                "service.instance.id": service_instance_id,
                "deployment.environment": deployment_env,
            },
        )

        # Instrument LLM frameworks – spans flow into the Arthur-managed tracer provider
        try:
            instrument_openai()
            logger.info("OpenAI instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument OpenAI: {e}")

        try:
            instrument_langchain()
            logger.info("LangChain instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument LangChain: {e}")

        logger.info(
            f"Telemetry initialised: service={service_name}, "
            f"instance={service_instance_id}, environment={deployment_env}, "
            f"arthur_task={arthur_task_id}"
        )
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
