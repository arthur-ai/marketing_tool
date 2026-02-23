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


class RedisSpanEnrichmentProcessor:
    """
    Span processor that enriches Redis spans with OpenInference-compliant metadata.

    This processor:
    1. Identifies Redis spans and assigns appropriate OpenInference span kinds
    2. Adds contextual metadata (operation context, duration, parent info)
    3. Extracts job_id and other context from parent spans
    4. Categorizes operations (job_queue, cache, analytics, etc.)

    Following OpenInference semantic conventions:
    https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
    """

    # Redis commands mapped to OpenInference span kinds
    # TOOL: Operations that modify state or perform actions (job queue, cache writes)
    TOOL_COMMANDS = {
        "PSETEX",
        "SETEX",
        "SET",
        "SETNX",
        "MSET",
        "MSETNX",
        "ZPOPMIN",
        "ZPOPMAX",
        "ZADD",
        "ZREM",
        "ZINCRBY",
        "LPUSH",
        "RPUSH",
        "LPOP",
        "RPOP",
        "LREM",
        "LSET",
        "HDEL",
        "HSET",
        "HSETNX",
        "HMSET",
        "HINCRBY",
        "DEL",
        "EXPIRE",
        "EXPIREAT",
        "PERSIST",
        "INCR",
        "INCRBY",
        "DECR",
        "DECRBY",
    }

    # RETRIEVER: Operations that retrieve data (reads, counts, queries)
    RETRIEVER_COMMANDS = {
        "GET",
        "MGET",
        "GETSET",
        "ZCARD",
        "ZCOUNT",
        "ZRANGE",
        "ZRANGEBYSCORE",
        "ZRANK",
        "ZREVRANK",
        "ZSCORE",
        "ZREVRANGE",
        "ZREVRANGEBYSCORE",
        "LLEN",
        "LRANGE",
        "LINDEX",
        "HGET",
        "HGETALL",
        "HMGET",
        "HEXISTS",
        "HKEYS",
        "HVALS",
        "HLEN",
        "EXISTS",
        "KEYS",
        "SCAN",
        "TYPE",
        "TTL",
        "PTTL",
    }

    def __init__(self, wrapped_processor):
        """
        Initialize the enrichment span processor.

        Args:
            wrapped_processor: The underlying span processor to forward spans to
        """
        self.wrapped_processor = wrapped_processor

    def _get_span_attributes(self, span):
        """Extract attributes from span, handling different span implementations."""
        attrs = {}
        # Try different ways to access attributes (OpenTelemetry SDK uses different structures)
        if hasattr(span, "attributes"):
            attrs = span.attributes if isinstance(span.attributes, dict) else {}
        elif hasattr(span, "_attributes"):
            attrs = span._attributes if isinstance(span._attributes, dict) else {}
        elif hasattr(span, "_readable_span") and hasattr(
            span._readable_span, "attributes"
        ):
            # ReadableSpan wrapper
            attrs = (
                span._readable_span.attributes
                if isinstance(span._readable_span.attributes, dict)
                else {}
            )
        return attrs

    def _set_span_attribute(self, span, key, value):
        """Set an attribute on a span, handling different span implementations."""
        # Try to modify the internal attributes dictionary
        if hasattr(span, "_attributes") and isinstance(span._attributes, dict):
            span._attributes[key] = value
        elif hasattr(span, "attributes") and isinstance(span.attributes, dict):
            span.attributes[key] = value
        elif hasattr(span, "_readable_span") and hasattr(
            span._readable_span, "_attributes"
        ):
            if isinstance(span._readable_span._attributes, dict):
                span._readable_span._attributes[key] = value

    def _get_redis_command(self, span, attrs):
        """Extract Redis command name from span."""
        span_name = getattr(span, "name", "")

        # Try db.operation first (most common)
        db_operation = attrs.get("db.operation", "")
        if db_operation:
            return str(db_operation).upper()

        # Try db.statement (may contain full command like "PSETEX ? ? ?")
        db_statement = attrs.get("db.statement", "")
        if db_statement:
            statement = str(db_statement).upper()
            # Extract command from statement (first word before space or ?)
            parts = statement.split()
            if parts:
                return parts[0]

        # Fallback to span name
        if span_name:
            return span_name.upper()

        return None

    def _determine_span_kind(self, command):
        """
        Determine OpenInference span kind based on Redis command.

        Returns: "TOOL", "RETRIEVER", or None if not a recognized command
        """
        if not command:
            return None

        command_upper = command.upper()

        if command_upper in self.TOOL_COMMANDS:
            return "TOOL"
        elif command_upper in self.RETRIEVER_COMMANDS:
            return "RETRIEVER"

        return None

    def _determine_operation_context(self, command, attrs):
        """
        Determine the operation context (job_queue, cache, analytics, etc.).

        Returns: Context string describing the operation purpose
        """
        if not command:
            return "redis_operation"

        command_upper = command.upper()

        # Job queue operations (ARQ uses sorted sets for job scheduling)
        if command_upper in {
            "ZRANGEBYSCORE",
            "ZPOPMIN",
            "ZPOPMAX",
            "ZADD",
            "ZREM",
            "ZCARD",
        }:
            # Check if key suggests job queue
            db_statement = str(attrs.get("db.statement", "")).lower()
            if any(
                keyword in db_statement
                for keyword in ["arq:", "job", "queue", "worker"]
            ):
                return "job_queue"
            return "sorted_set"

        # Cache operations (typically SET/SETEX/PSETEX with expiration)
        if command_upper in {"PSETEX", "SETEX", "SET", "GET", "MGET"}:
            return "cache"

        # Analytics/counting operations
        if command_upper in {"ZCARD", "ZCOUNT", "LLEN", "HLEN", "INCR", "DECR"}:
            return "analytics"

        # Hash operations (often used for structured data)
        if command_upper.startswith("H"):
            return "hash_operation"

        # List operations
        if command_upper.startswith("L"):
            return "list_operation"

        return "redis_operation"

    def _extract_parent_context(self, span):
        """
        Extract context from parent span (job_id, operation name, etc.).

        Returns: Dict with parent context information
        """
        context = {}

        try:
            # Try to get parent span context from span's parent attribute
            parent = None
            if hasattr(span, "parent") and span.parent:
                parent = span.parent
            elif hasattr(span, "parent_span_id") and span.parent_span_id:
                # Parent span ID is available, but we'd need tracer to get the actual span
                # For now, we'll try to get context from the current span's attributes
                # which might contain parent information
                pass

            # Try to extract job_id from current span's attributes (might be set by parent)
            attrs = self._get_span_attributes(span)
            job_id = (
                attrs.get("job_id")
                or attrs.get("arq.job_id")
                or attrs.get("metadata.job_id")
            )
            if job_id:
                context["parent.job_id"] = str(job_id)

            # If we have a parent span, extract more context
            if parent:
                parent_attrs = self._get_span_attributes(parent)

                # Extract job_id from parent if not found in current span
                if not job_id:
                    job_id = parent_attrs.get("job_id") or parent_attrs.get(
                        "arq.job_id"
                    )
                    if job_id:
                        context["parent.job_id"] = str(job_id)

                # Extract parent operation name
                parent_name = getattr(parent, "name", "")
                if parent_name:
                    context["parent.operation"] = parent_name

                # Extract parent span kind if it's an OpenInference span
                parent_kind = parent_attrs.get("openinference.span.kind")
                if parent_kind:
                    context["parent.span_kind"] = str(parent_kind)

        except Exception:
            # If we can't extract parent context, continue without it
            pass

        return context

    def _calculate_duration_ms(self, span):
        """Calculate span duration in milliseconds."""
        try:
            if hasattr(span, "start_time") and hasattr(span, "end_time"):
                if span.start_time and span.end_time:
                    duration_ns = span.end_time - span.start_time
                    duration_ms = (
                        duration_ns / 1_000_000
                    )  # Convert nanoseconds to milliseconds
                    return round(duration_ms, 2)
        except Exception:
            pass
        return None

    def _extract_redis_connection_info(self, attrs):
        """Extract Redis connection metadata."""
        info = {}

        # Database index
        db_index = attrs.get("db.redis.database_index")
        if db_index is not None:
            info["redis.database_index"] = int(db_index)

        # Network peer info
        peer_name = attrs.get("net.peer.name")
        if peer_name:
            info["redis.peer_name"] = str(peer_name)

        peer_port = attrs.get("net.peer.port")
        if peer_port is not None:
            info["redis.peer_port"] = int(peer_port)

        # Transport protocol
        transport = attrs.get("net.transport")
        if transport:
            info["redis.transport"] = str(transport)

        return info

    def _extract_key_pattern(self, attrs, command):
        """Extract and sanitize Redis key pattern from statement."""
        key_info = {}

        try:
            db_statement = str(attrs.get("db.statement", ""))
            if not db_statement or "?" in db_statement:
                # Parameterized query - extract pattern
                # Example: "PSETEX ? ? ?" -> extract first parameter pattern
                parts = db_statement.split("?")
                if len(parts) > 1:
                    # Try to infer key pattern from context
                    # For job queue operations, keys often start with "arq:"
                    if command and command.upper() in {
                        "ZRANGEBYSCORE",
                        "ZPOPMIN",
                        "ZPOPMAX",
                        "ZCARD",
                    }:
                        key_info["redis.key_pattern"] = "arq:*"
                    elif command and command.upper() in {
                        "PSETEX",
                        "SETEX",
                        "SET",
                        "GET",
                    }:
                        key_info["redis.key_pattern"] = "cache:*"
                return key_info

            # If we have actual key (not parameterized), extract prefix
            # Be careful not to expose sensitive data
            first_word = db_statement.split()[0] if db_statement else ""
            if first_word and first_word.upper() == command.upper():
                # Extract key from statement (second token typically)
                parts = db_statement.split()
                if len(parts) > 1:
                    key = parts[1]
                    # Only extract prefix pattern, not full key
                    if ":" in key:
                        prefix = key.split(":")[0] + ":*"
                        key_info["redis.key_pattern"] = prefix
                    elif len(key) > 0:
                        # Generic pattern
                        key_info["redis.key_pattern"] = (
                            f"{key[:10]}*" if len(key) > 10 else key
                        )
        except Exception:
            pass

        return key_info

    def _extract_result_metadata(self, attrs, command):
        """Extract metadata about operation results."""
        metadata = {}

        try:
            # Args length (from Redis instrumentation)
            args_length = attrs.get("db.redis.args_length")
            if args_length is not None:
                metadata["redis.args_count"] = int(args_length)

            # For commands that return counts, try to extract count info
            command_upper = command.upper() if command else ""
            if command_upper in {"ZCARD", "ZCOUNT", "LLEN", "HLEN"}:
                metadata["redis.returns_count"] = True
            elif command_upper in {"ZRANGE", "ZRANGEBYSCORE", "LRANGE", "HGETALL"}:
                metadata["redis.returns_collection"] = True

            # For commands with expiration
            if command_upper in {"PSETEX", "SETEX", "EXPIRE", "EXPIREAT"}:
                metadata["redis.has_expiration"] = True
        except Exception:
            pass

        return metadata

    def _extract_error_info(self, span, attrs):
        """Extract error and status information."""
        error_info = {}

        try:
            # Check span status
            if hasattr(span, "status"):
                status = span.status
                if hasattr(status, "status_code"):
                    status_code = str(status.status_code)
                    error_info["redis.status_code"] = status_code

                    # Check if error
                    if status_code.upper() in {"ERROR", "UNSET"}:
                        error_info["redis.has_error"] = status_code.upper() == "ERROR"

                        # Extract error message if available
                        if hasattr(status, "description") and status.description:
                            error_info["redis.error_message"] = str(status.description)

            # Check for exception attributes
            exception_type = attrs.get("exception.type")
            if exception_type:
                error_info["redis.exception_type"] = str(exception_type)

            exception_message = attrs.get("exception.message")
            if exception_message:
                error_info["redis.exception_message"] = str(exception_message)
        except Exception:
            pass

        return error_info

    def _extract_trace_correlation(self, span):
        """Extract trace and span correlation IDs."""
        correlation = {}

        try:
            # Trace ID
            if hasattr(span, "context") and span.context:
                trace_id = getattr(span.context, "trace_id", None)
                if trace_id:
                    correlation["trace_id"] = format(trace_id, "x")

            # Span ID
            if hasattr(span, "context") and span.context:
                span_id = getattr(span.context, "span_id", None)
                if span_id:
                    correlation["span_id"] = format(span_id, "x")

            # Parent span ID
            if hasattr(span, "parent_span_id") and span.parent_span_id:
                correlation["parent_span_id"] = format(span.parent_span_id, "x")
        except Exception:
            pass

        return correlation

    def _should_filter_span(self, command, attrs, span):
        """
        Determine if a Redis span should be filtered out from telemetry.

        Filters out frequent operations that create noise in telemetry:
        - ARQ polling operations (ZRANGEBYSCORE for job queue)
        - Frequent cache operations (PSETEX/SETEX from libraries)

        Args:
            command: Redis command name
            attrs: Span attributes dictionary
            span: The span object (for accessing span name)

        Returns: True if span should be filtered (not logged), False otherwise
        """
        # Get command from parameter or extract from span name as fallback
        command_to_check = command
        if not command_to_check:
            try:
                span_name = getattr(span, "name", "")
                if span_name:
                    command_to_check = str(span_name).upper()
            except Exception:
                pass

        if not command_to_check:
            return False

        command_upper = command_to_check.upper()

        # Filter out ARQ job queue polling operations
        # ARQ uses ZRANGEBYSCORE to poll for ready jobs, which happens very frequently
        if command_upper == "ZRANGEBYSCORE":
            # Get db.statement from attributes (try multiple ways)
            db_statement = str(attrs.get("db.statement", "")).lower()
            if not db_statement:
                # Try to get from span directly if not in attrs
                try:
                    if hasattr(span, "attributes") and isinstance(
                        span.attributes, dict
                    ):
                        db_statement = str(
                            span.attributes.get("db.statement", "")
                        ).lower()
                    elif hasattr(span, "_attributes") and isinstance(
                        span._attributes, dict
                    ):
                        db_statement = str(
                            span._attributes.get("db.statement", "")
                        ).lower()
                except Exception:
                    pass

            # If statement is parameterized (contains "?"), filter it
            # ARQ uses parameterized queries like "ZRANGEBYSCORE ? ? ? ? ? ?"
            # In this codebase, all parameterized ZRANGEBYSCORE operations are from ARQ
            if db_statement and "?" in db_statement:
                return True

            # Check for ARQ patterns in statement
            arq_patterns = ["arq:", "arq:queue", "arq:job"]
            if db_statement and any(
                keyword in db_statement for keyword in arq_patterns
            ):
                return True

            # As a fallback, filter ALL ZRANGEBYSCORE operations since they're
            # almost certainly from ARQ polling in this codebase
            # This is the most aggressive filter - if you have other uses of
            # ZRANGEBYSCORE, you can make this conditional
            return True

        # Filter out frequent cache operations from libraries
        # PSETEX/SETEX are used for caching, and libraries (like LangChain) may
        # call them very frequently for internal caching, creating telemetry noise
        if command_upper in {"PSETEX", "SETEX"}:
            db_statement = str(attrs.get("db.statement", "")).lower()
            # Check if this is a library-internal cache operation
            # Common patterns: langchain cache, openai cache, or other library caches
            library_cache_patterns = [
                "langchain",
                "openai",
                "cache:",
                ":cache",
                "llm_cache",
                "prompt_cache",
            ]
            if any(pattern in db_statement for pattern in library_cache_patterns):
                return True

        return False

    def _extract_performance_indicators(self, duration_ms, command):
        """Calculate performance indicators."""
        indicators = {}

        if duration_ms is not None:
            indicators["redis.duration_ms"] = duration_ms

            # Performance categorization
            if duration_ms < 1:
                indicators["redis.performance_category"] = "fast"
            elif duration_ms < 10:
                indicators["redis.performance_category"] = "normal"
            elif duration_ms < 50:
                indicators["redis.performance_category"] = "slow"
            else:
                indicators["redis.performance_category"] = "very_slow"

            # Latency threshold warnings
            if duration_ms > 100:
                indicators["redis.latency_warning"] = True

        return indicators

    def _extract_application_context(self, attrs):
        """Extract application-level context from span attributes."""
        context = {}

        try:
            # Service name
            if hasattr(attrs, "get"):
                service_name = attrs.get("service.name")
                if service_name:
                    context["service.name"] = str(service_name)

            # Service instance
            service_instance = attrs.get("service.instance.id")
            if service_instance:
                context["service.instance.id"] = str(service_instance)

            # Deployment environment
            deployment_env = attrs.get("deployment.environment")
            if deployment_env:
                context["deployment.environment"] = str(deployment_env)

            # User ID (if available in context)
            user_id = attrs.get("user.id") or attrs.get("metadata.user_id")
            if user_id:
                context["user.id"] = str(user_id)

            # Request ID (if available)
            request_id = attrs.get("request.id") or attrs.get("http.request_id")
            if request_id:
                context["request.id"] = str(request_id)

            # Content/Job type
            content_type = attrs.get("content.type") or attrs.get(
                "metadata.content_type"
            )
            if content_type:
                context["content.type"] = str(content_type)

            # Pipeline stage
            pipeline_stage = attrs.get("pipeline.stage") or attrs.get(
                "metadata.pipeline_stage"
            )
            if pipeline_stage:
                context["pipeline.stage"] = str(pipeline_stage)
        except Exception:
            pass

        return context

    def on_start(self, span, parent_context=None):
        """Called when a span starts - forward to wrapped processor."""
        self.wrapped_processor.on_start(span, parent_context)

    def on_end(self, span):
        """Enrich Redis spans with metadata before forwarding to wrapped processor."""
        attrs = self._get_span_attributes(span)

        # Check if this is a Redis span
        db_system = attrs.get("db.system", "")
        if str(db_system).lower() != "redis":
            # Not a Redis span, forward as-is
            self.wrapped_processor.on_end(span)
            return

        # Extract Redis command - do this early for filtering
        command = self._get_redis_command(span, attrs)

        # Also check span name directly as fallback (in case command extraction fails)
        # Try multiple ways to get the span name
        span_name = ""
        try:
            span_name = str(getattr(span, "name", "")).upper()
        except Exception:
            pass
        if not span_name:
            try:
                # Try accessing via readable_span if available
                if hasattr(span, "_readable_span"):
                    span_name = str(getattr(span._readable_span, "name", "")).upper()
            except Exception:
                pass

        # Filter out frequent ARQ polling operations to reduce telemetry noise
        # This must happen BEFORE any enrichment to prevent export
        # Check both the command and span name to ensure we catch all ZRANGEBYSCORE operations
        should_filter = False

        # Direct check: if command is ZRANGEBYSCORE, filter it
        if command and str(command).upper() == "ZRANGEBYSCORE":
            should_filter = True

        # Direct check: if span name is ZRANGEBYSCORE, filter it
        if span_name == "ZRANGEBYSCORE":
            should_filter = True

        # Also check via the filter method
        if not should_filter and self._should_filter_span(command, attrs, span):
            should_filter = True

        if should_filter:
            # Skip this span - don't forward to wrapped processor
            # This prevents the span from being exported
            return

        # Determine OpenInference span kind
        span_kind = self._determine_span_kind(command)

        # Only enrich if we can determine the span kind
        if span_kind:
            # Determine operation context
            operation_context = self._determine_operation_context(command, attrs)

            # Calculate duration first (needed for performance indicators)
            duration_ms = self._calculate_duration_ms(span)

            # === Core OpenInference Attributes ===
            self._set_span_attribute(span, "openinference.span.kind", span_kind)

            # === Redis Operation Metadata ===
            self._set_span_attribute(span, "redis.operation_context", operation_context)
            self._set_span_attribute(span, "redis.command", command)

            if duration_ms is not None:
                self._set_span_attribute(span, "redis.duration_ms", duration_ms)

            # === Redis Connection Info ===
            connection_info = self._extract_redis_connection_info(attrs)
            for key, value in connection_info.items():
                self._set_span_attribute(span, key, value)

            # === Key Pattern (Sanitized) ===
            key_info = self._extract_key_pattern(attrs, command)
            for key, value in key_info.items():
                self._set_span_attribute(span, key, value)

            # === Result Metadata ===
            result_metadata = self._extract_result_metadata(attrs, command)
            for key, value in result_metadata.items():
                self._set_span_attribute(span, key, value)

            # === Error Information ===
            error_info = self._extract_error_info(span, attrs)
            for key, value in error_info.items():
                self._set_span_attribute(span, key, value)

            # === Performance Indicators ===
            performance = self._extract_performance_indicators(duration_ms, command)
            for key, value in performance.items():
                self._set_span_attribute(span, key, value)

            # === Trace Correlation ===
            correlation = self._extract_trace_correlation(span)
            for key, value in correlation.items():
                self._set_span_attribute(span, key, value)

            # === Parent Context ===
            parent_context = self._extract_parent_context(span)
            for key, value in parent_context.items():
                self._set_span_attribute(span, key, value)

            # === Application Context ===
            app_context = self._extract_application_context(attrs)
            for key, value in app_context.items():
                self._set_span_attribute(span, key, value)

            # === Tags for Categorization ===
            tags = attrs.get("tag.tags", [])
            if not isinstance(tags, list):
                tags = []

            # Add operation context tag
            if operation_context not in tags:
                tags.append(operation_context)

            # Add performance category tag
            if duration_ms is not None:
                perf_category = performance.get("redis.performance_category")
                if perf_category and perf_category not in tags:
                    tags.append(f"perf:{perf_category}")

            # Add error tag if applicable
            if error_info.get("redis.has_error"):
                tags.append("error")

            self._set_span_attribute(span, "tag.tags", tags)

        # Forward enriched span to wrapped processor
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
            # BatchSpanProcessor exports asynchronously in a background thread,
            # preventing export timeouts from blocking request processing.
            # (SimpleSpanProcessor blocks the calling thread on every span end.)
            base_processor = BatchSpanProcessor(arthur_exporter)
            enriched_processor = RedisSpanEnrichmentProcessor(base_processor)
            _tracer_provider.add_span_processor(enriched_processor)
            exporters_configured.append(f"Arthur ({endpoint})")
            logger.info(
                f"Arthur OTLP exporter configured: {endpoint} (enriching Redis spans with OpenInference metadata)"
            )

        # Add console exporter if configured (for local development/Docker logs)
        if export_console:
            console_exporter = ConsoleSpanExporter()
            # Create span processor and wrap it with enrichment to add OpenInference metadata
            base_processor = SimpleSpanProcessor(console_exporter)
            enriched_processor = RedisSpanEnrichmentProcessor(base_processor)
            _tracer_provider.add_span_processor(enriched_processor)
            exporters_configured.append("Console (stdout/stderr)")
            logger.info(
                "Console span exporter enabled - spans will appear in Docker logs (enriching Redis spans with OpenInference metadata)"
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
