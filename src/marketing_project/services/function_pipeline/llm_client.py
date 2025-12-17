"""
LLM Client for making OpenAI API calls with structured outputs.
"""

import asyncio
import json
import logging
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

from marketing_project.services.function_pipeline.tracing import (
    add_span_event,
    close_span,
    create_span,
    ensure_span_has_minimum_metadata,
    extract_model_info,
    extract_quality_metrics,
    get_tracer,
    is_tracing_available,
    record_span_exception,
    set_llm_invocation_parameters,
    set_llm_messages,
    set_llm_response_format,
    set_llm_token_counts,
    set_span_attribute,
    set_span_duration,
    set_span_error,
    set_span_input,
    set_span_kind,
    set_span_output,
    set_span_status,
)

logger = logging.getLogger("marketing_project.services.function_pipeline.llm_client")

# Import Status for tracing
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    trace = None
    Status = None
    StatusCode = None


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for datetime and other non-serializable objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # Handle Pydantic BaseModel instances
    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump(mode="json")
        except (TypeError, ValueError):
            # Fallback to regular model_dump if mode='json' fails
            return obj.model_dump()
    raise TypeError(f"Type {type(obj)} not serializable")


class LLMClient:
    """Client for making OpenAI API calls with structured outputs."""

    def __init__(self, client: AsyncOpenAI):
        """
        Initialize LLM client.

        Args:
            client: AsyncOpenAI client instance
        """
        self.client = client

    async def build_context_messages(
        self,
        prompt: str,
        system_instruction: str,
        context: Optional[Dict[str, Any]],
        step_name: str,
        job_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build messages list with context from previous steps.

        Args:
            prompt: User prompt
            system_instruction: System instruction
            context: Optional context from previous steps
            step_name: Name of the step
            job_id: Optional job ID for context registry

        Returns:
            List of message dictionaries
        """
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        # Add context from previous steps if available
        context_span = None
        context_start_time = None
        # Initialize context registry metrics (used later if context registry is used)
        context_registry_queries = 0
        context_registry_hits = 0
        context_registry_misses = 0
        context_registry_resolved_keys = []
        context_registry_missing_keys = []
        uses_context_registry = False

        if context and is_tracing_available():
            context_start_time = time.time()
            context_span = create_span(
                f"pipeline.context_building.{step_name}",
                attributes={
                    "step_name": step_name,
                    "context_keys_count": len(context.keys()) if context else 0,
                },
                span_type="context_building",
            )
            if context_span:
                # Set OpenInference span kind
                set_span_kind(context_span, "TOOL")

                # Set input attributes (context being built) - always set, never blank
                set_span_input(context_span, context if context else {})

                # Add event
                add_span_event(
                    context_span,
                    "context_building.started",
                    {
                        "context_keys_count": len(context.keys()) if context else 0,
                    },
                )

                # Ensure minimum metadata
                ensure_span_has_minimum_metadata(
                    context_span,
                    f"pipeline.context_building.{step_name}",
                    "context_building",
                )

                if job_id:
                    set_span_attribute(context_span, "job_id", job_id)

        try:
            if context:
                # Try to use context references if context registry is available
                if job_id:
                    try:
                        from marketing_project.services.context_registry import (
                            get_context_registry,
                        )

                        context_registry = get_context_registry()
                        uses_context_registry = True
                        if context_span:
                            set_span_attribute(
                                context_span, "agentic.context_registry_used", True
                            )

                        # Build context message with references
                        context_refs = []
                        for key in context.keys():
                            if key not in (
                                "input_content",
                                "content_type",
                                "output_content_type",
                                "_execution_step_number",
                            ):
                                context_registry_queries += 1
                                ref = await context_registry.get_context_reference(
                                    job_id, key
                                )
                                if ref:
                                    context_registry_hits += 1
                                    context_registry_resolved_keys.append(key)
                                    context_refs.append(
                                        f"- {key}: [context reference: {ref.step_name}]"
                                    )
                                else:
                                    context_registry_misses += 1
                                    context_registry_missing_keys.append(key)

                        if context_refs:
                            context_msg = (
                                f"\n\n### Context from Previous Steps (References):\n"
                                + "\n".join(context_refs)
                            )
                            # Still include essential context directly
                            essential_context = {
                                k: v
                                for k, v in context.items()
                                if k
                                in (
                                    "input_content",
                                    "content_type",
                                    "output_content_type",
                                )
                            }
                            if essential_context:
                                context_msg += f"\n\n### Essential Context:\n```json\n{json.dumps(essential_context, indent=2, default=_json_serializer)}\n```"
                            messages[-1]["content"] += context_msg
                        else:
                            # Fallback to full context dump
                            context_msg = f"\n\n### Context from Previous Steps:\n```json\n{json.dumps(context, indent=2, default=_json_serializer)}\n```"
                            messages[-1]["content"] += context_msg
                    except Exception as e:
                        logger.debug(
                            f"Failed to use context references, using direct context: {e}"
                        )
                        # Fallback to full context dump
                        context_msg = f"\n\n### Context from Previous Steps:\n```json\n{json.dumps(context, indent=2, default=_json_serializer)}\n```"
                        messages[-1]["content"] += context_msg
                else:
                    # No job_id, use direct context
                    context_msg = f"\n\n### Context from Previous Steps:\n```json\n{json.dumps(context, indent=2, default=_json_serializer)}\n```"
                    messages[-1]["content"] += context_msg

                # Update context span with output (built messages with context)
                if context_span:
                    # Set output attributes (messages with context added) - always set, never blank
                    set_span_output(
                        context_span,
                        messages if messages else [],
                        output_mime_type="application/json",
                    )

                    context_size_bytes = len(
                        json.dumps(context, default=_json_serializer).encode("utf-8")
                    )
                    set_span_attribute(
                        context_span, "context_size_bytes", context_size_bytes
                    )

                    # Add context registry performance metrics
                    if uses_context_registry:
                        try:
                            set_span_attribute(
                                context_span,
                                "context_registry.queries_count",
                                context_registry_queries,
                            )
                            set_span_attribute(
                                context_span,
                                "context_registry.hits",
                                context_registry_hits,
                            )
                            set_span_attribute(
                                context_span,
                                "context_registry.misses",
                                context_registry_misses,
                            )
                            if context_registry_queries > 0:
                                hit_rate = (
                                    context_registry_hits / context_registry_queries
                                )
                                set_span_attribute(
                                    context_span, "context_registry.hit_rate", hit_rate
                                )
                            else:
                                set_span_attribute(
                                    context_span, "context_registry.hit_rate", 1.0
                                )
                            set_span_attribute(
                                context_span,
                                "context_registry.keys_resolved",
                                json.dumps(context_registry_resolved_keys),
                            )
                            set_span_attribute(
                                context_span,
                                "context_registry.keys_missing",
                                json.dumps(context_registry_missing_keys),
                            )
                        except Exception:
                            pass

                    # Set duration
                    if context_start_time:
                        set_span_duration(context_span, context_start_time)

                    # Add event
                    add_span_event(
                        context_span,
                        "context_building.completed",
                        {
                            "uses_context_registry": uses_context_registry,
                        },
                    )

                    if Status and StatusCode:
                        set_span_status(context_span, StatusCode.OK)
        except Exception as e:
            if context_span:
                record_span_exception(context_span, e)
                if Status and StatusCode:
                    set_span_status(context_span, StatusCode.ERROR, str(e))
            raise
        finally:
            close_span(context_span)

        return messages

    async def call_with_retries(
        self,
        messages: List[Dict[str, str]],
        response_model: type[BaseModel],
        step_name: str,
        step_number: int,
        step_model: str,
        step_temperature: float,
        step_max_retries: int,
        job_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> BaseModel:
        """
        Call OpenAI API with structured output and retry logic.

        Args:
            messages: List of message dictionaries
            response_model: Pydantic model for structured output
            step_name: Name of the step
            step_number: Step number
            step_model: Model to use
            step_temperature: Temperature setting
            step_max_retries: Maximum retry attempts
            job_id: Optional job ID
            context: Optional context

        Returns:
            Tuple of (parsed_result, response) where parsed_result is the Pydantic model
            and response is the full OpenAI response object

        Raises:
            Exception: If all retries fail
        """
        start_time = time.time()

        # Create LLM call span wrapping the retry loop
        llm_call_start_time = time.time()
        llm_call_span = create_span(
            f"pipeline.llm_call.{step_name}",
            attributes={
                "step_name": step_name,
                "step_number": step_number,
                "model": step_model,
                "temperature": step_temperature,
                "max_retries": step_max_retries,
            },
            span_type="llm_call",
        )
        if llm_call_span:
            # Set OpenInference span kind
            set_span_kind(llm_call_span, "LLM")

            # Extract and set model performance metrics
            model_info = extract_model_info(step_model)
            for key, value in model_info.items():
                set_span_attribute(llm_call_span, f"llm.{key}", value)

            # Set input attributes (call parameters) - always set, never blank
            set_span_input(
                llm_call_span,
                {
                    "step_name": step_name,
                    "step_number": step_number,
                    "model": step_model,
                    "temperature": step_temperature,
                    "max_retries": step_max_retries,
                },
            )

            # Set additional LLM attributes
            set_span_attribute(llm_call_span, "llm.model_name", step_model)
            set_span_attribute(llm_call_span, "llm.provider", "openai.responses")
            set_span_attribute(llm_call_span, "llm.structured_output", True)
            set_span_attribute(llm_call_span, "llm.streaming", False)

            # Ensure minimum metadata
            ensure_span_has_minimum_metadata(
                llm_call_span, f"pipeline.llm_call.{step_name}", "llm_call"
            )

            # Add event
            add_span_event(
                llm_call_span,
                "llm_call.started",
                {
                    "step_name": step_name,
                    "model": step_model,
                },
            )

            if job_id:
                set_span_attribute(llm_call_span, "job_id", job_id)

        # Track retry history
        retry_history = []

        try:
            for attempt in range(step_max_retries):
                try:
                    logger.info(
                        f"Step {step_number}: {step_name} (attempt {attempt + 1}/{step_max_retries}, model: {step_model})"
                    )

                    if llm_call_span:
                        set_span_attribute(llm_call_span, "retry_attempt", attempt + 1)
                        set_span_attribute(
                            llm_call_span, "retry.attempt_number", attempt + 1
                        )
                        set_span_attribute(
                            llm_call_span, "retry.max_attempts", step_max_retries
                        )

                        # Add retry event
                        if attempt > 0:
                            add_span_event(
                                llm_call_span,
                                "retry.attempt",
                                {
                                    "attempt": attempt + 1,
                                    "max_attempts": step_max_retries,
                                    "previous_errors": len(retry_history),
                                },
                            )

                    # Make the API call with tracing
                    parsed_result, response = await self._make_api_call(
                        messages=messages,
                        response_model=response_model,
                        step_name=step_name,
                        step_number=step_number,
                        step_model=step_model,
                        step_temperature=step_temperature,
                        attempt=attempt,
                        job_id=job_id,
                        context=context,
                    )

                    execution_time = time.time() - start_time

                    # Update LLM call span on success
                    if llm_call_span:
                        # Set output attributes (parsed result)
                        try:
                            if hasattr(parsed_result, "model_dump"):
                                output_data = parsed_result.model_dump()
                            elif hasattr(parsed_result, "dict"):
                                output_data = parsed_result.dict()
                            else:
                                output_data = parsed_result
                            set_span_output(llm_call_span, output_data)

                            # Extract quality metrics
                            extract_quality_metrics(llm_call_span, parsed_result)
                        except Exception as e:
                            logger.debug(f"Failed to set llm_call_span output: {e}")

                        # Set duration
                        set_span_duration(llm_call_span, llm_call_start_time)

                        # Add success event
                        add_span_event(
                            llm_call_span,
                            "llm_call.completed",
                            {
                                "attempt": attempt + 1,
                                "execution_time": execution_time,
                            },
                        )

                        if Status and StatusCode:
                            set_span_status(llm_call_span, StatusCode.OK)
                        if response.usage:
                            set_span_attribute(
                                llm_call_span,
                                "input_tokens",
                                response.usage.prompt_tokens or 0,
                            )
                            set_span_attribute(
                                llm_call_span,
                                "output_tokens",
                                response.usage.completion_tokens or 0,
                            )
                            set_span_attribute(
                                llm_call_span,
                                "total_tokens",
                                response.usage.total_tokens or 0,
                            )

                    logger.info(
                        f"Step {step_number} completed in {execution_time:.2f}s"
                    )
                    return parsed_result, response

                except Exception as e:
                    # Track retry history
                    retry_history.append(
                        {
                            "attempt": attempt + 1,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        }
                    )

                    # Update LLM call span on error
                    if llm_call_span:
                        set_span_error(
                            llm_call_span,
                            e,
                            {
                                "attempt": attempt + 1,
                                "max_attempts": step_max_retries,
                            },
                        )
                        set_span_attribute(llm_call_span, "retry_attempt", attempt + 1)
                        set_span_attribute(
                            llm_call_span, "retry.attempt_number", attempt + 1
                        )
                        set_span_attribute(llm_call_span, "retry.reason", str(e))
                        set_span_attribute(
                            llm_call_span, "retry.backoff_seconds", 2**attempt
                        )

                        # Store retry history
                        if retry_history:
                            import json

                            set_span_attribute(
                                llm_call_span,
                                "retry.previous_attempts",
                                json.dumps(retry_history),
                            )

                        # Add retry error event
                        add_span_event(
                            llm_call_span,
                            "retry.error",
                            {
                                "attempt": attempt + 1,
                                "error_type": type(e).__name__,
                                "will_retry": attempt < step_max_retries - 1,
                            },
                        )

                    # Approval is now handled via sentinel values, not exceptions
                    # This exception handler is for other errors only

                    logger.warning(
                        f"Step {step_number} failed (attempt {attempt + 1}): {e}"
                    )

                    if attempt == step_max_retries - 1:
                        # Final attempt failed
                        if llm_call_span:
                            # Set duration
                            set_span_duration(llm_call_span, llm_call_start_time)

                            # Add failure event
                            add_span_event(
                                llm_call_span,
                                "llm_call.failed",
                                {
                                    "all_retries_exhausted": True,
                                    "total_attempts": step_max_retries,
                                },
                            )

                            if Status and StatusCode:
                                set_span_status(llm_call_span, StatusCode.ERROR, str(e))
                            set_span_attribute(
                                llm_call_span, "all_retries_exhausted", True
                            )
                        raise

                    # Wait before retry (exponential backoff)
                    backoff_seconds = 2**attempt
                    await asyncio.sleep(backoff_seconds)

                    # Add backoff event
                    if llm_call_span:
                        add_span_event(
                            llm_call_span,
                            "retry.backoff",
                            {
                                "backoff_seconds": backoff_seconds,
                                "next_attempt": attempt + 2,
                            },
                        )
        except Exception as e:
            # Ensure LLM call span is closed on any exception
            close_span(llm_call_span, type(e), e, None)
            raise
        finally:
            # Close LLM call span if still open
            close_span(llm_call_span)

    async def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        response_model: type[BaseModel],
        step_name: str,
        step_number: int,
        step_model: str,
        step_temperature: float,
        attempt: int,
        job_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> BaseModel:
        """
        Make a single API call with tracing.

        Args:
            messages: List of message dictionaries
            response_model: Pydantic model for structured output
            step_name: Name of the step
            step_number: Step number
            step_model: Model to use
            step_temperature: Temperature setting
            attempt: Current attempt number
            job_id: Optional job ID
            context: Optional context

        Returns:
            Tuple of (parsed_result, response) where parsed_result is the Pydantic model
            and response is the full OpenAI response object
        """
        # Create OpenTelemetry span for this LLM call
        if is_tracing_available():
            try:
                tracer = get_tracer(__name__)
                with tracer.start_as_current_span(
                    f"function_pipeline.{step_name}",
                    kind=trace.SpanKind.CLIENT,
                ) as span:
                    # Set OpenInference span kind
                    set_span_kind(span, "LLM")

                    # Set input attributes (full context dict) - always set, never blank
                    set_span_input(span, context if context else {})
                    if context:
                        content_type = context.get("content_type")
                        if content_type:
                            set_span_attribute(span, "content_type", content_type)

                    # Ensure minimum metadata
                    ensure_span_has_minimum_metadata(
                        span, f"function_pipeline.{step_name}", "llm_call"
                    )

                    # Set LLM input messages (always set, never blank)
                    set_llm_messages(span, messages if messages else [])

                    # Ensure minimum metadata (already called above, but ensure it's set)
                    ensure_span_has_minimum_metadata(
                        span, f"function_pipeline.{step_name}", "llm_call"
                    )

                    # Extract and set model performance metrics
                    model_info = extract_model_info(step_model)
                    for key, value in model_info.items():
                        set_span_attribute(span, f"llm.{key}", value)

                    # Set span attributes
                    set_span_attribute(span, "step_name", step_name)
                    set_span_attribute(span, "step_number", step_number)
                    set_span_attribute(span, "model", step_model)
                    set_span_attribute(span, "llm.model_name", step_model)
                    set_span_attribute(span, "llm.provider", "openai.responses")
                    set_span_attribute(
                        span, "llm.structured_output", True
                    )  # We use structured output
                    set_span_attribute(
                        span, "llm.streaming", False
                    )  # Not using streaming
                    set_span_attribute(span, "temperature", step_temperature)
                    set_span_attribute(span, "attempt", attempt + 1)
                    if job_id:
                        set_span_attribute(span, "job_id", job_id)

                    # Set LLM response format (structured output)
                    set_llm_response_format(span, response_model)

                    # Set LLM invocation parameters
                    invocation_params = {
                        "temperature": step_temperature,
                        "model": step_model,
                    }
                    set_llm_invocation_parameters(span, invocation_params)

                    # Ensure minimum metadata
                    ensure_span_has_minimum_metadata(
                        span, f"function_pipeline.{step_name}", "llm_call"
                    )

                    # Schema generation span
                    schema_start_time = time.time()
                    schema_span = create_span(
                        f"pipeline.schema_generation.{step_name}",
                        attributes={
                            "step_name": step_name,
                            "response_model_name": response_model.__name__,
                        },
                        span_type="schema_generation",
                    )
                    if schema_span:
                        # Set OpenInference span kind
                        set_span_kind(schema_span, "TOOL")

                        # Set input attributes (response model info) - always set, never blank
                        try:
                            schema_dict = response_model.model_json_schema()
                            set_span_input(
                                schema_span,
                                {
                                    "response_model_name": response_model.__name__,
                                    "schema": schema_dict,
                                },
                            )

                            # Calculate and set schema metrics
                            from marketing_project.services.function_pipeline.tracing import (
                                calculate_schema_metrics,
                            )

                            schema_metrics = calculate_schema_metrics(schema_dict)
                            for key, value in schema_metrics.items():
                                set_span_attribute(schema_span, key, value)

                            schema_complexity = len(json.dumps(schema_dict))
                            set_span_attribute(
                                schema_span, "schema_complexity", schema_complexity
                            )

                            # Set output attributes (generated schema) - always set
                            set_span_output(schema_span, schema_dict)

                            # Ensure minimum metadata
                            ensure_span_has_minimum_metadata(
                                schema_span,
                                f"pipeline.schema_generation.{step_name}",
                                "schema_generation",
                            )

                            # Set duration
                            set_span_duration(schema_span, schema_start_time)
                        except Exception as e:
                            logger.debug(f"Failed to set schema metrics: {e}")
                            # Even on error, set minimal input/output
                            set_span_input(
                                schema_span,
                                {"response_model_name": response_model.__name__},
                            )
                            set_span_output(schema_span, {})

                    try:
                        # Track response time
                        api_call_start_time = time.time()
                        response = await self.client.beta.chat.completions.parse(
                            model=step_model,
                            messages=messages,
                            response_format=response_model,
                            temperature=step_temperature,
                        )
                        api_call_end_time = time.time()
                        response_time_ms = (
                            api_call_end_time - api_call_start_time
                        ) * 1000

                        # Set response time metric
                        set_span_attribute(
                            span, "llm.response_time_ms", response_time_ms
                        )

                        # Close schema span
                        if schema_span:
                            if Status and StatusCode:
                                set_span_status(schema_span, StatusCode.OK)
                            close_span(schema_span)

                        # Result parsing span
                        parsing_start_time = time.time()
                        parsing_span = create_span(
                            f"pipeline.result_parsing.{step_name}",
                            attributes={
                                "step_name": step_name,
                                "response_model_name": response_model.__name__,
                            },
                            span_type="result_parsing",
                        )
                        if parsing_span:
                            # Set OpenInference span kind
                            set_span_kind(parsing_span, "TOOL")

                            # Set input attributes (response to parse) - always set, never blank
                            try:
                                raw_response = (
                                    response.choices[0].message.content
                                    if hasattr(response.choices[0].message, "content")
                                    else None
                                )
                                set_span_input(
                                    parsing_span,
                                    {
                                        "response_model_name": response_model.__name__,
                                        "raw_response": raw_response or "",
                                    },
                                )
                            except Exception:
                                set_span_input(
                                    parsing_span,
                                    {
                                        "response_model_name": response_model.__name__,
                                    },
                                )

                            # Ensure minimum metadata
                            ensure_span_has_minimum_metadata(
                                parsing_span,
                                f"pipeline.result_parsing.{step_name}",
                                "result_parsing",
                            )

                        try:
                            parsed_result = response.choices[0].message.parsed

                            # Update parsing span
                            if parsing_span:
                                set_span_attribute(
                                    parsing_span, "parsing_success", True
                                )
                                if hasattr(parsed_result, "model_dump"):
                                    try:
                                        result_dict = parsed_result.model_dump()
                                        set_span_attribute(
                                            parsing_span,
                                            "result_keys_count",
                                            len(result_dict.keys()),
                                        )
                                        # Set output attributes (parsed result)
                                        set_span_output(parsing_span, result_dict)

                                        # Extract quality metrics
                                        extract_quality_metrics(
                                            parsing_span, parsed_result
                                        )
                                    except Exception:
                                        pass
                                elif parsed_result:
                                    # Set output even if not a Pydantic model
                                    set_span_output(parsing_span, parsed_result)

                                # Set duration
                                set_span_duration(parsing_span, parsing_start_time)

                                # Add event
                                add_span_event(
                                    parsing_span,
                                    "result_parsing.completed",
                                    {
                                        "parsing_success": True,
                                    },
                                )

                                if Status and StatusCode:
                                    set_span_status(parsing_span, StatusCode.OK)
                        except Exception as parse_error:
                            if parsing_span:
                                # Set output attributes (error result)
                                set_span_output(
                                    parsing_span,
                                    {
                                        "parsing_success": False,
                                        "error": str(parse_error),
                                        "error_type": type(parse_error).__name__,
                                    },
                                )

                                # Set duration
                                set_span_duration(parsing_span, parsing_start_time)

                                # Enhanced error handling
                                set_span_error(
                                    parsing_span,
                                    parse_error,
                                    {
                                        "response_model_name": response_model.__name__,
                                    },
                                )

                                # Add event
                                add_span_event(
                                    parsing_span,
                                    "result_parsing.failed",
                                    {
                                        "error_type": type(parse_error).__name__,
                                    },
                                )

                                if Status and StatusCode:
                                    set_span_status(
                                        parsing_span, StatusCode.ERROR, str(parse_error)
                                    )
                                set_span_attribute(
                                    parsing_span, "parsing_success", False
                                )
                            raise
                        finally:
                            close_span(parsing_span)

                    except Exception as schema_error:
                        if schema_span:
                            record_span_exception(schema_span, schema_error)
                            if Status and StatusCode:
                                set_span_status(
                                    schema_span, StatusCode.ERROR, str(schema_error)
                                )
                            close_span(
                                schema_span, type(schema_error), schema_error, None
                            )
                        raise

                    # Update span with response metadata
                    try:
                        # Set output attributes (parsed result)
                        set_span_output(
                            span,
                            (
                                parsed_result.model_dump()
                                if hasattr(parsed_result, "model_dump")
                                else parsed_result
                            ),
                        )

                        # Set LLM output messages
                        output_messages = []
                        if response.choices and len(response.choices) > 0:
                            choice = response.choices[0]
                            if choice.message:
                                output_messages.append(
                                    {
                                        "role": choice.message.role or "assistant",
                                        "content": getattr(
                                            choice.message, "content", None
                                        ),
                                    }
                                )
                        set_llm_messages(
                            span, None, output_messages if output_messages else None
                        )

                        # Set token counts using OpenInference format
                        if response.usage:
                            set_llm_token_counts(
                                span,
                                prompt_tokens=response.usage.prompt_tokens,
                                completion_tokens=response.usage.completion_tokens,
                                total_tokens=response.usage.total_tokens,
                            )
                            # Keep legacy attributes for backward compatibility
                            set_span_attribute(
                                span, "input_tokens", response.usage.prompt_tokens or 0
                            )
                            set_span_attribute(
                                span,
                                "output_tokens",
                                response.usage.completion_tokens or 0,
                            )
                            set_span_attribute(
                                span, "total_tokens", response.usage.total_tokens or 0
                            )
                        if Status and StatusCode:
                            set_span_status(span, StatusCode.OK)
                    except Exception as e:
                        logger.debug(f"Failed to update span with response: {e}")

                    return parsed_result, response
            except Exception as span_error:
                logger.debug(f"Failed to create span: {span_error}")
                # Fallback to non-instrumented call
                response = await self.client.beta.chat.completions.parse(
                    model=step_model,
                    messages=messages,
                    response_format=response_model,
                    temperature=step_temperature,
                )
                return response.choices[0].message.parsed, response
        else:
            # No tracing available, make direct call
            response = await self.client.beta.chat.completions.parse(
                model=step_model,
                messages=messages,
                response_format=response_model,
                temperature=step_temperature,
            )
            return response.choices[0].message.parsed, response
