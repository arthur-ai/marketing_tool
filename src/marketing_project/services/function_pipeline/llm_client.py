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
    close_span,
    create_span,
    get_tracer,
    is_tracing_available,
    record_span_exception,
    set_span_attribute,
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
        if context and is_tracing_available():
            context_span = create_span(
                f"pipeline.context_building.{step_name}",
                attributes={
                    "step_name": step_name,
                    "context_keys_count": len(context.keys()) if context else 0,
                },
            )
            if context_span and job_id:
                set_span_attribute(context_span, "job_id", job_id)

        try:
            if context:
                # Try to use context references if context registry is available
                uses_context_registry = False
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
                                ref = await context_registry.get_context_reference(
                                    job_id, key
                                )
                                if ref:
                                    context_refs.append(
                                        f"- {key}: [context reference: {ref.step_name}]"
                                    )

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

                # Update context span with size info
                if context_span:
                    context_size_bytes = len(
                        json.dumps(context, default=_json_serializer).encode("utf-8")
                    )
                    set_span_attribute(
                        context_span, "context_size_bytes", context_size_bytes
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
        llm_call_span = create_span(
            f"pipeline.llm_call.{step_name}",
            attributes={
                "step_name": step_name,
                "step_number": step_number,
                "model": step_model,
                "temperature": step_temperature,
                "max_retries": step_max_retries,
            },
        )
        if llm_call_span and job_id:
            set_span_attribute(llm_call_span, "job_id", job_id)

        try:
            for attempt in range(step_max_retries):
                try:
                    logger.info(
                        f"Step {step_number}: {step_name} (attempt {attempt + 1}/{step_max_retries}, model: {step_model})"
                    )

                    if llm_call_span:
                        set_span_attribute(llm_call_span, "retry_attempt", attempt + 1)

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
                    # Update LLM call span on error
                    if llm_call_span:
                        record_span_exception(llm_call_span, e)
                        set_span_attribute(
                            llm_call_span, "error.type", type(e).__name__
                        )
                        set_span_attribute(llm_call_span, "error.message", str(e))
                        set_span_attribute(llm_call_span, "retry_attempt", attempt + 1)

                    # Check if this is an ApprovalRequiredException - don't retry, just re-raise
                    from marketing_project.processors.approval_helper import (
                        ApprovalRequiredException,
                    )

                    if isinstance(e, ApprovalRequiredException):
                        # Approval required is not an error for the LLM call span
                        if llm_call_span:
                            set_span_attribute(llm_call_span, "approval_required", True)
                            if Status and StatusCode:
                                set_span_status(
                                    llm_call_span, StatusCode.OK
                                )  # Still OK
                        raise  # Re-raise immediately without retrying

                    logger.warning(
                        f"Step {step_number} failed (attempt {attempt + 1}): {e}"
                    )

                    if attempt == step_max_retries - 1:
                        # Final attempt failed
                        if llm_call_span:
                            if Status and StatusCode:
                                set_span_status(llm_call_span, StatusCode.ERROR, str(e))
                            set_span_attribute(
                                llm_call_span, "all_retries_exhausted", True
                            )
                        raise

                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(2**attempt)
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
                    # Set span attributes
                    set_span_attribute(span, "step_name", step_name)
                    set_span_attribute(span, "step_number", step_number)
                    set_span_attribute(span, "model", step_model)
                    set_span_attribute(span, "temperature", step_temperature)
                    set_span_attribute(span, "attempt", attempt + 1)
                    if job_id:
                        set_span_attribute(span, "job_id", job_id)
                    if context:
                        content_type = context.get("content_type")
                        if content_type:
                            set_span_attribute(span, "content_type", content_type)

                    # Schema generation span
                    schema_span = create_span(
                        f"pipeline.schema_generation.{step_name}",
                        attributes={
                            "step_name": step_name,
                            "response_model_name": response_model.__name__,
                        },
                    )
                    if schema_span:
                        try:
                            schema_dict = response_model.model_json_schema()
                            schema_complexity = len(json.dumps(schema_dict))
                            set_span_attribute(
                                schema_span, "schema_complexity", schema_complexity
                            )
                        except Exception:
                            pass

                    try:
                        response = await self.client.beta.chat.completions.parse(
                            model=step_model,
                            messages=messages,
                            response_format=response_model,
                            temperature=step_temperature,
                        )

                        # Close schema span
                        if schema_span:
                            if Status and StatusCode:
                                set_span_status(schema_span, StatusCode.OK)
                            close_span(schema_span)

                        # Result parsing span
                        parsing_span = create_span(
                            f"pipeline.result_parsing.{step_name}",
                            attributes={
                                "step_name": step_name,
                                "response_model_name": response_model.__name__,
                            },
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
                                    except Exception:
                                        pass
                                if Status and StatusCode:
                                    set_span_status(parsing_span, StatusCode.OK)
                        except Exception as parse_error:
                            if parsing_span:
                                record_span_exception(parsing_span, parse_error)
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
                        if response.usage:
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
