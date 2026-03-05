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

logger = logging.getLogger("marketing_project.services.function_pipeline.llm_client")


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
        ]
        _context_msg: Optional[str] = None

        if context:
            # Try to use context references if context registry is available
            if job_id:
                try:
                    from marketing_project.services.context_registry import (
                        get_context_registry,
                    )

                    context_registry = get_context_registry()

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
                        _context_msg = context_msg
                    else:
                        # Fallback to full context dump
                        _context_msg = f"\n\n### Context from Previous Steps:\n```json\n{json.dumps(context, indent=2, default=_json_serializer)}\n```"
                except Exception as e:
                    logger.debug(
                        f"Failed to use context references, using direct context: {e}"
                    )
                    # Fallback to full context dump
                    _context_msg = f"\n\n### Context from Previous Steps:\n```json\n{json.dumps(context, indent=2, default=_json_serializer)}\n```"
            else:
                # No job_id, use direct context
                _context_msg = f"\n\n### Context from Previous Steps:\n```json\n{json.dumps(context, indent=2, default=_json_serializer)}\n```"

            # Inject context as a separate turn so the task prompt stays clean
            if _context_msg:
                messages.append({"role": "user", "content": _context_msg})
                messages.append(
                    {
                        "role": "assistant",
                        "content": "I have reviewed the context from the previous pipeline steps and am ready to proceed.",
                    }
                )

        # Always append the task prompt as the final user turn
        messages.append({"role": "user", "content": prompt})
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
        provider_override: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
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

        for attempt in range(step_max_retries):
            try:
                logger.info(
                    f"Step {step_number}: {step_name} (attempt {attempt + 1}/{step_max_retries}, model: {step_model})"
                )

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
                    provider_override=provider_override,
                    model_config=model_config,
                )

                execution_time = time.time() - start_time
                logger.info(f"Step {step_number} completed in {execution_time:.2f}s")
                return parsed_result, response

            except Exception as e:
                # Approval is now handled via sentinel values, not exceptions
                # This exception handler is for other errors only
                logger.warning(
                    f"Step {step_number} failed (attempt {attempt + 1}): {e}"
                )

                if attempt == step_max_retries - 1:
                    raise

                # Wait before retry (exponential backoff)
                backoff_seconds = 2**attempt
                await asyncio.sleep(backoff_seconds)

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
        provider_override: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> BaseModel:
        """
        Make a single API call.

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
        from marketing_project.services.function_pipeline.providers import (
            call_llm_structured,
        )

        return await call_llm_structured(
            messages=messages,
            response_model=response_model,
            model=step_model,
            temperature=step_temperature,
            provider=provider_override,
            model_config=model_config,
            openai_client=self.client,
        )
