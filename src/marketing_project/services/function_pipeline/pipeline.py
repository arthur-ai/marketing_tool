"""
Function-based Pipeline Service using OpenAI Structured Outputs.

This service provides a deterministic, fast, and type-safe pipeline that replaces
the agent-based orchestration with direct function calling.

Key Benefits:
- Guaranteed structured JSON output via Pydantic models
- Faster execution (no agent reasoning loops)
- Predictable costs and token usage
- Easy debugging and testing
- Full type safety
"""

import asyncio
import copy
import json
import logging
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

# OpenTelemetry imports for tracing (kept for backward compatibility)
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


from marketing_project.models.pipeline_steps import (
    ArticleGenerationResult,
    BlogPostPreprocessingApprovalResult,
    ContentFormattingResult,
    DesignKitResult,
    MarketingBriefResult,
    PipelineConfig,
    PipelineResult,
    PipelineStepConfig,
    PipelineStepInfo,
    SEOKeywordsResult,
    SEOOptimizationResult,
    SuggestedLinksResult,
    TranscriptContentExtractionResult,
    TranscriptDurationExtractionResult,
    TranscriptPreprocessingApprovalResult,
    TranscriptSpeakersExtractionResult,
)
from marketing_project.plugins.context_utils import ContextTransformer
from marketing_project.plugins.registry import get_plugin_registry
from marketing_project.prompts.prompts import TEMPLATES, get_template, has_template
from marketing_project.services.function_pipeline.approval import check_step_approval

# Import refactored modules
from marketing_project.services.function_pipeline.helpers import PipelineHelpers
from marketing_project.services.function_pipeline.llm_client import LLMClient
from marketing_project.services.function_pipeline.orchestration import (
    build_initial_context,
    compile_pipeline_result,
    filter_active_plugins,
    load_pipeline_configs,
    register_step_output,
    update_job_progress,
)
from marketing_project.services.function_pipeline.step_results import save_step_result
from marketing_project.services.function_pipeline.tracing import (
    add_job_metadata_to_span,
    add_span_event,
    close_span,
    create_job_root_span,
    create_span,
    ensure_span_has_minimum_metadata,
    extract_content_characteristics,
    extract_quality_metrics,
    extract_step_business_metrics,
    is_tracing_available,
    link_spans,
    record_span_exception,
    set_job_output,
    set_span_attribute,
    set_span_duration,
    set_span_error,
    set_span_input,
    set_span_kind,
    set_span_output,
    set_span_status,
)

logger = logging.getLogger("marketing_project.services.function_pipeline")


class FunctionPipeline:
    """
    Direct function calling pipeline using OpenAI structured outputs.

    This replaces the agent-based orchestration with deterministic function calls,
    ensuring structured JSON output for every step.
    """

    def __init__(
        self,
        model: str = "gpt-5.1",
        temperature: float = 0.7,
        lang: str = "en",
        pipeline_config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize the function pipeline.

        Args:
            model: OpenAI model to use (default: gpt-5.1) - used if pipeline_config not provided
            temperature: Sampling temperature (default: 0.7) - used if pipeline_config not provided
            lang: Language for prompts (default: "en")
            pipeline_config: Optional PipelineConfig for per-step model configuration
        """
        self.client = AsyncOpenAI()
        self.lang = lang
        self.step_info: List[PipelineStepInfo] = []
        self.llm_client = LLMClient(self.client)

        # Support both old-style (model, temperature) and new-style (pipeline_config)
        if pipeline_config:
            self.pipeline_config = pipeline_config
            self.model = pipeline_config.default_model
            self.temperature = pipeline_config.default_temperature
        else:
            self.pipeline_config = PipelineConfig(
                default_model=model,
                default_temperature=temperature,
            )
            self.model = model
            self.temperature = temperature

        # Define optional steps that can fail without stopping the pipeline
        # These are typically enhancement steps that aren't critical for core functionality
        self.optional_steps = {
            "suggested_links",  # Internal links are nice-to-have
            "design_kit",  # Design kit is enhancement
        }

        # Initialize helpers
        self.helpers = PipelineHelpers(
            lang=self.lang, pipeline_config=self.pipeline_config
        )

    def _get_system_instruction(
        self, agent_name: str, context: Optional[Dict] = None
    ) -> str:
        """Load comprehensive system instruction from .j2 template."""
        return self.helpers.get_system_instruction(agent_name, context)

    def _get_user_prompt(self, step_name: str, context: Dict[str, Any]) -> str:
        """Load user prompt from .j2 template and render with context variables."""
        return self.helpers.get_user_prompt(step_name, context)

    def _get_step_model(self, step_name: str) -> str:
        """Get the model to use for a specific step."""
        return self.helpers.get_step_model(step_name)

    def _get_step_temperature(self, step_name: str) -> float:
        """Get the temperature to use for a specific step."""
        return self.helpers.get_step_temperature(step_name)

    def _get_step_max_retries(self, step_name: str) -> int:
        """Get the max retries for a specific step."""
        return self.helpers.get_step_max_retries(step_name)

    async def _execute_step_with_plugin(
        self,
        step_name: str,
        pipeline_context: Dict[str, Any],
        job_id: Optional[str] = None,
        execution_step_number: Optional[int] = None,
    ) -> BaseModel:
        """
        Execute a pipeline step using its plugin.

        Args:
            step_name: Name of the step to execute
            pipeline_context: Accumulated context from previous steps
            job_id: Optional job ID for tracking
            execution_step_number: Optional actual execution step number (for dynamic numbering)

        Returns:
            Pydantic model instance with step results
        """
        # Create step execution span
        step_start_time = time.time()
        context_keys_available = (
            list(pipeline_context.keys()) if pipeline_context else []
        )
        step_span = create_span(
            f"pipeline.step_execution.{step_name}",
            attributes={
                "step_name": step_name,
                "plugin_name": step_name,
                "context_keys_available": json.dumps(context_keys_available),
                "context_keys_count": len(context_keys_available),
            },
            span_type="step_execution",
        )
        if step_span:
            # Set OpenInference span kind
            set_span_kind(step_span, "AGENT")

            # Set input attributes (pipeline context) - always set, never blank
            set_span_input(step_span, pipeline_context if pipeline_context else {})

            # Store original job input (first input of the full job) for observability
            original_input = (
                pipeline_context.get("input_content") if pipeline_context else None
            )
            if original_input:
                try:
                    # Create a snapshot of the original input
                    if isinstance(original_input, dict):
                        input_snapshot = {
                            "content_type": pipeline_context.get(
                                "content_type", "unknown"
                            ),
                            "has_title": bool(original_input.get("title")),
                            "title": (
                                original_input.get("title", "")[:200]
                                if original_input.get("title")
                                else None
                            ),  # Truncate for size
                        }
                        # Store preview of content if available
                        if "content" in original_input:
                            content_str = str(original_input["content"])
                            input_snapshot["content_preview"] = content_str[
                                :500
                            ]  # First 500 chars
                            input_snapshot["content_size_bytes"] = len(
                                content_str.encode("utf-8")
                            )
                    else:
                        input_snapshot = {
                            "content_type": (
                                pipeline_context.get("content_type", "unknown")
                                if pipeline_context
                                else "unknown"
                            ),
                            "input_type": type(original_input).__name__,
                            "input_preview": str(original_input)[:500],
                        }
                    set_span_attribute(
                        step_span,
                        "job.original_input_snapshot",
                        json.dumps(input_snapshot, default=str),
                    )
                except Exception as e:
                    logger.debug(f"Failed to store original input snapshot: {e}")
            else:
                # Set default snapshot if no original input
                try:
                    input_snapshot = {
                        "content_type": (
                            pipeline_context.get("content_type", "unknown")
                            if pipeline_context
                            else "unknown"
                        ),
                        "has_title": False,
                    }
                    set_span_attribute(
                        step_span,
                        "job.original_input_snapshot",
                        json.dumps(input_snapshot, default=str),
                    )
                except Exception:
                    pass

            # Add step dependencies tracking
            try:
                registry = get_plugin_registry()
                plugin = registry.get_plugin(step_name)
                if plugin:
                    required_keys = plugin.get_required_context_keys()
                    available_keys = (
                        list(pipeline_context.keys()) if pipeline_context else []
                    )
                    missing_keys = [k for k in required_keys if k not in available_keys]

                    set_span_attribute(
                        step_span, "step.dependencies", json.dumps(required_keys)
                    )
                    set_span_attribute(
                        step_span,
                        "step.dependencies_available",
                        json.dumps(available_keys),
                    )
                    set_span_attribute(
                        step_span, "step.dependencies_missing", json.dumps(missing_keys)
                    )
                    set_span_attribute(
                        step_span, "step.dependencies_satisfied", len(missing_keys) == 0
                    )
                    set_span_attribute(
                        step_span, "step.execution_order", execution_step_number or 0
                    )
            except Exception:
                # Set defaults if plugin not found
                set_span_attribute(step_span, "step.dependencies", json.dumps([]))
                set_span_attribute(
                    step_span,
                    "step.dependencies_available",
                    json.dumps(context_keys_available),
                )
                set_span_attribute(
                    step_span, "step.dependencies_missing", json.dumps([])
                )
                set_span_attribute(step_span, "step.dependencies_satisfied", True)

            # Ensure minimum metadata
            ensure_span_has_minimum_metadata(
                step_span, f"pipeline.step_execution.{step_name}", "step_execution"
            )

            # Add event
            add_span_event(
                step_span,
                "step.started",
                {
                    "step_name": step_name,
                    "context_keys_count": len(context_keys_available),
                },
            )

            if execution_step_number:
                set_span_attribute(step_span, "step_number", execution_step_number)
            if job_id:
                set_span_attribute(step_span, "job_id", job_id)

        try:
            registry = get_plugin_registry()
            plugin = registry.get_plugin(step_name)

            if not plugin:
                raise ValueError(f"Plugin not found for step: {step_name}")

            # Try to resolve missing context keys from context registry if available
            if job_id:
                try:
                    from marketing_project.services.context_registry import (
                        get_context_registry,
                    )

                    context_registry = get_context_registry()
                    required_keys = plugin.get_required_context_keys()
                    missing_keys = [
                        key for key in required_keys if key not in pipeline_context
                    ]

                    # Resolve missing keys from context registry
                    if missing_keys:
                        resolved_context = await context_registry.query_context(
                            job_id=job_id, keys=missing_keys
                        )
                        # Merge resolved context into pipeline_context
                        resolved_keys = []
                        for key, value in resolved_context.items():
                            if key not in pipeline_context:
                                pipeline_context[key] = value
                                resolved_keys.append(key)
                                logger.debug(
                                    f"Resolved context key '{key}' from context registry for step {step_name}"
                                )

                        # Add context resolution event and metrics
                        if step_span:
                            if resolved_keys:
                                add_span_event(
                                    step_span,
                                    "context.resolved",
                                    {
                                        "resolved_keys": resolved_keys,
                                        "source": "context_registry",
                                    },
                                )
                                set_span_attribute(
                                    step_span,
                                    "context_registry.hits",
                                    len(resolved_keys),
                                )
                                set_span_attribute(
                                    step_span,
                                    "context_registry.keys_resolved",
                                    json.dumps(resolved_keys),
                                )
                            else:
                                set_span_attribute(
                                    step_span, "context_registry.hits", 0
                                )

                            # Track context registry metrics
                            set_span_attribute(
                                step_span,
                                "context_registry.queries_count",
                                len(missing_keys),
                            )
                            set_span_attribute(
                                step_span,
                                "context_registry.misses",
                                len(missing_keys) - len(resolved_keys),
                            )
                            if len(missing_keys) > 0:
                                hit_rate = len(resolved_keys) / len(missing_keys)
                                set_span_attribute(
                                    step_span, "context_registry.hit_rate", hit_rate
                                )
                            else:
                                set_span_attribute(
                                    step_span, "context_registry.hit_rate", 1.0
                                )
                except Exception as e:
                    logger.warning(
                        f"Failed to resolve context from registry for step {step_name}: {e}"
                    )

            # Validate context
            if not plugin.validate_context(pipeline_context):
                missing = [
                    key
                    for key in plugin.get_required_context_keys()
                    if key not in pipeline_context
                ]
                raise ValueError(
                    f"Missing required context keys for {step_name}: {missing}"
                )

            # Store execution step number in pipeline context for plugins to use
            if execution_step_number is not None:
                pipeline_context["_execution_step_number"] = execution_step_number

            # Set model configuration on plugin if available
            step_config = self.pipeline_config.get_step_config(step_name)
            plugin.model_config = step_config

            # Execute plugin
            result = await plugin.execute(
                context=pipeline_context, pipeline=self, job_id=job_id
            )

            # Check if result is ApprovalRequiredSentinel (approval required, stop execution)
            from marketing_project.processors.approval_helper import (
                ApprovalRequiredSentinel,
            )

            if isinstance(result, ApprovalRequiredSentinel):
                logger.info(
                    f"Pipeline execution stopped at step {step_name} due to approval requirement. "
                    f"Approval ID: {result.approval_result.approval_id}"
                )
                # Set output attributes (approval sentinel)
                if step_span:
                    set_span_output(
                        step_span,
                        {
                            "type": "ApprovalRequiredSentinel",
                            "approval_id": result.approval_result.approval_id,
                        },
                    )

                    # Set duration
                    set_span_duration(step_span, step_start_time)

                    # Add approval required event
                    add_span_event(
                        step_span,
                        "approval.required",
                        {
                            "approval_id": result.approval_result.approval_id,
                        },
                    )

                    # Link to approval span if we can get the span context
                    try:
                        from opentelemetry import trace

                        approval_span_context = (
                            trace.get_current_span().get_span_context()
                        )
                        if approval_span_context and approval_span_context.is_valid:
                            link_spans(
                                step_span,
                                approval_span_context,
                                {
                                    "relationship": "approval_required",
                                },
                            )
                    except Exception:
                        pass

                # Close span and return the sentinel to propagate up
                if Status and StatusCode:
                    set_span_status(step_span, StatusCode.OK)  # Not an error
                close_span(step_span)
                return result

            # Set output attributes (step result) - always set, never blank
            if step_span:
                try:
                    # Convert result to dict if it's a Pydantic model
                    if hasattr(result, "model_dump"):
                        output_data = result.model_dump()
                    elif hasattr(result, "dict"):
                        output_data = result.dict()
                    elif isinstance(result, dict):
                        output_data = result
                    else:
                        output_data = {"result_type": type(result).__name__}

                    # Always set output (never blank)
                    set_span_output(step_span, output_data)

                    # Store final output summary (last output of the step) for observability
                    try:
                        output_summary = {
                            "step_name": step_name,
                            "output_keys": (
                                list(output_data.keys())
                                if isinstance(output_data, dict)
                                else []
                            ),
                            "output_keys_count": (
                                len(output_data.keys())
                                if isinstance(output_data, dict)
                                else 0
                            ),
                            "has_confidence_score": isinstance(output_data, dict)
                            and "confidence_score" in output_data,
                            "output_size_bytes": len(
                                json.dumps(output_data, default=str).encode("utf-8")
                            ),
                        }
                        # Add step-specific key indicators
                        if isinstance(output_data, dict):
                            if "main_keyword" in output_data:
                                output_summary["has_main_keyword"] = True
                            if "target_audience" in output_data:
                                output_summary["has_target_audience"] = True
                            if "key_messages" in output_data:
                                output_summary["has_key_messages"] = True
                        set_span_attribute(
                            step_span,
                            "step.final_output_summary",
                            json.dumps(output_summary, default=str),
                        )
                    except Exception as e:
                        logger.debug(f"Failed to store final output summary: {e}")

                    # Extract quality metrics
                    extract_quality_metrics(step_span, result)

                    # Extract step-specific business metrics
                    business_metrics = extract_step_business_metrics(step_name, result)
                    for key, value in business_metrics.items():
                        set_span_attribute(step_span, key, value)

                    # Add content transformation metrics
                    try:
                        if pipeline_context:
                            input_size = len(
                                json.dumps(pipeline_context, default=str).encode(
                                    "utf-8"
                                )
                            )
                            output_size = len(
                                json.dumps(output_data, default=str).encode("utf-8")
                            )
                            set_span_attribute(
                                step_span, "transformation.input_size_bytes", input_size
                            )
                            set_span_attribute(
                                step_span,
                                "transformation.output_size_bytes",
                                output_size,
                            )
                            if input_size > 0:
                                size_change_percent = (
                                    (output_size - input_size) / input_size
                                ) * 100
                                set_span_attribute(
                                    step_span,
                                    "transformation.size_change_percent",
                                    size_change_percent,
                                )
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(f"Failed to set step execution output: {e}")
                    # Even on error, set minimal output
                    set_span_output(step_span, {"error": str(e)})

            # Set duration and add completion event
            if step_span:
                set_span_duration(step_span, step_start_time)
                add_span_event(
                    step_span,
                    "step.completed",
                    {
                        "step_name": step_name,
                    },
                )

            # Close step execution span
            if Status and StatusCode:
                set_span_status(step_span, StatusCode.OK)
            close_span(step_span)

            return result
        except Exception as e:
            # Update step span on error
            if step_span:
                # Set duration
                set_span_duration(step_span, step_start_time)

                # Enhanced error handling
                set_span_error(
                    step_span,
                    e,
                    {
                        "step_name": step_name,
                        "context_keys_count": len(context_keys_available),
                    },
                )

                # Add failure event
                add_span_event(
                    step_span,
                    "step.failed",
                    {
                        "step_name": step_name,
                        "error_type": type(e).__name__,
                    },
                )

                if Status and StatusCode:
                    set_span_status(step_span, StatusCode.ERROR, str(e))
                close_span(step_span, type(e), e, None)
            raise

    async def _call_function(
        self,
        prompt: str,
        system_instruction: str,
        response_model: type[BaseModel],
        step_name: str,
        step_number: int,
        context: Optional[Dict] = None,
        max_retries: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> BaseModel:
        """
        Call OpenAI with structured output using response_format.

        Integrates with approval system for human-in-the-loop review when enabled.

        Args:
            prompt: User prompt with content to process
            system_instruction: System instructions for this step
            response_model: Pydantic model defining expected output structure
            step_name: Name of the current step
            step_number: Step sequence number
            context: Additional context from previous steps
            max_retries: Maximum number of retry attempts (uses step config if None)
            job_id: Optional job ID for approval tracking

        Returns:
            Instance of response_model with structured data (potentially modified by approval)

        Raises:
            Exception: If function call fails after retries or approval is rejected
        """
        start_time = time.time()

        # Extract relative_step_number from context if present
        relative_step_number = None
        if context and "_relative_step_number" in context:
            relative_step_number = context.get("_relative_step_number")
        elif step_number:  # For initial execution, relative equals absolute
            relative_step_number = step_number

        # Get step-specific model and temperature
        step_model = self._get_step_model(step_name)
        step_temperature = self._get_step_temperature(step_name)
        step_max_retries = (
            max_retries
            if max_retries is not None
            else self._get_step_max_retries(step_name)
        )

        # Build messages with context using LLM client
        messages = await self.llm_client.build_context_messages(
            prompt=prompt,
            system_instruction=system_instruction,
            context=context,
            step_name=step_name,
            job_id=job_id,
        )

        try:
            # Call LLM with retries using LLM client
            parsed_result, response = await self.llm_client.call_with_retries(
                messages=messages,
                response_model=response_model,
                step_name=step_name,
                step_number=step_number,
                step_model=step_model,
                step_temperature=step_temperature,
                step_max_retries=step_max_retries,
                job_id=job_id,
                context=context,
            )

            execution_time = time.time() - start_time

            # ========================================
            # Human-in-the-Loop Approval Integration
            # ========================================
            if job_id:
                from marketing_project.processors.approval_helper import (
                    ApprovalRequiredSentinel,
                    ApprovalResult,
                )

                approval_result = await check_step_approval(
                    parsed_result=parsed_result,
                    step_name=step_name,
                    step_number=step_number,
                    job_id=job_id,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    context=context,
                    start_time=start_time,
                    step_info_list=self.step_info,
                )

                # If approval is required, return sentinel to signal pipeline should stop
                if approval_result.requires_approval:
                    logger.info(
                        f"Pipeline stopping at step {step_number} ({step_name}) "
                        f"due to approval requirement (approval_id: {approval_result.approval_id})"
                    )
                    return ApprovalRequiredSentinel(approval_result)

            # Track step info
            step_info = PipelineStepInfo(
                step_name=step_name,
                step_number=step_number,
                status="success",
                execution_time=execution_time,
                tokens_used=response.usage.total_tokens if response.usage else None,
            )
            self.step_info.append(step_info)

            # Save step result to disk if job_id is provided
            if job_id:
                await save_step_result(
                    parsed_result=parsed_result,
                    step_name=step_name,
                    step_number=step_number,
                    job_id=job_id,
                    context=context,
                    execution_time=execution_time,
                    response_usage=response.usage if response.usage else None,
                    status="success",
                    relative_step_number=relative_step_number,
                )

            logger.info(f"Step {step_number} completed in {execution_time:.2f}s")
            return parsed_result

        except Exception as e:
            execution_time = time.time() - start_time

            # Handle final failure
            step_info = PipelineStepInfo(
                step_name=step_name,
                step_number=step_number,
                status="failed",
                execution_time=execution_time,
                error_message=str(e),
            )
            self.step_info.append(step_info)

            # Save failed step result to disk if job_id is provided
            if job_id:
                await save_step_result(
                    parsed_result=None,  # No result for failed steps
                    step_name=step_name,
                    step_number=step_number,
                    job_id=job_id,
                    context=context,
                    execution_time=execution_time,
                    status="failed",
                    error_message=str(e),
                    relative_step_number=relative_step_number,
                )

            logger.error(
                f"Step {step_number}: {step_name} failed after {step_max_retries} attempts: {e}"
            )
            raise

    async def execute_pipeline(
        self,
        content_json: str,
        job_id: Optional[str] = None,
        content_type: str = "blog_post",
        output_content_type: Optional[str] = None,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete 7-step content pipeline using function calling.

        This method orchestrates all pipeline steps in sequence, passing results
        from one step to the next, and compiling the final structured output.

        Args:
            content_json: Input content as JSON string
            job_id: Optional job ID for tracking
            content_type: Type of content being processed
            output_content_type: Optional output content type (defaults to content_type)
            pipeline_config: Optional PipelineConfig for per-step model configuration

        Returns:
            Dictionary with complete pipeline results including all step outputs
        """
        # Update pipeline config if provided
        if pipeline_config:
            self.pipeline_config = pipeline_config
            self.model = pipeline_config.default_model
            self.temperature = pipeline_config.default_temperature
        pipeline_start = time.time()
        logger.info("=" * 80)
        logger.info(
            f"Starting Function Pipeline (job_id: {job_id}, type: {content_type})"
        )
        logger.info("=" * 80)

        # Reset step info
        self.step_info = []

        # Create job root span if one doesn't exist (e.g., if called directly, not from ARQ job)
        # If called from ARQ job, the job root span already exists and pipeline span will be a child
        # OpenTelemetry context propagation will automatically nest spans correctly
        job_root_span = None
        if is_tracing_available() and job_id:
            try:
                # Check if there's already a current span (from ARQ job)
                # If there's no active span, create a job root span
                current_span = trace.get_current_span()
                # Check if span context is valid (has a valid span_id)
                span_context = current_span.get_span_context() if current_span else None
                if span_context is None or span_context.span_id == 0:
                    # Get job to extract metadata
                    job = None
                    try:
                        from marketing_project.services.job_manager import (
                            get_job_manager,
                        )

                        job_manager = get_job_manager()
                        job = await job_manager.get_job(job_id)
                        # Ensure content_type is in metadata
                        if job and job.metadata and "content_type" not in job.metadata:
                            job.metadata["content_type"] = content_type
                    except Exception:
                        pass

                    job_root_span = create_job_root_span(
                        job_id=job_id,
                        job_type="pipeline",
                        input_value=content_json,
                        job=job,
                    )
            except Exception as e:
                logger.debug(f"Failed to check/create job root span: {e}")

        # Parse input content
        try:
            content = json.loads(content_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            raise ValueError(f"Invalid JSON input: {e}")

        # Create pipeline execution span (will be child of job root span if it exists)
        pipeline_span = None
        pipeline_start_time = time.time()
        if is_tracing_available():
            try:
                tracer = trace.get_tracer(__name__)
                pipeline_span = tracer.start_as_current_span(
                    "pipeline.execute", kind=trace.SpanKind.INTERNAL
                )
                pipeline_span.__enter__()

                # Store start time for duration calculation
                pipeline_span._pipeline_start_time = pipeline_start_time

                # Set OpenInference span kind
                set_span_kind(pipeline_span, "CHAIN")

                # Set input attributes (always set, never blank)
                set_span_input(pipeline_span, content if content else {})

                # Extract and add content characteristics
                content_chars = extract_content_characteristics(
                    content if content else {}
                )
                for key, value in content_chars.items():
                    if value is not None and value != "unknown":
                        set_span_attribute(pipeline_span, f"content.{key}", value)

                # Add pipeline configuration metadata
                try:
                    from marketing_project.prompts.prompts import TEMPLATE_VERSION

                    set_span_attribute(
                        pipeline_span, "pipeline.template_version", TEMPLATE_VERSION
                    )
                except Exception:
                    set_span_attribute(pipeline_span, "pipeline.template_version", "v1")

                # Add pipeline config version (could be from env or config)
                set_span_attribute(pipeline_span, "pipeline.config_version", "v1")

                # Add enabled steps (get from pipeline config)
                try:
                    enabled_steps = [
                        step.step_name for step in self.steps if step.enabled
                    ]
                    optional_steps = [
                        step.step_name for step in self.steps if step.optional
                    ]
                    set_span_attribute(
                        pipeline_span,
                        "pipeline.enabled_steps",
                        json.dumps(enabled_steps),
                    )
                    set_span_attribute(
                        pipeline_span,
                        "pipeline.optional_steps",
                        json.dumps(optional_steps),
                    )
                    set_span_attribute(
                        pipeline_span, "pipeline.total_steps", len(enabled_steps)
                    )
                except Exception:
                    pass

                # Ensure minimum metadata
                ensure_span_has_minimum_metadata(
                    pipeline_span, "pipeline.execute", "pipeline_execute"
                )

                # Add started event
                add_span_event(
                    pipeline_span,
                    "pipeline.started",
                    {
                        "content_type": content_type,
                        "output_content_type": output_content_type or "unknown",
                    },
                )

                # Set other attributes
                pipeline_span.set_attribute("agentic.workflow_type", "pipeline")
                pipeline_span.set_attribute("pipeline_type", "function_pipeline")
                pipeline_span.set_attribute("content_type", content_type)
                if output_content_type:
                    pipeline_span.set_attribute(
                        "output_content_type", output_content_type
                    )
                if job_id:
                    pipeline_span.set_attribute("job_id", job_id)
            except Exception as e:
                logger.debug(f"Failed to create pipeline execution span: {e}")
                pipeline_span = None

        # Store input content in job metadata if job_id is provided
        if job_id:
            try:
                from marketing_project.services.job_manager import get_job_manager

                job_manager = get_job_manager()
                job = await job_manager.get_job(job_id)
                if job:
                    job.metadata["input_content"] = content
                    # Also extract and store title for easier access
                    if isinstance(content, dict) and "title" in content:
                        job.metadata["title"] = content["title"]
                    await job_manager._save_job(job)
            except Exception as e:
                logger.warning(
                    f"Failed to store input content in job metadata for {job_id}: {e}"
                )

        # Load configurations
        configs = await load_pipeline_configs()
        internal_docs_config = configs["internal_docs_config"]
        design_kit_config = configs["design_kit_config"]

        # Use output_content_type if provided, otherwise default to content_type
        final_output_content_type = output_content_type or content_type
        if output_content_type and output_content_type != content_type:
            logger.info(
                f"Function Pipeline: Converting {content_type} to {final_output_content_type}"
            )
        else:
            logger.info(
                f"Function Pipeline: Processing {content_type} (output_content_type={final_output_content_type})"
            )

        # Build initial pipeline context
        pipeline_context = build_initial_context(
            content=content,
            content_type=content_type,
            output_content_type=final_output_content_type,
            internal_docs_config=internal_docs_config,
            design_kit_config=design_kit_config,
        )

        results = {}
        quality_warnings = []

        try:
            # Get all plugins in execution order
            registry = get_plugin_registry()

            # Validate dependencies before execution
            is_valid, errors = registry.validate_dependencies()
            if not is_valid:
                error_msg = "Pipeline dependency validation failed:\n" + "\n".join(
                    f"  - {e}" for e in errors
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Log approval configuration for debugging
            try:
                from marketing_project.services.approval_manager import (
                    get_approval_manager,
                )

                approval_manager = await get_approval_manager(reload_from_db=True)
                settings = approval_manager.settings
                logger.info(
                    f"[APPROVAL CONFIG] require_approval={settings.require_approval}, "
                    f"approval_agents={settings.approval_agents}, "
                    f"content_type={content_type}"
                )
            except Exception as e:
                logger.warning(f"Failed to load approval settings for logging: {e}")

            plugins = registry.get_plugins_in_order()

            # Filter out steps that should be skipped based on content type
            active_plugins = filter_active_plugins(plugins, content_type)

            # Execute each step using its plugin with dynamic step numbers
            failed_steps = []
            total_steps = len(active_plugins)
            pipeline_start_time = time.time()

            # Update pipeline span with total steps
            if pipeline_span:
                try:
                    pipeline_span.set_attribute("total_steps", total_steps)
                except Exception:
                    pass

            for execution_index, plugin in enumerate(active_plugins, start=1):
                # For initial execution, relative_step_number equals execution_index
                pipeline_context["_relative_step_number"] = execution_index

                step_start_time = time.time()
                logger.info(
                    f"Executing step {execution_index}/{total_steps} (plugin step_number={plugin.step_number}): {plugin.step_name}"
                )

                # Update progress with detailed information
                if job_id:
                    await update_job_progress(
                        job_id=job_id,
                        execution_index=execution_index,
                        total_steps=total_steps,
                        plugin=plugin,
                        pipeline_start_time=pipeline_start_time,
                    )

                is_optional = plugin.step_name in self.optional_steps

                try:
                    # Execute step using plugin
                    step_result = await self._execute_step_with_plugin(
                        step_name=plugin.step_name,
                        pipeline_context=pipeline_context,
                        job_id=job_id,
                        execution_step_number=execution_index,
                    )

                    # Check if approval is required (sentinel value)
                    from marketing_project.processors.approval_helper import (
                        ApprovalRequiredSentinel,
                    )

                    if isinstance(step_result, ApprovalRequiredSentinel):
                        # Approval required - stop pipeline execution
                        pipeline_end = time.time()
                        execution_time = pipeline_end - pipeline_start

                        # Update pipeline span for approval (not an error)
                        if pipeline_span:
                            try:
                                approval_output = {
                                    "status": "waiting_for_approval",
                                    "approval_id": step_result.approval_result.approval_id,
                                    "stopped_at_step": step_result.approval_result.step_number,
                                    "stopped_at_step_name": step_result.approval_result.step_name,
                                    "results_so_far": results,
                                }
                                from marketing_project.services.function_pipeline.tracing import (
                                    set_span_output,
                                )

                                set_span_output(pipeline_span, approval_output)
                            except Exception as span_error:
                                logger.warning(
                                    f"Failed to update pipeline span: {span_error}"
                                )

                        # Return early with approval status
                        from marketing_project.services.job_manager import (
                            JobStatus,
                            get_job_manager,
                        )

                        job_manager = get_job_manager()
                        await job_manager.update_job_status(
                            job_id, JobStatus.WAITING_FOR_APPROVAL
                        )

                        return {
                            "status": "waiting_for_approval",
                            "approval_id": step_result.approval_result.approval_id,
                            "step_name": step_result.approval_result.step_name,
                            "step_number": step_result.approval_result.step_number,
                            "results": results,
                            "execution_time": execution_time,
                            "metadata": {
                                "job_id": job_id,
                                "stopped_at_step": step_result.approval_result.step_number,
                            },
                        }

                    # Store result - use model_dump(mode='json') to ensure datetime objects are serialized
                    try:
                        results[plugin.step_name] = step_result.model_dump(mode="json")
                    except (TypeError, ValueError) as e:
                        # Fallback to regular model_dump if mode='json' fails
                        logger.warning(
                            f"Failed to dump {plugin.step_name} result with mode='json': {e}. "
                            f"Falling back to regular model_dump."
                        )
                        results[plugin.step_name] = step_result.model_dump()

                    # Validate result structure for blog_post_preprocessing_approval
                    if plugin.step_name == "blog_post_preprocessing_approval":
                        result_dict = results[plugin.step_name]
                        if not isinstance(result_dict, dict):
                            logger.error(
                                f"blog_post_preprocessing_approval result is not a dict: {type(result_dict)}"
                            )
                        else:
                            required_fields = ["is_valid", "requires_approval"]
                            missing_fields = [
                                f for f in required_fields if f not in result_dict
                            ]
                            if missing_fields:
                                logger.warning(
                                    f"blog_post_preprocessing_approval result missing fields: {missing_fields}"
                                )
                            else:
                                logger.info(
                                    f"blog_post_preprocessing_approval result validated: "
                                    f"is_valid={result_dict.get('is_valid')}, "
                                    f"requires_approval={result_dict.get('requires_approval')}"
                                )

                    # Register step output in context registry for zero data loss
                    if job_id:
                        await register_step_output(
                            job_id=job_id,
                            step_name=plugin.step_name,
                            step_number=execution_index,
                            output_data=results[plugin.step_name],
                            pipeline_context=pipeline_context,
                            required_context_keys=plugin.get_required_context_keys(),
                        )

                    # Add result to pipeline context for next steps
                    pipeline_context[plugin.step_name] = results[plugin.step_name]

                    # Log context update for debugging
                    logger.debug(
                        f"Added {plugin.step_name} result to pipeline context. "
                        f"Context now contains: {list(pipeline_context.keys())}"
                    )

                    # Special logging for blog_post_preprocessing_approval to verify input_content is updated
                    if plugin.step_name == "blog_post_preprocessing_approval":
                        input_content = pipeline_context.get("input_content", {})
                        logger.info(
                            f"After {plugin.step_name}, input_content keys: {list(input_content.keys()) if isinstance(input_content, dict) else 'not a dict'}, "
                            f"author={input_content.get('author') if isinstance(input_content, dict) else 'N/A'}, "
                            f"category={input_content.get('category') if isinstance(input_content, dict) else 'N/A'}"
                        )

                except Exception as e:
                    error_msg = str(e)
                    logger.error(
                        f"Step {execution_index}/{total_steps} (plugin step_number={plugin.step_number}, {plugin.step_name}) failed: {error_msg}"
                    )

                    if is_optional:
                        # Optional step failed - log warning and continue
                        logger.warning(
                            f"Optional step {plugin.step_name} failed, continuing pipeline: {error_msg}"
                        )
                        failed_steps.append(
                            {
                                "step_name": plugin.step_name,
                                "step_number": execution_index,
                                "error": error_msg,
                                "optional": True,
                            }
                        )
                        quality_warnings.append(
                            f"Optional step '{plugin.step_name}' failed: {error_msg}"
                        )
                        # Don't add to results or context, but continue
                        continue
                    else:
                        # Critical step failed - stop pipeline
                        failed_steps.append(
                            {
                                "step_name": plugin.step_name,
                                "step_number": execution_index,
                                "error": error_msg,
                                "optional": False,
                            }
                        )
                        raise

            # ========================================
            # Compile Final Result
            # ========================================
            pipeline_end = time.time()
            execution_time = pipeline_end - pipeline_start

            # Update and close pipeline span
            if pipeline_span:
                try:
                    # Set output attributes (always set, never blank)
                    set_span_output(pipeline_span, result if result else {})

                    pipeline_span.set_attribute("steps_completed", len(results))
                    pipeline_span.set_attribute(
                        "execution_time_seconds", execution_time
                    )

                    # Add business intelligence metrics
                    try:
                        total_steps = (
                            len(self.steps) if hasattr(self, "steps") else len(results)
                        )
                        steps_completed_rate = (
                            len(results) / total_steps if total_steps > 0 else 0.0
                        )
                        set_span_attribute(
                            pipeline_span,
                            "business.steps_completed_rate",
                            steps_completed_rate,
                        )
                        set_span_attribute(
                            pipeline_span, "business.steps_completed", len(results)
                        )
                        set_span_attribute(
                            pipeline_span, "business.total_steps", total_steps
                        )

                        # Calculate success rate (steps that didn't fail)
                        if failed_steps:
                            success_rate = (
                                (total_steps - len(failed_steps)) / total_steps
                                if total_steps > 0
                                else 0.0
                            )
                            set_span_attribute(
                                pipeline_span, "business.success_rate", success_rate
                            )
                            set_span_attribute(
                                pipeline_span,
                                "business.failed_steps_count",
                                len(failed_steps),
                            )
                        else:
                            set_span_attribute(
                                pipeline_span, "business.success_rate", 1.0
                            )
                            set_span_attribute(
                                pipeline_span, "business.failed_steps_count", 0
                            )
                    except Exception:
                        pass

                    # Set duration (pipeline_start was set earlier)
                    if hasattr(pipeline_span, "_pipeline_start_time"):
                        set_span_duration(
                            pipeline_span, pipeline_span._pipeline_start_time
                        )

                    # Add completion event
                    add_span_event(
                        pipeline_span,
                        "pipeline.completed",
                        {
                            "steps_completed": len(results),
                            "execution_time": execution_time,
                        },
                    )

                    if Status and StatusCode:
                        pipeline_span.set_status(Status(StatusCode.OK))
                    pipeline_span.__exit__(None, None, None)
                except Exception as e:
                    logger.debug(f"Failed to close pipeline span: {e}")

            # Close job root span if we created it
            if job_root_span and job_id:
                try:
                    # Refresh job to get updated metadata
                    from marketing_project.services.job_manager import get_job_manager

                    job_manager = get_job_manager()
                    updated_job = await job_manager.get_job(job_id)
                    if updated_job:
                        add_job_metadata_to_span(
                            job_root_span, updated_job, job_id, "pipeline"
                        )
                    set_job_output(job_root_span, result)
                    set_span_status(
                        job_root_span, StatusCode.OK if StatusCode else None
                    )
                    close_span(job_root_span)
                except Exception as e:
                    logger.debug(f"Failed to close job root span: {e}")

            logger.info("=" * 80)
            logger.info(f"Pipeline completed successfully in {execution_time:.2f}s")
            logger.info("=" * 80)

            # Calculate total tokens used
            total_tokens = sum(
                step.tokens_used for step in self.step_info if step.tokens_used
            )

            # Compile final result using orchestration utility
            result = compile_pipeline_result(
                results=results,
                content=content,
                content_type=content_type,
                execution_time=execution_time,
                total_tokens=total_tokens,
                model=self.model,
                step_info=self.step_info,
                failed_steps=failed_steps if failed_steps else None,
                quality_warnings=quality_warnings if quality_warnings else None,
            )
            result["metadata"]["job_id"] = job_id  # Add job_id to metadata
            return result

        except Exception as e:
            # Check if this is an approval required exception
            from marketing_project.processors.approval_helper import (
                ApprovalRequiredException,
            )
            from marketing_project.services.job_manager import (
                JobStatus,
                get_job_manager,
            )

            if (
                False
            ):  # Removed ApprovalRequiredException check - approvals use sentinels now
                # Approval required - pipeline stops, job completes with WAITING_FOR_APPROVAL status
                pipeline_end = time.time()
                execution_time = pipeline_end - pipeline_start

                # Update pipeline span for approval (not an error)
                if pipeline_span:
                    try:
                        # Set output attributes with approval information
                        approval_output = {
                            "status": "waiting_for_approval",
                            "approval_id": e.approval_id,
                            "stopped_at_step": e.step_number,
                            "stopped_at_step_name": e.step_name,
                            "results_so_far": results,
                        }
                        set_span_output(pipeline_span, approval_output)

                        pipeline_span.set_attribute("approval_required", True)
                        pipeline_span.set_attribute("approval_id", e.approval_id)
                        pipeline_span.set_attribute("stopped_at_step", e.step_number)
                        pipeline_span.set_attribute("stopped_at_step_name", e.step_name)
                        pipeline_span.set_status(
                            Status(StatusCode.OK)
                        )  # Still OK, just needs approval
                        pipeline_span.__exit__(None, None, None)
                    except Exception as span_err:
                        logger.debug(
                            f"Failed to update pipeline span for approval: {span_err}"
                        )

                # Close job root span if we created it (approval is not a failure)
                if job_root_span:
                    try:
                        set_span_attribute(
                            job_root_span, "job.status", "waiting_for_approval"
                        )
                        set_span_attribute(job_root_span, "approval_id", e.approval_id)
                        # Set output with approval information
                        approval_output = {
                            "pipeline_status": "waiting_for_approval",
                            "approval_id": e.approval_id,
                            "stopped_at_step": e.step_number,
                            "stopped_at_step_name": e.step_name,
                            "step_results": results if "results" in locals() else [],
                        }
                        set_job_output(job_root_span, approval_output)
                        set_span_status(
                            job_root_span, StatusCode.OK if StatusCode else None
                        )
                        close_span(job_root_span)
                    except Exception as span_err:
                        logger.debug(f"Failed to close job root span: {span_err}")

                logger.info(
                    f"[APPROVAL] Pipeline stopped for approval at step {e.step_number} ({e.step_name}) "
                    f"after {execution_time:.2f}s. Job {e.job_id} marked as WAITING_FOR_APPROVAL"
                )

                # Ensure job status is updated
                job_manager = get_job_manager()
                await job_manager.update_job_status(
                    e.job_id, JobStatus.WAITING_FOR_APPROVAL
                )
                await job_manager.update_job_progress(
                    e.job_id, 90, f"Waiting for approval at step {e.step_number}"
                )

                # Return partial results for display
                return {
                    "pipeline_status": "waiting_for_approval",
                    "step_results": results,
                    "quality_warnings": quality_warnings,
                    "final_content": None,
                    "metadata": {
                        "job_id": e.job_id,
                        "content_id": content.get("id"),
                        "content_type": content_type,
                        "title": content.get("title"),
                        "steps_completed": e.step_number - 1,
                        "execution_time_seconds": execution_time,
                        "total_tokens_used": sum(
                            step.tokens_used
                            for step in self.step_info
                            if step.tokens_used
                        ),
                        "model": self.model,
                        "stopped_at_step": e.step_number,
                        "stopped_at_step_name": e.step_name,
                        "approval_id": e.approval_id,
                        "step_info": [
                            (
                                step.model_dump(mode="json")
                                if hasattr(step, "model_dump")
                                else (
                                    step.model_dump()
                                    if hasattr(step, "model_dump")
                                    else step
                                )
                            )
                            for step in self.step_info
                        ],
                    },
                }

            # Other exceptions - pipeline failed
            pipeline_end = time.time()
            execution_time = pipeline_end - pipeline_start

            logger.error(f"Pipeline failed after {execution_time:.2f}s: {e}")

            # Ensure pipeline span is closed if not already
            if pipeline_span:
                try:
                    # Set output with error information
                    error_output = {
                        "pipeline_status": "failed",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "step_results": results if "results" in locals() else {},
                    }
                    set_span_output(pipeline_span, error_output)
                    pipeline_span.__exit__(type(e), e, None)
                except Exception:
                    pass

            # Close job root span if we created it
            if job_root_span:
                try:
                    set_span_attribute(job_root_span, "job.status", "failed")
                    # Set output with error information
                    error_output = {
                        "pipeline_status": "failed",
                        "error": str(e),
                        "step_results": results if "results" in locals() else [],
                    }
                    set_job_output(job_root_span, error_output)
                    record_span_exception(job_root_span, e)
                    set_span_status(
                        job_root_span, StatusCode.ERROR if StatusCode else None, str(e)
                    )
                    close_span(job_root_span)
                except Exception as span_err:
                    logger.debug(f"Failed to close job root span: {span_err}")

            # Return partial results if available
            return {
                "pipeline_status": "failed",
                "step_results": results,
                "quality_warnings": quality_warnings + [f"Pipeline failed: {str(e)}"],
                "final_content": "",
                "metadata": {
                    "job_id": job_id,
                    "content_id": content.get("id"),
                    "content_type": content_type,
                    "steps_completed": len(results),
                    "execution_time_seconds": execution_time,
                    "error": str(e),
                    "step_info": [
                        (
                            step.model_dump(mode="json")
                            if hasattr(step, "model_dump")
                            else (
                                step.model_dump()
                                if hasattr(step, "model_dump")
                                else step
                            )
                        )
                        for step in self.step_info
                    ],
                },
            }

    async def resume_pipeline(
        self,
        context_data: Dict[str, Any],
        job_id: Optional[str] = None,
        content_type: str = "blog_post",
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> Dict[str, Any]:
        """
        Resume pipeline execution from a saved context (after approval).

        Args:
            context_data: Saved context from approval_manager containing:
                - context: Accumulated context from previous steps
                - last_step: Name of last completed step
                - last_step_number: Step number that was completed
                - step_result: Result from the last step
            job_id: Optional job ID for tracking
            content_type: Type of content being processed
            pipeline_config: Optional PipelineConfig for per-step model configuration

        Returns:
            Dictionary with complete pipeline results
        """
        # Update pipeline config if provided
        if pipeline_config:
            self.pipeline_config = pipeline_config
            self.model = pipeline_config.default_model
            self.temperature = pipeline_config.default_temperature
        pipeline_start = time.time()
        logger.info("=" * 80)
        logger.info(
            f"Resuming Function Pipeline from step {context_data.get('last_step_number')} (job_id: {job_id})"
        )
        logger.info("=" * 80)

        # Load saved context
        saved_context = context_data.get("context", {})
        last_step_number = context_data.get("last_step_number", 0)
        last_step_result = context_data.get("step_result", {})
        last_step_name = context_data.get("last_step", "")

        # Get original content from context_data
        content = context_data.get("original_content")
        if not content:
            raise ValueError(
                "Cannot resume pipeline: original content not found in saved context"
            )

        # Load configurations
        configs = await load_pipeline_configs()
        internal_docs_config = configs["internal_docs_config"]
        design_kit_config = configs["design_kit_config"]

        # Reset step info
        self.step_info = []

        # Start with saved context results
        # Get all plugin step names dynamically
        registry = get_plugin_registry()
        all_step_names = [
            plugin.step_name for plugin in registry.get_plugins_in_order()
        ]

        results = {}
        for step_name in all_step_names:
            if step_name in saved_context:
                results[step_name] = saved_context[step_name]

        # Add the last step result that was approved
        # Handle both "seo_keywords" and "Step 1: seo_keywords" formats
        normalized_last_step = last_step_name
        if ":" in last_step_name:
            normalized_last_step = last_step_name.split(":")[-1].strip()

        if normalized_last_step in all_step_names:
            # Always use last_step_result if available (from approval) - this is the most up-to-date
            if last_step_result:
                results[normalized_last_step] = last_step_result
                logger.debug(
                    f"Loaded {normalized_last_step} from step_result (approved result)"
                )
            elif normalized_last_step in saved_context:
                # Fallback: use from saved_context if last_step_result is empty
                results[normalized_last_step] = saved_context[normalized_last_step]
                logger.debug(f"Loaded {normalized_last_step} from saved_context")
            else:
                logger.warning(
                    f"Step {normalized_last_step} not found in step_result or saved_context"
                )

        # Debug: Log what we have in results
        logger.info(
            f"Resume: Last step was {last_step_name} (normalized: {normalized_last_step}) (step {last_step_number})"
        )
        logger.info(f"Resume: Results loaded: {list(results.keys())}")
        logger.info(f"Resume: Saved context keys: {list(saved_context.keys())}")
        logger.info(
            f"Resume: last_step_result type: {type(last_step_result)}, empty: {not last_step_result if last_step_result else True}"
        )

        # For seo_keywords, check if approval has filtered keywords (from keyword selection)
        if normalized_last_step == "seo_keywords" and results.get("seo_keywords"):
            try:
                from marketing_project.services.approval_manager import (
                    get_approval_manager,
                )

                approval_manager = await get_approval_manager()
                # Find approval for this job and step
                approvals = await approval_manager.list_approvals(
                    job_id=context_data.get("job_id"), status="approved"
                )
                for approval in approvals:
                    if (
                        approval.pipeline_step == "seo_keywords"
                        and approval.modified_output
                    ):
                        # Use filtered keywords from approval (includes main_keyword)
                        results["seo_keywords"] = approval.modified_output
                        logger.info(
                            f"Using filtered keywords from approval {approval.id} for resume, main_keyword: {approval.modified_output.get('main_keyword', 'N/A')}"
                        )
                        break
            except Exception as e:
                logger.warning(
                    f"Could not load approval for filtered keywords: {e}. Using original keywords."
                )
                # Ensure main_keyword exists in results if not from approval
                if "main_keyword" not in results.get("seo_keywords", {}):
                    primary_keywords = results.get("seo_keywords", {}).get(
                        "primary_keywords", []
                    )
                    if primary_keywords:
                        results["seo_keywords"]["main_keyword"] = primary_keywords[0]
                        logger.warning(
                            f"No main_keyword found, using first primary keyword: {primary_keywords[0]}"
                        )

        quality_warnings = []

        try:
            # Rebuild pipeline_context from saved results
            # Preserve output_content_type from original run
            saved_output_content_type = context_data.get("output_content_type")
            pipeline_context = {
                "input_content": content,
                "content_type": context_data.get("content_type", "blog_post"),
                "output_content_type": saved_output_content_type
                or context_data.get("content_type", "blog_post"),
                "internal_docs_config": (
                    internal_docs_config.model_dump() if internal_docs_config else None
                ),
                "design_kit_config": (
                    design_kit_config.model_dump() if design_kit_config else None
                ),
            }
            # Add all saved step results to context
            for step_name in all_step_names:
                if step_name in results:
                    pipeline_context[step_name] = results[step_name]

            # Resume from the step after the approval step
            resume_from = last_step_number + 1

            logger.info(f"Resuming pipeline from step {resume_from}")

            # Get all plugins in execution order
            registry = get_plugin_registry()
            plugins = registry.get_plugins_in_order()

            # Filter out steps that should be skipped
            active_plugins = []
            content_type_resume = context_data.get("content_type", "blog_post")
            # First filter by content type
            filtered_plugins = filter_active_plugins(plugins, content_type_resume)

            for plugin in filtered_plugins:
                # Skip steps that have already been completed
                if plugin.step_number < resume_from:
                    continue

                # Skip if result already exists
                if plugin.step_name in results:
                    logger.info(f"Skipping {plugin.step_name} (already completed)")
                    continue

                active_plugins.append(plugin)

            # Execute remaining steps with dynamic step numbers
            for execution_index, plugin in enumerate(active_plugins, start=resume_from):
                # Calculate relative step number (1-indexed)
                relative_step_number = execution_index - last_step_number

                logger.info(f"Executing step {execution_index}: {plugin.step_name}")

                # Store relative_step_number in context (similar to _execution_step_number)
                pipeline_context["_relative_step_number"] = relative_step_number

                # Validate context before executing
                if not plugin.validate_context(pipeline_context):
                    missing = [
                        key
                        for key in plugin.get_required_context_keys()
                        if key not in pipeline_context
                    ]
                    available_results = list(results.keys())
                    available_context_keys = list(pipeline_context.keys())
                    raise ValueError(
                        f"Cannot resume pipeline: Missing required context keys for {plugin.step_name}: {missing}. "
                        f"Last step was {last_step_name} (step {last_step_number}). "
                        f"Results available: {available_results if available_results else 'none'}. "
                        f"Context keys: {available_context_keys if available_context_keys else 'none'}. "
                        f"step_result present: {bool(last_step_result)}"
                    )

                # Execute step using plugin with dynamic step number
                step_result = await self._execute_step_with_plugin(
                    step_name=plugin.step_name,
                    pipeline_context=pipeline_context,
                    job_id=job_id,
                    execution_step_number=execution_index,
                )

                # Check if approval is required (sentinel value)
                from marketing_project.processors.approval_helper import (
                    ApprovalRequiredSentinel,
                )

                if isinstance(step_result, ApprovalRequiredSentinel):
                    # Approval required - stop pipeline execution
                    pipeline_end = time.time()
                    execution_time = pipeline_end - pipeline_start

                    from marketing_project.services.job_manager import (
                        JobStatus,
                        get_job_manager,
                    )

                    job_manager = get_job_manager()
                    await job_manager.update_job_status(
                        job_id, JobStatus.WAITING_FOR_APPROVAL
                    )

                    return {
                        "status": "waiting_for_approval",
                        "approval_id": step_result.approval_result.approval_id,
                        "step_name": step_result.approval_result.step_name,
                        "step_number": step_result.approval_result.step_number,
                        "results": results,
                        "execution_time": execution_time,
                        "metadata": {
                            "job_id": job_id,
                            "stopped_at_step": step_result.approval_result.step_number,
                        },
                    }

                # Store result - use model_dump(mode='json') to ensure datetime objects are serialized
                try:
                    results[plugin.step_name] = step_result.model_dump(mode="json")
                except (TypeError, ValueError):
                    # Fallback to regular model_dump if mode='json' fails
                    results[plugin.step_name] = step_result.model_dump()
                pipeline_context[plugin.step_name] = results[plugin.step_name]

            # Compile final result
            pipeline_end = time.time()
            execution_time = pipeline_end - pipeline_start

            logger.info("=" * 80)
            logger.info(
                f"Resume Pipeline completed successfully in {execution_time:.2f}s"
            )
            logger.info("=" * 80)

            # Get final formatting result if available
            final_content = None
            if "content_formatting" in results:
                try:
                    from marketing_project.models.pipeline_steps import (
                        ContentFormattingResult,
                    )

                    formatting = ContentFormattingResult(
                        **results["content_formatting"]
                    )
                    final_content = formatting.formatted_html
                except Exception as e:
                    logger.warning(
                        f"Failed to parse content_formatting result: {e}. "
                        "Skipping final_content extraction."
                    )
                    # Try to extract formatted_html directly if it's a dict
                    if isinstance(results["content_formatting"], dict):
                        final_content = results["content_formatting"].get(
                            "formatted_html"
                        )

            # Get input content from context
            input_content = context_data.get("input_content") or context_data.get(
                "original_content"
            )

            # Compile final result using orchestration utility
            result = compile_pipeline_result(
                results=results,
                content=input_content or {},
                content_type=context_data.get("content_type", "blog_post"),
                execution_time=execution_time,
                total_tokens=sum(
                    step.tokens_used for step in self.step_info if step.tokens_used
                ),
                model=self.model,
                step_info=self.step_info,
                failed_steps=None,
                quality_warnings=quality_warnings if quality_warnings else None,
            )
            result["metadata"]["job_id"] = job_id
            result["metadata"]["resumed_from_step"] = last_step_number
            result["input_content"] = input_content  # Include input content in result
            return result

        except Exception as e:
            # Check if this is an approval required exception
            from marketing_project.processors.approval_helper import (
                ApprovalRequiredException,
            )
            from marketing_project.services.job_manager import (
                JobStatus,
                get_job_manager,
            )

            if (
                False
            ):  # Removed ApprovalRequiredException check - approvals use sentinels now
                # Approval required - pipeline stops, job completes with WAITING_FOR_APPROVAL status
                pipeline_end = time.time()
                execution_time = pipeline_end - pipeline_start

                logger.info(
                    f"[APPROVAL] Resume pipeline stopped for approval at step {e.step_number} ({e.step_name}) "
                    f"after {execution_time:.2f}s. Job {e.job_id} marked as WAITING_FOR_APPROVAL"
                )

                # Ensure job status is updated
                job_manager = get_job_manager()
                await job_manager.update_job_status(
                    e.job_id, JobStatus.WAITING_FOR_APPROVAL
                )
                await job_manager.update_job_progress(
                    e.job_id, 90, f"Waiting for approval at step {e.step_number}"
                )

                # Save pipeline context for resume
                from marketing_project.services.approval_manager import (
                    get_approval_manager,
                )

                approval_manager = await get_approval_manager()
                original_content = context_data.get("original_content")
                await approval_manager.save_pipeline_context(
                    job_id=e.job_id,
                    context=pipeline_context,
                    step_name=e.step_name,
                    step_number=e.step_number,
                    step_result={},  # Will be filled by approval_manager from approval
                    original_content=original_content,
                )

                # Return partial results for display
                return {
                    "pipeline_status": "waiting_for_approval",
                    "step_results": results,
                    "quality_warnings": quality_warnings,
                    "final_content": None,
                    "metadata": {
                        "job_id": e.job_id,
                        "resumed_from_step": last_step_number,
                        "steps_completed": len(results),
                        "execution_time_seconds": execution_time,
                        "approval_id": e.approval_id,
                        "approval_step": e.step_number,
                        "approval_step_name": e.step_name,
                        "step_info": [
                            (
                                step.model_dump(mode="json")
                                if hasattr(step, "model_dump")
                                else (
                                    step.model_dump()
                                    if hasattr(step, "model_dump")
                                    else step
                                )
                            )
                            for step in self.step_info
                        ],
                    },
                }

            # Other exceptions - log and re-raise
            logger.error(f"Resume pipeline failed: {e}")
            raise

    async def execute_single_step(
        self,
        step_name: str,
        content_json: str,
        context: Dict[str, Any],
        job_id: Optional[str] = None,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single pipeline step independently.

        This method allows executing individual pipeline steps with user-provided
        context, separate from the full pipeline execution.

        Args:
            step_name: Name of the step to execute (e.g., "seo_keywords")
            content_json: Input content as JSON string
            context: Dictionary containing all required context keys for the step
            job_id: Optional job ID for tracking and result persistence
            pipeline_config: Optional PipelineConfig for per-step model configuration

        Returns:
            Dictionary with step result and execution metadata

        Raises:
            ValueError: If step not found or required context keys are missing
        """
        # Update pipeline config if provided
        if pipeline_config:
            self.pipeline_config = pipeline_config
            self.model = pipeline_config.default_model
            self.temperature = pipeline_config.default_temperature
        import time

        step_start = time.time()
        logger.info("=" * 80)
        logger.info(f"Starting Single Step Execution: {step_name} (job_id: {job_id})")
        logger.info("=" * 80)

        # Reset step info
        self.step_info = []

        # Parse input content
        try:
            content = json.loads(content_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            raise ValueError(f"Invalid JSON input: {e}")

        # Get plugin registry and validate step exists
        registry = get_plugin_registry()
        plugin = registry.get_plugin(step_name)

        if not plugin:
            available_steps = ", ".join(registry.get_all_plugins().keys())
            raise ValueError(
                f"Step '{step_name}' not found. Available steps: {available_steps}"
            )

        # Validate required context keys
        required_keys = plugin.get_required_context_keys()
        missing_keys = [key for key in required_keys if key not in context]

        if missing_keys:
            raise ValueError(
                f"Missing required context keys for {step_name}: {missing_keys}. "
                f"Required keys: {required_keys}"
            )

        # Build pipeline context
        pipeline_context = {
            "input_content": content,
            "content_type": context.get("content_type", "blog_post"),
            "output_content_type": context.get(
                "output_content_type", context.get("content_type", "blog_post")
            ),
        }

        # Add all provided context keys
        for key, value in context.items():
            if key not in ("content_type", "output_content_type"):
                pipeline_context[key] = value

        # Load internal docs configuration if available
        internal_docs_config = None
        try:
            from marketing_project.services.internal_docs_manager import (
                get_internal_docs_manager,
            )

            internal_docs_manager = await get_internal_docs_manager()
            internal_docs_config = await internal_docs_manager.get_active_config()
            if internal_docs_config:
                logger.info("Loaded internal docs configuration")
        except Exception as e:
            logger.warning(
                f"Failed to load internal docs configuration: {e} - continuing without it"
            )

        if internal_docs_config:
            pipeline_context["internal_docs_config"] = internal_docs_config.model_dump(
                mode="json"
            )

        # Load design kit configuration if available
        design_kit_config = None
        try:
            from marketing_project.services.design_kit_manager import (
                get_design_kit_manager,
            )

            design_kit_manager = await get_design_kit_manager()
            design_kit_config = await design_kit_manager.get_active_config()
            if design_kit_config:
                logger.info("Loaded design kit configuration")
        except Exception as e:
            logger.warning(
                f"Failed to load design kit configuration: {e} - continuing without it"
            )

        if design_kit_config:
            pipeline_context["design_kit_config"] = design_kit_config.model_dump(
                mode="json"
            )

        try:
            # Execute step using plugin
            logger.info(f"Executing step {plugin.step_number}: {step_name}")

            step_result = await self._execute_step_with_plugin(
                step_name=step_name,
                pipeline_context=pipeline_context,
                job_id=job_id,
            )

            # Convert result to dict
            try:
                result_dict = step_result.model_dump(mode="json")
            except (TypeError, ValueError):
                result_dict = step_result.model_dump()

            step_end = time.time()
            execution_time = step_end - step_start

            logger.info("=" * 80)
            logger.info(
                f"Single Step Execution completed successfully in {execution_time:.2f}s"
            )
            logger.info("=" * 80)

            # Calculate tokens used
            total_tokens = sum(
                step.tokens_used for step in self.step_info if step.tokens_used
            )

            return {
                "step_name": step_name,
                "step_number": plugin.step_number,
                "result": result_dict,
                "execution_time_seconds": execution_time,
                "total_tokens_used": total_tokens,
                "model": self.model,
                "step_info": [
                    (
                        step.model_dump(mode="json")
                        if hasattr(step, "model_dump")
                        else (
                            step.model_dump() if hasattr(step, "model_dump") else step
                        )
                    )
                    for step in self.step_info
                ],
            }

        except Exception as e:
            step_end = time.time()
            execution_time = step_end - step_start

            logger.error(
                f"Single step execution failed after {execution_time:.2f}s: {e}"
            )
            raise
