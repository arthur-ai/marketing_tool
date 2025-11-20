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
import json
import logging
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel


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
    PipelineResult,
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

logger = logging.getLogger("marketing_project.services.function_pipeline")


class FunctionPipeline:
    """
    Direct function calling pipeline using OpenAI structured outputs.

    This replaces the agent-based orchestration with deterministic function calls,
    ensuring structured JSON output for every step.
    """

    def __init__(
        self, model: str = "gpt-5.1", temperature: float = 0.7, lang: str = "en"
    ):
        """
        Initialize the function pipeline.

        Args:
            model: OpenAI model to use (default: gpt-5.1)
            temperature: Sampling temperature (default: 0.7)
            lang: Language for prompts (default: "en")
        """
        self.client = AsyncOpenAI()
        self.model = model
        self.temperature = temperature
        self.lang = lang
        self.step_info: List[PipelineStepInfo] = []

    def _get_system_instruction(
        self, agent_name: str, context: Optional[Dict] = None
    ) -> str:
        """
        Load comprehensive system instruction from .j2 template and enhance for function calling.

        This method loads the detailed agent prompts (100-170 lines) from the existing
        prompt templates and appends quality scoring requirements for structured output.

        Args:
            agent_name: Name of the agent (e.g., "seo_keywords", "marketing_brief")
            context: Optional context variables for Jinja2 template rendering

        Returns:
            Complete system instruction with quality metrics requirements
        """
        template_name = f"{agent_name}_agent_instructions"

        # Check if template exists using helper function
        if has_template(self.lang, template_name):
            try:
                # Load template using helper function (handles caching and fallback)
                template = get_template(self.lang, template_name)
                base_instruction = template.render(**(context or {}))

                # Append quality scoring requirements for function calling
                quality_addendum = """

## Quality Metrics for Structured Output

**IMPORTANT**: In addition to all requirements above, you MUST provide quality metrics in your response.

The output schema includes confidence and quality score fields. Provide realistic assessments:
- **confidence_score** (0-1): Your confidence in the output quality and accuracy
- **Additional quality scores**: As defined in the output schema (e.g., relevance_score, readability_score, seo_score)

These scores are critical for:
- Approval workflows and human review prioritization
- Quality monitoring and performance tracking
- Automated quality assurance and validation

Be honest in your assessments - scores should reflect actual quality, not aspirational targets."""

                return base_instruction + quality_addendum
            except Exception as e:
                logger.warning(
                    f"Failed to load template {template_name} for language {self.lang}: {e}. "
                    "Using fallback instruction."
                )
        else:
            # Fallback to basic instruction if template not found
            logger.warning(
                f"Template not found for {template_name} in language {self.lang}, "
                "using fallback instruction"
            )

        # Fallback instruction
        return f"""You are a {agent_name.replace('_', ' ')} specialist.

Analyze the provided content and generate structured output according to the schema.
Include confidence_score (0-1) and any other quality metrics defined in the output model."""

    def _get_user_prompt(self, step_name: str, context: Dict[str, Any]) -> str:
        """
        Load user prompt from .j2 template and render with context variables.

        This method loads the user prompt template for a given step and renders it
        with the provided context variables. Pydantic models are automatically
        converted to dicts for safe template rendering.

        Args:
            step_name: Name of the step (e.g., "seo_keywords", "marketing_brief")
            context: Context variables for Jinja2 template rendering

        Returns:
            Rendered user prompt string
        """
        template_name = f"{step_name}_user_prompt"

        # Check if template exists using helper function
        if has_template(self.lang, template_name):
            try:
                # Load template using helper function (handles caching and fallback)
                template = get_template(self.lang, template_name)

                # Prepare context for template rendering using ContextTransformer
                template_context = ContextTransformer.prepare_template_context(context)

                # Add 'content' as alias for 'input_content' if it exists (for template compatibility)
                # Templates like seo_keywords_user_prompt.j2 and design_kit_user_prompt.j2 expect 'content'
                if (
                    "input_content" in template_context
                    and "content" not in template_context
                ):
                    template_context["content"] = template_context["input_content"]

                # Handle content truncation for seo_keywords step
                if step_name == "seo_keywords" and "content" in template_context:
                    content = template_context.get("content", {})
                    if isinstance(content, dict):
                        content_str = content.get("content", "")
                        template_context["content_content_preview"] = (
                            content_str[:8000] if content_str else ""
                        )
                    else:
                        template_context["content_content_preview"] = ""

                # Render template with context
                return template.render(**template_context)
            except Exception as e:
                logger.warning(
                    f"Failed to load or render template {template_name} for language {self.lang}: {e}. "
                    "Using fallback prompt."
                )
                logger.debug(f"Template rendering error details: {e}", exc_info=True)
        else:
            # Fallback to basic prompt if template not found
            logger.warning(
                f"Template not found for {template_name} in language {self.lang}, "
                "using fallback prompt"
            )

        # Fallback prompt
        return f"Process the content for {step_name.replace('_', ' ')} step."

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
        registry = get_plugin_registry()
        plugin = registry.get_plugin(step_name)

        if not plugin:
            raise ValueError(f"Plugin not found for step: {step_name}")

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

        # Execute plugin
        result = await plugin.execute(
            context=pipeline_context, pipeline=self, job_id=job_id
        )

        return result

    async def _call_function(
        self,
        prompt: str,
        system_instruction: str,
        response_model: type[BaseModel],
        step_name: str,
        step_number: int,
        context: Optional[Dict] = None,
        max_retries: int = 2,
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
            max_retries: Maximum number of retry attempts
            job_id: Optional job ID for approval tracking

        Returns:
            Instance of response_model with structured data (potentially modified by approval)

        Raises:
            Exception: If function call fails after retries or approval is rejected
        """
        start_time = time.time()

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        # Add context from previous steps if available
        if context:
            context_msg = f"\n\n### Context from Previous Steps:\n```json\n{json.dumps(context, indent=2, default=_json_serializer)}\n```"
            messages[-1]["content"] += context_msg

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Step {step_number}: {step_name} (attempt {attempt + 1}/{max_retries})"
                )

                response = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_model,
                    temperature=self.temperature,
                )

                execution_time = time.time() - start_time
                parsed_result = response.choices[0].message.parsed

                # ========================================
                # Human-in-the-Loop Approval Integration
                # ========================================
                if job_id:
                    try:
                        from marketing_project.processors.approval_helper import (
                            ApprovalRequiredException,
                            check_and_create_approval_request,
                        )
                        from marketing_project.services.job_manager import (
                            JobStatus,
                            get_job_manager,
                        )

                        logger.info(
                            f"[APPROVAL] Checking approval for step {step_number} ({step_name}) in job {job_id}"
                        )

                        # Convert result to dict for approval system
                        try:
                            result_dict = parsed_result.model_dump(mode="json")
                        except (TypeError, ValueError):
                            result_dict = parsed_result.model_dump()

                        # Extract confidence score if available
                        confidence = result_dict.get("confidence_score")

                        # Prepare input data for approval context
                        # Include original content from pipeline context if available
                        pipeline_content = (
                            context.get("input_content") if context else None
                        )
                        # Get content from context or use a placeholder if not available
                        content_for_approval = (
                            pipeline_content
                            if pipeline_content
                            else {"title": "N/A", "content": "N/A"}
                        )
                        input_data = {
                            "prompt": prompt[:500],  # Truncate for readability
                            "system_instruction": system_instruction[:200],
                            "context_keys": list(context.keys()) if context else [],
                            "original_content": pipeline_content
                            or content_for_approval,
                        }

                        # Check if approval is needed (raises ApprovalRequiredException if required)
                        await check_and_create_approval_request(
                            job_id=job_id,
                            agent_name=step_name,
                            step_name=f"Step {step_number}: {step_name}",
                            step_number=step_number,
                            input_data=input_data,
                            output_data=result_dict,
                            context=context or {},
                            confidence_score=confidence,
                            suggestions=[
                                f"Review {step_name} output quality",
                                "Check alignment with content goals",
                                "Verify accuracy and appropriateness",
                            ],
                        )
                        # If no exception raised, approval not needed or auto-approved, continue
                        logger.info(
                            f"[APPROVAL] No approval required for step {step_number} ({step_name}), continuing pipeline"
                        )

                    except ApprovalRequiredException as e:
                        # Approval required - pipeline should stop
                        logger.info(
                            f"[APPROVAL] Step {step_number} ({step_name}) requires approval. "
                            f"Pipeline stopping. Approval ID: {e.approval_id}"
                        )

                        # Update job status to WAITING_FOR_APPROVAL
                        job_manager = get_job_manager()
                        await job_manager.update_job_status(
                            job_id, JobStatus.WAITING_FOR_APPROVAL
                        )
                        await job_manager.update_job_progress(
                            job_id,
                            90,
                            f"Waiting for approval at step {step_number}: {step_name}",
                        )

                        # Track step info
                        execution_time = time.time() - start_time
                        step_info = PipelineStepInfo(
                            step_name=step_name,
                            step_number=step_number,
                            status="waiting_for_approval",
                            execution_time=execution_time,
                        )
                        self.step_info.append(step_info)

                        # Save step result to disk if job_id is provided (even if waiting for approval)
                        if job_id:
                            try:
                                from marketing_project.services.step_result_manager import (
                                    get_step_result_manager,
                                )

                                step_manager = get_step_result_manager()
                                # Use model_dump(mode='json') to ensure datetime objects are serialized to strings
                                if hasattr(parsed_result, "model_dump"):
                                    try:
                                        result_data = parsed_result.model_dump(
                                            mode="json"
                                        )
                                    except (TypeError, ValueError):
                                        # Fallback to regular model_dump if mode='json' fails
                                        result_data = parsed_result.model_dump()
                                else:
                                    result_data = parsed_result

                                await step_manager.save_step_result(
                                    job_id=job_id,
                                    step_number=step_number,
                                    step_name=step_name,
                                    result_data=result_data,
                                    metadata={
                                        "execution_time": execution_time,
                                        "status": "waiting_for_approval",
                                    },
                                    execution_context_id=None,  # Will be auto-determined
                                    root_job_id=None,  # Will be auto-determined
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to save step result to disk for step {step_number}: {e}"
                                )

                        # Re-raise to signal pipeline should stop
                        raise

                    except Exception as e:
                        # Approval system error - log but continue
                        logger.warning(
                            f"Step {step_number}: Approval check failed (continuing): {e}"
                        )

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
                    try:
                        from marketing_project.services.step_result_manager import (
                            get_step_result_manager,
                        )

                        step_manager = get_step_result_manager()
                        # Use model_dump(mode='json') to ensure datetime objects are serialized to strings
                        if hasattr(parsed_result, "model_dump"):
                            try:
                                result_data = parsed_result.model_dump(mode="json")
                            except (TypeError, ValueError):
                                # Fallback to regular model_dump if mode='json' fails
                                result_data = parsed_result.model_dump()
                        else:
                            result_data = parsed_result

                        await step_manager.save_step_result(
                            job_id=job_id,
                            step_number=step_number,
                            step_name=step_name,
                            result_data=result_data,
                            metadata={
                                "execution_time": execution_time,
                                "tokens_used": (
                                    response.usage.total_tokens
                                    if response.usage
                                    else None
                                ),
                                "status": "success",
                            },
                            execution_context_id=None,  # Will be auto-determined
                            root_job_id=None,  # Will be auto-determined
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to save step result to disk for step {step_number}: {e}"
                        )

                logger.info(f"Step {step_number} completed in {execution_time:.2f}s")
                return parsed_result

            except Exception as e:
                # Check if this is an ApprovalRequiredException - don't retry, just re-raise
                from marketing_project.processors.approval_helper import (
                    ApprovalRequiredException,
                )

                if isinstance(e, ApprovalRequiredException):
                    raise  # Re-raise immediately without retrying

                logger.warning(
                    f"Step {step_number} failed (attempt {attempt + 1}): {e}"
                )

                if attempt == max_retries - 1:
                    # Final attempt failed
                    execution_time = time.time() - start_time
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
                        try:
                            from marketing_project.services.step_result_manager import (
                                get_step_result_manager,
                            )

                            step_manager = get_step_result_manager()
                            await step_manager.save_step_result(
                                job_id=job_id,
                                step_number=step_number,
                                step_name=step_name,
                                result_data={},
                                metadata={
                                    "execution_time": execution_time,
                                    "status": "failed",
                                    "error_message": str(e),
                                },
                                execution_context_id=None,  # Will be auto-determined
                                root_job_id=None,  # Will be auto-determined
                            )
                        except Exception as e2:
                            logger.warning(
                                f"Failed to save failed step result to disk for step {step_number}: {e2}"
                            )

                    logger.error(
                        f"Step {step_number}: {step_name} failed after {max_retries} attempts: {e}"
                    )
                    raise

                # Wait before retry (exponential backoff)
                await asyncio.sleep(2**attempt)

    async def execute_pipeline(
        self,
        content_json: str,
        job_id: Optional[str] = None,
        content_type: str = "blog_post",
        output_content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete 7-step content pipeline using function calling.

        This method orchestrates all pipeline steps in sequence, passing results
        from one step to the next, and compiling the final structured output.

        Args:
            content_json: Input content as JSON string
            job_id: Optional job ID for tracking
            content_type: Type of content being processed

        Returns:
            Dictionary with complete pipeline results including all step outputs
        """
        pipeline_start = time.time()
        logger.info("=" * 80)
        logger.info(
            f"Starting Function Pipeline (job_id: {job_id}, type: {content_type})"
        )
        logger.info("=" * 80)

        # Reset step info
        self.step_info = []

        # Parse input content
        try:
            content = json.loads(content_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            raise ValueError(f"Invalid JSON input: {e}")

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

        # Load internal docs configuration (if available)
        internal_docs_config = None
        try:
            from marketing_project.services.internal_docs_manager import (
                get_internal_docs_manager,
            )

            internal_docs_manager = await get_internal_docs_manager()
            internal_docs_config = await internal_docs_manager.get_active_config()
            if internal_docs_config:
                logger.info("Loaded internal docs configuration")
            else:
                logger.warning(
                    "No internal docs configuration found - pipeline will run without it"
                )
        except Exception as e:
            logger.warning(
                f"Failed to load internal docs configuration: {e} - pipeline will run without it"
            )

        # Load design kit configuration (if available)
        design_kit_config = None
        try:
            from marketing_project.services.design_kit_manager import (
                get_design_kit_manager,
            )

            design_kit_manager = await get_design_kit_manager()
            design_kit_config = await design_kit_manager.get_active_config()
            if design_kit_config:
                logger.info("Loaded design kit configuration")
            else:
                logger.warning(
                    "No design kit configuration found - pipeline will run without it (allowing pipeline to continue)"
                )
        except Exception as e:
            logger.warning(
                f"Failed to load design kit configuration: {e} - pipeline will run without it (allowing pipeline to continue)"
            )

        # Use output_content_type if provided, otherwise default to content_type
        final_output_content_type = output_content_type or content_type

        # Store original content in context for first step (needed for resume)
        pipeline_context = {
            "input_content": content,
            "content_type": content_type,
            "output_content_type": final_output_content_type,
            "internal_docs_config": (
                internal_docs_config.model_dump(mode="json")
                if internal_docs_config
                else None
            ),
            "design_kit_config": (
                design_kit_config.model_dump(mode="json") if design_kit_config else None
            ),
        }

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

            plugins = registry.get_plugins_in_order()

            # Filter out steps that should be skipped based on content type
            # This allows us to calculate dynamic step numbers based on actual execution
            active_plugins = []
            for plugin in plugins:
                # Skip transcript_preprocessing_approval for non-transcript content
                if (
                    plugin.step_name == "transcript_preprocessing_approval"
                    and content_type != "transcript"
                ):
                    logger.info(
                        f"Skipping step {plugin.step_name} (not transcript content)"
                    )
                    continue
                # Skip blog_post_preprocessing_approval for non-blog_post content
                if (
                    plugin.step_name == "blog_post_preprocessing_approval"
                    and content_type != "blog_post"
                ):
                    logger.info(
                        f"Skipping step {plugin.step_name} (not blog_post content)"
                    )
                    continue
                active_plugins.append(plugin)

            # Execute each step using its plugin with dynamic step numbers
            for execution_index, plugin in enumerate(active_plugins, start=1):
                logger.info(f"Executing step {execution_index}: {plugin.step_name}")

                # Execute step using plugin
                step_result = await self._execute_step_with_plugin(
                    step_name=plugin.step_name,
                    pipeline_context=pipeline_context,
                    job_id=job_id,
                )

                # Store result - use model_dump(mode='json') to ensure datetime objects are serialized
                try:
                    results[plugin.step_name] = step_result.model_dump(mode="json")
                except (TypeError, ValueError):
                    # Fallback to regular model_dump if mode='json' fails
                    results[plugin.step_name] = step_result.model_dump()
                pipeline_context[plugin.step_name] = results[plugin.step_name]

            # ========================================
            # Compile Final Result
            # ========================================
            pipeline_end = time.time()
            execution_time = pipeline_end - pipeline_start

            logger.info("=" * 80)
            logger.info(f"Pipeline completed successfully in {execution_time:.2f}s")
            logger.info("=" * 80)

            # Calculate total tokens used
            total_tokens = sum(
                step.tokens_used for step in self.step_info if step.tokens_used
            )

            # Get final content from formatting result if available
            final_content = None
            if "content_formatting" in results:
                from marketing_project.models.pipeline_steps import (
                    ContentFormattingResult,
                )

                formatting = ContentFormattingResult(**results["content_formatting"])
                final_content = formatting.formatted_html

            return {
                "pipeline_status": (
                    "completed_with_warnings" if quality_warnings else "completed"
                ),
                "step_results": results,
                "quality_warnings": quality_warnings,
                "final_content": final_content,
                "input_content": content,  # Include input content in result
                "metadata": {
                    "job_id": job_id,
                    "content_id": content.get("id"),
                    "content_type": content_type,
                    "title": content.get("title"),
                    "steps_completed": len(results),
                    "execution_time_seconds": execution_time,
                    "total_tokens_used": total_tokens,
                    "model": self.model,
                    "completed_at": datetime.utcnow().isoformat(),
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

        except Exception as e:
            # Check if this is an approval required exception
            from marketing_project.processors.approval_helper import (
                ApprovalRequiredException,
            )
            from marketing_project.services.job_manager import (
                JobStatus,
                get_job_manager,
            )

            if isinstance(e, ApprovalRequiredException):
                # Approval required - pipeline stops, job completes with WAITING_FOR_APPROVAL status
                pipeline_end = time.time()
                execution_time = pipeline_end - pipeline_start

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

        Returns:
            Dictionary with complete pipeline results
        """
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

        # Load internal docs configuration (if available)
        internal_docs_config = None
        try:
            from marketing_project.services.internal_docs_manager import (
                get_internal_docs_manager,
            )

            internal_docs_manager = await get_internal_docs_manager()
            internal_docs_config = await internal_docs_manager.get_active_config()
            if internal_docs_config:
                logger.info("Loaded internal docs configuration for resume")
            else:
                logger.warning(
                    "No internal docs configuration found - pipeline will run without it"
                )
        except Exception as e:
            logger.warning(
                f"Failed to load internal docs configuration: {e} - pipeline will run without it"
            )

        # Load design kit configuration (if available)
        design_kit_config = None
        try:
            from marketing_project.services.design_kit_manager import (
                get_design_kit_manager,
            )

            design_kit_manager = await get_design_kit_manager()
            design_kit_config = await design_kit_manager.get_active_config()
            if design_kit_config:
                logger.info("Loaded design kit configuration for resume")
            else:
                logger.warning(
                    "No design kit configuration found - pipeline will run without it (allowing pipeline to continue)"
                )
        except Exception as e:
            logger.warning(
                f"Failed to load design kit configuration: {e} - pipeline will run without it (allowing pipeline to continue)"
            )

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
            for plugin in plugins:
                # Skip steps that have already been completed
                if plugin.step_number < resume_from:
                    continue

                # Skip if result already exists
                if plugin.step_name in results:
                    logger.info(f"Skipping {plugin.step_name} (already completed)")
                    continue

                # Skip transcript_preprocessing_approval for non-transcript content
                if (
                    plugin.step_name == "transcript_preprocessing_approval"
                    and content_type_resume != "transcript"
                ):
                    logger.info(f"Skipping {plugin.step_name} (not transcript content)")
                    continue

                # Skip blog_post_preprocessing_approval for non-blog_post content
                if (
                    plugin.step_name == "blog_post_preprocessing_approval"
                    and content_type_resume != "blog_post"
                ):
                    logger.info(f"Skipping {plugin.step_name} (not blog_post content)")
                    continue

                active_plugins.append(plugin)

            # Execute remaining steps with dynamic step numbers
            for execution_index, plugin in enumerate(active_plugins, start=resume_from):
                logger.info(f"Executing step {execution_index}: {plugin.step_name}")

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
                from marketing_project.models.pipeline_steps import (
                    ContentFormattingResult,
                )

                formatting = ContentFormattingResult(**results["content_formatting"])
                final_content = formatting.formatted_html

            # Get input content from context
            input_content = context_data.get("input_content") or context_data.get(
                "original_content"
            )

            return {
                "pipeline_status": "completed",
                "step_results": results,
                "quality_warnings": quality_warnings,
                "final_content": final_content,
                "input_content": input_content,  # Include input content in result
                "metadata": {
                    "job_id": job_id,
                    "resumed_from_step": last_step_number,
                    "steps_completed": len(results),
                    "execution_time_seconds": execution_time,
                    "total_tokens_used": sum(
                        step.tokens_used for step in self.step_info if step.tokens_used
                    ),
                    "model": self.model,
                    "completed_at": datetime.utcnow().isoformat(),
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

        except Exception as e:
            # Check if this is an approval required exception
            from marketing_project.processors.approval_helper import (
                ApprovalRequiredException,
            )
            from marketing_project.services.job_manager import (
                JobStatus,
                get_job_manager,
            )

            if isinstance(e, ApprovalRequiredException):
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

        Returns:
            Dictionary with step result and execution metadata

        Raises:
            ValueError: If step not found or required context keys are missing
        """
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
