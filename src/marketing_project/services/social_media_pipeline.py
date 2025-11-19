"""
Social Media Pipeline Service using OpenAI Structured Outputs.

This service provides a separate pipeline specifically for generating social media posts
from blog posts. It has 4 steps: SEO Keywords, Marketing Brief, Angle & Hook, Post Generation.
"""

import asyncio
import json
import logging
import time
from datetime import date, datetime
from typing import Any, Dict, Optional

from openai import AsyncOpenAI


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for datetime and other non-serializable objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


from marketing_project.models.pipeline_steps import (
    AngleHookResult,
    PipelineResult,
    PipelineStepInfo,
    SEOKeywordsResult,
    SocialMediaMarketingBriefResult,
    SocialMediaPostResult,
)
from marketing_project.plugins.context_utils import ContextTransformer
from marketing_project.plugins.registry import get_plugin_registry
from marketing_project.plugins.seo_keywords.tasks import SEOKeywordsPlugin
from marketing_project.plugins.social_media_angle_hook.tasks import (
    SocialMediaAngleHookPlugin,
)
from marketing_project.plugins.social_media_marketing_brief.tasks import (
    SocialMediaMarketingBriefPlugin,
)
from marketing_project.plugins.social_media_post_generation.tasks import (
    SocialMediaPostGenerationPlugin,
)
from marketing_project.prompts.prompts import get_template, has_template

logger = logging.getLogger("marketing_project.services.social_media_pipeline")


class SocialMediaPipeline:
    """
    Social media pipeline for generating platform-specific posts.

    This pipeline has 4 steps:
    1. SEO Keywords - Extract keywords from blog post
    2. Marketing Brief - Generate platform-specific marketing brief
    3. Angle & Hook - Create engaging angles and hooks
    4. Post Generation - Generate final platform-specific post
    """

    def __init__(
        self, model: str = "gpt-4o-mini", temperature: float = 0.7, lang: str = "en"
    ):
        """
        Initialize the social media pipeline.

        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
            temperature: Sampling temperature (default: 0.7)
            lang: Language for prompts (default: "en")
        """
        self.client = AsyncOpenAI()
        self.model = model
        self.temperature = temperature
        self.lang = lang
        self.step_info: list[PipelineStepInfo] = []

    def _get_system_instruction(
        self, agent_name: str, context: Optional[Dict] = None
    ) -> str:
        """
        Load system instruction from .j2 template.

        Args:
            agent_name: Name of the agent (e.g., "social_media_marketing_brief")
            context: Optional context variables for Jinja2 template rendering

        Returns:
            Complete system instruction
        """
        template_name = f"{agent_name}_agent_instructions"

        if has_template(self.lang, template_name):
            try:
                template = get_template(self.lang, template_name)
                base_instruction = template.render(**(context or {}))

                # Append quality scoring requirements
                quality_addendum = """

## Quality Metrics for Structured Output

**IMPORTANT**: In addition to all requirements above, you MUST provide quality metrics in your response.

The output schema includes confidence and quality score fields. Provide realistic assessments:
- **confidence_score** (0-1): Your confidence in the output quality and accuracy
- **Additional quality scores**: As defined in the output schema (e.g., engagement_score)

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

        Args:
            step_name: Name of the step
            context: Context variables for Jinja2 template rendering

        Returns:
            Rendered user prompt string
        """
        template_name = f"{step_name}_user_prompt"

        if has_template(self.lang, template_name):
            try:
                template = get_template(self.lang, template_name)
                template_context = ContextTransformer.prepare_template_context(context)

                # Add 'content' as alias for 'input_content' if it exists
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

                return template.render(**template_context)
            except Exception as e:
                logger.warning(
                    f"Failed to load or render template {template_name} for language {self.lang}: {e}. "
                    "Using fallback prompt."
                )
                logger.debug(f"Template rendering error details: {e}", exc_info=True)
        else:
            logger.warning(
                f"Template not found for {template_name} in language {self.lang}, "
                "using fallback prompt"
            )

        # Fallback prompt
        return f"Process the content for {step_name.replace('_', ' ')} step."

    async def _call_function(
        self,
        prompt: str,
        system_instruction: str,
        response_model: type,
        step_name: str,
        step_number: int,
        context: Dict[str, Any],
        job_id: Optional[str] = None,
    ) -> Any:
        """
        Call OpenAI API with structured output.

        Integrates with approval system for human-in-the-loop review when enabled.

        Args:
            prompt: User prompt
            system_instruction: System instruction
            response_model: Pydantic model class for structured output
            step_name: Name of the step
            step_number: Step number
            context: Execution context
            job_id: Optional job ID for tracking

        Returns:
            Instance of response_model

        Raises:
            ApprovalRequiredException: If approval is required (pipeline should stop)
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

        try:
            # Call OpenAI API with structured output
            response = await self.client.beta.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": step_name,
                        "strict": True,
                        "schema": response_model.model_json_schema(),
                    },
                },
                temperature=self.temperature,
            )

            # Parse response
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI API")

            # Parse JSON and validate against model
            result_data = json.loads(content)
            result = response_model(**result_data)

            execution_time = time.time() - start_time
            tokens_used = response.usage.total_tokens if response.usage else None

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
                        result_dict = result.model_dump(mode="json")
                    except (TypeError, ValueError):
                        result_dict = result.model_dump()

                    # Extract confidence score if available
                    confidence = result_dict.get("confidence_score")

                    # Prepare input data for approval context
                    pipeline_content = context.get("input_content") if context else None
                    content_for_approval = (
                        pipeline_content
                        if pipeline_content
                        else {"title": "N/A", "content": "N/A"}
                    )
                    input_data = {
                        "prompt": prompt[:500],  # Truncate for readability
                        "system_instruction": system_instruction[:200],
                        "context_keys": list(context.keys()) if context else [],
                        "original_content": pipeline_content or content_for_approval,
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

                    # Re-raise to be handled by execute_pipeline
                    raise

            # Record step info
            step_info = PipelineStepInfo(
                step_name=step_name,
                step_number=step_number,
                status="success",
                execution_time=execution_time,
                tokens_used=tokens_used,
            )
            self.step_info.append(step_info)

            logger.info(
                f"Step {step_number}: {step_name} completed in {execution_time:.2f}s "
                f"(tokens: {tokens_used})"
            )

            return result

        except ApprovalRequiredException:
            # Re-raise approval exceptions
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            step_info = PipelineStepInfo(
                step_name=step_name,
                step_number=step_number,
                status="failed",
                execution_time=execution_time,
                error_message=str(e),
            )
            self.step_info.append(step_info)
            logger.error(f"Step {step_number}: {step_name} failed: {e}")
            raise

    async def _execute_step_with_plugin(
        self,
        plugin: Any,
        pipeline_context: Dict[str, Any],
        job_id: Optional[str] = None,
    ) -> Any:
        """
        Execute a pipeline step using its plugin.

        Args:
            plugin: Plugin instance
            pipeline_context: Accumulated context from previous steps
            job_id: Optional job ID for tracking

        Returns:
            Step result as Pydantic model
        """
        # Validate context
        if not plugin.validate_context(pipeline_context):
            missing = [
                key
                for key in plugin.get_required_context_keys()
                if key not in pipeline_context
            ]
            raise ValueError(
                f"Missing required context keys for {plugin.step_name}: {missing}"
            )

        # Execute step using plugin
        result = await plugin.execute(pipeline_context, self, job_id)

        # Save step result if job_id is provided
        if job_id:
            try:
                from marketing_project.services.step_result_manager import (
                    get_step_result_manager,
                )

                step_manager = get_step_result_manager()
                await step_manager.save_step_result(
                    job_id=job_id,
                    step_number=plugin.step_number,
                    step_name=plugin.step_name,
                    result_data=result.model_dump(mode="json"),
                    metadata={
                        "execution_time": (
                            self.step_info[-1].execution_time
                            if self.step_info
                            else None
                        ),
                        "status": "success",
                    },
                    execution_context_id=None,
                    root_job_id=None,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to save step result to disk for step {plugin.step_number}: {e}"
                )

        return result

    async def execute_pipeline(
        self,
        content_json: str,
        job_id: Optional[str] = None,
        content_type: str = "blog_post",
        social_media_platform: str = "linkedin",
        email_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete 4-step social media pipeline.

        Args:
            content_json: Input blog post content as JSON string
            job_id: Optional job ID for tracking
            content_type: Type of content being processed (default: blog_post)
            social_media_platform: Platform (linkedin, hackernews, or email)
            email_type: Email type if platform is email (newsletter or promotional)

        Returns:
            Dictionary with complete pipeline results
        """
        pipeline_start = time.time()
        logger.info("=" * 80)
        logger.info(
            f"Starting Social Media Pipeline (job_id: {job_id}, platform: {social_media_platform})"
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
                    if isinstance(content, dict) and "title" in content:
                        job.metadata["title"] = content["title"]
                    await job_manager._save_job(job)
            except Exception as e:
                logger.warning(
                    f"Failed to store input content in job metadata for {job_id}: {e}"
                )

        # Initialize pipeline context
        pipeline_context = {
            "input_content": content,
            "content_type": content_type,
            "social_media_platform": social_media_platform,
            "email_type": email_type,
        }

        results = {}
        quality_warnings = []

        try:
            # Define the 4 steps in order
            plugins = [
                SEOKeywordsPlugin(),  # Step 1
                SocialMediaMarketingBriefPlugin(),  # Step 2
                SocialMediaAngleHookPlugin(),  # Step 3
                SocialMediaPostGenerationPlugin(),  # Step 4
            ]

            # Execute each step
            for plugin in plugins:
                logger.info(f"Executing step {plugin.step_number}: {plugin.step_name}")

                # Check for approval requirements
                try:
                    step_result = await self._execute_step_with_plugin(
                        plugin=plugin,
                        pipeline_context=pipeline_context,
                        job_id=job_id,
                    )
                except Exception as e:
                    # Check if this is an approval required exception
                    from marketing_project.processors.approval_helper import (
                        ApprovalRequiredException,
                    )

                    if isinstance(e, ApprovalRequiredException):
                        # Re-raise to handle in outer try/except
                        raise

                    # Other exceptions - step failed
                    raise

                # Store result
                try:
                    results[plugin.step_name] = step_result.model_dump(mode="json")
                except (TypeError, ValueError):
                    results[plugin.step_name] = step_result.model_dump()
                pipeline_context[plugin.step_name] = results[plugin.step_name]

            # Compile final result
            pipeline_end = time.time()
            execution_time = pipeline_end - pipeline_start

            logger.info("=" * 80)
            logger.info(
                f"Social Media Pipeline completed successfully in {execution_time:.2f}s"
            )
            logger.info("=" * 80)

            # Calculate total tokens used
            total_tokens = sum(
                step.tokens_used for step in self.step_info if step.tokens_used
            )

            # Get final post content
            final_content = None
            if "social_media_post_generation" in results:
                post_result = SocialMediaPostResult(
                    **results["social_media_post_generation"]
                )
                final_content = post_result.content

            return {
                "pipeline_status": (
                    "completed_with_warnings" if quality_warnings else "completed"
                ),
                "step_results": results,
                "quality_warnings": quality_warnings,
                "final_content": final_content,
                "input_content": content,
                "metadata": {
                    "job_id": job_id,
                    "content_id": content.get("id"),
                    "content_type": content_type,
                    "platform": social_media_platform,
                    "email_type": email_type,
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
                # Approval required - pipeline stops
                pipeline_end = time.time()
                execution_time = pipeline_end - pipeline_start

                logger.info(
                    f"[APPROVAL] Social Media Pipeline stopped for approval at step {e.step_number} ({e.step_name}) "
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

                # Return partial results
                return {
                    "pipeline_status": "waiting_for_approval",
                    "step_results": results,
                    "quality_warnings": quality_warnings,
                    "final_content": None,
                    "metadata": {
                        "job_id": e.job_id,
                        "content_id": content.get("id"),
                        "content_type": content_type,
                        "platform": social_media_platform,
                        "email_type": email_type,
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

            logger.error(
                f"Social Media Pipeline failed after {execution_time:.2f}s: {e}"
            )

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
                    "platform": social_media_platform,
                    "email_type": email_type,
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
