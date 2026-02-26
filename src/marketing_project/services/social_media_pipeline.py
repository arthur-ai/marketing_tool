"""
Social Media Pipeline Service using OpenAI Structured Outputs.

This service provides a separate pipeline specifically for generating social media posts
from blog posts. It has 4 steps: SEO Keywords, Marketing Brief, Angle & Hook, Post Generation.
"""

import asyncio
import copy
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from openai import AsyncOpenAI

# OpenTelemetry imports for tracing
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    _tracing_available = True
except ImportError:
    _tracing_available = False
    logger.debug("OpenTelemetry not available, tracing disabled")


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for datetime and other non-serializable objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


from marketing_project.models.pipeline_steps import (
    AngleHookResult,
    BlogPostPreprocessingApprovalResult,
    PipelineConfig,
    PipelineResult,
    PipelineStepConfig,
    PipelineStepInfo,
    SEOKeywordsResult,
    SocialMediaMarketingBriefResult,
    SocialMediaPostResult,
)
from marketing_project.plugins.blog_post_preprocessing_approval.tasks import (
    BlogPostPreprocessingApprovalPlugin,
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
from marketing_project.services.function_pipeline.tracing import (
    ensure_span_has_minimum_metadata,
    extract_model_info,
    set_llm_invocation_parameters,
    set_llm_messages,
    set_llm_response_format,
    set_llm_system_and_provider,
    set_llm_token_counts,
    set_span_input,
    set_span_kind,
    set_span_output,
)

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
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        lang: str = "en",
        pipeline_config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize the social media pipeline.

        Args:
            model: OpenAI model to use (deprecated: use pipeline_config instead)
            temperature: Sampling temperature (deprecated: use pipeline_config instead)
            lang: Language for prompts (default: "en")
            pipeline_config: Optional PipelineConfig for per-step model configuration
        """
        self.client = AsyncOpenAI()
        self.lang = lang
        self.step_info: list[PipelineStepInfo] = []
        self._platform_config: Optional[Dict[str, Any]] = None

        # Support both old-style (model, temperature) and new-style (pipeline_config)
        if pipeline_config:
            self.pipeline_config = pipeline_config
            self.model = pipeline_config.default_model
            self.temperature = pipeline_config.default_temperature
        else:
            # Use provided values or defaults (but note: models should be configured in settings)
            self.model = model or "gpt-5.1"
            self.temperature = temperature if temperature is not None else 0.7
            self.pipeline_config = PipelineConfig(
                default_model=self.model,
                default_temperature=self.temperature,
                default_max_retries=2,
                step_configs={},
            )

    def _load_platform_config(self) -> Dict[str, Any]:
        """
        Load platform configuration from YAML file.

        Returns:
            Platform configuration dictionary
        """
        if self._platform_config is not None:
            return self._platform_config

        try:
            # Use importlib to find the package directory reliably
            import importlib.util

            import marketing_project

            # Get the package directory
            package_dir = Path(marketing_project.__file__).parent
            config_file = package_dir / "config" / "platform_config.yml"

            if not config_file.exists():
                logger.warning(
                    f"Platform config file not found: {config_file}. Using defaults."
                )
                self._platform_config = {}
                return self._platform_config

            with open(config_file, "r") as f:
                self._platform_config = yaml.safe_load(f) or {}

            logger.debug(f"Loaded platform config from {config_file}")
            return self._platform_config
        except Exception as e:
            logger.warning(f"Failed to load platform config: {e}. Using defaults.")
            self._platform_config = {}
            return self._platform_config

    def _get_platform_config(self, platform: str) -> Dict[str, Any]:
        """
        Get configuration for a specific platform.

        Args:
            platform: Platform name (linkedin, hackernews, email)

        Returns:
            Platform-specific configuration
        """
        config = self._load_platform_config()
        platforms = config.get("platforms", {})
        return platforms.get(platform, {})

    def _validate_content_length(
        self, content: str, platform: str, email_type: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate content length against platform limits.

        Args:
            content: Content to validate
            platform: Platform name
            email_type: Email type if platform is email

        Returns:
            Tuple of (is_valid, warning_message)
        """
        platform_config = self._get_platform_config(platform)
        if not platform_config:
            return True, None

        # Handle email type-specific limits
        if platform == "email" and email_type:
            email_types = platform_config.get("types", {})
            type_config = email_types.get(email_type, {})
            limit = type_config.get("character_limit") or platform_config.get(
                "character_limit", 5000
            )
            warning_threshold = type_config.get("character_limit_warning") or limit
        else:
            limit = platform_config.get("character_limit", 3000)
            warning_threshold = platform_config.get("character_limit_warning", limit)

        content_length = len(content)
        if content_length > limit:
            return (
                False,
                f"Content exceeds {platform} limit of {limit} characters ({content_length} chars)",
            )
        elif content_length > warning_threshold:
            return (
                True,
                f"Content is approaching {platform} limit ({content_length}/{limit} chars)",
            )

        return True, None

    def _assess_platform_quality(
        self, result: SocialMediaPostResult, platform: str
    ) -> Dict[str, Optional[float]]:
        """
        Assess platform-specific quality scores from result.

        Args:
            result: SocialMediaPostResult with quality scores
            platform: Platform name

        Returns:
            Dictionary with platform-specific quality scores
        """
        scores = {}
        platform_config = self._get_platform_config(platform)

        if platform == "linkedin":
            scores["linkedin_score"] = result.linkedin_score
        elif platform == "hackernews":
            scores["hackernews_score"] = result.hackernews_score
        elif platform == "email":
            scores["email_score"] = result.email_score

        # Log quality assessment
        if platform_config:
            quality_metrics = platform_config.get("quality_metrics", {})
            if quality_metrics:
                logger.debug(
                    f"Platform quality metrics weights for {platform}: {quality_metrics}"
                )

        return scores

    async def _generate_variations(
        self,
        pipeline_context: Dict[str, Any],
        base_result: SocialMediaPostResult,
        num_variations: int,
        job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate variations of a post with different approaches/temperatures.

        Args:
            pipeline_context: Pipeline context with angle/hook and brief
            base_result: Base post result to generate variations from
            num_variations: Number of variations to generate (1-2)
            job_id: Optional job ID for tracking

        Returns:
            List of variation dictionaries
        """
        variations = []
        platform = pipeline_context.get("social_media_platform", "linkedin")

        # Different temperature settings for variations
        variation_temperatures = [
            0.8,
            0.9,
        ]  # Slightly higher temperatures for more creativity

        for i in range(num_variations):
            logger.info(f"Generating variation {i + 1}/{num_variations}")

            # Use different temperature for variation
            original_temperature = self.temperature
            self.temperature = variation_temperatures[i % len(variation_temperatures)]

            try:
                # Re-execute post generation with different temperature
                post_plugin = SocialMediaPostGenerationPlugin()
                variation_result = await post_plugin.execute(
                    pipeline_context, self, job_id
                )

                variation_dict = {
                    "variation_id": f"variation_{i + 1}",
                    "content": variation_result.content,
                    "subject_line": variation_result.subject_line,
                    "hashtags": variation_result.hashtags,
                    "call_to_action": variation_result.call_to_action,
                    "confidence_score": variation_result.confidence_score,
                    "engagement_score": variation_result.engagement_score,
                    "temperature_used": self.temperature,
                }

                # Add platform-specific scores
                if platform == "linkedin" and variation_result.linkedin_score:
                    variation_dict["linkedin_score"] = variation_result.linkedin_score
                elif platform == "hackernews" and variation_result.hackernews_score:
                    variation_dict["hackernews_score"] = (
                        variation_result.hackernews_score
                    )
                elif platform == "email" and variation_result.email_score:
                    variation_dict["email_score"] = variation_result.email_score

                variations.append(variation_dict)
            except Exception as e:
                logger.warning(f"Failed to generate variation {i + 1}: {e}")
            finally:
                # Restore original temperature
                self.temperature = original_temperature

        return variations

    def _get_system_instruction(
        self, agent_name: str, context: Optional[Dict] = None
    ) -> str:
        """
        Load system instruction from .j2 template.
        Supports platform-specific templates for social media agents.

        Args:
            agent_name: Name of the agent (e.g., "social_media_marketing_brief")
            context: Optional context variables for Jinja2 template rendering

        Returns:
            Complete system instruction
        """
        # Check for platform-specific template first
        platform = None
        if context:
            platform = context.get("social_media_platform") or context.get("platform")

        # Try platform-specific template if platform is specified and agent is social media related
        template_name = f"{agent_name}_agent_instructions"
        if platform and agent_name.startswith("social_media"):
            platform_template_name = f"{agent_name}_{platform}_agent_instructions"
            if has_template(self.lang, platform_template_name):
                template_name = platform_template_name
                logger.debug(
                    f"Using platform-specific template: {platform_template_name}"
                )

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

    def _get_step_model(self, step_name: str) -> str:
        """
        Get model for a specific step from pipeline config.

        Args:
            step_name: Name of the step

        Returns:
            Model name to use for this step
        """
        return self.pipeline_config.get_step_model(step_name)

    def _get_step_temperature(self, step_name: str) -> float:
        """
        Get temperature for a specific step from pipeline config.

        Args:
            step_name: Name of the step

        Returns:
            Temperature to use for this step
        """
        return self.pipeline_config.get_step_temperature(step_name)

    def _fix_schema_additional_properties(
        self, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively fix JSON schema to add additionalProperties: false to all object types
        and ensure required array includes all non-optional fields.

        OpenAI's structured outputs require that all object types in the schema
        explicitly set additionalProperties: false for strict mode, and that the
        required array includes all properties that don't have default values.

        Args:
            schema: JSON schema dictionary

        Returns:
            Fixed schema with additionalProperties set to false and required array fixed
        """
        if not isinstance(schema, dict):
            return schema

        # Create a copy to avoid modifying the original
        fixed_schema = copy.deepcopy(schema)

        # If this is an object type, ensure additionalProperties is set
        if fixed_schema.get("type") == "object":
            if "additionalProperties" not in fixed_schema:
                fixed_schema["additionalProperties"] = False

        # Recursively fix properties
        if "properties" in fixed_schema:
            for prop_name, prop_schema in fixed_schema["properties"].items():
                fixed_schema["properties"][prop_name] = (
                    self._fix_schema_additional_properties(prop_schema)
                )

        # Fix required array: ensure all non-optional properties are included
        if "properties" in fixed_schema and fixed_schema.get("type") == "object":
            properties = fixed_schema["properties"]
            required = fixed_schema.get("required", [])

            # Find all properties that should be required (not optional)
            for prop_name, prop_schema in properties.items():
                # Check if property is optional (has default, is in anyOf with null, etc.)
                is_optional = self._is_property_optional(prop_schema)

                # If not optional and not in required, add it
                if not is_optional and prop_name not in required:
                    required.append(prop_name)

            # Update required array
            if required:
                fixed_schema["required"] = sorted(
                    list(set(required))
                )  # Remove duplicates and sort
            elif "required" in fixed_schema:
                # If we had a required array but it's now empty, keep it as empty list
                fixed_schema["required"] = []

        # Fix anyOf schemas (for Optional types)
        if "anyOf" in fixed_schema:
            fixed_schema["anyOf"] = [
                self._fix_schema_additional_properties(sub_schema)
                for sub_schema in fixed_schema["anyOf"]
            ]

        # Fix oneOf schemas
        if "oneOf" in fixed_schema:
            fixed_schema["oneOf"] = [
                self._fix_schema_additional_properties(sub_schema)
                for sub_schema in fixed_schema["oneOf"]
            ]

        # Fix allOf schemas
        if "allOf" in fixed_schema:
            fixed_schema["allOf"] = [
                self._fix_schema_additional_properties(sub_schema)
                for sub_schema in fixed_schema["allOf"]
            ]

        # Fix items in arrays
        if "items" in fixed_schema:
            fixed_schema["items"] = self._fix_schema_additional_properties(
                fixed_schema["items"]
            )

        # Fix definitions/defs (for referenced schemas)
        if "definitions" in fixed_schema:
            for def_name, def_schema in fixed_schema["definitions"].items():
                fixed_schema["definitions"][def_name] = (
                    self._fix_schema_additional_properties(def_schema)
                )

        if "$defs" in fixed_schema:
            for def_name, def_schema in fixed_schema["$defs"].items():
                fixed_schema["$defs"][def_name] = (
                    self._fix_schema_additional_properties(def_schema)
                )

        return fixed_schema

    def _is_property_optional(self, prop_schema: Dict[str, Any]) -> bool:
        """
        Check if a property schema represents an optional field.

        Args:
            prop_schema: Property schema dictionary

        Returns:
            True if the property is optional, False otherwise
        """
        # If it has a default value, it's optional
        if "default" in prop_schema:
            return True

        # If it's wrapped in anyOf with null type, it's optional
        if "anyOf" in prop_schema:
            for sub_schema in prop_schema["anyOf"]:
                if isinstance(sub_schema, dict) and sub_schema.get("type") == "null":
                    return True

        # If it's wrapped in oneOf with null type, it's optional
        if "oneOf" in prop_schema:
            for sub_schema in prop_schema["oneOf"]:
                if isinstance(sub_schema, dict) and sub_schema.get("type") == "null":
                    return True

        # If type is explicitly null, it's optional
        if prop_schema.get("type") == "null":
            return True

        return False

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
        Uses model and temperature from pipeline_config for this step.

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
        # Import at method level so it's available in exception handlers
        from marketing_project.processors.approval_helper import (
            ApprovalRequiredException,
        )

        start_time = time.time()

        # Get step-specific model and temperature from pipeline config
        step_model = self._get_step_model(step_name)
        step_temperature = self._get_step_temperature(step_name)

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        # Add context from previous steps if available
        if context:
            context_msg = f"\n\n### Context from Previous Steps:\n```json\n{json.dumps(context, indent=2, default=_json_serializer)}\n```"
            messages[-1]["content"] += context_msg

        # Add trend context for post generation step
        if step_name == "social_media_post_generation":
            try:
                from marketing_project.services.trend_integration import (
                    get_trend_service,
                )

                platform = context.get("social_media_platform", "linkedin")
                trend_service = get_trend_service()
                trending_hashtags = await trend_service.get_trending_hashtags(
                    platform, limit=5
                )
                if trending_hashtags:
                    trend_context = trend_service.get_trend_context_for_prompt(
                        platform, trending_hashtags
                    )
                    messages[-1]["content"] += trend_context
            except Exception as e:
                logger.debug(f"Failed to add trend context to prompt: {e}")

        try:
            # Generate JSON schema and fix additionalProperties for OpenAI compatibility
            schema = response_model.model_json_schema()
            schema = self._fix_schema_additional_properties(schema)

            # Call OpenAI API with structured output using step-specific model
            logger.debug(
                f"Calling {step_model} for step {step_name} with temperature {step_temperature}"
            )

            # Initialize parsed_result for all code paths
            parsed_result = None
            response = None

            # Create OpenTelemetry span for this LLM call
            if _tracing_available:
                try:
                    tracer = trace.get_tracer(__name__)
                    with tracer.start_as_current_span(
                        f"social_media_pipeline.{step_name}",
                        kind=trace.SpanKind.CLIENT,
                    ) as span:
                        # Set OpenInference span kind
                        set_span_kind(span, "LLM")

                        # Set input attributes (full context dict) - always set, never blank
                        set_span_input(span, context if context else {})
                        if context:
                            content_type = context.get("content_type")
                            if content_type:
                                span.set_attribute("content_type", content_type)
                            platform = context.get("social_media_platform")
                            if platform:
                                span.set_attribute("platform", platform)
                                span.set_attribute("social_media_platform", platform)

                        # Extract and set model performance metrics
                        model_info = extract_model_info(step_model)
                        for key, value in model_info.items():
                            span.set_attribute(f"llm.{key}", value)

                        # Set LLM input messages (always set, never blank)
                        set_llm_messages(span, messages if messages else [])

                        # Ensure minimum metadata
                        ensure_span_has_minimum_metadata(
                            span, f"social_media_pipeline.{step_name}", "llm_call"
                        )

                        # Set span attributes
                        span.set_attribute("step_name", step_name)
                        span.set_attribute("step_number", step_number)
                        span.set_attribute("model", step_model)
                        span.set_attribute("llm.model_name", step_model)
                        set_llm_system_and_provider(
                            span, system="openai", provider="openai"
                        )
                        span.set_attribute("temperature", step_temperature)
                        if job_id:
                            span.set_attribute("job_id", job_id)

                        # Set LLM response format (JSON schema)
                        response_format = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": step_name,
                                "strict": True,
                                "schema": schema,
                            },
                        }
                        set_llm_response_format(span, response_format)

                        # Set LLM invocation parameters
                        invocation_params = {
                            "temperature": step_temperature,
                            "model": step_model,
                        }
                        set_llm_invocation_parameters(span, invocation_params)

                        response = await self.client.chat.completions.create(
                            model=step_model,
                            messages=messages,
                            response_format=response_format,
                            temperature=step_temperature,
                        )

                        # Parse response to get result for output
                        content = response.choices[0].message.content
                        result_data = None
                        result = None
                        if content:
                            try:
                                result_data = json.loads(content)
                                result = response_model(**result_data)
                            except Exception as parse_error:
                                logger.debug(
                                    f"Failed to parse response in span: {parse_error}"
                                )

                        # Update span with response metadata
                        try:
                            # Set output attributes (parsed result or raw content)
                            if result:
                                set_span_output(
                                    span,
                                    (
                                        result.model_dump()
                                        if hasattr(result, "model_dump")
                                        else result_data
                                    ),
                                )
                            elif content:
                                set_span_output(
                                    span, content, output_mime_type="application/json"
                                )

                            # Set LLM output messages
                            output_messages = []
                            if response.choices and len(response.choices) > 0:
                                choice = response.choices[0]
                                if choice.message:
                                    output_messages.append(
                                        {
                                            "role": choice.message.role or "assistant",
                                            "content": choice.message.content,
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
                                span.set_attribute(
                                    "input_tokens", response.usage.prompt_tokens or 0
                                )
                                span.set_attribute(
                                    "output_tokens",
                                    response.usage.completion_tokens or 0,
                                )
                                span.set_attribute(
                                    "total_tokens", response.usage.total_tokens or 0
                                )
                            span.set_status(Status(StatusCode.OK))
                        except Exception as e:
                            logger.debug(f"Failed to update span with response: {e}")

                        # Store parsed result for return after span closes
                        parsed_result = result
                except Exception as span_error:
                    logger.debug(f"Failed to create span: {span_error}")
                    # Fallback to non-instrumented call - response will be set here
                    response = await self.client.chat.completions.create(
                        model=step_model,
                        messages=messages,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": step_name,
                                "strict": True,
                                "schema": schema,
                            },
                        },
                        temperature=step_temperature,
                    )
            else:
                # No tracing available, make direct call - response will be set here
                response = await self.client.chat.completions.create(
                    model=step_model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": step_name,
                            "strict": True,
                            "schema": schema,
                        },
                    },
                    temperature=step_temperature,
                )

            # Parse response (only if not already parsed in span)
            if parsed_result is None and response:
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from OpenAI API")

                # Parse JSON and validate against model
                result_data = json.loads(content)
                parsed_result = response_model(**result_data)

            # Return parsed result
            if parsed_result:
                return parsed_result
            else:
                raise ValueError("Failed to parse response from OpenAI API")

            execution_time = time.time() - start_time
            tokens_used = response.usage.total_tokens if response.usage else None

            # ========================================
            # Human-in-the-Loop Approval Integration
            # ========================================
            if job_id:
                try:
                    from marketing_project.processors.approval_helper import (
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
        execution_step_number: Optional[int] = None,
    ) -> Any:
        """
        Execute a pipeline step using its plugin.

        Args:
            plugin: Plugin instance
            pipeline_context: Accumulated context from previous steps
            job_id: Optional job ID for tracking
            execution_step_number: Optional execution step number (overrides plugin.step_number for saving)

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

        # Check if result is ApprovalRequiredSentinel (approval required, don't save step result)
        from marketing_project.processors.approval_helper import (
            ApprovalRequiredSentinel,
        )

        if isinstance(result, ApprovalRequiredSentinel):
            # Return early - don't save step result for approval sentinel
            return result

        # Save step result if job_id is provided
        if job_id:
            try:
                from marketing_project.services.step_result_manager import (
                    get_step_result_manager,
                )

                # Capture input snapshot and context keys used
                input_snapshot = None
                context_keys_used = []
                if pipeline_context:
                    input_snapshot = copy.deepcopy(pipeline_context)
                    context_keys_used = list(pipeline_context.keys())

                # Use execution_step_number if provided, otherwise use plugin.step_number
                step_number_for_saving = (
                    execution_step_number
                    if execution_step_number is not None
                    else plugin.step_number
                )

                step_manager = get_step_result_manager()
                await step_manager.save_step_result(
                    job_id=job_id,
                    step_number=step_number_for_saving,
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
                    input_snapshot=input_snapshot,
                    context_keys_used=context_keys_used,
                )
            except Exception as e:
                # Use execution_step_number if provided, otherwise use plugin.step_number for error message
                step_number_for_error = (
                    execution_step_number
                    if execution_step_number is not None
                    else plugin.step_number
                )
                logger.warning(
                    f"Failed to save step result to disk for step {step_number_for_error}: {e}"
                )

        return result

    async def execute_pipeline(
        self,
        content_json: str,
        job_id: Optional[str] = None,
        content_type: str = "blog_post",
        social_media_platform: str = "linkedin",
        email_type: Optional[str] = None,
        generate_variations: int = 1,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete 4-step social media pipeline.

        Args:
            content_json: Input blog post content as JSON string
            job_id: Optional job ID for tracking
            content_type: Type of content being processed (default: blog_post)
            social_media_platform: Platform (linkedin, hackernews, or email)
            email_type: Email type if platform is email (newsletter or promotional)
            generate_variations: Number of variations to generate (1-3, default: 1)
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
            f"Starting Social Media Pipeline (job_id: {job_id}, platform: {social_media_platform})"
        )
        logger.info("=" * 80)

        # Reset step info
        self.step_info = []

        # Update job progress to indicate pipeline started
        if job_id:
            try:
                from marketing_project.services.job_manager import get_job_manager

                job_manager = get_job_manager()
                await job_manager.update_job_progress(
                    job_id, 5, "Starting social media pipeline"
                )
            except Exception as e:
                logger.warning(f"Failed to update job progress: {e}")

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

        # Initialize context registry for this job if job_id is provided
        context_registry = None
        if job_id:
            try:
                from marketing_project.services.context_registry import ContextRegistry

                context_registry = ContextRegistry()
                logger.debug(f"Initialized context registry for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize context registry: {e}")

        results = {}
        quality_warnings = []

        try:
            # Define the steps in order
            # For blog_post content, start with blog_post_preprocessing_approval (step 1)
            plugins = []
            if content_type == "blog_post":
                plugins.append(BlogPostPreprocessingApprovalPlugin())  # Step 1

            # Add the standard social media pipeline steps
            plugins.extend(
                [
                    SEOKeywordsPlugin(),  # Step 2 (or 1 if not blog_post)
                    SocialMediaMarketingBriefPlugin(),  # Step 3 (or 2 if not blog_post)
                    SocialMediaAngleHookPlugin(),  # Step 4 (or 3 if not blog_post)
                    SocialMediaPostGenerationPlugin(),  # Step 5 (or 4 if not blog_post)
                ]
            )

            # Execute each step
            for step_index, plugin in enumerate(plugins, start=1):
                logger.info(
                    f"Executing step {step_index}/{len(plugins)}: {plugin.step_name}"
                )

                # Set plugin model config from pipeline config
                step_config = self.pipeline_config.get_step_config(plugin.step_name)
                plugin.model_config = PipelineStepConfig(
                    step_name=plugin.step_name,
                    model=step_config.model,
                    temperature=step_config.temperature,
                    max_retries=step_config.max_retries,
                )

                # Update job progress
                if job_id:
                    try:
                        from marketing_project.services.job_manager import (
                            get_job_manager,
                        )

                        job_manager = get_job_manager()
                        # Calculate progress: each step is equal percentage (100% / total steps)
                        step_progress = int((step_index / len(plugins)) * 100)
                        # Format step name for display (keep the actual step_name for mapping)
                        step_display_name = plugin.step_name.replace("_", " ").replace(
                            "social media ", ""
                        )
                        await job_manager.update_job_progress(
                            job_id,
                            step_progress,
                            f"Step {step_index}/{len(plugins)}: {step_display_name}",
                        )
                        # Also update current_step with the actual step_name for frontend mapping
                        job = await job_manager.get_job(job_id)
                        if job:
                            job.current_step = plugin.step_name
                            await job_manager._save_job(job)
                    except Exception as e:
                        logger.warning(f"Failed to update job progress: {e}")

                # Build optimized context for this step (reduce token usage)
                try:
                    from marketing_project.services.context_summarizer import (
                        ContextSummarizer,
                    )

                    optimized_context = ContextSummarizer.build_optimized_context(
                        full_context=pipeline_context,
                        step_name=plugin.step_name,
                        context_registry=context_registry,
                        job_id=job_id,
                    )
                    logger.debug(
                        f"Optimized context for {plugin.step_name}: {len(optimized_context)} keys (from {len(pipeline_context)} keys)"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to optimize context for {plugin.step_name}: {e}. Using full context."
                    )
                    optimized_context = pipeline_context

                # Check for approval requirements
                try:
                    step_result = await self._execute_step_with_plugin(
                        plugin=plugin,
                        pipeline_context=optimized_context,
                        job_id=job_id,
                        execution_step_number=step_index,
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

                # Check if result is ApprovalRequiredSentinel (approval required, stop execution)
                from marketing_project.processors.approval_helper import (
                    ApprovalRequiredSentinel,
                )

                if isinstance(step_result, ApprovalRequiredSentinel):
                    # Approval required - stop pipeline execution
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
                        "platform": social_media_platform,
                    }

                # Store result - use model_dump(mode='json') to ensure datetime objects are serialized
                try:
                    results[plugin.step_name] = step_result.model_dump(mode="json")
                except (TypeError, ValueError):
                    # Fallback to regular model_dump if mode='json' fails
                    results[plugin.step_name] = step_result.model_dump()

                # Add platform field to social_media_post_generation result
                if plugin.step_name == "social_media_post_generation":
                    results[plugin.step_name]["platform"] = social_media_platform

                pipeline_context[plugin.step_name] = results[plugin.step_name]

                # Register step output in context registry for efficient context passing
                if job_id:
                    try:
                        from marketing_project.services.context_registry import (
                            ContextRegistry,
                        )

                        context_registry = ContextRegistry()
                        await context_registry.register_step_output(
                            job_id=job_id,
                            step_name=plugin.step_name,
                            step_number=plugin.step_number,
                            output_data=results[plugin.step_name],
                        )
                        logger.debug(
                            f"Registered step output for {plugin.step_name} in context registry"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to register step output in context registry: {e}"
                        )

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

            # Validate generate_variations parameter
            generate_variations = max(
                1, min(3, generate_variations)
            )  # Clamp between 1 and 3
            pipeline_context["generate_variations"] = generate_variations

            # Get final post content and validate
            final_content = None
            platform_quality_scores = {}
            variations_list = []
            if "social_media_post_generation" in results:
                post_result = SocialMediaPostResult(
                    **results["social_media_post_generation"]
                )
                final_content = post_result.content

                # Generate variations if requested
                if generate_variations > 1:
                    logger.info(
                        f"Generating {generate_variations} variations of the post"
                    )
                    variations_list = await self._generate_variations(
                        pipeline_context=pipeline_context,
                        base_result=post_result,
                        num_variations=generate_variations
                        - 1,  # -1 because we already have the base
                        job_id=job_id,
                    )
                    # Add base result as first variation
                    variations_list.insert(
                        0,
                        {
                            "variation_id": "base",
                            "content": post_result.content,
                            "subject_line": post_result.subject_line,
                            "hashtags": post_result.hashtags,
                            "call_to_action": post_result.call_to_action,
                            "confidence_score": post_result.confidence_score,
                            "engagement_score": post_result.engagement_score,
                        },
                    )

                # Validate content length against platform limits
                is_valid, warning = self._validate_content_length(
                    final_content, social_media_platform, email_type
                )
                if not is_valid:
                    quality_warnings.append(warning)
                    logger.warning(f"Content validation failed: {warning}")
                elif warning:
                    quality_warnings.append(warning)
                    logger.info(f"Content validation warning: {warning}")

                # Assess platform-specific quality scores
                platform_quality_scores = self._assess_platform_quality(
                    post_result, social_media_platform
                )
                logger.info(
                    f"Platform quality scores for {social_media_platform}: {platform_quality_scores}"
                )

            return {
                "pipeline_status": (
                    "completed_with_warnings" if quality_warnings else "completed"
                ),
                "step_results": results,
                "quality_warnings": quality_warnings,
                "final_content": final_content,
                "variations": variations_list if variations_list else None,
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
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "platform_quality_scores": platform_quality_scores,
                    "variations_generated": (
                        len(variations_list) if variations_list else 0
                    ),
                    "context_optimization": {
                        "enabled": context_registry is not None,
                        "full_context_stored": context_registry is not None,
                    },
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

    async def execute_multi_platform_pipeline(
        self,
        content_json: str,
        platforms: list[str],
        job_id: Optional[str] = None,
        content_type: str = "blog_post",
        email_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute pipeline for multiple platforms in parallel.
        Shares SEO Keywords and Marketing Brief across platforms.

        Args:
            content_json: Input blog post content as JSON string
            platforms: List of platforms (linkedin, hackernews, email)
            job_id: Optional job ID for tracking
            content_type: Type of content being processed (default: blog_post)
            email_type: Email type if email is in platforms (newsletter or promotional)

        Returns:
            Dictionary with results for each platform
        """
        pipeline_start = time.time()
        logger.info("=" * 80)
        logger.info(
            f"Starting Multi-Platform Social Media Pipeline (job_id: {job_id}, platforms: {platforms})"
        )
        logger.info("=" * 80)

        # Validate platforms
        valid_platforms = {"linkedin", "hackernews", "email"}
        invalid_platforms = [p for p in platforms if p not in valid_platforms]
        if invalid_platforms:
            raise ValueError(f"Invalid platforms: {invalid_platforms}")

        # Check max platforms limit
        platform_config = self._load_platform_config()
        max_platforms = platform_config.get("max_platforms_per_batch", 5)
        if len(platforms) > max_platforms:
            raise ValueError(
                f"Maximum {max_platforms} platforms allowed per batch, got {len(platforms)}"
            )

        # Parse input content
        try:
            content = json.loads(content_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            raise ValueError(f"Invalid JSON input: {e}")

        # Execute blog_post_preprocessing_approval step first if content_type is blog_post
        if content_type == "blog_post":
            logger.info("Executing blog post preprocessing approval step...")
            preprocessing_plugin = BlogPostPreprocessingApprovalPlugin()
            preprocessing_context = {
                "input_content": content,
                "content_type": content_type,
            }
            preprocessing_result = await preprocessing_plugin.execute(
                preprocessing_context, self, job_id
            )

            # Check if result is ApprovalRequiredSentinel
            from marketing_project.processors.approval_helper import (
                ApprovalRequiredSentinel,
            )

            if isinstance(preprocessing_result, ApprovalRequiredSentinel):
                # Approval required - stop pipeline execution
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
                    "approval_id": preprocessing_result.approval_result.approval_id,
                    "step_name": preprocessing_result.approval_result.step_name,
                    "step_number": preprocessing_result.approval_result.step_number,
                }

            # Update content with any extracted data from preprocessing
            if isinstance(preprocessing_result, BlogPostPreprocessingApprovalResult):
                # The preprocessing plugin already updated input_content in context
                # but we need to update our local content dict
                if preprocessing_result.author and not content.get("author"):
                    content["author"] = preprocessing_result.author
                if preprocessing_result.category and not content.get("category"):
                    content["category"] = preprocessing_result.category
                if preprocessing_result.tags and (
                    not content.get("tags") or len(content.get("tags", [])) == 0
                ):
                    content["tags"] = preprocessing_result.tags
                if (
                    preprocessing_result.word_count is not None
                    and content.get("word_count") is None
                ):
                    content["word_count"] = preprocessing_result.word_count
                if (
                    preprocessing_result.reading_time is not None
                    and content.get("reading_time") is None
                ):
                    content["reading_time"] = preprocessing_result.reading_time

        # Execute SEO Keywords step once (shared across platforms)
        logger.info("Executing shared SEO Keywords step...")
        seo_plugin = SEOKeywordsPlugin()
        seo_context = {
            "input_content": content,
            "content_type": content_type,
        }
        seo_result = await seo_plugin.execute(seo_context, self, job_id)

        # Check if result is ApprovalRequiredSentinel
        from marketing_project.processors.approval_helper import (
            ApprovalRequiredSentinel,
        )

        if isinstance(seo_result, ApprovalRequiredSentinel):
            # Approval required - stop pipeline execution
            from marketing_project.services.job_manager import (
                JobStatus,
                get_job_manager,
            )

            job_manager = get_job_manager()
            await job_manager.update_job_status(job_id, JobStatus.WAITING_FOR_APPROVAL)
            return {
                "status": "waiting_for_approval",
                "approval_id": seo_result.approval_result.approval_id,
                "step_name": seo_result.approval_result.step_name,
                "step_number": seo_result.approval_result.step_number,
            }

        seo_result_dict = (
            seo_result.model_dump(mode="json")
            if hasattr(seo_result, "model_dump")
            else seo_result.model_dump()
        )

        # Execute Marketing Brief step once (shared across platforms)
        logger.info("Executing shared Marketing Brief step...")
        brief_plugin = SocialMediaMarketingBriefPlugin()
        brief_context = {
            "input_content": content,
            "content_type": content_type,
            "seo_keywords": seo_result_dict,
            "social_media_platform": platforms[0],  # Use first platform for brief
        }
        brief_result = await brief_plugin.execute(brief_context, self, job_id)

        # Check if result is ApprovalRequiredSentinel
        if isinstance(brief_result, ApprovalRequiredSentinel):
            # Approval required - stop pipeline execution
            from marketing_project.services.job_manager import (
                JobStatus,
                get_job_manager,
            )

            job_manager = get_job_manager()
            await job_manager.update_job_status(job_id, JobStatus.WAITING_FOR_APPROVAL)
            return {
                "status": "waiting_for_approval",
                "approval_id": brief_result.approval_result.approval_id,
                "step_name": brief_result.approval_result.step_name,
                "step_number": brief_result.approval_result.step_number,
            }

        brief_result_dict = (
            brief_result.model_dump(mode="json")
            if hasattr(brief_result, "model_dump")
            else brief_result.model_dump()
        )

        # Execute platform-specific steps in parallel
        async def execute_platform_pipeline(
            platform: str,
        ) -> tuple[str, Dict[str, Any]]:
            """Execute pipeline for a single platform."""
            logger.info(f"Executing pipeline for platform: {platform}")

            # Create platform-specific context
            platform_context = {
                "input_content": content,
                "content_type": content_type,
                "seo_keywords": seo_result_dict,
                "social_media_marketing_brief": brief_result_dict,
                "social_media_platform": platform,
                "email_type": email_type if platform == "email" else None,
            }

            # Execute Angle & Hook step
            angle_hook_plugin = SocialMediaAngleHookPlugin()
            angle_hook_result = await angle_hook_plugin.execute(
                platform_context, self, job_id
            )

            # Check if result is ApprovalRequiredSentinel
            if isinstance(angle_hook_result, ApprovalRequiredSentinel):
                # Approval required - stop pipeline execution
                from marketing_project.services.job_manager import (
                    JobStatus,
                    get_job_manager,
                )

                job_manager = get_job_manager()
                await job_manager.update_job_status(
                    job_id, JobStatus.WAITING_FOR_APPROVAL
                )
                return platform, {
                    "status": "waiting_for_approval",
                    "approval_id": angle_hook_result.approval_result.approval_id,
                    "step_name": angle_hook_result.approval_result.step_name,
                    "step_number": angle_hook_result.approval_result.step_number,
                    "platform": platform,
                }

            angle_hook_result_dict = (
                angle_hook_result.model_dump(mode="json")
                if hasattr(angle_hook_result, "model_dump")
                else angle_hook_result.model_dump()
            )
            platform_context["social_media_angle_hook"] = angle_hook_result_dict

            # Execute Post Generation step
            post_plugin = SocialMediaPostGenerationPlugin()
            post_result = await post_plugin.execute(platform_context, self, job_id)

            # Check if result is ApprovalRequiredSentinel
            if isinstance(post_result, ApprovalRequiredSentinel):
                # Approval required - stop pipeline execution
                from marketing_project.services.job_manager import (
                    JobStatus,
                    get_job_manager,
                )

                job_manager = get_job_manager()
                await job_manager.update_job_status(
                    job_id, JobStatus.WAITING_FOR_APPROVAL
                )
                return platform, {
                    "status": "waiting_for_approval",
                    "approval_id": post_result.approval_result.approval_id,
                    "step_name": post_result.approval_result.step_name,
                    "step_number": post_result.approval_result.step_number,
                    "platform": platform,
                }

            post_result_dict = (
                post_result.model_dump(mode="json")
                if hasattr(post_result, "model_dump")
                else post_result.model_dump()
            )

            # Ensure platform field is included in post_result_dict
            if "platform" not in post_result_dict:
                post_result_dict["platform"] = platform

            # Validate content length
            final_content = post_result_dict.get("content", "")
            is_valid, warning = self._validate_content_length(
                final_content, platform, email_type if platform == "email" else None
            )

            # Assess quality scores
            platform_quality_scores = self._assess_platform_quality(
                post_result, platform
            )

            return platform, {
                "platform": platform,
                "step_results": {
                    "seo_keywords": seo_result_dict,
                    "social_media_marketing_brief": brief_result_dict,
                    "social_media_angle_hook": angle_hook_result_dict,
                    "social_media_post_generation": post_result_dict,
                },
                "final_content": final_content,
                "quality_warnings": [warning] if warning else [],
                "platform_quality_scores": platform_quality_scores,
            }

        # Execute all platforms in parallel
        platform_results = await asyncio.gather(
            *[execute_platform_pipeline(platform) for platform in platforms]
        )

        # Organize results by platform
        results_by_platform = {
            platform: result for platform, result in platform_results
        }

        pipeline_end = time.time()
        execution_time = pipeline_end - pipeline_start

        logger.info(
            f"Multi-Platform Pipeline completed in {execution_time:.2f}s for {len(platforms)} platforms"
        )

        return {
            "pipeline_status": "completed",
            "platforms": platforms,
            "results_by_platform": results_by_platform,
            "shared_steps": {
                "seo_keywords": seo_result_dict,
                "social_media_marketing_brief": brief_result_dict,
            },
            "input_content": content,
            "metadata": {
                "job_id": job_id,
                "content_id": content.get("id"),
                "content_type": content_type,
                "platforms": platforms,
                "email_type": email_type,
                "title": content.get("title"),
                "execution_time_seconds": execution_time,
                "model": self.model,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        }
