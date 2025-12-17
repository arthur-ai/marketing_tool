"""
Helper methods for prompt generation and configuration.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from marketing_project.plugins.context_utils import ContextTransformer
from marketing_project.prompts.prompts import get_template, has_template
from marketing_project.services.function_pipeline.tracing import (
    add_span_event,
    ensure_span_has_minimum_metadata,
    set_prompt_template,
    set_prompt_template_variables,
    set_span_duration,
    set_span_input,
    set_span_kind,
    set_span_output,
)

logger = logging.getLogger("marketing_project.services.function_pipeline.helpers")

# OpenTelemetry imports for tracing
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    _tracing_available = True
except ImportError:
    _tracing_available = False


class PipelineHelpers:
    """Helper methods for pipeline operations."""

    def __init__(self, lang: str = "en", pipeline_config=None):
        """
        Initialize helpers.

        Args:
            lang: Language for prompts
            pipeline_config: PipelineConfig instance
        """
        self.lang = lang
        self.pipeline_config = pipeline_config

    def get_system_instruction(
        self, agent_name: str, context: Optional[Dict] = None
    ) -> str:
        """
        Load comprehensive system instruction from .j2 template and enhance for function calling.

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

    def get_user_prompt(self, step_name: str, context: Dict[str, Any]) -> str:
        """
        Load user prompt from .j2 template and render with context variables.

        Args:
            step_name: Name of the step (e.g., "seo_keywords", "marketing_brief")
            context: Context variables for Jinja2 template rendering

        Returns:
            Rendered user prompt string
        """
        template_name = f"{step_name}_user_prompt"

        # Create prompt preparation span
        prompt_span = None
        prompt_start_time = None
        if _tracing_available:
            try:
                prompt_start_time = time.time()
                tracer = trace.get_tracer(__name__)
                prompt_span = tracer.start_as_current_span(
                    f"pipeline.prompt_preparation.{step_name}",
                    kind=trace.SpanKind.INTERNAL,
                )
                prompt_span.__enter__()

                # Set OpenInference span kind
                set_span_kind(prompt_span, "TOOL")

                # Add span event
                add_span_event(
                    prompt_span,
                    "prompt_preparation.started",
                    {
                        "step_name": step_name,
                        "template_name": template_name,
                    },
                )

                # Set input attributes (template name + context dict) - always set, never blank
                input_data = {
                    "template_name": template_name,
                    "template_language": self.lang,
                    "context": context or {},
                }
                set_span_input(prompt_span, input_data)

                # Ensure minimum metadata
                ensure_span_has_minimum_metadata(
                    prompt_span,
                    f"pipeline.prompt_preparation.{step_name}",
                    "prompt_preparation",
                )

                # Set prompt template variables
                context_keys_used = list(context.keys()) if context else []
                set_prompt_template_variables(prompt_span, context_keys_used)

                # Ensure minimum metadata
                ensure_span_has_minimum_metadata(
                    prompt_span,
                    f"pipeline.prompt_preparation.{step_name}",
                    "prompt_preparation",
                )

                # Set other attributes
                prompt_span.set_attribute("step_name", step_name)
                prompt_span.set_attribute("template_name", template_name)
                prompt_span.set_attribute("template_language", self.lang)
                prompt_span.set_attribute(
                    "context_keys_used", json.dumps(context_keys_used)
                )
            except Exception as e:
                logger.debug(f"Failed to create prompt preparation span: {e}")
                prompt_span = None

        try:
            # Check if template exists using helper function
            if has_template(self.lang, template_name):
                try:
                    # Load template using helper function (handles caching and fallback)
                    template = get_template(self.lang, template_name)

                    # Set prompt template content and version in OpenInference format
                    if prompt_span:
                        try:
                            # Get template source if available
                            template_source = getattr(template, "source", None)
                            if template_source:
                                from marketing_project.prompts.prompts import (
                                    TEMPLATE_VERSION,
                                )

                                set_prompt_template(
                                    prompt_span, template_source, TEMPLATE_VERSION
                                )
                        except Exception:
                            pass

                    # Prepare context for template rendering using ContextTransformer
                    template_context = ContextTransformer.prepare_template_context(
                        context
                    )

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
                    rendered_prompt = template.render(**template_context)

                    # Update span with prompt info
                    if prompt_span:
                        try:
                            # Set output attributes (rendered prompt string) - always set, never blank
                            set_span_output(
                                prompt_span,
                                rendered_prompt if rendered_prompt else "",
                                output_mime_type="text/plain",
                            )

                            prompt_span.set_attribute(
                                "prompt_length", len(rendered_prompt)
                            )

                            # Add template and prompt metrics
                            try:
                                # Get template source for complexity calculation
                                template_source = getattr(template, "source", None)
                                if template_source:
                                    # Calculate template complexity (rough estimate)
                                    template_complexity = len(template_source)
                                    prompt_span.set_attribute(
                                        "prompt.template_complexity",
                                        template_complexity,
                                    )

                                # Template variable count
                                prompt_span.set_attribute(
                                    "prompt.variable_count", len(context_keys_used)
                                )

                                # Rendered prompt length
                                prompt_span.set_attribute(
                                    "prompt.rendered_length", len(rendered_prompt)
                                )

                                # Estimated tokens (rough: ~4 chars per token)
                                estimated_tokens = len(rendered_prompt) // 4
                                prompt_span.set_attribute(
                                    "prompt.estimated_tokens", estimated_tokens
                                )

                                # Check for conditional logic in template
                                has_conditionals = (
                                    "{% if" in str(template_source)
                                    if template_source
                                    else False
                                )
                                prompt_span.set_attribute(
                                    "prompt.has_conditional_logic", has_conditionals
                                )

                                # Template version
                                from marketing_project.prompts.prompts import (
                                    TEMPLATE_VERSION,
                                )

                                prompt_span.set_attribute(
                                    "prompt.template_version", TEMPLATE_VERSION
                                )
                            except Exception:
                                pass

                            # Set duration
                            if prompt_start_time:
                                set_span_duration(prompt_span, prompt_start_time)

                            # Add completion event
                            add_span_event(
                                prompt_span,
                                "prompt_preparation.completed",
                                {
                                    "prompt_length": len(rendered_prompt),
                                },
                            )

                            prompt_span.set_status(Status(StatusCode.OK))
                        except Exception:
                            pass

                    return rendered_prompt
                except Exception as e:
                    if prompt_span:
                        try:
                            from marketing_project.services.function_pipeline.tracing import (
                                set_span_error,
                            )

                            set_span_error(
                                prompt_span,
                                e,
                                {
                                    "template_name": template_name,
                                    "step_name": step_name,
                                },
                            )
                            prompt_span.set_status(Status(StatusCode.ERROR, str(e)))
                        except Exception:
                            pass
                    logger.warning(
                        f"Failed to load or render template {template_name} for language {self.lang}: {e}. "
                        "Using fallback prompt."
                    )
                    logger.debug(
                        f"Template rendering error details: {e}", exc_info=True
                    )
            else:
                # Fallback to basic prompt if template not found
                logger.warning(
                    f"Template not found for {template_name} in language {self.lang}, "
                    "using fallback prompt"
                )

            # Fallback prompt
            fallback_prompt = (
                f"Process the content for {step_name.replace('_', ' ')} step."
            )
            if prompt_span:
                try:
                    # Set output attributes (fallback prompt string)
                    set_span_output(
                        prompt_span, fallback_prompt, output_mime_type="text/plain"
                    )

                    prompt_span.set_attribute("used_fallback", True)
                    prompt_span.set_attribute("prompt_length", len(fallback_prompt))

                    # Set duration
                    if prompt_start_time:
                        set_span_duration(prompt_span, prompt_start_time)

                    # Add fallback event
                    add_span_event(
                        prompt_span,
                        "prompt_preparation.fallback_used",
                        {
                            "template_name": template_name,
                        },
                    )

                    prompt_span.set_status(Status(StatusCode.OK))
                except Exception:
                    pass
            return fallback_prompt
        finally:
            if prompt_span:
                try:
                    # Set duration if not already set
                    if prompt_start_time:
                        set_span_duration(prompt_span, prompt_start_time)
                    prompt_span.__exit__(None, None, None)
                except Exception:
                    pass

    def get_step_model(self, step_name: str) -> str:
        """Get the model to use for a specific step."""
        return self.pipeline_config.get_step_model(step_name)

    def get_step_temperature(self, step_name: str) -> float:
        """Get the temperature to use for a specific step."""
        return self.pipeline_config.get_step_temperature(step_name)

    def get_step_max_retries(self, step_name: str) -> int:
        """Get the max retries for a specific step."""
        return self.pipeline_config.get_step_max_retries(step_name)
