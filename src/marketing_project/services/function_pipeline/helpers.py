"""
Helper methods for prompt generation and configuration.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.plugins.context_utils import ContextTransformer
from marketing_project.prompts.prompts import get_template, has_template

logger = logging.getLogger("marketing_project.services.function_pipeline.helpers")


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

    def get_step_model(self, step_name: str) -> str:
        """Get the model to use for a specific step."""
        return self.pipeline_config.get_step_model(step_name)

    def get_step_temperature(self, step_name: str) -> float:
        """Get the temperature to use for a specific step."""
        return self.pipeline_config.get_step_temperature(step_name)

    def get_step_max_retries(self, step_name: str) -> int:
        """Get the max retries for a specific step."""
        return self.pipeline_config.get_step_max_retries(step_name)
