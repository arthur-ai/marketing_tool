"""
Marketing Brief plugin tasks for Marketing Project.

This plugin handles marketing brief generation.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import MarketingBriefResult
from marketing_project.plugins.base import PipelineStepPlugin

logger = logging.getLogger("marketing_project.plugins.marketing_brief")


class MarketingBriefPlugin(PipelineStepPlugin):
    """Plugin for Marketing Brief generation step."""

    @property
    def step_name(self) -> str:
        return "marketing_brief"

    @property
    def step_number(self) -> int:
        return 2

    @property
    def response_model(self) -> type[MarketingBriefResult]:
        return MarketingBriefResult

    def get_required_context_keys(self) -> list[str]:
        return ["seo_keywords", "content_type"]  # design_kit_config is optional

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build prompt context for marketing brief step.

        Converts seo_keywords to model and prepares context for template.
        Optionally includes design_kit_config if available.
        """
        from marketing_project.models.design_kit_config import DesignKitConfig
        from marketing_project.models.pipeline_steps import SEOKeywordsResult

        # Get and convert seo_keywords to model
        seo_result = self._get_context_model(context, "seo_keywords", SEOKeywordsResult)
        content_type = context.get("content_type", "blog_post")
        output_content_type = context.get("output_content_type", content_type)

        # Get design kit config if available (optional)
        design_kit_config = None
        design_kit_dict = context.get("design_kit_config")
        if design_kit_dict:
            try:
                design_kit_config = DesignKitConfig(**design_kit_dict)
                # Get content-type-specific config
                design_kit_config = design_kit_config.get_content_type_config(
                    output_content_type
                )
            except Exception as e:
                logger.warning(f"Failed to parse design_kit_config: {e}")

        # Build prompt context
        prompt_context = {
            "content_type": content_type,
            "output_content_type": output_content_type,
            "seo_result": seo_result,
            "design_kit_config": design_kit_config,
        }

        return prompt_context

    async def execute(
        self, context: Dict[str, Any], pipeline: Any, job_id: Optional[str] = None
    ) -> MarketingBriefResult:
        """
        Execute marketing brief generation step.

        Args:
            context: Context containing seo_keywords and content_type
            pipeline: FunctionPipeline instance
            job_id: Optional job ID for tracking

        Returns:
            MarketingBriefResult with generated brief
        """
        # Use the common execution pattern
        return await self._execute_step(context, pipeline, job_id)
