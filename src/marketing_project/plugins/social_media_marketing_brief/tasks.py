"""
Social Media Marketing Brief plugin tasks for Marketing Project.

This plugin handles platform-specific marketing brief generation for social media.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import (
    SEOKeywordsResult,
    SocialMediaMarketingBriefResult,
)
from marketing_project.plugins.base import PipelineStepPlugin

logger = logging.getLogger("marketing_project.plugins.social_media_marketing_brief")


class SocialMediaMarketingBriefPlugin(PipelineStepPlugin):
    """Plugin for Social Media Marketing Brief generation step."""

    @property
    def step_name(self) -> str:
        return "social_media_marketing_brief"

    @property
    def step_number(self) -> int:
        return 2  # Second step in social media pipeline (after SEO Keywords)

    @property
    def response_model(self) -> type[SocialMediaMarketingBriefResult]:
        return SocialMediaMarketingBriefResult

    def get_required_context_keys(self) -> list[str]:
        return ["seo_keywords", "social_media_platform"]

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build prompt context for social media marketing brief step.

        Converts seo_keywords to model and prepares context for template.
        Includes platform and email_type information.
        """
        # Get and convert seo_keywords to model
        seo_result = self._get_context_model(context, "seo_keywords", SEOKeywordsResult)
        platform = context.get("social_media_platform", "linkedin")
        email_type = context.get("email_type")

        # Build prompt context
        prompt_context = {
            "seo_result": seo_result,
            "platform": platform,
            "email_type": email_type,
            "input_content": context.get("input_content"),
        }

        return prompt_context

    async def execute(
        self, context: Dict[str, Any], pipeline: Any, job_id: Optional[str] = None
    ) -> SocialMediaMarketingBriefResult:
        """
        Execute social media marketing brief generation step.

        Args:
            context: Context containing seo_keywords and social_media_platform
            pipeline: Pipeline instance
            job_id: Optional job ID for tracking

        Returns:
            SocialMediaMarketingBriefResult with generated brief
        """
        # Use the common execution pattern
        return await self._execute_step(context, pipeline, job_id)
