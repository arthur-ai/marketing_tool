"""
Social Media Post Generation plugin tasks for Marketing Project.

This plugin handles final post generation for social media platforms.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import (
    AngleHookResult,
    SocialMediaPostResult,
)
from marketing_project.plugins.base import PipelineStepPlugin

logger = logging.getLogger("marketing_project.plugins.social_media_post_generation")


class SocialMediaPostGenerationPlugin(PipelineStepPlugin):
    """Plugin for Social Media Post Generation step."""

    @property
    def step_name(self) -> str:
        return "social_media_post_generation"

    @property
    def step_number(self) -> int:
        return 4  # Fourth step in social media pipeline

    @property
    def response_model(self) -> type[SocialMediaPostResult]:
        return SocialMediaPostResult

    def get_required_context_keys(self) -> list[str]:
        return ["social_media_angle_hook", "social_media_platform"]

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build prompt context for post generation step.

        Converts angle & hook result to model and prepares context for template.
        """
        # Get and convert angle & hook result to model
        angle_hook_result = self._get_context_model(
            context, "social_media_angle_hook", AngleHookResult
        )
        platform = context.get("social_media_platform", "linkedin")
        email_type = context.get("email_type")

        # Get marketing brief if available (for additional context)
        brief_result = None
        brief_dict = context.get("social_media_marketing_brief")
        if brief_dict:
            try:
                from marketing_project.models.pipeline_steps import (
                    SocialMediaMarketingBriefResult,
                )

                brief_result = SocialMediaMarketingBriefResult(**brief_dict)
            except Exception as e:
                logger.warning(f"Failed to parse marketing brief: {e}")

        # Build prompt context
        prompt_context = {
            "angle_hook_result": angle_hook_result,
            "brief_result": brief_result,
            "platform": platform,
            "email_type": email_type,
            "input_content": context.get("input_content"),
        }

        return prompt_context

    async def execute(
        self, context: Dict[str, Any], pipeline: Any, job_id: Optional[str] = None
    ) -> SocialMediaPostResult:
        """
        Execute social media post generation step.

        Args:
            context: Context containing social_media_angle_hook and social_media_platform
            pipeline: Pipeline instance
            job_id: Optional job ID for tracking

        Returns:
            SocialMediaPostResult with generated post
        """
        # Use the common execution pattern
        return await self._execute_step(context, pipeline, job_id)
