"""
Social Media Angle & Hook plugin tasks for Marketing Project.

This plugin handles angle and hook generation for social media posts.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import (
    AngleHookResult,
    SocialMediaMarketingBriefResult,
)
from marketing_project.plugins.base import PipelineStepPlugin

logger = logging.getLogger("marketing_project.plugins.social_media_angle_hook")


class SocialMediaAngleHookPlugin(PipelineStepPlugin):
    """Plugin for Angle & Hook Generation step."""

    @property
    def step_name(self) -> str:
        return "social_media_angle_hook"

    @property
    def step_number(self) -> int:
        return 3  # Third step in social media pipeline

    @property
    def response_model(self) -> type[AngleHookResult]:
        return AngleHookResult

    def get_required_context_keys(self) -> list[str]:
        return ["social_media_marketing_brief", "social_media_platform"]

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build prompt context for angle & hook generation step.

        Converts marketing brief to model and prepares context for template.
        """
        # Get and convert marketing brief to model
        brief_result = self._get_context_model(
            context, "social_media_marketing_brief", SocialMediaMarketingBriefResult
        )
        platform = context.get("social_media_platform", "linkedin")
        email_type = context.get("email_type")

        # Build prompt context
        prompt_context = {
            "brief_result": brief_result,
            "platform": platform,
            "email_type": email_type,
            "input_content": context.get("input_content"),
        }

        return prompt_context

    async def execute(
        self, context: Dict[str, Any], pipeline: Any, job_id: Optional[str] = None
    ) -> AngleHookResult:
        """
        Execute angle & hook generation step.

        Args:
            context: Context containing social_media_marketing_brief and social_media_platform
            pipeline: Pipeline instance
            job_id: Optional job ID for tracking

        Returns:
            AngleHookResult with generated angles and hooks
        """
        # Use the common execution pattern
        return await self._execute_step(context, pipeline, job_id)
