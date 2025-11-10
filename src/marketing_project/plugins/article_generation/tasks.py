"""
Article Generation plugin tasks for Marketing Project.

This plugin handles article generation from marketing brief.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import ArticleGenerationResult
from marketing_project.plugins.base import PipelineStepPlugin

logger = logging.getLogger("marketing_project.plugins.article_generation")


class ArticleGenerationPlugin(PipelineStepPlugin):
    """Plugin for Article Generation step."""

    @property
    def step_name(self) -> str:
        return "article_generation"

    @property
    def step_number(self) -> int:
        return 3

    @property
    def response_model(self) -> type[ArticleGenerationResult]:
        return ArticleGenerationResult

    def get_required_context_keys(self) -> list[str]:
        return ["seo_keywords", "marketing_brief"]  # design_kit_config is optional

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build prompt context for article generation step.

        Converts context values to models and extracts ALL approved keywords.
        Optionally includes design_kit_config if available.
        """
        from marketing_project.models.design_kit_config import DesignKitConfig
        from marketing_project.models.pipeline_steps import (
            MarketingBriefResult,
            SEOKeywordsResult,
        )

        # Get and convert context values to models
        seo_result = self._get_context_model(context, "seo_keywords", SEOKeywordsResult)
        brief_result = self._get_context_model(
            context, "marketing_brief", MarketingBriefResult
        )
        output_content_type = context.get(
            "output_content_type", context.get("content_type", "blog_post")
        )

        # Extract ALL approved keywords (all categories)
        # Primary keywords (including main)
        primary_keywords = seo_result.primary_keywords or []
        # Supporting keywords (excluding main from primary)
        supporting_keywords = [
            k for k in primary_keywords if k != seo_result.main_keyword
        ]
        # Secondary keywords
        secondary_keywords = seo_result.secondary_keywords or []
        # LSI keywords
        lsi_keywords = seo_result.lsi_keywords or []
        # Long-tail keywords
        long_tail_keywords = seo_result.long_tail_keywords or []

        # Combine all keywords for comprehensive usage
        all_keywords = {
            "main_keyword": seo_result.main_keyword,
            "primary": primary_keywords,
            "supporting": supporting_keywords,
            "secondary": secondary_keywords,
            "lsi": lsi_keywords,
            "long_tail": long_tail_keywords,
        }

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
            "supporting_keywords": supporting_keywords,
            "all_keywords": all_keywords,
            "brief_result": brief_result,
            "seo_result": seo_result,  # Include full SEO result for access to new fields
            "output_content_type": output_content_type,
            "design_kit_config": design_kit_config,
        }

        return prompt_context

    async def execute(
        self, context: Dict[str, Any], pipeline: Any, job_id: Optional[str] = None
    ) -> ArticleGenerationResult:
        """
        Execute article generation step.

        Args:
            context: Context containing seo_keywords and marketing_brief
            pipeline: FunctionPipeline instance
            job_id: Optional job ID for tracking

        Returns:
            ArticleGenerationResult with generated article
        """
        # Use the common execution pattern
        return await self._execute_step(context, pipeline, job_id)
