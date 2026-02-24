"""
Content Formatting plugin tasks for Marketing Project.

This plugin handles content formatting for publication.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import ContentFormattingResult
from marketing_project.plugins.base import PipelineStepPlugin

logger = logging.getLogger("marketing_project.plugins.content_formatting")


class ContentFormattingPlugin(PipelineStepPlugin):
    """Plugin for Content Formatting step."""

    @property
    def step_name(self) -> str:
        return "content_formatting"

    @property
    def step_number(self) -> int:
        return 8

    @property
    def response_model(self) -> type[ContentFormattingResult]:
        return ContentFormattingResult

    def get_required_context_keys(self) -> list[str]:
        return [
            "seo_optimization",
            "article_generation",
        ]  # internal_docs, design_kit_config and internal_docs_config are optional

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build prompt context for content formatting step.

        Converts context values to models for template rendering.
        Optionally includes design_kit_config and internal_docs_config if available.
        """
        from marketing_project.models.internal_docs_config import InternalDocsConfig
        from marketing_project.models.pipeline_steps import (
            ArticleGenerationResult,
            SEOOptimizationResult,
            SuggestedLinksResult,
        )

        # Get and convert context values to models
        seo_opt_result = self._get_context_model(
            context, "seo_optimization", SEOOptimizationResult
        )
        article_result = self._get_context_model(
            context, "article_generation", ArticleGenerationResult
        )
        output_content_type = context.get(
            "output_content_type", context.get("content_type", "blog_post")
        )

        # Get suggested links result if available (optional - contains specific link suggestions)
        suggested_links_result = None
        suggested_links_result_dict = context.get("suggested_links")
        if suggested_links_result_dict:
            try:
                suggested_links_result = SuggestedLinksResult(
                    **suggested_links_result_dict
                )
                logger.info(
                    "Loaded suggested links result with link suggestions for content formatting"
                )
            except Exception as e:
                logger.warning(f"Failed to parse suggested_links result: {e}")

        # Get brand kit config if available (optional)
        brand_kit_config = None
        brand_kit_full = None
        brand_kit_dict = context.get("brand_kit_config")
        if brand_kit_dict:
            try:
                from marketing_project.models.brand_kit_config import BrandKitConfig

                brand_kit_full = BrandKitConfig(**brand_kit_dict)
                # Get content-type-specific config
                brand_kit_config = brand_kit_full.get_content_type_config(
                    output_content_type
                )
            except Exception as e:
                logger.warning(f"Failed to parse brand_kit_config: {e}")

        # Get internal docs config if available (optional - contains general interlinking rules)
        internal_docs_config = None
        internal_docs_dict = context.get("internal_docs_config")
        if internal_docs_dict:
            try:
                internal_docs_config = InternalDocsConfig(**internal_docs_dict)
            except Exception as e:
                logger.warning(f"Failed to parse internal_docs_config: {e}")

        # Build prompt context
        prompt_context = {
            "seo_opt_result": seo_opt_result,
            "article_result": article_result,
            "output_content_type": output_content_type,
            "suggested_links_result": suggested_links_result,
            "brand_kit_config": brand_kit_config,
            "design_kit_config": brand_kit_config,  # backward compat alias
            "internal_docs_config": internal_docs_config,
            # New brand intelligence fields (top-level for easy template access)
            "brand_about": brand_kit_full.about_the_brand if brand_kit_full else None,
            "brand_pov": brand_kit_full.brand_point_of_view if brand_kit_full else None,
            "brand_competitors": brand_kit_full.competitors if brand_kit_full else None,
            "brand_differentiation": (
                brand_kit_full.competitive_differentiation_angle
                if brand_kit_full
                else None
            ),
            "brand_icp": (
                brand_kit_full.ideal_customer_profile if brand_kit_full else None
            ),
            "brand_author_persona": (
                brand_kit_full.author_persona if brand_kit_full else None
            ),
            "brand_success_metrics": (
                brand_kit_full.success_metrics if brand_kit_full else None
            ),
            "brand_writing_samples": (
                brand_kit_full.writing_samples if brand_kit_full else None
            ),
        }

        return prompt_context

    async def execute(
        self, context: Dict[str, Any], pipeline: Any, job_id: Optional[str] = None
    ) -> ContentFormattingResult:
        """
        Execute content formatting step.

        Args:
            context: Context containing seo_optimization and article_generation
            pipeline: FunctionPipeline instance
            job_id: Optional job ID for tracking

        Returns:
            ContentFormattingResult with formatted content
        """
        # Use the common execution pattern
        return await self._execute_step(context, pipeline, job_id)
