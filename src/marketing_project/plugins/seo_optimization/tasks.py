"""
SEO Optimization plugin tasks for Marketing Project.

This plugin handles SEO optimization of generated content.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import SEOOptimizationResult
from marketing_project.plugins.base import PipelineStepPlugin

logger = logging.getLogger("marketing_project.plugins.seo_optimization")


class SEOOptimizationPlugin(PipelineStepPlugin):
    """Plugin for SEO Optimization step."""

    @property
    def step_name(self) -> str:
        return "seo_optimization"

    @property
    def step_number(self) -> int:
        return 5

    @property
    def response_model(self) -> type[SEOOptimizationResult]:
        return SEOOptimizationResult

    def get_required_context_keys(self) -> list[str]:
        return ["article_generation", "seo_keywords", "marketing_brief"]

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build prompt context for SEO optimization step.

        Converts context values to models for template rendering.
        """
        from marketing_project.models.pipeline_steps import (
            ArticleGenerationResult,
            MarketingBriefResult,
            SEOKeywordsResult,
        )

        # Get and convert context values to models
        article_result = self._get_context_model(
            context, "article_generation", ArticleGenerationResult
        )
        seo_result = self._get_context_model(context, "seo_keywords", SEOKeywordsResult)
        brief_result = self._get_context_model(
            context, "marketing_brief", MarketingBriefResult
        )

        # Build prompt context
        prompt_context = {
            "article_result": article_result,
            "seo_result": seo_result,
            "brief_result": brief_result,
        }

        return prompt_context

    async def execute(
        self, context: Dict[str, Any], pipeline: Any, job_id: Optional[str] = None
    ) -> SEOOptimizationResult:
        """
        Execute SEO optimization step.

        Args:
            context: Context containing article_generation, seo_keywords, and marketing_brief
            pipeline: FunctionPipeline instance
            job_id: Optional job ID for tracking

        Returns:
            SEOOptimizationResult with optimized content
        """
        # Use the common execution pattern
        return await self._execute_step(context, pipeline, job_id)
