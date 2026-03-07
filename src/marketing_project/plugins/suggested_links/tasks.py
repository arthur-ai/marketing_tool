"""
Suggested Links plugin tasks for Marketing Project.

This plugin suggests where to add internal links in the generated article,
using the InternalDocsConfig as a reference for available pages and linking patterns.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import SuggestedLinksResult
from marketing_project.plugins.base import PipelineStepPlugin

logger = logging.getLogger("marketing_project.plugins.suggested_links")


class SuggestedLinksPlugin(PipelineStepPlugin):
    """Plugin for Suggested Links step.

    This step analyzes the generated article and suggests specific places to add
    internal links, using the InternalDocsConfig to know what pages/categories
    are available and what anchor text patterns to use.
    """

    @property
    def step_name(self) -> str:
        return "suggested_links"

    @property
    def step_number(self) -> int:
        return 6

    @property
    def response_model(self) -> type[SuggestedLinksResult]:
        return SuggestedLinksResult

    def get_required_context_keys(self) -> list[str]:
        return [
            "article_generation",
            "seo_keywords",
            "seo_optimization",
            "input_content",
        ]
        # internal_docs_config is optional - step will work without it but with limited suggestions

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build prompt context for internal linking suggestions step.

        Converts context values to models for template rendering and includes
        InternalDocsConfig if available. Relevant documents are pre-fetched
        asynchronously in _execute_step and injected as "_db_relevant_documents".
        """
        from marketing_project.models.internal_docs_config import InternalDocsConfig
        from marketing_project.models.pipeline_steps import (
            ArticleGenerationResult,
            SEOKeywordsResult,
            SEOOptimizationResult,
        )

        # Get and convert context values to models
        article_result = self._get_context_model(
            context, "article_generation", ArticleGenerationResult
        )
        seo_result = self._get_context_model(context, "seo_keywords", SEOKeywordsResult)
        seo_opt_result = self._get_context_model(
            context, "seo_optimization", SEOOptimizationResult
        )
        content = context.get("input_content", {})

        # Load InternalDocsConfig if available
        internal_docs_config = None
        internal_docs_dict = context.get("internal_docs_config")
        if internal_docs_dict:
            try:
                internal_docs_config = InternalDocsConfig(**internal_docs_dict)
                logger.info(
                    "Loaded internal docs configuration for linking suggestions"
                )
            except Exception as e:
                logger.warning(f"Failed to parse internal_docs_config: {e}")
        else:
            logger.warning(
                "No internal docs configuration found - linking suggestions will be limited"
            )

        # Relevant documents were pre-fetched asynchronously in _execute_step
        relevant_documents = context.get("_db_relevant_documents", [])

        return {
            "content": content,
            "article_result": article_result,
            "seo_result": seo_result,
            "seo_opt_result": seo_opt_result,
            "internal_docs_config": internal_docs_config,
            "relevant_documents": (
                [doc.model_dump() for doc in relevant_documents]
                if relevant_documents
                else []
            ),
        }

    async def _execute_step(
        self,
        context: Dict[str, Any],
        pipeline: Any,
        job_id: Optional[str] = None,
    ) -> SuggestedLinksResult:
        """Pre-fetch relevant documents from DB (async), then run the common step."""
        from marketing_project.models.pipeline_steps import SEOKeywordsResult
        from marketing_project.services.scanned_document_db import (
            get_scanned_document_db,
        )

        try:
            seo_result = self._get_context_model(
                context, "seo_keywords", SEOKeywordsResult
            )
            keywords = []
            if seo_result:
                if seo_result.main_keyword:
                    keywords.append(seo_result.main_keyword)
                if seo_result.primary_keywords:
                    keywords.extend(seo_result.primary_keywords[:5])

            relevant_documents = []
            if keywords:
                db = get_scanned_document_db()
                relevant_documents = await db.search_by_keywords(keywords, limit=20)
                logger.info(
                    f"Found {len(relevant_documents)} relevant documents from database for link suggestions"
                )
        except Exception as e:
            logger.warning(f"Failed to query database for relevant documents: {e}")
            relevant_documents = []

        enriched_context = {**context, "_db_relevant_documents": relevant_documents}
        return await super()._execute_step(enriched_context, pipeline, job_id)

    async def execute(
        self, context: Dict[str, Any], pipeline: Any, job_id: Optional[str] = None
    ) -> SuggestedLinksResult:
        """
        Execute suggested links step.

        Args:
            context: Context containing article_generation, seo_keywords, seo_optimization, and input_content
            pipeline: FunctionPipeline instance
            job_id: Optional job ID for tracking

        Returns:
            SuggestedLinksResult with link suggestions
        """
        return await self._execute_step(context, pipeline, job_id)
