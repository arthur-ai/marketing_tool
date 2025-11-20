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
        return 3

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
        For transcripts, includes all transcript-specific metadata.
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

        # For transcripts, include all transcript-specific metadata
        if content_type == "transcript":
            input_content = context.get("input_content", {})
            if isinstance(input_content, dict):
                # Include core transcript fields
                prompt_context["transcript_speakers"] = input_content.get(
                    "speakers", []
                )
                prompt_context["transcript_duration"] = input_content.get("duration")
                prompt_context["transcript_type"] = input_content.get(
                    "transcript_type", "podcast"
                )
                prompt_context["transcript_title"] = input_content.get("title", "")
                prompt_context["transcript_snippet"] = input_content.get("snippet", "")

                # Include enhanced parsing metadata
                prompt_context["parsing_confidence"] = input_content.get(
                    "parsing_confidence"
                )
                prompt_context["detected_format"] = input_content.get("detected_format")
                prompt_context["parsing_warnings"] = input_content.get(
                    "parsing_warnings", []
                )
                prompt_context["quality_metrics"] = input_content.get(
                    "quality_metrics", {}
                )
                prompt_context["speaking_time_per_speaker"] = input_content.get(
                    "speaking_time_per_speaker", {}
                )
                prompt_context["detected_language"] = input_content.get(
                    "detected_language"
                )
                prompt_context["key_topics"] = input_content.get("key_topics", [])
                prompt_context["conversation_flow"] = input_content.get(
                    "conversation_flow", {}
                )

                # Include speaker analysis if available
                speaker_analysis = input_content.get("speaker_analysis", {})
                if speaker_analysis:
                    prompt_context["speaker_analysis"] = speaker_analysis

                # Include additional transcript metadata
                prompt_context["speaker_mapping"] = input_content.get(
                    "speaker_mapping", {}
                )
                prompt_context["timestamps"] = input_content.get("timestamps", {})
                prompt_context["word_count"] = input_content.get("word_count")

        # For blog posts, include blog post-specific metadata
        elif content_type == "blog_post":
            input_content = context.get("input_content", {})
            if isinstance(input_content, dict):
                # Extract blog post metadata
                prompt_context["blog_post_headings"] = input_content.get("headings", [])
                prompt_context["blog_post_tags"] = input_content.get("tags", [])
                prompt_context["blog_post_category"] = input_content.get("category", "")
                prompt_context["blog_post_author"] = input_content.get("author", "")
                prompt_context["blog_post_reading_time"] = input_content.get(
                    "reading_time"
                )
                prompt_context["blog_post_word_count"] = input_content.get("word_count")
                prompt_context["blog_post_links"] = input_content.get("links", [])
                prompt_context["blog_post_source_url"] = input_content.get(
                    "source_url", ""
                )
                prompt_context["blog_post_title"] = input_content.get("title", "")
                prompt_context["blog_post_snippet"] = input_content.get("snippet", "")

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
