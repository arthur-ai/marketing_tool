"""
Blog Post Preprocessing Approval plugin tasks for Marketing Project.

This plugin handles validation and approval of blog post preprocessing data
before proceeding to SEO keywords extraction.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import BlogPostPreprocessingApprovalResult
from marketing_project.plugins.base import PipelineStepPlugin

logger = logging.getLogger("marketing_project.plugins.blog_post_preprocessing_approval")


class BlogPostPreprocessingApprovalPlugin(PipelineStepPlugin):
    """Plugin for Blog Post Preprocessing Approval step."""

    @property
    def step_name(self) -> str:
        return "blog_post_preprocessing_approval"

    @property
    def step_number(self) -> int:
        return 1

    @property
    def response_model(self) -> type[BlogPostPreprocessingApprovalResult]:
        return BlogPostPreprocessingApprovalResult

    def get_required_context_keys(self) -> list[str]:
        return ["input_content"]

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build prompt context for blog post preprocessing approval step.

        Extracts blog post-specific fields for validation.
        """
        prompt_context = super()._build_prompt_context(context)

        # Get input content
        input_content = context.get("input_content", {})
        if isinstance(input_content, dict):
            # Extract blog post fields
            prompt_context["blog_post_id"] = input_content.get("id", "N/A")
            prompt_context["blog_post_title"] = input_content.get("title", "N/A")
            prompt_context["blog_post_content"] = input_content.get("content", "")
            prompt_context["blog_post_snippet"] = input_content.get("snippet", "")
            prompt_context["blog_post_author"] = input_content.get("author")
            prompt_context["blog_post_category"] = input_content.get("category")
            prompt_context["blog_post_tags"] = input_content.get("tags", [])
            prompt_context["blog_post_word_count"] = input_content.get("word_count")
            prompt_context["blog_post_reading_time"] = input_content.get("reading_time")
            prompt_context["blog_post_metadata"] = input_content.get("metadata", {})

            # Include parsing information if available
            prompt_context["parsing_confidence"] = input_content.get(
                "parsing_confidence"
            )
            prompt_context["detected_format"] = input_content.get("detected_format")
            prompt_context["parsing_warnings"] = input_content.get(
                "parsing_warnings", []
            )
            prompt_context["quality_metrics"] = input_content.get("quality_metrics", {})

            # Create content summary (first 500 chars)
            content_str = prompt_context.get("blog_post_content", "")
            if content_str:
                prompt_context["content_summary"] = (
                    content_str[:500] + "..." if len(content_str) > 500 else content_str
                )
            else:
                prompt_context["content_summary"] = "No content available"

        # Add content type
        prompt_context["content_type"] = context.get("content_type", "blog_post")

        return prompt_context

    async def execute(
        self,
        context: Dict[str, Any],
        pipeline: Any,
        job_id: Optional[str] = None,
    ) -> BlogPostPreprocessingApprovalResult:
        """
        Execute blog post preprocessing approval step.

        This step validates blog post fields and requires approval if issues are found.
        Only runs when content_type is "blog_post".

        Args:
            context: Context containing input_content
            pipeline: FunctionPipeline instance
            job_id: Optional job ID for tracking

        Returns:
            BlogPostPreprocessingApprovalResult with validation status
        """
        # Check if this is blog post content - if not, skip validation
        content_type = context.get("content_type", "blog_post")
        if content_type != "blog_post":
            logger.info(
                f"Skipping blog post preprocessing approval for content_type: {content_type}"
            )
            # Return a default valid result for non-blog_post content
            return BlogPostPreprocessingApprovalResult(
                is_valid=True,
                title_validated=True,
                content_validated=True,
                author_validated=True,
                category_validated=True,
                tags_validated=True,
                validation_issues=[],
                author=None,
                category=None,
                tags=[],
                word_count=None,
                reading_time=None,
                content_summary=None,
                confidence_score=1.0,
                requires_approval=False,
                approval_suggestions=[],
            )

        logger.info("Executing blog post preprocessing approval step")

        # Execute the step using the base implementation
        result = await self._execute_step(context, pipeline, job_id)

        # Merge AI-extracted data back into input_content for subsequent steps
        # This allows the pipeline to use extracted author, category, tags, etc.
        input_content = context.get("input_content", {})
        if isinstance(input_content, dict) and isinstance(
            result, BlogPostPreprocessingApprovalResult
        ):
            # Update author if AI extracted it and it's missing from input
            if result.author and not input_content.get("author"):
                input_content["author"] = result.author
                logger.info(
                    f"Merged AI-extracted author into input_content: {result.author}"
                )

            # Update category if AI extracted it and it's missing from input
            if result.category and not input_content.get("category"):
                input_content["category"] = result.category
                logger.info(
                    f"Merged AI-extracted category into input_content: {result.category}"
                )

            # Update tags if AI extracted them and they're missing from input
            if result.tags and (
                not input_content.get("tags") or len(input_content.get("tags", [])) == 0
            ):
                input_content["tags"] = result.tags
                logger.info(
                    f"Merged AI-extracted tags into input_content: {result.tags}"
                )

            # Update word_count if AI calculated it and it's missing from input
            if (
                result.word_count is not None
                and input_content.get("word_count") is None
            ):
                input_content["word_count"] = result.word_count
                logger.info(
                    f"Merged AI-calculated word_count into input_content: {result.word_count}"
                )

            # Update reading_time if AI calculated it and it's missing from input
            if (
                result.reading_time is not None
                and input_content.get("reading_time") is None
            ):
                input_content["reading_time"] = result.reading_time
                logger.info(
                    f"Merged AI-calculated reading_time into input_content: {result.reading_time} minutes"
                )

            # Update snippet if AI generated it and it's missing from input
            if result.content_summary and not input_content.get("snippet"):
                # Use content_summary as snippet if snippet is missing
                input_content["snippet"] = (
                    result.content_summary[:200]
                    if len(result.content_summary) > 200
                    else result.content_summary
                )
                logger.info("Updated snippet from content_summary")

            # Merge sentiment and analysis data back to input_content
            if result.overall_sentiment:
                input_content["overall_sentiment"] = result.overall_sentiment
            if result.sentiment_score is not None:
                input_content["sentiment_score"] = result.sentiment_score
            if result.emotional_tone:
                input_content["emotional_tone"] = result.emotional_tone
            if result.readability_score is not None:
                input_content["readability_score"] = result.readability_score
            if result.content_type:
                input_content["content_type_classification"] = result.content_type
            if result.target_audience:
                input_content["target_audience"] = result.target_audience
            if result.key_topics:
                input_content["key_topics"] = result.key_topics
            if result.detected_language:
                input_content["detected_language"] = result.detected_language
            if result.potential_keywords:
                input_content["potential_keywords"] = result.potential_keywords

            # Merge parsing information from result back to input_content
            if result.parsing_confidence is not None:
                input_content["parsing_confidence"] = result.parsing_confidence
            if result.detected_format:
                input_content["detected_format"] = result.detected_format
            if result.parsing_warnings:
                input_content["parsing_warnings"] = result.parsing_warnings
            if result.quality_metrics:
                input_content["quality_metrics"] = result.quality_metrics

            # Update context with modified input_content
            context["input_content"] = input_content

            # Log if AI successfully auto-fixed issues
            if (
                result.title_validated
                and result.content_validated
                and result.author_validated
                and result.category_validated
                and not result.requires_approval
            ):
                logger.info(
                    "AI successfully extracted missing data - approval not required"
                )

        return result
