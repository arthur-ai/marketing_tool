"""
Blog Post Preprocessing Approval plugin tasks for Marketing Project.

This plugin handles validation and approval of blog post preprocessing data
before proceeding to SEO keywords extraction.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import (
    BlogPostPreprocessingApprovalLLMResult,
    BlogPostPreprocessingApprovalResult,
)
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
    def response_model(self) -> type[BlogPostPreprocessingApprovalLLMResult]:
        return BlogPostPreprocessingApprovalLLMResult

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

        # Execute the step using the base implementation — returns BlogPostPreprocessingApprovalLLMResult
        result = await self._execute_step(context, pipeline, job_id)

        # Check if result is ApprovalRequiredSentinel (approval required, stop execution)
        from marketing_project.processors.approval_helper import (
            ApprovalRequiredSentinel,
        )

        if isinstance(result, ApprovalRequiredSentinel):
            return result

        # Upgrade from LLM base type to full result type
        full_result = BlogPostPreprocessingApprovalResult(**result.model_dump())

        # Programmatically fill deterministic enrichment fields
        input_content = context.get("input_content", {})
        content_str = (
            input_content.get("content", "") if isinstance(input_content, dict) else ""
        )
        if content_str:
            if full_result.word_count is None:
                full_result.word_count = len(content_str.split())
            if full_result.reading_time is None:
                full_result.reading_time = round(full_result.word_count / 200, 1)
            if full_result.content_summary is None:
                full_result.content_summary = (
                    content_str[:500] + "..." if len(content_str) > 500 else content_str
                )
        if full_result.confidence_score is None:
            full_result.confidence_score = 1.0

        # Merge AI-extracted data back into input_content for subsequent steps
        if isinstance(input_content, dict):
            # Update author if AI extracted it and it's missing from input
            if full_result.author and not input_content.get("author"):
                input_content["author"] = full_result.author
                logger.info(
                    f"Merged AI-extracted author into input_content: {full_result.author}"
                )

            # Update category if AI extracted it and it's missing from input
            if full_result.category and not input_content.get("category"):
                input_content["category"] = full_result.category
                logger.info(
                    f"Merged AI-extracted category into input_content: {full_result.category}"
                )

            # Update tags if AI extracted them and they're missing from input
            if full_result.tags and (
                not input_content.get("tags") or len(input_content.get("tags", [])) == 0
            ):
                input_content["tags"] = full_result.tags
                logger.info(
                    f"Merged AI-extracted tags into input_content: {full_result.tags}"
                )

            # Update word_count if computed and it's missing from input
            if (
                full_result.word_count is not None
                and input_content.get("word_count") is None
            ):
                input_content["word_count"] = full_result.word_count
                logger.info(
                    f"Merged computed word_count into input_content: {full_result.word_count}"
                )

            # Update reading_time if computed and it's missing from input
            if (
                full_result.reading_time is not None
                and input_content.get("reading_time") is None
            ):
                input_content["reading_time"] = full_result.reading_time
                logger.info(
                    f"Merged computed reading_time into input_content: {full_result.reading_time} minutes"
                )

            # Update snippet if content_summary generated it and it's missing from input
            if full_result.content_summary and not input_content.get("snippet"):
                input_content["snippet"] = (
                    full_result.content_summary[:200]
                    if len(full_result.content_summary) > 200
                    else full_result.content_summary
                )
                logger.info("Updated snippet from content_summary")

            # Update context with modified input_content
            context["input_content"] = input_content
            logger.info(
                f"Updated input_content in context with extracted data. "
                f"Author: {input_content.get('author')}, "
                f"Category: {input_content.get('category')}, "
                f"Tags: {input_content.get('tags')}, "
                f"Word count: {input_content.get('word_count')}"
            )

            # Log if AI successfully auto-fixed issues
            if (
                full_result.title_validated
                and full_result.content_validated
                and full_result.author_validated
                and full_result.category_validated
                and not full_result.requires_approval
            ):
                logger.info(
                    "AI successfully extracted missing data - approval not required"
                )

        logger.info(
            f"Blog post preprocessing approval step completed. "
            f"is_valid={full_result.is_valid}, requires_approval={full_result.requires_approval}, "
            f"validation_issues={len(full_result.validation_issues)}"
        )
        return full_result
