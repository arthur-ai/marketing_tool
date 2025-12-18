"""
Context summarization utilities for reducing token usage while preserving full data.

This module provides utilities to summarize context before passing to pipeline steps,
while keeping the full context in the context registry for later retrieval.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("marketing_project.services.context_summarizer")


class ContextSummarizer:
    """Utility class for summarizing context to reduce token usage."""

    @staticmethod
    def summarize_step_output(
        output_data: Dict[str, Any], max_length: int = 500
    ) -> Dict[str, Any]:
        """
        Summarize step output data while preserving key information.

        Args:
            output_data: Full step output data
            max_length: Maximum length for summarized text fields

        Returns:
            Summarized output data
        """
        summarized = {}
        for key, value in output_data.items():
            if isinstance(value, str):
                if len(value) > max_length:
                    summarized[key] = value[:max_length] + "... [truncated]"
                else:
                    summarized[key] = value
            elif isinstance(value, (list, dict)):
                # Keep structure but limit depth/size
                summarized[key] = ContextSummarizer._summarize_nested(
                    value, max_items=10
                )
            else:
                summarized[key] = value
        return summarized

    @staticmethod
    def _summarize_nested(
        data: Any, max_items: int = 10, depth: int = 0, max_depth: int = 2
    ) -> Any:
        """
        Summarize nested data structures.

        Args:
            data: Data to summarize
            max_items: Maximum items to keep in lists/dicts
            depth: Current depth
            max_depth: Maximum depth to traverse

        Returns:
            Summarized data
        """
        if depth > max_depth:
            return "[max depth reached]"

        if isinstance(data, list):
            if len(data) <= max_items:
                return [
                    ContextSummarizer._summarize_nested(
                        item, max_items, depth + 1, max_depth
                    )
                    for item in data
                ]
            else:
                summarized = [
                    ContextSummarizer._summarize_nested(
                        item, max_items, depth + 1, max_depth
                    )
                    for item in data[:max_items]
                ]
                summarized.append(f"... and {len(data) - max_items} more items")
                return summarized

        elif isinstance(data, dict):
            if len(data) <= max_items:
                return {
                    k: ContextSummarizer._summarize_nested(
                        v, max_items, depth + 1, max_depth
                    )
                    for k, v in data.items()
                }
            else:
                items = list(data.items())[:max_items]
                summarized = {
                    k: ContextSummarizer._summarize_nested(
                        v, max_items, depth + 1, max_depth
                    )
                    for k, v in items
                }
                summarized["_truncated"] = f"{len(data) - max_items} more keys"
                return summarized

        return data

    @staticmethod
    def get_relevant_context_keys(step_name: str) -> List[str]:
        """
        Get list of relevant context keys for a specific step.

        This helps reduce token usage by only passing relevant context.

        Args:
            step_name: Name of the step

        Returns:
            List of relevant context key names
        """
        # Define context dependencies for each step
        context_dependencies = {
            "blog_post_preprocessing_approval": ["input_content"],
            "transcript_preprocessing_approval": ["input_content"],
            "seo_keywords": ["input_content"],
            "social_media_marketing_brief": ["seo_keywords", "input_content"],
            "social_media_angle_hook": [
                "social_media_marketing_brief",
                "input_content",
            ],
            "social_media_post_generation": [
                "social_media_angle_hook",
                "social_media_marketing_brief",
                "input_content",
            ],
        }

        return context_dependencies.get(step_name, [])

    @staticmethod
    def build_optimized_context(
        full_context: Dict[str, Any],
        step_name: str,
        context_registry: Optional[Any] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build optimized context for a step by only including relevant keys.

        Uses context registry to store full context while passing summarized version.

        Args:
            full_context: Full pipeline context
            step_name: Name of the step
            context_registry: Optional context registry instance
            job_id: Optional job ID for context registry lookup

        Returns:
            Optimized context dictionary
        """
        relevant_keys = ContextSummarizer.get_relevant_context_keys(step_name)
        optimized_context = {}

        for key in relevant_keys:
            if key in full_context:
                value = full_context[key]
                # Summarize if it's a large data structure
                if isinstance(value, (dict, list)):
                    optimized_context[key] = ContextSummarizer.summarize_step_output(
                        value if isinstance(value, dict) else {"items": value},
                        max_length=300,
                    )
                else:
                    optimized_context[key] = value

        # Always include input_content and content_type as safety net (required by many steps)
        if "input_content" in full_context:
            # Only add if not already included from relevant_keys
            if "input_content" not in optimized_context:
                value = full_context["input_content"]
                # Summarize if it's a large data structure
                if isinstance(value, (dict, list)):
                    optimized_context["input_content"] = (
                        ContextSummarizer.summarize_step_output(
                            value if isinstance(value, dict) else {"items": value},
                            max_length=300,
                        )
                    )
                else:
                    optimized_context["input_content"] = value

        if "content_type" in full_context:
            optimized_context["content_type"] = full_context["content_type"]

        # Always include platform and email_type
        if "social_media_platform" in full_context:
            optimized_context["social_media_platform"] = full_context[
                "social_media_platform"
            ]
        if "email_type" in full_context:
            optimized_context["email_type"] = full_context["email_type"]

        return optimized_context
