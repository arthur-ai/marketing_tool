"""
Utility functions for the Marketing Project.

This module provides common utilities for context conversion, data validation,
and other shared functionality across the marketing project.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from marketing_project.core.models import (
    AppContext,
    BlogPostContext,
    ContentContext,
    EmailContext,
    ReleaseNotesContext,
    TranscriptContext,
)

logger = logging.getLogger("marketing_project.core.utils")


def convert_dict_to_content_context(data: Dict[str, Any]) -> ContentContext:
    """
    Convert dictionary to appropriate ContentContext object.

    Args:
        data: Dictionary containing content data

    Returns:
        ContentContext: Appropriate ContentContext object

    Raises:
        ValueError: If data doesn't contain required fields
    """
    # Validate required fields
    required_fields = ["id", "title", "content", "snippet"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Determine content type based on available fields
    if "speakers" in data or "transcript_type" in data:
        return TranscriptContext(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            snippet=data["snippet"],
            created_at=data.get("created_at"),
            source_url=data.get("source_url"),
            metadata=data.get("metadata", {}),
            speakers=data.get("speakers", []),
            duration=data.get("duration"),
            transcript_type=data.get("transcript_type", "podcast"),
            timestamps=data.get("timestamps"),
        )
    elif "version" in data or "changes" in data:
        return ReleaseNotesContext(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            snippet=data["snippet"],
            created_at=data.get("created_at"),
            source_url=data.get("source_url"),
            metadata=data.get("metadata", {}),
            version=data["version"],
            release_date=data.get("release_date"),
            changes=data.get("changes", []),
            breaking_changes=data.get("breaking_changes", []),
            features=data.get("features", []),
            bug_fixes=data.get("bug_fixes", []),
        )
    elif "author" in data or "tags" in data:
        return BlogPostContext(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            snippet=data["snippet"],
            created_at=data.get("created_at"),
            source_url=data.get("source_url"),
            metadata=data.get("metadata", {}),
            author=data.get("author"),
            tags=data.get("tags", []),
            category=data.get("category"),
            word_count=data.get("word_count"),
            reading_time=data.get("reading_time"),
        )
    elif "thread_id" in data or "subject" in data:
        return EmailContext(
            id=data["id"],
            thread_id=data["thread_id"],
            subject=data["subject"],
            snippet=data["snippet"],
            body=data["content"],
        )
    else:
        # Default to base ContentContext
        return ContentContext(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            snippet=data["snippet"],
            created_at=data.get("created_at"),
            source_url=data.get("source_url"),
            metadata=data.get("metadata", {}),
        )


def ensure_content_context(
    content: Union[ContentContext, Dict[str, Any]],
) -> ContentContext:
    """
    Ensure content is a ContentContext object.

    Args:
        content: Content to convert if needed

    Returns:
        ContentContext: ContentContext object
    """
    # Check for any content context type (from either import path)
    if hasattr(content, 'id') and hasattr(content, 'title') and hasattr(content, 'content') and hasattr(content, 'snippet'):
        return content
    elif isinstance(content, dict):
        return convert_dict_to_content_context(content)
    else:
        raise ValueError(f"Unsupported content type: {type(content)}")


def create_standard_task_result(
    success: bool = True,
    data: Any = None,
    error: Optional[str] = None,
    task_name: str = "unknown_task",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a standardized task result for Any Agent compatibility.

    Args:
        success: Whether the task succeeded
        data: Task result data
        error: Error message if task failed
        task_name: Name of the task
        metadata: Additional metadata

    Returns:
        Dict[str, Any]: Standardized task result
    """
    result = {
        "success": success,
        "task_name": task_name,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }

    if error:
        result["error"] = error

    if metadata:
        result["metadata"] = metadata

    return result


def validate_content_for_processing(content: ContentContext) -> Dict[str, Any]:
    """
    Validate content is ready for processing.

    Args:
        content: Content to validate

    Returns:
        Dict[str, Any]: Validation results
    """
    validation = {"is_valid": True, "issues": [], "warnings": []}

    # Check required fields
    if not content.title or not content.title.strip():
        validation["issues"].append("Missing or empty title")
        validation["is_valid"] = False

    if not content.content or not content.content.strip():
        validation["issues"].append("Missing or empty content")
        validation["is_valid"] = False

    if not content.snippet or not content.snippet.strip():
        validation["warnings"].append("Missing snippet - consider adding one")

    # Check content length
    word_count = len(content.content.split()) if content.content else 0
    if word_count < 100:
        validation["warnings"].append("Content is very short (less than 100 words)")
    elif word_count > 5000:
        validation["warnings"].append("Content is very long (more than 5000 words)")

    return validation


def extract_content_metadata_for_pipeline(content: ContentContext) -> Dict[str, Any]:
    """
    Extract metadata needed for pipeline processing.

    Args:
        content: Content to extract metadata from

    Returns:
        Dict[str, Any]: Extracted metadata
    """
    metadata = {
        "content_type": type(content).__name__.replace("Context", "").lower(),
        "id": content.id,
        "title": content.title,
        "word_count": len(content.content.split()) if content.content else 0,
        "has_snippet": bool(content.snippet),
        "has_metadata": bool(content.metadata),
        "created_at": content.created_at.isoformat() if content.created_at else None,
        "source_url": content.source_url,
    }

    # Add type-specific metadata
    if isinstance(content, TranscriptContext):
        metadata.update(
            {
                "speakers": content.speakers,
                "duration": content.duration,
                "transcript_type": content.transcript_type,
            }
        )
    elif isinstance(content, BlogPostContext):
        metadata.update(
            {
                "author": content.author,
                "tags": content.tags,
                "category": content.category,
                "reading_time": content.reading_time,
            }
        )
    elif isinstance(content, ReleaseNotesContext):
        metadata.update(
            {
                "version": content.version,
                "changes_count": len(content.changes),
                "features_count": len(content.features),
                "bug_fixes_count": len(content.bug_fixes),
            }
        )

    return metadata


def merge_task_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple task results into a single result.

    Args:
        results: List of task results to merge

    Returns:
        Dict[str, Any]: Merged result
    """
    if not results:
        return create_standard_task_result(success=False, error="No results to merge")

    # Check if all tasks succeeded
    all_successful = all(result.get("success", False) for result in results)

    # Merge data
    merged_data = {}
    for result in results:
        if result.get("success") and result.get("data"):
            task_name = result.get("task_name", "unknown")
            merged_data[task_name] = result["data"]

    # Collect errors
    errors = []
    for result in results:
        if not result.get("success") and result.get("error"):
            errors.append(result["error"])

    return create_standard_task_result(
        success=all_successful,
        data=merged_data,
        error="; ".join(errors) if errors else None,
        task_name="merged_results",
        metadata={
            "total_tasks": len(results),
            "successful_tasks": sum(1 for r in results if r.get("success", False)),
            "failed_tasks": sum(1 for r in results if not r.get("success", False)),
        },
    )
