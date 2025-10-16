"""
Content models for API endpoints.

This module defines Pydantic models for different types of content
that can be processed by the marketing pipeline.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Content type enumeration."""

    BLOG_POST = "blog_post"
    TRANSCRIPT = "transcript"
    RELEASE_NOTES = "release_notes"
    ARTICLE = "article"
    INTERNAL_DOCS = "internal_docs"


class ContentContext(BaseModel):
    """Base content context model."""

    id: str = Field(..., description="Unique identifier for the content")
    title: Optional[str] = Field(None, description="Content title")
    content: Optional[str] = Field(None, description="Main content body")
    snippet: Optional[str] = Field(None, description="Content snippet or summary")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    source_url: Optional[str] = Field(None, description="Source URL if applicable")
    content_type: Optional[ContentType] = Field(None, description="Type of content")


class BlogPostContext(ContentContext):
    """Blog post specific context."""

    author: Optional[str] = Field(None, description="Author name")
    tags: List[str] = Field(default_factory=list, description="Content tags")
    category: Optional[str] = Field(None, description="Content category")
    word_count: Optional[int] = Field(None, description="Word count")
    content_type: ContentType = Field(
        default=ContentType.BLOG_POST, description="Content type"
    )


class TranscriptContext(ContentContext):
    """Transcript specific context."""

    speakers: List[str] = Field(default_factory=list, description="List of speakers")
    duration: Optional[int] = Field(None, description="Duration in seconds")
    transcript_type: Optional[str] = Field(None, description="Type of transcript")
    content_type: ContentType = Field(
        default=ContentType.TRANSCRIPT, description="Content type"
    )


class ReleaseNotesContext(ContentContext):
    """Release notes specific context."""

    version: Optional[str] = Field(None, description="Version number")
    release_date: Optional[datetime] = Field(None, description="Release date")
    changes: List[str] = Field(default_factory=list, description="List of changes")
    features: List[str] = Field(default_factory=list, description="New features")
    bug_fixes: List[str] = Field(default_factory=list, description="Bug fixes")
    content_type: ContentType = Field(
        default=ContentType.RELEASE_NOTES, description="Content type"
    )
