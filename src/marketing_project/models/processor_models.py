"""
Request and response models for deterministic processor endpoints.

These models are specifically for the direct processor endpoints that bypass
the orchestrator and go straight to deterministic processing workflows.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .content_models import BlogPostContext, ReleaseNotesContext, TranscriptContext

# Import PipelineConfig at runtime to avoid forward reference issues
# Using string annotations (from __future__ import annotations) allows us to
# import at runtime without circular import issues
from .pipeline_steps import PipelineConfig


class BlogProcessorRequest(BaseModel):
    """Request model for blog post processor."""

    content: BlogPostContext = Field(..., description="Blog post content to process")
    output_content_type: Optional[str] = Field(
        default="blog_post",
        description="Output content type: blog_post, press_release, case_study, or social_media_post",
    )
    social_media_platform: Optional[str] = Field(
        default=None,
        description="Social media platform: linkedin, hackernews, or email (required when output_content_type is social_media_post). Deprecated: use social_media_platforms for multi-platform support.",
    )
    social_media_platforms: Optional[List[str]] = Field(
        default=None,
        description="List of social media platforms for batch generation: linkedin, hackernews, or email (required when output_content_type is social_media_post). If provided, overrides social_media_platform.",
    )
    email_type: Optional[str] = Field(
        default=None,
        description="Email type: newsletter or promotional (only relevant when email is in social_media_platforms or social_media_platform is email)",
    )
    variations_count: Optional[int] = Field(
        default=1,
        ge=1,
        le=3,
        description="Number of content variations to generate (1-3, default: 1)",
    )
    pipeline_config: Optional[PipelineConfig] = Field(
        None,
        description="Optional pipeline configuration for per-step model selection (models must be configured in settings)",
    )
    campaign_id: Optional[str] = Field(
        None, description="Optional campaign ID for grouping multiple content items"
    )
    content_ids: Optional[List[str]] = Field(
        None, description="Optional list of content IDs for batch processing"
    )
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Processing options"
    )


class ReleaseNotesProcessorRequest(BaseModel):
    """Request model for release notes processor."""

    content: ReleaseNotesContext = Field(
        ..., description="Release notes content to process"
    )
    output_content_type: Optional[str] = Field(
        default="blog_post",
        description="Output content type: blog_post, press_release, or case_study",
    )
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Processing options"
    )


class TranscriptProcessorRequest(BaseModel):
    """Request model for transcript processor."""

    content: TranscriptContext = Field(..., description="Transcript content to process")
    output_content_type: Optional[str] = Field(
        default="blog_post",
        description="Output content type: blog_post, press_release, or case_study",
    )
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Processing options"
    )


class ProcessorResponse(BaseModel):
    """Base response model for processor endpoints."""

    success: bool = Field(..., description="Whether processing was successful")
    message: str = Field(..., description="Response message")
    content_id: str = Field(..., description="Content ID that was processed")
    content_type: str = Field(..., description="Type of content processed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )


class BlogProcessorResponse(ProcessorResponse):
    """Response model for blog post processor."""

    blog_type: Optional[str] = Field(None, description="Detected blog post type")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extracted metadata")
    pipeline_result: Optional[Dict[str, Any]] = Field(
        None, description="Content pipeline results"
    )
    validation: Optional[str] = Field(None, description="Validation status")
    processing_steps_completed: Optional[List[str]] = Field(
        None, description="List of completed processing steps"
    )


class ReleaseNotesProcessorResponse(ProcessorResponse):
    """Response model for release notes processor."""

    release_type: Optional[str] = Field(None, description="Detected release type")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extracted metadata")
    pipeline_result: Optional[Dict[str, Any]] = Field(
        None, description="Content pipeline results"
    )
    validation: Optional[str] = Field(None, description="Validation status")
    processing_steps_completed: Optional[List[str]] = Field(
        None, description="List of completed processing steps"
    )


class TranscriptProcessorResponse(ProcessorResponse):
    """Response model for transcript processor."""

    transcript_type: Optional[str] = Field(None, description="Detected transcript type")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extracted metadata")
    pipeline_result: Optional[Dict[str, Any]] = Field(
        None, description="Content pipeline results"
    )
    validation: Optional[str] = Field(None, description="Validation status")
    processing_steps_completed: Optional[List[str]] = Field(
        None, description="List of completed processing steps"
    )
