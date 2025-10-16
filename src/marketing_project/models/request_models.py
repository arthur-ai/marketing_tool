"""
Request models for API endpoints.

This module defines Pydantic models for incoming API requests.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field

from .content_models import ContentContext, BlogPostContext, TranscriptContext, ReleaseNotesContext


class AnalyzeRequest(BaseModel):
    """Request model for content analysis."""
    content: Union[ContentContext, BlogPostContext, TranscriptContext, ReleaseNotesContext] = Field(
        ..., description="Content to analyze"
    )


class PipelineRequest(BaseModel):
    """Request model for full pipeline execution."""
    content: Union[ContentContext, BlogPostContext, TranscriptContext, ReleaseNotesContext] = Field(
        ..., description="Content to process through pipeline"
    )
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Pipeline options")


class WebhookRequest(BaseModel):
    """Request model for webhook endpoints."""
    data: Dict[str, Any] = Field(..., description="Webhook payload data")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="Request headers")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Webhook timestamp")
