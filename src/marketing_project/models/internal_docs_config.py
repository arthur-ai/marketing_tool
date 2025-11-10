"""
Internal Docs Configuration Model.

This model defines the structure for internal documentation configuration
that is used across pipeline runs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScannedDocument(BaseModel):
    """Represents a scanned internal document."""

    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Document URL or path")
    scanned_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this document was scanned"
    )
    relevance_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Relevance score (0-1)"
    )


class InternalDocsConfig(BaseModel):
    """Configuration for internal documentation linking and references."""

    scanned_documents: List[ScannedDocument] = Field(
        default_factory=list,
        description="List of internal documents that were scanned to generate this configuration",
    )
    commonly_referenced_pages: List[str] = Field(
        default_factory=list, description="List of commonly referenced page slugs/URLs"
    )
    commonly_referenced_categories: List[str] = Field(
        default_factory=list,
        description="List of commonly referenced content categories",
    )
    anchor_phrasing_patterns: List[str] = Field(
        default_factory=list,
        description="Patterns for how anchor text is typically phrased",
    )
    interlinking_rules: Dict[str, Any] = Field(
        default_factory=dict, description="Rules and guidelines for internal linking"
    )
    version: str = Field(
        default="1.0.0", description="Version identifier for this configuration"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when configuration was created",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when configuration was last updated",
    )
    is_active: bool = Field(
        default=True,
        description="Whether this configuration version is currently active",
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
