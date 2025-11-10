"""
Database Model for Scanned Internal Documents.

This model stores scanned internal documents with rich metadata
for use by Design Kit and Suggested Links services.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScannedDocumentMetadata(BaseModel):
    """Rich metadata for a scanned internal document."""

    # Content information
    content_text: Optional[str] = Field(None, description="Full extracted content text")
    content_summary: Optional[str] = Field(
        None, description="Brief summary of the content"
    )
    word_count: Optional[int] = Field(None, description="Word count of the content")

    # Structure information
    headings: List[str] = Field(
        default_factory=list, description="List of headings (H1, H2, H3)"
    )
    sections: List[Dict[str, Any]] = Field(
        default_factory=list, description="Document sections with structure"
    )
    content_type: Optional[str] = Field(
        None, description="Type of content (blog, docs, guide, etc.)"
    )

    # Keywords and topics
    extracted_keywords: List[str] = Field(
        default_factory=list, description="Keywords extracted from content"
    )
    topics: List[str] = Field(
        default_factory=list, description="Main topics/themes identified"
    )
    categories: List[str] = Field(
        default_factory=list, description="Content categories"
    )

    # Internal linking information
    internal_links_found: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Internal links found in this document (anchor_text, target_url)",
    )
    anchor_text_patterns: List[str] = Field(
        default_factory=list, description="Anchor text patterns used in this document"
    )
    outbound_link_count: int = Field(0, description="Number of internal links found")

    # SEO and metadata
    meta_description: Optional[str] = Field(
        None, description="Meta description if available"
    )
    meta_keywords: Optional[List[str]] = Field(
        None, description="Meta keywords if available"
    )
    canonical_url: Optional[str] = Field(None, description="Canonical URL if specified")

    # Additional metadata
    author: Optional[str] = Field(None, description="Document author if available")
    last_modified: Optional[datetime] = Field(
        None, description="Last modified date from document"
    )
    language: Optional[str] = Field(None, description="Content language")
    reading_time_minutes: Optional[float] = Field(
        None, description="Estimated reading time"
    )

    # Quality metrics
    readability_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Readability score"
    )
    completeness_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Content completeness score"
    )


class ScannedDocumentDB(BaseModel):
    """Database model for scanned internal documents with full metadata."""

    id: Optional[int] = Field(None, description="Database ID (auto-generated)")
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Document URL (unique identifier)")
    scanned_at: datetime = Field(
        default_factory=datetime.utcnow, description="When document was scanned"
    )
    last_scanned_at: Optional[datetime] = Field(
        None, description="Last time document was re-scanned"
    )

    # Rich metadata
    metadata: ScannedDocumentMetadata = Field(
        default_factory=ScannedDocumentMetadata,
        description="Rich metadata extracted from the document",
    )

    # Status and tracking
    is_active: bool = Field(
        True, description="Whether this document is currently active"
    )
    scan_count: int = Field(
        1, description="Number of times this document has been scanned"
    )
    relevance_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Relevance score (0-1)"
    )

    # Relationships
    related_documents: List[str] = Field(
        default_factory=list,
        description="URLs of related documents (based on content similarity)",
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
