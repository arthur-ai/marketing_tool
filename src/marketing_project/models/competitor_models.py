"""
Models for competitor research feature.

This module defines models for analyzing competitors' blogs and social media
posts to understand why their content performs well.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ContentStructureAnalysis(BaseModel):
    """Analysis of content structure (headings, length, format)."""

    word_count: Optional[int] = Field(None, description="Total word count")
    heading_count: Optional[int] = Field(None, description="Number of headings (H1-H3)")
    paragraph_count: Optional[int] = Field(None, description="Number of paragraphs")
    has_images: Optional[bool] = Field(
        None, description="Whether content includes images"
    )
    has_video: Optional[bool] = Field(
        None, description="Whether content includes video"
    )
    has_lists: Optional[bool] = Field(
        None, description="Whether content includes bullet/numbered lists"
    )
    has_code_blocks: Optional[bool] = Field(
        None, description="Whether content includes code examples"
    )
    content_format: Optional[str] = Field(
        None,
        description="Primary format: article, listicle, how-to, case-study, news, opinion",
    )
    estimated_read_time_minutes: Optional[float] = Field(
        None, description="Estimated reading time"
    )


class SEOSignals(BaseModel):
    """SEO signals detected in the content."""

    primary_keyword: Optional[str] = Field(None, description="Main keyword targeted")
    keyword_in_title: Optional[bool] = Field(
        None, description="Primary keyword in title"
    )
    keyword_in_first_paragraph: Optional[bool] = Field(
        None, description="Keyword in intro paragraph"
    )
    meta_description_quality: Optional[str] = Field(
        None, description="Quality assessment: excellent, good, fair, poor, missing"
    )
    internal_links_count: Optional[int] = Field(
        None, description="Number of internal links"
    )
    external_links_count: Optional[int] = Field(
        None, description="Number of external links"
    )
    schema_markup_detected: Optional[bool] = Field(
        None, description="Whether schema markup is present"
    )
    url_structure_quality: Optional[str] = Field(
        None, description="URL structure assessment"
    )


class EngagementSignals(BaseModel):
    """Engagement and performance signals."""

    estimated_shares: Optional[int] = Field(
        None, description="Estimated social shares if available"
    )
    estimated_comments: Optional[int] = Field(
        None, description="Estimated comment count if available"
    )
    call_to_action_present: Optional[bool] = Field(
        None, description="Whether CTAs are present"
    )
    call_to_action_quality: Optional[str] = Field(
        None, description="CTA quality: strong, moderate, weak, none"
    )
    emotional_hooks: Optional[List[str]] = Field(
        None, description="Emotional triggers used (curiosity, fear, aspiration, etc.)"
    )
    social_proof_present: Optional[bool] = Field(
        None, description="Testimonials, stats, case studies"
    )
    personalization_level: Optional[str] = Field(
        None, description="How personalized: high, medium, low"
    )


class SocialMediaSignals(BaseModel):
    """Social-media-specific performance signals (for social posts)."""

    platform: Optional[str] = Field(
        None, description="Platform: linkedin, twitter, instagram, facebook"
    )
    hashtag_count: Optional[int] = Field(None, description="Number of hashtags used")
    hashtags: Optional[List[str]] = Field(None, description="Hashtags used")
    hook_strength: Optional[str] = Field(
        None, description="Opening hook strength: very_strong, strong, moderate, weak"
    )
    hook_text: Optional[str] = Field(None, description="First line or hook of the post")
    media_type: Optional[str] = Field(
        None, description="Media used: none, image, video, carousel, poll, document"
    )
    post_length_category: Optional[str] = Field(
        None,
        description="Length category: short (<100w), medium (100-300w), long (>300w)",
    )
    engagement_format: Optional[str] = Field(
        None,
        description="Format: question, story, tip, list, announcement, behind-scenes, thought-leadership",
    )


class ContentStrengthFactor(BaseModel):
    """A specific reason why this content performs well."""

    factor: str = Field(..., description="Name of the strength factor")
    description: str = Field(..., description="Detailed explanation of why this works")
    impact: Literal["high", "medium", "low"] = Field(
        ..., description="Estimated impact on performance"
    )


class ContentWeaknessFactor(BaseModel):
    """A specific weakness or opportunity gap in the content."""

    factor: str = Field(..., description="Name of the weakness/gap")
    description: str = Field(..., description="Explanation of the weakness")
    opportunity: str = Field(
        ..., description="How you can do better than this competitor"
    )


class CompetitorContentAnalysis(BaseModel):
    """
    Full analysis of a single piece of competitor content.

    Captures structure, SEO, engagement, and actionable insights
    about why the content performs well or poorly.
    """

    url: Optional[str] = Field(None, description="URL of the analyzed content")
    title: Optional[str] = Field(None, description="Title of the content")
    content_type: Literal["blog", "social_media", "unknown"] = Field(
        default="unknown", description="Type of content"
    )
    platform: Optional[str] = Field(
        None,
        description="Platform (for social media): linkedin, twitter, instagram, etc.",
    )
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Content characteristics
    structure: Optional[ContentStructureAnalysis] = None
    seo_signals: Optional[SEOSignals] = None
    engagement_signals: Optional[EngagementSignals] = None
    social_signals: Optional[SocialMediaSignals] = None

    # Performance assessment
    overall_quality_score: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="Overall quality score out of 10"
    )
    performance_tier: Optional[
        Literal["top", "above_average", "average", "below_average"]
    ] = Field(None, description="Estimated performance tier")

    # What makes it work / where it falls short
    strength_factors: List[ContentStrengthFactor] = Field(
        default_factory=list, description="Why this content performs well"
    )
    weakness_factors: List[ContentWeaknessFactor] = Field(
        default_factory=list, description="Gaps and opportunities"
    )

    # Strategic takeaways
    key_topics_covered: List[str] = Field(
        default_factory=list, description="Main topics or themes covered"
    )
    tone_and_voice: Optional[str] = Field(
        None,
        description="Writing tone: professional, conversational, technical, inspirational, etc.",
    )
    target_audience: Optional[str] = Field(None, description="Inferred target audience")
    unique_angle: Optional[str] = Field(
        None, description="What unique perspective or angle this content takes"
    )
    actionable_insights: List[str] = Field(
        default_factory=list,
        description="Specific things you can apply to your own content strategy",
    )

    # Raw content snapshot (truncated for storage)
    content_snippet: Optional[str] = Field(
        None, max_length=2000, description="First 2000 chars of the content"
    )


class CompetitorResearchRequest(BaseModel):
    """Request to analyze competitor content."""

    competitor_urls: Optional[List[str]] = Field(
        None,
        description="List of competitor blog post or social media post URLs to analyze",
    )
    competitor_content: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Raw content to analyze. Each item: {title, content, url, platform, content_type}",
    )
    content_type: Literal["blog", "social_media", "both"] = Field(
        default="both", description="Type of competitor content to focus on"
    )
    your_niche: Optional[str] = Field(
        None, description="Your niche or industry for more targeted analysis"
    )
    your_content_goals: Optional[str] = Field(
        None,
        description="What you're trying to achieve (e.g., 'increase organic traffic', 'build brand awareness')",
    )
    focus_platforms: Optional[List[str]] = Field(
        None,
        description="Social media platforms to focus on: linkedin, twitter, instagram, facebook",
    )


class CompetitorResearchSummary(BaseModel):
    """Cross-competitor summary of patterns and insights."""

    top_content_patterns: List[str] = Field(
        default_factory=list,
        description="Common patterns seen across top-performing competitor content",
    )
    winning_content_formats: List[str] = Field(
        default_factory=list, description="Content formats that appear most effective"
    )
    common_topics: List[str] = Field(
        default_factory=list, description="Topics competitors frequently cover"
    )
    content_gaps: List[str] = Field(
        default_factory=list,
        description="Topics or angles competitors are missing that you could own",
    )
    seo_opportunities: List[str] = Field(
        default_factory=list,
        description="SEO gaps or opportunities identified from competitor analysis",
    )
    social_media_tactics: List[str] = Field(
        default_factory=list,
        description="Effective social media tactics competitors are using",
    )
    recommended_content_strategy: Optional[str] = Field(
        None, description="High-level recommended strategy based on competitor research"
    )
    quick_wins: List[str] = Field(
        default_factory=list,
        description="Immediate actions to take based on competitor insights",
    )


class CompetitorResearchResult(BaseModel):
    """Full result of a competitor research job."""

    job_id: str = Field(..., description="ID of the research job")
    status: Literal["pending", "processing", "completed", "failed"] = Field(
        default="pending"
    )
    request: CompetitorResearchRequest
    analyses: List[CompetitorContentAnalysis] = Field(
        default_factory=list, description="Individual content analyses"
    )
    summary: Optional[CompetitorResearchSummary] = Field(
        None, description="Cross-competitor summary and strategic insights"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class CompetitorResearchListItem(BaseModel):
    """Lightweight list view of a competitor research job."""

    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    content_type: str
    competitor_count: int
    your_niche: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime] = None
