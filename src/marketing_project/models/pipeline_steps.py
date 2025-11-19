"""
Pydantic models for each step in the content pipeline.

These models define the structured output expected from each pipeline step,
ensuring type safety and predictable results.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class KeywordMetadata(BaseModel):
    """Metadata for individual keywords with search metrics."""

    keyword: str = Field(description="The keyword")
    search_volume: Optional[int] = Field(
        None, description="Estimated monthly search volume"
    )
    difficulty_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Keyword difficulty score (0-100, higher is more difficult)",
    )
    cpc_estimate: Optional[float] = Field(
        None, description="Estimated cost per click (CPC)"
    )
    trend_direction: Optional[str] = Field(
        None, description="Trend direction: 'rising', 'stable', or 'declining'"
    )
    trend_percentage: Optional[float] = Field(
        None, description="Percentage change in trend"
    )
    relevance_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Relevance score for this keyword (0-100)"
    )


class KeywordDensityAnalysis(BaseModel):
    """Detailed keyword density analysis with placement recommendations."""

    keyword: str = Field(description="The keyword being analyzed")
    current_density: float = Field(
        description="Current keyword density percentage in content"
    )
    optimal_density: float = Field(description="Recommended optimal density percentage")
    occurrences: int = Field(description="Number of times keyword appears in content")
    placement_locations: List[str] = Field(
        description="Locations where keyword appears (e.g., 'title', 'h1', 'first_paragraph')"
    )


class KeywordCluster(BaseModel):
    """Grouped related keywords by topic cluster."""

    cluster_name: str = Field(description="Name or theme of the keyword cluster")
    keywords: List[str] = Field(description="Keywords in this cluster")
    primary_keyword: Optional[str] = Field(
        None, description="Primary keyword for this cluster"
    )
    topic_theme: Optional[str] = Field(
        None, description="Main topic or theme of this cluster"
    )


class SEOKeywordsResult(BaseModel):
    """Result from SEO Keywords extraction step."""

    main_keyword: str = Field(
        description="The single most important keyword for this content - the primary focus"
    )
    primary_keywords: List[str] = Field(
        description="Main keywords for the content (3-5 keywords, including main_keyword)"
    )
    secondary_keywords: Optional[List[str]] = Field(
        description="Supporting keywords (5-10 keywords)", default=None
    )
    lsi_keywords: Optional[List[str]] = Field(
        description="Latent Semantic Indexing keywords for context", default=None
    )
    long_tail_keywords: Optional[List[str]] = Field(
        description="Long-tail keyword opportunities (5-8 keywords)", default=None
    )
    keyword_density: Optional[Dict[str, float]] = Field(
        description="Keyword frequency analysis (deprecated: use keyword_density_analysis instead)",
        default=None,
    )
    keyword_density_analysis: Optional[List[KeywordDensityAnalysis]] = Field(
        description="Detailed keyword density analysis with placement recommendations",
        default=None,
    )
    search_intent: str = Field(
        description="User search intent (informational/transactional/navigational/commercial)"
    )
    keyword_difficulty: Optional[Dict[str, float]] = Field(
        description="Keyword difficulty scores per keyword (0-100 scale)", default=None
    )

    # Keyword metadata
    primary_keywords_metadata: Optional[List[KeywordMetadata]] = Field(
        description="Metadata for primary keywords (search volume, difficulty, trends)",
        default=None,
    )
    secondary_keywords_metadata: Optional[List[KeywordMetadata]] = Field(
        description="Metadata for secondary keywords", default=None
    )
    long_tail_keywords_metadata: Optional[List[KeywordMetadata]] = Field(
        description="Metadata for long-tail keywords", default=None
    )

    # Keyword clustering and analysis
    keyword_clusters: Optional[List[KeywordCluster]] = Field(
        description="Grouped related keywords by topic cluster", default=None
    )
    search_volume_summary: Optional[Dict[str, int]] = Field(
        description="Total estimated monthly search volume by keyword category",
        default=None,
    )
    optimization_recommendations: Optional[List[str]] = Field(
        description="Specific actions to improve keyword performance", default=None
    )

    # Quality and confidence metrics
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model's confidence in keyword analysis quality (0-1)",
    )
    relevance_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Overall keyword relevance score (0-100)"
    )


class MarketingBriefResult(BaseModel):
    """Result from Marketing Brief generation step."""

    target_audience: List[str] = Field(
        description="Target audience personas with demographics"
    )
    key_messages: List[str] = Field(description="Core messaging points (3-5 messages)")
    content_strategy: str = Field(
        description="Recommended content strategy and approach"
    )
    kpis: Optional[List[str]] = Field(
        description="Key performance indicators to track", default=None
    )
    distribution_channels: Optional[List[str]] = Field(
        description="Recommended distribution channels", default=None
    )
    tone_and_voice: Optional[str] = Field(
        description="Recommended tone and voice for the content", default=None
    )
    competitive_angle: Optional[str] = Field(
        description="Unique competitive positioning", default=None
    )

    # Quality and confidence metrics
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model's confidence in marketing brief quality (0-1)",
    )
    strategy_alignment_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="How well strategy aligns with business objectives (0-100)",
    )


class ArticleGenerationResult(BaseModel):
    """Result from Article generation step."""

    article_title: str = Field(
        description="Generated article title optimized for clicks and engagement"
    )
    article_content: str = Field(
        description="Generated article content created from scratch based on marketing brief"
    )
    outline: List[str] = Field(
        description="Article structure outline with section headers"
    )
    call_to_action: str = Field(description="Recommended call-to-action")
    hook: Optional[str] = Field(
        description="Opening hook to capture attention", default=None
    )
    key_takeaways: Optional[List[str]] = Field(
        description="Main takeaways from the content", default=None
    )
    word_count: Optional[int] = Field(
        description="Word count of generated article content", default=None
    )

    # Quality and confidence metrics
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Model's confidence in article quality (0-1)"
    )
    readability_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Flesch-Kincaid readability score (0-100, higher is easier)",
    )
    engagement_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Predicted engagement potential (0-100)"
    )


class HeaderStructure(BaseModel):
    """Header structure analysis with H1-H3 hierarchy."""

    model_config = ConfigDict(extra="forbid")

    h1_count: Optional[int] = Field(None, description="Number of H1 headings found")
    h2_count: Optional[int] = Field(None, description="Number of H2 headings found")
    h3_count: Optional[int] = Field(None, description="Number of H3 headings found")
    h1_headings: Optional[List[str]] = Field(
        None, description="List of H1 heading text"
    )
    h2_headings: Optional[List[str]] = Field(
        None, description="List of H2 heading text"
    )
    h3_headings: Optional[List[str]] = Field(
        None, description="List of H3 heading text"
    )
    hierarchy_valid: Optional[bool] = Field(
        None,
        description="Whether the header hierarchy is valid (single H1, proper nesting)",
    )
    validation_notes: Optional[List[str]] = Field(
        None, description="Notes about header structure validation"
    )


class KeywordPlacement(BaseModel):
    """Keyword placement information."""

    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(description="The keyword")
    locations: Optional[List[str]] = Field(
        None,
        description="Locations where the keyword appears (e.g., 'title', 'intro', 'h2-1')",
    )
    count: Optional[int] = Field(
        None, description="Number of times the keyword appears"
    )


class KeywordMap(BaseModel):
    """Primary and related keywords with placement locations."""

    model_config = ConfigDict(extra="forbid")

    primary_keyword: Optional[str] = Field(None, description="The primary keyword")
    related_keywords: Optional[List[str]] = Field(None, description="Related keywords")
    placements: Optional[List[KeywordPlacement]] = Field(
        None, description="Detailed placement information for each keyword"
    )


class ReadabilityOptimization(BaseModel):
    """Readability analysis results."""

    model_config = ConfigDict(extra="forbid")

    score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Readability score (0-100, higher is easier)",
    )
    grade_level: Optional[float] = Field(None, description="Flesch-Kincaid grade level")
    active_voice_percentage: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Percentage of sentences using active voice (0-100)",
    )
    average_sentence_length: Optional[float] = Field(
        None, description="Average number of words per sentence"
    )
    recommendations: Optional[List[str]] = Field(
        None, description="Readability improvement recommendations"
    )


class SEOOptimizationResult(BaseModel):
    """Result from SEO Optimization step."""

    model_config = ConfigDict(extra="forbid")

    optimized_content: str = Field(
        description="SEO-optimized version of the full article content (complete, not truncated)"
    )
    meta_title: str = Field(description="SEO meta title (50-60 characters)")
    meta_description: str = Field(
        description="SEO meta description (150-160 characters)"
    )
    slug: str = Field(description="URL-friendly slug")
    alt_texts: Optional[Dict[str, str]] = Field(
        description="Alt text suggestions for images", default=None
    )
    schema_markup: Optional[str] = Field(
        description="JSON-LD schema markup as JSON string", default=None
    )
    canonical_url: Optional[str] = Field(
        description="Recommended canonical URL", default=None
    )
    og_tags: Optional[Dict[str, str]] = Field(
        description="Open Graph tags for social sharing", default=None
    )

    # Quality and confidence metrics
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model's confidence in SEO optimization quality (0-1)",
    )
    seo_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Overall SEO optimization score (0-100)"
    )
    keyword_optimization_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Keyword integration and density score (0-100)",
    )

    # SEO Workflow Analysis Fields
    header_structure: Optional[HeaderStructure] = Field(
        None, description="H1-H3 hierarchy analysis with validation results"
    )
    keyword_map: Optional[KeywordMap] = Field(
        None,
        description="Primary and related keywords with placement locations in content",
    )
    readability_optimization: Optional[ReadabilityOptimization] = Field(
        None,
        description="Readability analysis including score, grade level, and active voice percentage",
    )
    modification_report: Optional[List[str]] = Field(
        None,
        description="Summary of changes made during SEO optimization (e.g., 'Meta description regenerated', 'H2 hierarchy corrected')",
    )


class InternalLinkSuggestion(BaseModel):
    """A specific internal link suggestion with placement context."""

    anchor_text: str = Field(description="The anchor text to use for the link")
    target_url: str = Field(description="The URL/path of the target internal page")
    target_title: Optional[str] = Field(
        None, description="Title of the target page (if known)"
    )
    placement_context: str = Field(
        description="Context where this link should be placed (e.g., section name, paragraph snippet)"
    )
    section: Optional[str] = Field(
        None, description="Article section where this link should be added"
    )
    relevance_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Relevance score for this specific link (0-1)"
    )
    reasoning: Optional[str] = Field(
        None, description="Brief explanation of why this link is suggested"
    )


class SuggestedLinksResult(BaseModel):
    """Result from Suggested Links step.

    This step suggests where to add internal links in the generated article,
    using the InternalDocsConfig as a reference for available pages.
    """

    internal_links: List[InternalLinkSuggestion] = Field(
        description="Specific internal link suggestions with placement context",
        default_factory=list,
    )

    # Summary metrics
    total_suggestions: int = Field(
        description="Total number of link suggestions", default=0
    )
    high_confidence_links: int = Field(
        description="Number of high-confidence link suggestions (relevance > 0.7)",
        default=0,
    )
    matched_pages: Optional[List[str]] = Field(
        None, description="List of pages from InternalDocsConfig that were matched"
    )
    matched_categories: Optional[List[str]] = Field(
        None, description="List of categories from InternalDocsConfig that were matched"
    )

    # Quality metrics
    average_relevance: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Average relevance score across all suggestions (0-1)",
    )


class ContentFormattingResult(BaseModel):
    """Result from Content Formatting step."""

    formatted_html: str = Field(
        description="HTML-formatted content ready for publication"
    )
    formatted_markdown: str = Field(description="Markdown-formatted content")
    sections: Optional[List[Dict[str, str]]] = Field(
        description="Content sections with headings and content", default=None
    )
    reading_time: Optional[int] = Field(
        description="Estimated reading time in minutes", default=None
    )
    table_of_contents: Optional[List[Dict[str, str]]] = Field(
        description="Table of contents with anchors", default=None
    )
    formatting_notes: Optional[List[str]] = Field(
        description="Formatting recommendations or notes", default=None
    )

    # Quality and confidence metrics
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model's confidence in formatting quality (0-1)",
    )
    accessibility_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Accessibility compliance score (0-100)"
    )
    formatting_quality_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Overall formatting quality (0-100)"
    )


class DesignKitResult(BaseModel):
    """Result from Design Kit application step."""

    visual_components: Optional[List[Dict[str, str]]] = Field(
        description="Visual elements to add (images, charts, infographics)",
        default=None,
    )
    color_scheme: Optional[Dict[str, str]] = Field(
        description="Recommended color palette with hex codes", default=None
    )
    typography: Optional[Dict[str, str]] = Field(
        description="Typography recommendations (fonts, sizes)", default=None
    )
    layout_suggestions: Optional[List[str]] = Field(
        description="Layout and spacing recommendations", default=None
    )
    hero_image_concept: Optional[str] = Field(
        description="Hero image concept or description", default=None
    )
    accessibility_notes: Optional[List[str]] = Field(
        description="Accessibility recommendations", default=None
    )

    # Quality and confidence metrics
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model's confidence in design recommendations (0-1)",
    )
    design_quality_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Overall design quality assessment (0-100)"
    )
    brand_consistency_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Brand consistency score (0-100)"
    )


class PipelineResult(BaseModel):
    """Complete pipeline result containing all step outputs."""

    pipeline_status: str = Field(
        description="Status: completed, completed_with_warnings, or failed"
    )
    step_results: Optional[Dict[str, Any]] = Field(
        description="Results from each pipeline step", default=None
    )
    quality_warnings: Optional[List[str]] = Field(
        description="Quality issues or warnings identified", default=None
    )
    final_content: str = Field(description="Final processed and formatted content")
    metadata: Optional[Dict[str, Any]] = Field(
        description="Pipeline execution metadata", default=None
    )
    execution_time_seconds: Optional[float] = Field(
        description="Total pipeline execution time", default=None
    )


class PipelineStepInfo(BaseModel):
    """Information about a pipeline step execution."""

    step_name: str = Field(description="Name of the pipeline step")
    step_number: int = Field(description="Step sequence number")
    status: str = Field(description="Step status (success/failed/skipped)")
    execution_time: Optional[float] = Field(
        description="Step execution time in seconds", default=None
    )
    error_message: Optional[str] = Field(
        description="Error message if step failed", default=None
    )
    tokens_used: Optional[int] = Field(
        description="Tokens consumed by this step", default=None
    )


class SocialMediaMarketingBriefResult(BaseModel):
    """Result from Social Media Marketing Brief generation step."""

    platform: str = Field(
        description="Social media platform: linkedin, hackernews, or email"
    )
    target_audience: List[str] = Field(
        description="Target audience personas with demographics for this platform"
    )
    key_messages: List[str] = Field(description="Core messaging points (3-5 messages)")
    tone_and_voice: str = Field(
        description="Recommended tone and voice for this platform"
    )
    content_strategy: str = Field(
        description="Platform-specific content strategy and approach"
    )
    distribution_strategy: str = Field(
        description="Platform-specific distribution and engagement strategy"
    )
    platform_specific_notes: Optional[Dict[str, Any]] = Field(
        description="Platform-specific recommendations and notes", default=None
    )

    # Quality and confidence metrics
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model's confidence in marketing brief quality (0-1)",
    )


class AngleHookResult(BaseModel):
    """Result from Angle & Hook Generation step."""

    platform: str = Field(
        description="Social media platform: linkedin, hackernews, or email"
    )
    angles: List[str] = Field(
        description="Multiple angle options for approaching the content (3-5 angles)"
    )
    hooks: List[str] = Field(
        description="Hook variations for capturing attention (3-5 hooks)"
    )
    recommended_angle: str = Field(
        description="The recommended angle to use for this platform"
    )
    recommended_hook: str = Field(
        description="The recommended hook to use for this platform"
    )
    rationale: str = Field(
        description="Explanation of why the recommended angle and hook were chosen"
    )

    # Quality and confidence metrics
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model's confidence in angle/hook quality (0-1)",
    )


class SocialMediaPostResult(BaseModel):
    """Result from Social Media Post Generation step."""

    platform: str = Field(
        description="Social media platform: linkedin, hackernews, or email"
    )
    content: str = Field(description="Generated social media post content")
    subject_line: Optional[str] = Field(
        description="Email subject line (for email platform only)", default=None
    )
    hashtags: Optional[List[str]] = Field(
        description="Recommended hashtags (for LinkedIn platform only)", default=None
    )
    call_to_action: Optional[str] = Field(
        description="Recommended call-to-action", default=None
    )
    metadata: Optional[Dict[str, Any]] = Field(
        description="Additional platform-specific metadata", default=None
    )

    # Quality and confidence metrics
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model's confidence in post quality (0-1)",
    )
    engagement_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Predicted engagement potential (0-100)"
    )
