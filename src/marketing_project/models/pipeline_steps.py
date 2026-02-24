"""
Pydantic models for each step in the content pipeline.

These models define the structured output expected from each pipeline step,
ensuring type safety and predictable results.
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    @field_validator("schema_markup", mode="before")
    @classmethod
    def convert_schema_markup_to_string(cls, v: Any) -> Optional[str]:
        """
        Convert schema_markup from dict to JSON string if needed.

        The LLM may return schema_markup as a dict (JSON-LD object), but the model
        expects it as a JSON string. This validator handles the conversion.
        """
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return json.dumps(v)
        # If it's already a string or something else, try to convert it
        return str(v) if v else None


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


class BrandKitResult(BaseModel):
    """Result from Brand Kit application step."""

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


# Backward-compatibility alias
DesignKitResult = BrandKitResult


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
    # Platform-specific quality scores
    linkedin_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="LinkedIn-specific quality score (professional tone, thought leadership) (0-100)",
    )
    hackernews_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="HackerNews-specific quality score (technical depth, discussion potential) (0-100)",
    )
    email_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Email-specific quality score (open rate potential, engagement) (0-100)",
    )
    # Content variations
    variations: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Multiple variations of the post for A/B testing (max 3)",
    )
    variation_id: Optional[str] = Field(
        None,
        description="ID of the selected variation (if one was chosen)",
    )


class QualityMetrics(BaseModel):
    """Quality metrics for transcript parsing."""

    completeness: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Completeness score (0-1)",
    )
    speaker_clarity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Speaker clarity score (0-1)",
    )
    timestamp_accuracy: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Timestamp accuracy score (0-1)",
    )
    content_quality: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Content quality score (0-1)",
    )


class ConversationFlow(BaseModel):
    """Analysis of conversation flow structure."""

    flow_type: Optional[str] = Field(
        None,
        description="Type of conversation flow (e.g., 'interview', 'discussion', 'presentation')",
    )
    question_count: Optional[int] = Field(
        None,
        description="Number of questions detected",
    )
    answer_count: Optional[int] = Field(
        None,
        description="Number of answers detected",
    )
    patterns: Optional[List[str]] = Field(
        None,
        description="Detected conversation patterns",
    )


class TranscriptContentExtractionResult(BaseModel):
    """Result from transcript content extraction step."""

    extracted_content: str = Field(
        description="Main transcript content extracted from raw input"
    )
    content_validated: bool = Field(
        description="Content validation status - true if content is valid"
    )
    content_summary: Optional[str] = Field(
        None, description="Summary of content for review (first 500 chars)"
    )
    validation_issues: List[str] = Field(
        default_factory=list,
        description="List of content-specific validation issues found (empty if valid)",
    )
    quality_metrics: Optional[QualityMetrics] = Field(
        None,
        description="Detailed quality metrics from content extraction",
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in content extraction quality (0-1)",
    )
    transcript_type: Optional[str] = Field(
        None, description="Detected transcript type (podcast, meeting, interview, etc.)"
    )


class TranscriptSpeakersExtractionResult(BaseModel):
    """Result from transcript speakers extraction step."""

    speakers: List[str] = Field(
        default_factory=list,
        description="Extracted speaker names from transcript content",
    )
    speakers_validated: bool = Field(
        description="Speakers validation status - true if speakers list is valid"
    )
    validation_issues: List[str] = Field(
        default_factory=list,
        description="List of speaker-specific validation issues found (empty if valid)",
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in speakers extraction quality (0-1)",
    )


class TranscriptDurationExtractionResult(BaseModel):
    """Result from transcript duration extraction step."""

    duration: Optional[int] = Field(
        None, description="Duration in seconds extracted/calculated from transcript"
    )
    duration_validated: bool = Field(
        description="Duration validation status - true if duration is valid"
    )
    validation_issues: List[str] = Field(
        default_factory=list,
        description="List of duration-specific validation issues found (empty if valid)",
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in duration extraction quality (0-1)",
    )
    extraction_method: Optional[str] = Field(
        None,
        description="Method used to extract duration (timestamps, word_count, estimated, etc.)",
    )


class TranscriptPreprocessingApprovalResult(BaseModel):
    """Result from Transcript Preprocessing Approval step."""

    is_valid: bool = Field(
        description="Overall validation status - true if all transcript fields are valid"
    )
    speakers_validated: bool = Field(
        description="Speakers validation status - true if speakers list is valid"
    )
    duration_validated: bool = Field(
        description="Duration validation status - true if duration is valid"
    )
    content_validated: bool = Field(
        description="Content validation status - true if content is valid"
    )
    transcript_type_validated: bool = Field(
        description="Transcript type validation status - true if transcript_type is valid"
    )
    validation_issues: List[str] = Field(
        default_factory=list,
        description="List of validation issues found (empty if all valid)",
    )
    speakers: List[str] = Field(
        default_factory=list,
        description="Confirmed speaker list after validation",
    )
    duration: Optional[int] = Field(
        None, description="Confirmed duration in seconds after validation"
    )
    transcript_type: Optional[str] = Field(
        None, description="Confirmed transcript type after validation"
    )
    content_summary: Optional[str] = Field(
        None, description="Summary of content for review (first 500 chars)"
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in preprocessing quality (0-1)",
    )
    requires_approval: bool = Field(
        description="Whether human approval is required before proceeding"
    )
    approval_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for reviewer on what to check",
    )
    # Enhanced parsing fields
    parsing_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in parsing accuracy (0-1)",
    )
    detected_format: Optional[str] = Field(
        None,
        description="Detected transcript format (webvtt, srt, json, csv, ttml, plain)",
    )
    parsing_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings from parsing process",
    )
    quality_metrics: Optional[QualityMetrics] = Field(
        None,
        description="Detailed quality metrics from parsing",
    )
    speaking_time_per_speaker: Optional[Dict[str, int]] = Field(
        None,
        description="Speaking time in seconds per speaker",
    )
    detected_language: Optional[str] = Field(
        None,
        description="Detected language code (e.g., 'en')",
    )
    key_topics: List[str] = Field(
        default_factory=list,
        description="Key topics/themes extracted from content",
    )
    conversation_flow: Optional[ConversationFlow] = Field(
        None,
        description="Analysis of conversation flow structure",
    )


def _fix_anyof_additional_properties(schema: Dict[str, Any], model_class: type) -> None:
    """
    Fix anyOf schemas to include additionalProperties: false for OpenAI compatibility.

    OpenAI requires that when using anyOf in JSON schemas, each schema in the anyOf
    array must explicitly set additionalProperties: false.

    This fixes all Optional[Dict[str, Any]] fields in the schema.
    """
    if "properties" not in schema:
        return

    # Fix all fields that have anyOf with object types
    for field_name, field_schema in schema["properties"].items():
        # Check if the field has an anyOf
        if "anyOf" in field_schema:
            # Fix each schema in the anyOf array
            for anyof_schema in field_schema["anyOf"]:
                # Skip the null schema (for Optional types)
                if anyof_schema.get("type") == "null":
                    continue
                # For object/dict schemas, ensure additionalProperties is set to false
                if anyof_schema.get("type") == "object" or "properties" in anyof_schema:
                    anyof_schema["additionalProperties"] = False


class BlogPostPreprocessingApprovalResult(BaseModel):
    """Result from Blog Post Preprocessing Approval step."""

    is_valid: bool = Field(
        description="Overall validation status - true if all blog post fields are valid"
    )
    title_validated: bool = Field(
        description="Title validation status - true if title is valid"
    )
    content_validated: bool = Field(
        description="Content validation status - true if content is valid"
    )
    author_validated: bool = Field(
        description="Author validation status - true if author is valid or extracted"
    )
    category_validated: bool = Field(
        description="Category validation status - true if category is valid or extracted"
    )
    tags_validated: bool = Field(
        description="Tags validation status - true if tags are valid or extracted"
    )
    validation_issues: List[str] = Field(
        default_factory=list,
        description="List of validation issues found (empty if all valid)",
    )
    author: Optional[str] = Field(None, description="Extracted/confirmed author name")
    category: Optional[str] = Field(None, description="Extracted/confirmed category")
    tags: List[str] = Field(
        default_factory=list,
        description="Extracted/confirmed tags list",
    )
    word_count: Optional[int] = Field(None, description="Calculated word count")
    reading_time: Optional[int] = Field(
        None, description="Calculated reading time in minutes"
    )
    content_summary: Optional[str] = Field(
        None, description="Summary of content for review (first 500 chars)"
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in preprocessing quality (0-1)",
    )
    requires_approval: bool = Field(
        description="Whether human approval is required before proceeding"
    )
    approval_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for reviewer on what to check",
    )
    # Sentiment fields
    overall_sentiment: Optional[str] = Field(
        None,
        description="Overall sentiment (positive, negative, neutral)",
    )
    sentiment_score: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Sentiment score (-1.0 to 1.0, where -1 is very negative, 1 is very positive)",
    )
    sentiment_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in sentiment analysis (0-1)",
    )
    emotional_tone: Optional[str] = Field(
        None,
        description="Emotional tone (professional, casual, technical, friendly, etc.)",
    )
    sentiment_by_section: Optional[Dict[str, Any]] = Field(
        None,
        description="Sentiment analysis per section (if content has clear sections)",
    )
    # Content analysis fields
    readability_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Readability score (0-100, higher is more readable)",
    )
    completeness_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Completeness score (0-100, based on structure and content)",
    )
    content_type: Optional[str] = Field(
        None,
        description="Content type classification (tutorial, guide, article, opinion, news, etc.)",
    )
    target_audience: Optional[str] = Field(
        None,
        description="Target audience (beginner, intermediate, advanced, general)",
    )
    headings: List[str] = Field(
        default_factory=list,
        description="List of headings (H1, H2, H3) extracted from content",
    )
    sections: List[str] = Field(
        default_factory=list,
        description="List of section titles identified in content",
    )
    paragraph_count: Optional[int] = Field(
        None, description="Number of paragraphs in content"
    )
    list_count: Optional[int] = Field(
        None, description="Number of lists (bulleted or numbered) in content"
    )
    link_count: Optional[int] = Field(
        None, description="Number of links (internal and external) in content"
    )
    # SEO/Marketing fields
    potential_keywords: List[str] = Field(
        default_factory=list,
        description="Potential keywords extracted from content",
    )
    seo_opportunities: List[str] = Field(
        default_factory=list,
        description="SEO optimization opportunities identified",
    )
    engagement_potential: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Engagement potential score (0-100)",
    )
    shareability_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Shareability score (0-100)",
    )
    cta_detected: Optional[bool] = Field(
        None, description="Whether call-to-action is detected in content"
    )
    # Language/Topics
    detected_language: Optional[str] = Field(
        None,
        description="Detected language code (e.g., 'en')",
    )
    key_topics: List[str] = Field(
        default_factory=list,
        description="Key topics/themes extracted from content",
    )
    inferred_categories: List[str] = Field(
        default_factory=list,
        description="Categories inferred from content analysis",
    )
    # Optional parsing fields
    parsing_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in parsing accuracy (0-1)",
    )
    detected_format: Optional[str] = Field(
        None,
        description="Detected content format (html, markdown, plain, etc.)",
    )
    parsing_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings from parsing process",
    )
    quality_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed quality metrics from parsing/analysis",
    )
    content_structure: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed content structure analysis",
    )

    model_config = ConfigDict(json_schema_extra=_fix_anyof_additional_properties)


class StepInputOutput(BaseModel):
    """Input and output data for a single pipeline step."""

    step_name: str = Field(description="Name of the pipeline step")
    step_number: int = Field(description="Step sequence number")
    input_snapshot: Dict[str, Any] = Field(
        description="Snapshot of input context used for this step"
    )
    output: Dict[str, Any] = Field(description="Output generated by this step")
    context_keys_used: List[str] = Field(
        default_factory=list,
        description="List of context keys consumed by this step",
    )
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (timing, tokens, status, etc.)",
    )


class PipelineFlowResponse(BaseModel):
    """Complete pipeline flow visualization data."""

    job_id: str = Field(description="Job identifier")
    input_content: Dict[str, Any] = Field(
        description="Original input content to the pipeline"
    )
    steps: List[StepInputOutput] = Field(
        description="List of step results with inputs and outputs"
    )
    final_output: Dict[str, Any] = Field(
        description="Final processed content from the pipeline"
    )
    execution_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution summary (total time, tokens, etc.)",
    )


class EngineConfig(BaseModel):
    """Configuration for engine selection with field-level overrides."""

    default_engine: str = Field(
        default="llm",
        description="Default engine type to use (e.g., 'llm', 'local_semantic')",
    )
    field_overrides: Optional[Dict[str, str]] = Field(
        None,
        description="Optional field-level engine overrides mapping field names to engine types",
    )
    serp_analysis_model: Optional[str] = Field(
        None,
        description="Model to use for LLM-based SERP analysis (defaults to pipeline default if not set)",
    )


class PipelineStepConfig(BaseModel):
    """Configuration for a single pipeline step."""

    step_name: str = Field(description="Name of the pipeline step")
    model: Optional[str] = Field(
        None,
        description="OpenAI model to use for this step (e.g., 'gpt-4o-mini', 'gpt-4o', 'gpt-5.1', 'gpt-5.2')",
    )
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature for this step"
    )
    max_retries: Optional[int] = Field(
        None, ge=0, description="Maximum number of retries for this step"
    )
    seo_keywords_engine_config: Optional[EngineConfig] = Field(
        None,
        description="Engine configuration for SEO keywords step (field-level engine selection)",
    )


class PipelineConfig(BaseModel):
    """Configuration for the entire pipeline."""

    default_model: str = Field(
        default="gpt-5.1", description="Default OpenAI model for all steps"
    )
    default_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Default sampling temperature"
    )
    default_max_retries: int = Field(
        default=2, ge=0, description="Default maximum number of retries"
    )
    step_configs: Dict[str, PipelineStepConfig] = Field(
        default_factory=dict,
        description="Step-specific configurations keyed by step name",
    )
    seo_keywords_engine_config: Optional[EngineConfig] = Field(
        None,
        description="Engine configuration for SEO keywords step (field-level engine selection)",
    )

    def get_step_config(self, step_name: str) -> PipelineStepConfig:
        """
        Get configuration for a specific step, with defaults applied.

        Args:
            step_name: Name of the step

        Returns:
            PipelineStepConfig with step-specific values or defaults
        """
        step_config = self.step_configs.get(step_name)
        if step_config:
            return step_config

        # Return default config for this step
        return PipelineStepConfig(
            step_name=step_name,
            model=self.default_model,
            temperature=self.default_temperature,
            max_retries=self.default_max_retries,
        )

    def get_step_model(self, step_name: str) -> str:
        """Get model for a specific step."""
        return self.get_step_config(step_name).model or self.default_model

    def get_step_temperature(self, step_name: str) -> float:
        """Get temperature for a specific step."""
        step_config = self.get_step_config(step_name)
        return (
            step_config.temperature
            if step_config.temperature is not None
            else self.default_temperature
        )

    def get_step_max_retries(self, step_name: str) -> int:
        """Get max retries for a specific step."""
        step_config = self.get_step_config(step_name)
        return (
            step_config.max_retries
            if step_config.max_retries is not None
            else self.default_max_retries
        )
