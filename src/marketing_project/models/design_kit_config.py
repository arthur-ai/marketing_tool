"""
Design Kit Configuration Model.

This model defines the comprehensive structure for design kit configuration
that is used across pipeline runs. It includes visual design, voice & tone,
structure patterns, SEO patterns, CTA patterns, compliance rules, and more.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContentTypeConfig(BaseModel):
    """Static configuration for a specific content type."""

    section_order: Optional[List[str]] = Field(
        default=None, description="Content-type-specific section order override"
    )
    word_count_range: Optional[Dict[str, int]] = Field(
        default=None, description="Content-type-specific word count range override"
    )
    paragraph_length_range: Optional[Dict[str, int]] = Field(
        default=None,
        description="Content-type-specific paragraph length range override",
    )
    heading_depth: Optional[str] = Field(
        default=None, description="Content-type-specific heading depth override"
    )
    include_tldr: Optional[bool] = Field(
        default=None, description="Content-type-specific TL;DR inclusion override"
    )
    include_summary: Optional[bool] = Field(
        default=None, description="Content-type-specific summary inclusion override"
    )


class DesignKitConfig(BaseModel):
    """Comprehensive configuration for design kit and brand guidelines."""

    # Visual Design (existing)
    visual_components: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Visual elements to add (images, charts, infographics)",
    )
    color_scheme: Optional[Dict[str, str]] = Field(
        default=None, description="Recommended color palette with hex codes"
    )
    typography: Optional[Dict[str, str]] = Field(
        default=None, description="Typography recommendations (fonts, sizes)"
    )
    layout_suggestions: Optional[List[str]] = Field(
        default=None, description="Layout and spacing recommendations"
    )
    hero_image_concept: Optional[str] = Field(
        default=None, description="Hero image concept or description"
    )
    accessibility_notes: Optional[List[str]] = Field(
        default=None, description="Accessibility recommendations"
    )

    # Voice & Tone (new)
    voice_adjectives: Optional[List[str]] = Field(
        default=None, description="Voice adjectives (e.g., ['confident', 'practical'])"
    )
    point_of_view: Optional[str] = Field(
        default=None, description="Point of view: we/you/neutral"
    )
    sentence_length_tempo: Optional[str] = Field(
        default=None,
        description="Sentence length and tempo (short/medium/long, fast/medium/slow)",
    )
    lexical_preferences: Optional[List[str]] = Field(
        default=None,
        description="Preferred lexical terms (e.g., ['guardrails', 'observability', 'drift'])",
    )

    # Structure (new)
    section_order: Optional[List[str]] = Field(
        default=None,
        description="Common section order (e.g., ['intro', 'problem', 'approach', 'proof', 'cta'])",
    )
    heading_depth: Optional[str] = Field(
        default=None, description="Heading depth preference (H2/H3/H4)"
    )
    list_usage_preference: Optional[str] = Field(
        default=None, description="List usage preference (frequent/moderate/minimal)"
    )
    paragraph_length_range: Optional[Dict[str, int]] = Field(
        default=None, description="Paragraph length range in words (min/max)"
    )
    include_tldr: Optional[bool] = Field(
        default=None, description="Whether to include TL;DR section"
    )
    include_summary: Optional[bool] = Field(
        default=None, description="Whether to include summary section"
    )

    # SEO Patterns (new)
    title_format: Optional[str] = Field(
        default=None, description="Title format template/pattern"
    )
    meta_description_style: Optional[str] = Field(
        default=None,
        description="Meta description style (descriptive/action-oriented/etc)",
    )
    slug_casing: Optional[str] = Field(
        default=None,
        description="Slug casing convention (kebab-case/snake_case/camelCase)",
    )
    tag_conventions: Optional[List[str]] = Field(
        default=None, description="Tagging patterns and conventions"
    )
    internal_link_anchor_style: Optional[str] = Field(
        default=None,
        description="Internal link anchor style (exact-match/natural-language/etc)",
    )
    external_citation_style: Optional[str] = Field(
        default=None, description="Format for citing external sources"
    )

    # CTA Patterns (new)
    cta_language: Optional[List[str]] = Field(
        default=None, description="Common CTA phrases"
    )
    cta_positions: Optional[List[str]] = Field(
        default=None, description="Where CTAs typically appear"
    )
    cta_verbs: Optional[List[str]] = Field(
        default=None, description="Action verbs used in CTAs"
    )
    typical_link_targets: Optional[List[str]] = Field(
        default=None, description="Common link destinations"
    )

    # Compliance & Brand (new)
    must_use_names_terms: Optional[List[str]] = Field(
        default=None, description="Required terminology that must be used"
    )
    prohibited_phrases: Optional[List[str]] = Field(
        default=None, description="Forbidden terms/phrases"
    )
    disclaimer_boilerplate: Optional[str] = Field(
        default=None, description="Disclaimer boilerplate text"
    )
    date_format: Optional[str] = Field(
        default=None, description="Date format (e.g., 'YYYY-MM-DD', 'Month DD, YYYY')"
    )
    numbers_formatting_rules: Optional[Dict[str, str]] = Field(
        default=None,
        description="Number formatting rules (e.g., {'percentage': '60%', 'currency': '$1.2M'})",
    )

    # Interlinking Rules (new - pulled from Internal Docs)
    commonly_referenced_pages: Optional[List[str]] = Field(
        default=None,
        description="Commonly referenced page slugs/URLs (from internal_docs_config)",
    )
    commonly_referenced_categories: Optional[List[str]] = Field(
        default=None,
        description="Commonly referenced categories (from internal_docs_config)",
    )
    anchor_phrasing_patterns: Optional[List[str]] = Field(
        default=None, description="Anchor phrasing patterns (from internal_docs_config)"
    )

    # Attribution (new)
    author_name_style: Optional[str] = Field(
        default=None, description="Format for author names"
    )
    bio_length_range: Optional[Dict[str, int]] = Field(
        default=None, description="Author bio length range in words (min/max)"
    )
    sign_off_patterns: Optional[List[str]] = Field(
        default=None, description="Common sign-off patterns"
    )

    # Quant/Targets (new)
    word_count_range: Optional[Dict[str, int]] = Field(
        default=None, description="Typical word count range (min/max)"
    )
    heading_density: Optional[str] = Field(
        default=None, description="Heading density preference (low/medium/high)"
    )
    keyword_density_band: Optional[str] = Field(
        default=None, description="Keyword density band (low/medium/high)"
    )

    # Reusable Snippets (new)
    opening_lines: Optional[List[str]] = Field(
        default=None, description="Common opening phrases"
    )
    transition_sentences: Optional[List[str]] = Field(
        default=None, description="Transition phrases"
    )
    proof_statements: Optional[List[str]] = Field(
        default=None, description="Proof/evidence phrases"
    )
    conclusion_frames: Optional[List[str]] = Field(
        default=None, description="Conclusion templates"
    )
    common_faqs: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Common FAQ question/answer pairs"
    )

    # Content Type Variations (static configs)
    blog_config: Optional[ContentTypeConfig] = Field(
        None, description="Blog post specific configuration overrides"
    )
    press_config: Optional[ContentTypeConfig] = Field(
        None, description="Press release specific configuration overrides"
    )
    case_config: Optional[ContentTypeConfig] = Field(
        None, description="Case study specific configuration overrides"
    )

    # Metadata
    version: str = Field(description="Version identifier for this configuration")
    created_at: datetime = Field(description="Timestamp when configuration was created")
    updated_at: datetime = Field(
        description="Timestamp when configuration was last updated"
    )
    is_active: bool = Field(
        description="Whether this configuration version is currently active"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def get_content_type_config(self, content_type: str) -> Dict[str, Any]:
        """
        Get content-type-specific configuration, merging with global defaults.

        Args:
            content_type: Content type (blog_post, press_release, case_study)

        Returns:
            Merged configuration dict for the content type
        """
        base_config = self.model_dump(
            exclude={
                "blog_config",
                "press_config",
                "case_config",
                "version",
                "created_at",
                "updated_at",
                "is_active",
            }
        )

        # Get the appropriate content type config
        type_config_obj = None
        if content_type == "blog_post" and self.blog_config:
            type_config_obj = self.blog_config
        elif content_type == "press_release" and self.press_config:
            type_config_obj = self.press_config
        elif content_type == "case_study" and self.case_config:
            type_config_obj = self.case_config

        # Merge content-type-specific overrides (only non-None values)
        if type_config_obj:
            type_config = type_config_obj.model_dump(
                exclude_unset=True, exclude_none=True
            )
            for key, value in type_config.items():
                if key in base_config:
                    base_config[key] = value

        return base_config
