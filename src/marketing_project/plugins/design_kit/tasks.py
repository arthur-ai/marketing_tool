"""
Design Kit plugin for Marketing Project.

This plugin generates DesignKitConfig (brand guidelines configuration) using AI/LLM.
It is not part of the content pipeline but is used to generate configuration.

The generation process:
1. Fetches all content from internal_docs (blog posts, articles, guides, documentation, etc.)
2. Analyzes each content piece individually to extract design kit patterns
3. Aggregates all individual analyses
4. Performs final LLM call to synthesize into one general design kit config
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from marketing_project.models.design_kit_config import DesignKitConfig
from marketing_project.services.function_pipeline import FunctionPipeline
from marketing_project.services.scanned_document_db import get_scanned_document_db

logger = logging.getLogger("marketing_project.plugins.design_kit")


class ContentAnalysis(BaseModel):
    """Analysis result from a single content piece (blog post, article, guide, etc.)."""

    voice_adjectives: List[str] = Field(default_factory=list)
    point_of_view: Optional[str] = Field(None)
    sentence_length_tempo: Optional[str] = Field(None)
    lexical_preferences: List[str] = Field(default_factory=list)
    section_order: List[str] = Field(default_factory=list)
    heading_depth: Optional[str] = Field(None)
    cta_language: List[str] = Field(default_factory=list)
    cta_positions: List[str] = Field(default_factory=list)
    cta_verbs: List[str] = Field(default_factory=list)
    opening_lines: List[str] = Field(default_factory=list)
    transition_sentences: List[str] = Field(default_factory=list)
    proof_statements: List[str] = Field(default_factory=list)
    conclusion_frames: List[str] = Field(default_factory=list)
    typical_link_targets: List[str] = Field(default_factory=list)
    must_use_names_terms: List[str] = Field(default_factory=list)
    tag_conventions: List[str] = Field(default_factory=list)


class DesignKitPlugin:
    """
    Plugin for generating DesignKitConfig using AI/LLM.

    This generates comprehensive brand guidelines configuration with all fields
    populated based on best practices and common patterns.
    """

    async def _analyze_content(
        self,
        pipeline: FunctionPipeline,
        content_doc: Dict[str, Any],
        index: int,
        total: int,
    ) -> Dict[str, Any]:
        """
        Analyze a single content piece to extract design kit patterns.

        Args:
            pipeline: FunctionPipeline instance
            content_doc: Content document from internal_docs (blog post, article, guide, etc.)
            index: Current content index (for logging)
            total: Total number of content pieces (for logging)

        Returns:
            Dictionary with extracted design kit patterns from this content piece
        """
        try:
            content_type = content_doc.get("metadata", {}).get(
                "content_type", "unknown"
            )
            logger.info(
                f"Analyzing {content_type} {index + 1}/{total}: {content_doc.get('title', 'Unknown')}"
            )

            # Extract relevant content from document
            title = content_doc.get("title", "")
            content = content_doc.get("metadata", {}).get("content_text", "")
            headings = content_doc.get("metadata", {}).get("headings", [])
            meta_description = content_doc.get("metadata", {}).get(
                "meta_description", ""
            )
            author = content_doc.get("metadata", {}).get("author", "")
            internal_links = content_doc.get("metadata", {}).get(
                "internal_links_found", []
            )

            system_prompt = """You are an expert content analyst specializing in brand voice and content patterns.
Your task is to analyze a piece of content (blog post, article, guide, documentation, etc.) and extract ACTUAL design kit patterns that are present in the content.

IMPORTANT: Extract REAL patterns from the content, not generic examples. If a pattern is not found, leave the field empty or use an empty list.

Focus on:
- Voice & tone: What adjectives describe the writing style? What point of view is used (we/you/neutral)? What's the sentence style?
- Structure: What's the actual section order based on headings? What heading levels are used?
- CTAs: Extract actual CTA phrases, verbs, and where they appear in the content (if applicable)
- Snippets: Extract actual opening lines, transitions, proof statements, and conclusion patterns from the text
- Terminology: What specific terms, product names, or brand language is used?
- Links: What are the actual link targets mentioned?

Be precise and extract only what you can observe in the content. Different content types may have different patterns - extract what's actually there."""

            # Include more content for better analysis (first 5000 chars)
            content_preview = content[:5000] if content else ""
            # Also include last 1000 chars to capture conclusions
            if len(content) > 5000:
                content_preview += (
                    "\n\n[... content continues ...]\n\n" + content[-1000:]
                )

            # Extract actual link targets
            link_targets = (
                [link.get("target_url", "") for link in internal_links[:20]]
                if internal_links
                else []
            )

            user_prompt = f"""Analyze this {content_type} content and extract ACTUAL design kit patterns from the content:

Content Type: {content_type}
Title: {title}
Meta Description: {meta_description}
Author: {author}
Headings Structure: {', '.join(headings[:15]) if headings else 'None found'}
Content: {content_preview}
Link Targets Found: {', '.join(link_targets[:10]) if link_targets else 'None'}

Extract and return ONLY patterns you can observe in the content above:

- voice_adjectives: List of 3-5 adjectives that describe the actual voice/tone (e.g., "confident", "technical", "friendly")
- point_of_view: "we", "you", or "neutral" - based on pronoun usage
- sentence_length_tempo: Describe actual sentence style (e.g., "medium/fast", "short/medium")
- lexical_preferences: List of 3-8 specific terms, product names, or brand language actually used
- section_order: List of section names in order based on headings (e.g., ["intro", "problem", "solution", "cta"])
- heading_depth: Most common heading level used: "H2", "H3", or "H4"
- cta_language: List of actual CTA phrases found (e.g., "Get Started", "Learn More", "Try Now")
- cta_positions: Where CTAs appear: "intro", "mid-content", "conclusion", or combinations
- cta_verbs: List of action verbs used in CTAs (e.g., "start", "explore", "discover")
- opening_lines: List of 1-3 actual opening phrases/sentences from the content
- transition_sentences: List of 2-5 actual transition phrases found between sections
- proof_statements: List of 2-5 actual evidence/social proof phrases (e.g., "used by X companies", "proven results")
- conclusion_frames: List of 1-3 actual conclusion patterns/phrases used
- typical_link_targets: List of actual link destinations mentioned (URLs or page names)
- must_use_names_terms: List of important product names, brand terms, or required terminology found
- tag_conventions: List of any visible tagging patterns (if metadata/tags are visible)

If a pattern is not clearly present, use an empty list [] or null. Only include what you can extract from the actual content."""

            # Call LLM to analyze this content piece
            analysis = await pipeline._call_function(
                prompt=user_prompt,
                system_instruction=system_prompt,
                response_model=ContentAnalysis,
                step_name=f"content_analysis_{index}",
                step_number=0,
                context=None,
                max_retries=1,
                job_id=None,
            )

            # Convert Pydantic model to dict
            return analysis.model_dump()

        except Exception as e:
            logger.warning(f"Error analyzing content {index + 1}: {e}")
            return {}

    async def generate_config(
        self,
        use_internal_docs: bool = True,
        model: Optional[str] = None,
        temperature: float = 0.7,
        job_id: Optional[str] = None,
    ) -> DesignKitConfig:
        """
        Generate a comprehensive design kit configuration using AI/LLM.

        Process:
        1. Fetches all content from internal_docs (blog posts, articles, guides, documentation, etc.)
        2. Analyzes each content piece individually to extract design kit patterns
        3. Aggregates all individual analyses
        4. Performs final LLM call to synthesize into one general design kit config

        Args:
            use_internal_docs: Whether to fetch and analyze all content from internal_docs
            model: OpenAI model to use (defaults to OPENAI_MODEL env var or gpt-5.1)
            temperature: Sampling temperature (default: 0.7)

        Returns:
            Generated DesignKitConfig with AI-populated fields
        """
        try:
            logger.info("Generating design kit configuration using AI...")

            # Helper to update progress if job_id provided
            async def update_progress(progress: int, message: str):
                if job_id:
                    from marketing_project.services.job_manager import get_job_manager

                    job_manager = get_job_manager()
                    await job_manager.update_job_progress(job_id, progress, message)
                logger.info(f"Design kit generation: {message} ({progress}%)")

            await update_progress(40, "Initializing AI pipeline")

            # Use FunctionPipeline infrastructure for consistency
            pipeline = FunctionPipeline(
                model=model or os.getenv("OPENAI_MODEL", "gpt-5.1"),
                temperature=temperature,
            )

            content_analyses: List[Dict[str, Any]] = []

            # Step 1: Fetch all content from internal_docs if requested
            if use_internal_docs:
                await update_progress(45, "Fetching content from internal_docs")
                try:
                    db = get_scanned_document_db()
                    # Get all active documents from internal_docs (all content types)
                    all_docs = db.get_all_active_documents()

                    # Filter to only content that has text content (exclude empty or metadata-only docs)
                    content_docs = [
                        doc
                        for doc in all_docs
                        if doc.metadata.content_text
                        and len(doc.metadata.content_text.strip()) > 100
                    ]

                    # Limit to max 20 content pieces to avoid timeout (prioritize by recency)
                    MAX_CONTENT_PIECES = 20
                    if len(content_docs) > MAX_CONTENT_PIECES:
                        # Sort by scanned_at (most recent first) and take top N
                        content_docs = sorted(
                            content_docs,
                            key=lambda d: d.scanned_at or datetime.min,
                            reverse=True,
                        )[:MAX_CONTENT_PIECES]
                        logger.info(
                            f"Limited to {MAX_CONTENT_PIECES} most recent content pieces "
                            f"(out of {len(all_docs)} total) to avoid timeout"
                        )

                    if content_docs:
                        # Group by content type for logging
                        content_types = {}
                        for doc in content_docs:
                            content_type = doc.metadata.content_type or "unknown"
                            content_types[content_type] = (
                                content_types.get(content_type, 0) + 1
                            )

                        logger.info(
                            f"Found {len(content_docs)} content pieces in internal_docs "
                            f"({', '.join(f'{count} {ctype}' for ctype, count in content_types.items())}). "
                            f"Analyzing each one..."
                        )

                        await update_progress(
                            50, f"Analyzing {len(content_docs)} content pieces"
                        )

                        # Step 2: Analyze each content piece individually
                        for idx, content_doc in enumerate(content_docs):
                            # Update progress for each analysis
                            analysis_progress = 50 + int(
                                (idx / len(content_docs)) * 20
                            )  # 50-70%
                            await update_progress(
                                analysis_progress,
                                f"Analyzing content {idx + 1}/{len(content_docs)}: {content_doc.title[:50]}...",
                            )

                            analysis = await self._analyze_content(
                                pipeline,
                                content_doc.model_dump(),
                                idx,
                                len(content_docs),
                            )
                            if analysis:
                                content_analyses.append(analysis)

                        logger.info(
                            f"Completed analysis of {len(content_analyses)} content pieces"
                        )
                        await update_progress(
                            70,
                            f"Completed analysis of {len(content_analyses)} content pieces",
                        )
                    else:
                        logger.warning(
                            "No content found in internal_docs. Generating generic config."
                        )
                except Exception as e:
                    logger.warning(
                        f"Error fetching content from internal_docs: {e}. Generating generic config."
                    )

            # Step 3: Final LLM call to synthesize all analyses into one general config
            if content_analyses:
                await update_progress(
                    75,
                    f"Synthesizing {len(content_analyses)} analyses into unified config",
                )
                logger.info(
                    f"Synthesizing {len(content_analyses)} content analyses into unified design kit config"
                )

                # Synthesize from actual content analyses
                system_prompt = """You are an expert content strategist and brand guidelines specialist.
You have been provided with individual design kit pattern analyses extracted from multiple content pieces (blog posts, articles, guides, documentation, etc.).
Your task is to synthesize these into one comprehensive, general design kit configuration.

SYNTHESIS RULES:
1. For list fields (voice_adjectives, cta_language, etc.):
   - Combine all unique values from all analyses
   - Remove duplicates
   - Keep the most common/representative examples (5-15 items)
   - Prioritize patterns that appear in multiple posts

2. For single-value fields (point_of_view, heading_depth, etc.):
   - Use the most common value across all posts
   - If there's a tie or variation, choose the most representative one

3. For fields with variations:
   - Identify the most common pattern
   - Note any significant variations in comments if needed

4. For missing fields:
   - If a field is missing from all analyses, use industry best practices
   - If a field is present in some but not all, include it if it appears in 2+ posts

5. Quality over quantity:
   - Prefer actual extracted patterns over generic examples
   - Only include patterns that are clearly present in the source content
   - Ensure all list fields have at least 3-5 examples where possible

The result should be a practical, general-purpose design kit that reflects the actual patterns found in your content."""

                # Format analyses for the prompt
                analyses_text = "\n\n".join(
                    [
                        f"Content Analysis {i+1}:\n{self._format_analysis(analysis)}"
                        for i, analysis in enumerate(content_analyses)
                    ]
                )

                user_prompt = f"""Synthesize the following {len(content_analyses)} content analyses into one comprehensive design kit configuration:

{analyses_text}

Generate a complete DesignKitConfig JSON structure following these guidelines:

1. VOICE & TONE:
   - voice_adjectives: Combine all unique adjectives, keep 5-8 most common
   - point_of_view: Use the most common value (we/you/neutral)
   - sentence_length_tempo: Use the most common pattern
   - lexical_preferences: Combine all unique terms, keep 5-10 most frequent

2. STRUCTURE:
   - section_order: Use the most common section sequence
   - heading_depth: Use the most common heading level
   - paragraph_length_range: Calculate from actual content if available, or use defaults
   - list_usage_preference: Infer from content patterns

3. SEO PATTERNS:
   - title_format: Analyze title patterns if visible
   - meta_description_style: Analyze meta description patterns
   - slug_casing: Infer from URLs if available
   - internal_link_anchor_style: Analyze from link patterns

4. CTA PATTERNS:
   - cta_language: Combine all unique CTAs, keep 8-12 most common
   - cta_positions: Combine all positions found
   - cta_verbs: Combine all unique verbs, keep 8-12 most common
   - typical_link_targets: Combine all unique link targets

5. REUSABLE SNIPPETS:
   - opening_lines: Combine all unique opening lines, keep 5-8 best examples
   - transition_sentences: Combine all unique transitions, keep 5-8 best
   - proof_statements: Combine all unique proof statements, keep 5-8 best
   - conclusion_frames: Combine all unique conclusions, keep 5-8 best
   - common_faqs: Extract if found, otherwise leave empty

6. COMPLIANCE & BRAND:
   - must_use_names_terms: Combine all unique terms from all posts
   - prohibited_phrases: Include if found in multiple posts
   - date_format: Infer from content if visible
   - numbers_formatting_rules: Use defaults if not found

7. ATTRIBUTION:
   - author_name_style: Analyze from author fields if available
   - bio_length_range: Use defaults if not found
   - sign_off_patterns: Extract if found in conclusions

8. QUANT/TARGETS:
   - word_count_range: Calculate from actual content if available
   - heading_density: Infer from heading patterns
   - keyword_density_band: Use "medium" as default

Return a complete DesignKitConfig JSON structure with ALL fields populated. Use actual extracted patterns where available, and sensible defaults for missing fields."""
            else:
                await update_progress(
                    75, "Generating generic design kit config (no content found)"
                )
                logger.info(
                    "No content analyses available, generating generic design kit config"
                )

                # Fallback to generic generation if no blog posts found
                system_prompt = """You are an expert content strategist and brand guidelines specialist.
Generate a comprehensive design kit configuration that includes best practices for:
- Voice & tone (adjectives, point of view, sentence style, lexical preferences)
- Content structure (section order, heading depth, paragraph length, list usage)
- SEO patterns (title formats, meta descriptions, slug conventions, link styles)
- CTA patterns (language, positions, action verbs, link targets)
- Compliance & brand (required terms, prohibited phrases, formatting rules)
- Attribution (author styles, bio length, sign-offs)
- Content metrics (word count ranges, heading density, keyword density)
- Reusable snippets (opening lines, transitions, proof statements, conclusions, FAQs)

Generate realistic, practical values that would be useful for a modern tech/SaaS company's content marketing.
Provide multiple examples for list fields (at least 5-10 items where appropriate).
Use industry-standard best practices."""

                user_prompt = """Generate a complete design kit configuration with all fields populated.
Include:
- Voice adjectives: 5-8 descriptive words (e.g., confident, practical, approachable)
- Point of view: we/you/neutral
- Sentence length/tempo: short/medium/long, fast/medium/slow
- Lexical preferences: 5-10 preferred terms
- Section order: typical content flow (e.g., intro, problem, solution, proof, cta)
- CTA language: 8-12 common CTA phrases
- CTA positions: where CTAs appear (e.g., intro, mid-content, conclusion)
- CTA verbs: 8-12 action verbs
- Opening lines: 5-8 engaging opening phrases
- Transition sentences: 5-8 transition phrases
- Proof statements: 5-8 evidence/social proof phrases
- Conclusion frames: 5-8 conclusion templates
- Common FAQs: 3-5 question/answer pairs
- Tag conventions: 5-8 tagging patterns
- Typical link targets: 5-8 common destinations
- Must-use terms: 3-5 required terminology
- Prohibited phrases: 3-5 forbidden terms
- Sign-off patterns: 3-5 sign-off styles

Return a complete DesignKitConfig JSON structure with all fields populated."""

            # Step 4: Final synthesis call
            await update_progress(80, "Calling AI to generate final design kit config")
            logger.info(
                "Calling OpenAI API to generate final design kit configuration..."
            )

            generated_config = await pipeline._call_function(
                prompt=user_prompt,
                system_instruction=system_prompt,
                response_model=DesignKitConfig,
                step_name="design_kit_config_synthesis",
                step_number=0,
                context=None,
                max_retries=2,
                job_id=job_id,
            )

            await update_progress(95, "Design kit config generated successfully")
            logger.info("Successfully generated design kit configuration using AI")
            return generated_config

        except Exception as e:
            logger.error(
                f"Error generating design kit config with AI: {e}", exc_info=True
            )
            raise

    async def _synthesize_config(
        self,
        pipeline: FunctionPipeline,
        content_analyses: List[Dict[str, Any]],
        job_id: Optional[str] = None,
    ) -> DesignKitConfig:
        """
        Synthesize multiple content analyses into a unified design kit config.

        Args:
            pipeline: FunctionPipeline instance
            content_analyses: List of analysis dictionaries from individual content pieces
            job_id: Optional job ID for progress tracking

        Returns:
            Synthesized DesignKitConfig
        """
        try:

            async def update_progress(progress: int, message: str):
                if job_id:
                    from marketing_project.services.job_manager import get_job_manager

                    job_manager = get_job_manager()
                    await job_manager.update_job_progress(job_id, progress, message)
                logger.info(f"Design kit synthesis: {message} ({progress}%)")

            if not content_analyses:
                # No analyses - generate generic config
                await update_progress(50, "Generating generic config (no analyses)")
                system_prompt = """You are an expert content strategist and brand guidelines specialist.
Generate a comprehensive design kit configuration that includes best practices for:
- Voice & tone (adjectives, point of view, sentence style, lexical preferences)
- Content structure (section order, heading depth, paragraph length, list usage)
- SEO patterns (title formats, meta descriptions, slug conventions, link styles)
- CTA patterns (language, positions, action verbs, link targets)
- Compliance & brand (required terms, prohibited phrases, formatting rules)
- Attribution (author styles, bio length, sign-offs)
- Content metrics (word count ranges, heading density, keyword density)
- Reusable snippets (opening lines, transitions, proof statements, conclusions, FAQs)

Generate realistic, practical values that would be useful for a modern tech/SaaS company's content marketing.
Provide multiple examples for list fields (at least 5-10 items where appropriate).
Use industry-standard best practices."""

                user_prompt = """Generate a complete design kit configuration with all fields populated.
Include:
- Voice adjectives: 5-8 descriptive words (e.g., confident, practical, approachable)
- Point of view: we/you/neutral
- Sentence length/tempo: short/medium/long, fast/medium/slow
- Lexical preferences: 5-10 preferred terms
- Section order: typical content flow (e.g., intro, problem, solution, proof, cta)
- CTA language: 8-12 common CTA phrases
- CTA positions: where CTAs appear (e.g., intro, mid-content, conclusion)
- CTA verbs: 8-12 action verbs
- Opening lines: 5-8 engaging opening phrases
- Transition sentences: 5-8 transition phrases
- Proof statements: 5-8 evidence/social proof phrases
- Conclusion frames: 5-8 conclusion templates
- Common FAQs: 3-5 question/answer pairs
- Tag conventions: 5-8 tagging patterns
- Typical link targets: 5-8 common destinations
- Must-use terms: 3-5 required terminology
- Prohibited phrases: 3-5 forbidden terms
- Sign-off patterns: 3-5 sign-off styles

Return a complete DesignKitConfig JSON structure with all fields populated."""
            else:
                # Synthesize from actual content analyses
                await update_progress(
                    50, f"Synthesizing {len(content_analyses)} analyses"
                )

                system_prompt = """You are an expert content strategist and brand guidelines specialist.
You have been provided with individual design kit pattern analyses extracted from multiple content pieces (blog posts, articles, guides, documentation, etc.).
Your task is to synthesize these into one comprehensive, general design kit configuration.

SYNTHESIS RULES:
1. For list fields (voice_adjectives, cta_language, etc.):
   - Combine all unique values from all analyses
   - Remove duplicates
   - Keep the most common/representative examples (5-15 items)
   - Prioritize patterns that appear in multiple posts

2. For single-value fields (point_of_view, heading_depth, etc.):
   - Use the most common value across all posts
   - If there's a tie or variation, choose the most representative one

3. For fields with variations:
   - Identify the most common pattern
   - Note any significant variations in comments if needed

4. For missing fields:
   - If a field is missing from all analyses, use industry best practices
   - If a field is present in some but not all, include it if it appears in 2+ posts

5. Quality over quantity:
   - Prefer actual extracted patterns over generic examples
   - Only include patterns that are clearly present in the source content
   - Ensure all list fields have at least 3-5 examples where possible

The result should be a practical, general-purpose design kit that reflects the actual patterns found in your content."""

                # Format analyses for the prompt
                analyses_text = "\n\n".join(
                    [
                        f"Content Analysis {i+1}:\n{self._format_analysis(analysis)}"
                        for i, analysis in enumerate(content_analyses)
                    ]
                )

                user_prompt = f"""Synthesize the following {len(content_analyses)} content analyses into one comprehensive design kit configuration:

{analyses_text}

Generate a complete DesignKitConfig JSON structure following these guidelines:

1. VOICE & TONE:
   - voice_adjectives: Combine all unique adjectives, keep 5-8 most common
   - point_of_view: Use the most common value (we/you/neutral)
   - sentence_length_tempo: Use the most common pattern
   - lexical_preferences: Combine all unique terms, keep 5-10 most frequent

2. STRUCTURE:
   - section_order: Use the most common section sequence
   - heading_depth: Use the most common heading level
   - paragraph_length_range: Calculate from actual content if available, or use defaults
   - list_usage_preference: Infer from content patterns

3. SEO PATTERNS:
   - title_format: Analyze title patterns if visible
   - meta_description_style: Analyze meta description patterns
   - slug_casing: Infer from URLs if available
   - internal_link_anchor_style: Analyze from link patterns

4. CTA PATTERNS:
   - cta_language: Combine all unique CTAs, keep 8-12 most common
   - cta_positions: Combine all positions found
   - cta_verbs: Combine all unique verbs, keep 8-12 most common
   - typical_link_targets: Combine all unique link targets

5. REUSABLE SNIPPETS:
   - opening_lines: Combine all unique opening lines, keep 5-8 best examples
   - transition_sentences: Combine all unique transitions, keep 5-8 best
   - proof_statements: Combine all unique proof statements, keep 5-8 best
   - conclusion_frames: Combine all unique conclusions, keep 5-8 best
   - common_faqs: Extract if found, otherwise leave empty

6. COMPLIANCE & BRAND:
   - must_use_names_terms: Combine all unique terms from all posts
   - prohibited_phrases: Include if found in multiple posts
   - date_format: Infer from content if visible
   - numbers_formatting_rules: Use defaults if not found

7. ATTRIBUTION:
   - author_name_style: Analyze from author fields if available
   - bio_length_range: Use defaults if not found
   - sign_off_patterns: Extract if found in conclusions

8. QUANT/TARGETS:
   - word_count_range: Calculate from actual content if available
   - heading_density: Infer from heading patterns
   - keyword_density_band: Use "medium" as default

Return a complete DesignKitConfig JSON structure with ALL fields populated. Use actual extracted patterns where available, and sensible defaults for missing fields."""

            await update_progress(80, "Calling AI to generate final design kit config")
            logger.info("Calling OpenAI API to synthesize design kit configuration...")

            generated_config = await pipeline._call_function(
                prompt=user_prompt,
                system_instruction=system_prompt,
                response_model=DesignKitConfig,
                step_name="design_kit_config_synthesis",
                step_number=0,
                context=None,
                max_retries=2,
                job_id=job_id,
            )

            await update_progress(95, "Design kit config synthesized successfully")
            logger.info("Successfully synthesized design kit configuration using AI")
            return generated_config

        except Exception as e:
            logger.error(f"Error synthesizing design kit config: {e}", exc_info=True)
            raise

    def _format_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format an analysis dictionary for display in prompt."""
        lines = []
        for key, value in analysis.items():
            if value:
                if isinstance(value, list):
                    lines.append(
                        f"{key}: {', '.join(str(v) for v in value[:10])}"
                    )  # Limit to 10 items
                else:
                    lines.append(f"{key}: {value}")
        return "\n".join(lines)
