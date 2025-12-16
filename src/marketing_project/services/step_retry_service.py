"""
Step Retry Service.

Handles re-execution of individual pipeline steps that were rejected during approval.
This allows users to retry a specific step without re-running the entire pipeline.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from marketing_project.services.function_pipeline.tracing import (
    close_span,
    create_span,
    is_tracing_available,
    record_span_exception,
    set_span_attribute,
    set_span_input,
    set_span_kind,
    set_span_output,
    set_span_status,
)

# Import Status for tracing
try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None

from marketing_project.models.pipeline_steps import (
    AngleHookResult,
    ArticleGenerationResult,
    BlogPostPreprocessingApprovalResult,
    ContentFormattingResult,
    DesignKitResult,
    MarketingBriefResult,
    SEOKeywordsResult,
    SEOOptimizationResult,
    SocialMediaMarketingBriefResult,
    SocialMediaPostResult,
    SuggestedLinksResult,
    TranscriptPreprocessingApprovalResult,
)
from marketing_project.services.function_pipeline import FunctionPipeline

logger = logging.getLogger("marketing_project.services.step_retry")

# Map step names to their response models
STEP_MODEL_MAP = {
    "transcript_preprocessing_approval": TranscriptPreprocessingApprovalResult,
    "blog_post_preprocessing_approval": BlogPostPreprocessingApprovalResult,
    "seo_keywords": SEOKeywordsResult,
    "marketing_brief": MarketingBriefResult,
    "article_generation": ArticleGenerationResult,
    "seo_optimization": SEOOptimizationResult,
    "suggested_links": SuggestedLinksResult,
    "content_formatting": ContentFormattingResult,
    "design_kit": DesignKitResult,
    "social_media_marketing_brief": SocialMediaMarketingBriefResult,
    "social_media_angle_hook": AngleHookResult,
    "social_media_post_generation": SocialMediaPostResult,
}

# Map step names to their step numbers (for logging/tracking)
STEP_NUMBER_MAP = {
    "transcript_preprocessing_approval": 1,
    "blog_post_preprocessing_approval": 1,
    "seo_keywords": 1,
    "marketing_brief": 2,
    "article_generation": 3,
    "seo_optimization": 4,
    "suggested_links": 5,
    "content_formatting": 6,
    "design_kit": 7,
    "social_media_marketing_brief": 2,
    "social_media_angle_hook": 3,
    "social_media_post_generation": 4,
}


class StepRetryService:
    """Service for retrying individual pipeline steps."""

    def __init__(
        self, model: str = "gpt-5.1", temperature: float = 0.7, lang: str = "en"
    ):
        """
        Initialize the step retry service.

        Args:
            model: OpenAI model to use
            temperature: Sampling temperature
            lang: Language for prompts
        """
        self.pipeline = FunctionPipeline(
            model=model, temperature=temperature, lang=lang
        )

    async def retry_step(
        self,
        step_name: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        user_guidance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retry a single pipeline step.

        Args:
            step_name: Name of the step to retry (e.g., "seo_keywords", "marketing_brief")
            input_data: Input data for the step (includes original content and prompts)
            context: Optional context from previous steps
            job_id: Optional job ID for tracking

        Returns:
            Dictionary with step result including:
                - step_name: Name of the step
                - status: "success" or "error"
                - result: The step output data
                - execution_time: Time taken to execute
                - error_message: Error message if failed

        Raises:
            ValueError: If step_name is invalid
        """
        start_time = time.time()

        # Create telemetry span for step retry execution
        retry_step_span = None
        if is_tracing_available():
            retry_step_span = create_span(
                "step_retry.execute",
                attributes={
                    "step_name": step_name,
                    "job_id": job_id or "unknown",
                    "has_user_guidance": bool(user_guidance),
                },
            )
            if retry_step_span:
                # Set OpenInference span kind
                set_span_kind(retry_step_span, "AGENT")

                # Set input attributes (input_data + context)
                input_data_for_span = {
                    "input_data": input_data,
                    "context": context or {},
                    "user_guidance": user_guidance,
                }
                set_span_input(retry_step_span, input_data_for_span)

                if context:
                    content_type = context.get("content_type")
                    if content_type:
                        set_span_attribute(
                            retry_step_span, "content_type", content_type
                        )
                if user_guidance:
                    set_span_attribute(
                        retry_step_span, "user_guidance_length", len(user_guidance)
                    )

        # Validate step name
        if step_name not in STEP_MODEL_MAP:
            if retry_step_span:
                set_span_status(retry_step_span, StatusCode.ERROR, "Invalid step name")
                set_span_attribute(retry_step_span, "error.type", "ValueError")
            raise ValueError(
                f"Invalid step name: {step_name}. "
                f"Valid steps: {', '.join(STEP_MODEL_MAP.keys())}"
            )

        logger.info(f"Retrying step '{step_name}' for job {job_id}")

        try:
            # Get the appropriate response model
            response_model = STEP_MODEL_MAP[step_name]
            step_number = STEP_NUMBER_MAP[step_name]

            if retry_step_span:
                set_span_attribute(retry_step_span, "step_number", step_number)

            # Build the prompt from input data
            prompt = self._build_prompt(step_name, input_data, context, user_guidance)

            # Execute the step using the function pipeline
            # Skip approval check when retrying (step has already been approved)
            result = await self.pipeline._call_function(
                prompt=prompt,
                system_instruction=self.pipeline._get_system_instruction(
                    step_name, context
                ),
                response_model=response_model,
                step_name=step_name,
                step_number=step_number,
                context=context,
                job_id=None,  # Skip approval check during retry (already approved)
            )

            execution_time = time.time() - start_time

            retry_result = {
                "step_name": step_name,
                "status": "success",
                "result": result.model_dump(),
                "execution_time": execution_time,
                "retry_timestamp": datetime.utcnow().isoformat(),
                "error_message": None,
            }

            if retry_step_span:
                # Set output attributes
                set_span_output(retry_step_span, retry_result)

                set_span_status(
                    retry_step_span, StatusCode.OK, "Step retry completed successfully"
                )
                set_span_attribute(retry_step_span, "execution_time", execution_time)
                set_span_attribute(retry_step_span, "status", "success")

            logger.info(
                f"Step '{step_name}' retry completed successfully "
                f"in {execution_time:.2f}s for job {job_id}"
            )

            return retry_result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            error_result = {
                "step_name": step_name,
                "status": "error",
                "result": None,
                "execution_time": execution_time,
                "retry_timestamp": datetime.utcnow().isoformat(),
                "error_message": error_msg,
            }

            if retry_step_span:
                # Set output attributes (error result)
                set_span_output(retry_step_span, error_result)

                record_span_exception(retry_step_span, e)
                set_span_status(retry_step_span, StatusCode.ERROR, error_msg)
                set_span_attribute(retry_step_span, "error.type", type(e).__name__)
                set_span_attribute(retry_step_span, "execution_time", execution_time)
                set_span_attribute(retry_step_span, "status", "error")

            logger.error(
                f"Step '{step_name}' retry failed after {execution_time:.2f}s "
                f"for job {job_id}: {error_msg}"
            )

            return error_result
        finally:
            close_span(retry_step_span)

    def _build_prompt(
        self,
        step_name: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        user_guidance: Optional[str] = None,
    ) -> str:
        """
        Build the prompt for a specific step based on input data.

        Args:
            step_name: Name of the step
            input_data: Input data containing original content
            context: Optional context from previous steps

        Returns:
            Formatted prompt string for the step
        """
        context = context or {}

        # Extract content from input_data
        content = input_data.get("content", {})
        title = content.get("title", "N/A")
        content_text = content.get("content", "")
        snippet = content.get("snippet", "")
        category = content.get("category", "N/A")
        tags = content.get("tags", [])

        # Build step-specific prompts
        if step_name == "seo_keywords":
            prompt = f"""Analyze this content and extract comprehensive SEO keywords:

Title: {title}
Content: {content_text[:8000]}
Category: {category}
Tags: {', '.join(tags)}

IMPORTANT: Identify the SINGLE most important keyword (main_keyword) that should be the primary focus for this content. This should be:
- The most relevant keyword for the content
- The keyword with the best balance of search volume and competition
- The keyword that best represents the main topic
- A keyword that can realistically rank (difficulty score 20-70)

Then extract:

1. Main keyword (1): The single most important keyword
   - Must be included in primary_keywords list
   - Should have estimated search volume > 100/month
   - Difficulty score should be between 20-70 for achievable ranking

2. Primary keywords (3-5 total, including main_keyword):
   - Main keywords for the content
   - Each should have metadata: search_volume, difficulty_score, relevance_score
   - Prioritize keywords with good volume/difficulty ratio

3. Secondary keywords (5-10):
   - Supporting keywords with lower search volume but high relevance
   - Include semantic variations and synonyms
   - Each should have metadata with search_volume and difficulty_score

4. LSI keywords (5-10):
   - Semantically related terms for content depth
   - Terms that frequently co-occur with primary keywords
   - Contextually relevant to the main topic

5. Long-tail keywords (5-8):
   - Specific, less competitive keyword opportunities
   - Question-based keywords (what, how, why, when, where)
   - Include geographic or modifier variations if relevant
   - Example: "how to [main keyword]", "best [main keyword] for [audience]"

6. Keyword density analysis:
   - Calculate current density for each primary keyword
   - Recommend optimal density (1-3% for primary, 0.5-1% for secondary)
   - Identify placement locations (title, H1, first paragraph, etc.)
   - Provide keyword_density_analysis with current_density, optimal_density, occurrences, and placement_locations

7. Search intent classification:
   - Categorize as: informational, transactional, navigational, or commercial
   - Provide reasoning for intent classification

8. Keyword difficulty assessment:
   - Provide numeric score (0-100) for each keyword in keyword_difficulty Dict
   - Consider competition level, domain authority needed, content quality required
   - Flag keywords with difficulty > 70 as "challenging"
   - Include difficulty_score in keyword metadata

9. Search volume estimates:
   - Provide realistic monthly search volume estimates
   - Use industry benchmarks and content context
   - Mark low-volume keywords (< 100/month) appropriately
   - Include search_volume in keyword metadata

10. Keyword clusters:
    - Group related keywords by topic/thematic similarity
    - Identify parent-child keyword relationships
    - Map keyword hierarchy
    - Provide keyword_clusters with cluster_name, keywords, primary_keyword, and topic_theme

11. Optimization recommendations:
    - Specific actions to improve keyword performance
    - Content gaps to fill
    - Keyword placement suggestions
    - Provide optimization_recommendations as a list of actionable items

12. Search volume summary:
    - Calculate total estimated monthly search volume by keyword category
    - Provide search_volume_summary Dict with totals for primary, secondary, long_tail

Focus on keywords that will drive organic traffic while maintaining natural, high-quality content. Ensure all metadata fields are populated for primary, secondary, and long-tail keywords."""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the keyword extraction."

            return prompt

        elif step_name == "marketing_brief":
            seo_keywords = context.get("seo_keywords", {})
            main_keyword = seo_keywords.get("main_keyword", "")
            primary_keywords = seo_keywords.get("primary_keywords", [])
            search_intent = seo_keywords.get("search_intent", "informational")
            long_tail_keywords = seo_keywords.get("long_tail_keywords", [])
            keyword_clusters = seo_keywords.get("keyword_clusters", [])
            content_type = input_data.get(
                "content_type", context.get("content_type", "blog_post")
            )

            cluster_names = (
                [
                    c.get("cluster_name", "") if isinstance(c, dict) else ""
                    for c in keyword_clusters
                ]
                if keyword_clusters
                else []
            )

            prompt = f"""Create a comprehensive marketing brief for a {content_type} focused on the keyword: {main_keyword}

Focus the marketing brief on the main keyword: {main_keyword}

KEYWORD CONTEXT:
- Main Keyword: {main_keyword}
- Primary Keywords: {', '.join(primary_keywords)}
- Search Intent: {search_intent}
{f"- Long-tail Opportunities: {', '.join(long_tail_keywords)}" if long_tail_keywords else ""}
{f"- Keyword Clusters: {', '.join(cluster_names)}" if cluster_names else ""}

Define target audience personas, key messaging, content strategy, KPIs, and distribution channels that align with:
- The search intent ({search_intent})
{f"- The keyword clusters and themes identified" if cluster_names else ""}
{f"- Long-tail keyword opportunities for specific targeting" if long_tail_keywords else ""}"""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the marketing brief."

            return prompt

        elif step_name == "article_generation":
            seo_keywords = context.get("seo_keywords", {})
            brief = context.get("marketing_brief", {})
            main_keyword = seo_keywords.get("main_keyword", "")
            primary_keywords = seo_keywords.get("primary_keywords", [])
            long_tail_keywords = seo_keywords.get("long_tail_keywords", [])
            keyword_clusters = seo_keywords.get("keyword_clusters", [])
            # Extract supporting keywords (excluding main keyword)
            supporting_keywords = [k for k in primary_keywords if k != main_keyword]

            cluster_names = (
                [
                    c.get("cluster_name", "") if isinstance(c, dict) else ""
                    for c in keyword_clusters
                ]
                if keyword_clusters
                else []
            )

            prompt = f"""Generate a complete article from scratch based on the marketing brief:

Supporting Keywords: {', '.join(supporting_keywords) if supporting_keywords else 'None'}
{f"Long-tail Keywords: {', '.join(long_tail_keywords)}" if long_tail_keywords else ""}
{f"Keyword Clusters: {', '.join(cluster_names)}" if cluster_names else ""}

Marketing Brief:
- Target Audience: {', '.join(brief.get('target_audience', []))}
- Key Messages: {', '.join(brief.get('key_messages', []))}
- Content Strategy: {brief.get('content_strategy', 'Not specified')}
- Tone and Voice: {brief.get('tone_and_voice', 'Professional and engaging')}
- Competitive Angle: {brief.get('competitive_angle', 'Not specified')}
- KPIs: {', '.join(brief.get('kpis', [])) if brief.get('kpis') else 'Not specified'}
- Distribution Channels: {', '.join(brief.get('distribution_channels', [])) if brief.get('distribution_channels') else 'Not specified'}

Generate a complete article that:
- Is created from scratch, aligned with the marketing strategy
- Uses the supporting keywords naturally throughout the content
{f"- Incorporates long-tail keywords naturally for better SEO targeting" if long_tail_keywords else ""}
{f"- Organizes content around keyword clusters for thematic depth" if cluster_names else ""}
- Speaks directly to the target audience personas
- Incorporates all key messages strategically
- Follows the recommended content strategy
- Matches the specified tone and voice: {brief.get('tone_and_voice', 'Professional and engaging')}
- Leverages the competitive angle for differentiation
- Is designed to achieve the specified KPIs
- Is optimized for the recommended distribution channels

Create:
- Compelling article title optimized for clicks and engagement
- Well-structured, engaging article content (full article, not a summary)
- Clear outline with section headers
- Strong opening hook that captures attention
- Key takeaways that reinforce the main messages
- Compelling call-to-action aligned with the strategy"""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the article generation."

            return prompt

        elif step_name == "seo_optimization":
            article = context.get("article_generation", {})
            seo_keywords = context.get("seo_keywords", {})
            brief = context.get("marketing_brief", {})
            article_content = article.get("article_content", content_text)
            article_title = article.get("article_title", title)
            main_keyword = seo_keywords.get("main_keyword", "")
            primary_keywords = seo_keywords.get("primary_keywords", [])
            secondary_keywords = seo_keywords.get("secondary_keywords", [])
            long_tail_keywords = seo_keywords.get("long_tail_keywords", [])
            search_intent = seo_keywords.get("search_intent", "informational")
            keyword_difficulty = seo_keywords.get("keyword_difficulty", {})
            keyword_density_analysis = seo_keywords.get("keyword_density_analysis", [])
            optimization_recommendations = seo_keywords.get(
                "optimization_recommendations", []
            )

            prompt = f"""Perform comprehensive SEO optimization workflow with validation and generation:

ARTICLE CONTENT (FULL):
Title: {article_title}
Content: {article_content}

FROM STEP 1 (SEO Keywords):
- Main Keyword: {main_keyword}
- Primary Keywords: {', '.join(primary_keywords)}
- Secondary Keywords: {', '.join(secondary_keywords) if secondary_keywords else 'None'}
- Long-tail Keywords: {', '.join(long_tail_keywords) if long_tail_keywords else 'None'}
- Search Intent: {search_intent}
{f"- Keyword Difficulty Scores: Available for {len(keyword_difficulty)} keywords (focus on difficulty 20-70 range)" if keyword_difficulty and isinstance(keyword_difficulty, dict) else ""}
{f"- Keyword Density Analysis: Available for {len(keyword_density_analysis)} keywords" if keyword_density_analysis else ""}
{f"- Optimization Recommendations: {'; '.join(optimization_recommendations)}" if optimization_recommendations else ""}

FROM STEP 2 (Marketing Brief):
- Target Audience: {', '.join(brief.get('target_audience', []))}
- Tone and Voice: {brief.get('tone_and_voice', 'Professional and engaging')}
- Content Strategy: {brief.get('content_strategy', 'Not specified')}
- Competitive Angle: {brief.get('competitive_angle', 'Not specified')}
- Key Messages: {', '.join(brief.get('key_messages', []))}

FROM STEP 3 (Article Generation):
- Article Title: {article_title}
- Article Outline: {', '.join(article.get('outline', []))}
- Key Takeaways: {', '.join(article.get('key_takeaways', [])) if article.get('key_takeaways') else 'None'}

SEO OPTIMIZATION WORKFLOW:

1. PARSE CONTENT:
   - Analyze structure and detect title, headings (H1-H3), and section layout
   - Identify H1-H3 hierarchy and note any missing or duplicate elements

2. KEYWORD MAPPING:
   - Identify and confirm the primary keyword ({main_keyword}) and 3-5 semantically related terms
   - Ensure related keywords align with the post topic and search intent ({search_intent})
   {f"- Incorporate long-tail keywords naturally where appropriate for better targeting" if long_tail_keywords else ""}
   {f"- Use keyword difficulty scores to prioritize which keywords to emphasize (focus on difficulty 20-70 range)" if keyword_difficulty and isinstance(keyword_difficulty, dict) else ""}
   {f"- Follow optimization recommendations from keyword analysis" if optimization_recommendations else ""}
   {f"- Reference keyword density analysis to ensure optimal keyword distribution" if keyword_density_analysis else ""}
   {f"- Use placement locations from density analysis to guide keyword placement" if keyword_density_analysis else ""}

3. TITLE OPTIMIZATION/GENERATION:
   - Validate existing title includes primary keyword, is under 60 characters, and is compelling
   - If missing or weak: generate a new SEO title that includes the primary keyword and uses action-oriented language

4. META DESCRIPTION GENERATION:
   - If meta description exists: verify it's between 140-160 characters, includes the keyword once, and has a clear CTA or value
   - If missing: generate a new one following the same rules, incorporating key messages from the marketing brief

5. URL SLUG GENERATION:
   - If slug exists: check that it's short (3-5 words), lowercase, hyphen-separated, and contains the keyword
   - If missing: generate one automatically from the keyword or title

6. HEADER STRUCTURE OPTIMIZATION:
   - Ensure only one H1 exists
   - Validate that H2s represent logical sections and include relevant keywords
   - Adjust or generate sub-headings for clarity and SEO structure

7. KEYWORD PLACEMENT OPTIMIZATION:
   - Validate that the primary keyword appears naturally in the title, introduction, at least one H2, and near the conclusion
   - Check secondary keyword distribution and adjust if density is too low or too high
   - Ensure natural readability (avoid keyword stuffing)

8. READABILITY OPTIMIZATION:
   - Evaluate sentence length, transitions, and clarity
   - Aim for Grade 8-9 reading level
   - Rewrite for active voice and scan-friendly formatting (short paragraphs, bullet points, clear transitions)

9. SCHEMA GENERATION:
   - Create or validate a JSON-LD "BlogPosting" schema block
   - Include title, description, author (if available), date, and keywords
   - Output schema as a valid JSON-LD string

OUTPUT:
- SEO-optimized content version (full content, not truncated)
- Meta title (50-60 chars)
- Meta description (140-160 chars)
- URL slug
- Alt text suggestions for images
- Schema markup (JSON-LD as string)
- Open Graph tags for social sharing
- Header structure analysis (H1-H3 hierarchy with validation)
- Keyword map (primary + related keywords with placement locations)
- Readability optimization analysis (score, grade level, active voice percentage)
- Modification report (list of changes made, e.g., "Meta description regenerated", "H2 hierarchy corrected", "Keyword placement optimized")"""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the SEO optimization."

            return prompt

        elif step_name == "suggested_links":
            article = context.get("article_generation", {})
            seo = context.get("seo_optimization", {})
            seo_keywords = context.get("seo_keywords", {})
            internal_docs_config = context.get("internal_docs_config", {})

            # Extract data safely
            article_title = article.get("article_title", title)
            full_content = seo.get("optimized_content", content_text)
            main_keyword = seo_keywords.get("main_keyword", "")
            primary_keywords = seo_keywords.get("primary_keywords", [])
            secondary_keywords = seo_keywords.get("secondary_keywords", [])
            long_tail_keywords = seo_keywords.get("long_tail_keywords", [])
            search_intent = seo_keywords.get("search_intent", "informational")
            article_outline = article.get("outline", [])
            key_takeaways = article.get("key_takeaways", [])
            header_structure = seo.get("header_structure")

            # Build prompt
            prompt = f"""Analyze this article and suggest specific places to add internal links.

ARTICLE TO ANALYZE:
Title: {article_title}
Full Content: {full_content}

ARTICLE STRUCTURE:
- Outline: {', '.join(article_outline) if article_outline else 'None'}
- Key Takeaways: {', '.join(key_takeaways) if key_takeaways else 'None'}
- Header Structure: {str(header_structure) if header_structure else 'Not available'}

KEYWORDS FOR MATCHING:
- Main Keyword: {main_keyword}
- Primary Keywords: {', '.join(primary_keywords) if primary_keywords else 'None'}
- Secondary Keywords: {', '.join(secondary_keywords) if secondary_keywords else 'None'}
- Long-tail Keywords: {', '.join(long_tail_keywords) if long_tail_keywords else 'None'}
- Search Intent: {search_intent}
"""

            # Add InternalDocsConfig if available
            if internal_docs_config:
                prompt += "\nAVAILABLE INTERNAL DOCUMENTATION:\n"

                scanned_docs = internal_docs_config.get("scanned_documents", [])
                if scanned_docs:
                    prompt += f"Scanned Documents ({len(scanned_docs)}):\n"
                    for doc in scanned_docs:
                        prompt += f"- {doc.get('title', 'Unknown')} ({doc.get('url', 'N/A')})\n"

                common_pages = internal_docs_config.get("commonly_referenced_pages", [])
                if common_pages:
                    prompt += "\nCommonly Referenced Pages:\n"
                    for page in common_pages:
                        prompt += f"- {page}\n"

                common_categories = internal_docs_config.get(
                    "commonly_referenced_categories", []
                )
                if common_categories:
                    prompt += "\nCommonly Referenced Categories:\n"
                    for category in common_categories:
                        prompt += f"- {category}\n"

                anchor_patterns = internal_docs_config.get(
                    "anchor_phrasing_patterns", []
                )
                if anchor_patterns:
                    prompt += "\nAnchor Text Patterns to Follow:\n"
                    for pattern in anchor_patterns:
                        prompt += f"- {pattern}\n"
            else:
                prompt += "\nNOTE: No InternalDocsConfig found. You can still suggest links, but you won't have a reference list of available pages.\n"

            prompt += """
TASK:
Suggest specific internal links to add to this article. For each suggestion, provide:
1. **Anchor Text**: Natural, keyword-optimized anchor text that follows the anchor phrasing patterns (if provided)
2. **Target URL**: A specific page from the available internal documentation (if config provided) or a logical URL path
3. **Placement Context**: Where in the article this link should be placed (include a snippet of surrounding text or section name)
4. **Section**: Which article section this belongs to (from the outline)
5. **Relevance Score**: How relevant this link is (0.0-1.0)
6. **Reasoning**: Brief explanation of why this link is suggested

LINKING STRATEGY:
- Match article content and keywords to available pages/categories from InternalDocsConfig
- Use anchor text patterns from the config when suggesting anchor text
- Place links contextually where they add value (not just at the end)
- Focus on high-relevance matches (relevance > 0.6)
- Suggest 3-8 links total, prioritizing the most relevant connections
- Use keywords to find semantic matches between article content and available pages
- Consider search intent when matching content

IMPORTANT:
- Only suggest links to pages that exist in the InternalDocsConfig (if provided)
- Use the anchor phrasing patterns as guidance for anchor text style
- Provide specific placement context so links can be inserted accurately
- Prioritize links that enhance user experience and content discoverability
"""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the internal linking suggestions."

            return prompt

        elif step_name == "content_formatting":
            article = context.get("article_generation", {})
            seo = context.get("seo_optimization", {})

            # Extract data safely
            article_title = article.get("article_title", title)
            optimized_content = seo.get(
                "optimized_content", article.get("article_content", content_text)
            )
            article_outline = article.get("outline", [])
            header_structure = seo.get("header_structure")
            keyword_map = seo.get("keyword_map")
            readability_optimization = seo.get("readability_optimization")
            meta_title = seo.get("meta_title", "")
            meta_description = seo.get("meta_description", "")
            schema_markup = seo.get("schema_markup")

            prompt = f"""Format this SEO-optimized content for publication while preserving all SEO optimizations:

CONTENT TO FORMAT (FULL):
Content: {optimized_content}

FROM STEP 3 (Article Generation):
- Article Outline: {', '.join(article_outline) if article_outline else 'None'}

FROM STEP 4 (SEO Optimization - CRITICAL TO PRESERVE):
- Header Structure: {str(header_structure) if header_structure else 'Not available'}
  IMPORTANT: Preserve this exact H1-H3 hierarchy when formatting
- Keyword Map: {str(keyword_map) if keyword_map else 'Not available'}
  IMPORTANT: Preserve keyword placement locations during formatting
- Readability Optimization: {str(readability_optimization) if readability_optimization else 'Not available'}
  IMPORTANT: Maintain grade level and active voice percentage
- Meta Title: {meta_title}
- Meta Description: {meta_description}
- Schema Markup: {schema_markup if schema_markup else 'Not available'}

FORMATTING REQUIREMENTS:

1. PRESERVE SEO-OPTIMIZED STRUCTURE:
   - Maintain the exact header_structure hierarchy (H1-H3) from Step 4
   - Do NOT modify or reorganize headings - preserve SEO-optimized structure
   - Ensure only one H1 exists as per header_structure analysis
   - Keep H2s and H3s in their optimized positions

2. PRESERVE KEYWORD PLACEMENT:
   - Use keyword_map to ensure keywords remain in their optimized locations
   - Do NOT move or remove keywords from their placement locations
   - Maintain keyword density and natural placement as per keyword_map

3. MAINTAIN READABILITY OPTIMIZATION:
   - Keep readability_optimization standards (grade level, active voice percentage)
   - Preserve sentence length and structure optimizations
   - Maintain scan-friendly formatting (short paragraphs, bullet points, clear transitions)

4. INCLUDE SEO METADATA IN HTML:
   - Include meta_title in HTML <head> section as <title>
   - Include meta_description in HTML <head> section as <meta name="description">
   - Preserve schema_markup in HTML (include as JSON-LD script tag)

5. VALIDATE STRUCTURE:
   - Compare formatted structure against article outline
   - Ensure section organization matches outline
   - Verify all sections from outline are present

6. CREATE FORMATTED OUTPUT:
   - Clean, semantic HTML formatting (preserving SEO structure)
   - Markdown version (preserving SEO structure)
   - Section structure with proper headings (using preserved header_structure)
   - Table of contents with anchors (based on preserved header structure)
   - Reading time estimation
   - Formatting recommendations (ensuring SEO preservation)

CRITICAL: Do NOT break SEO optimizations during formatting. All SEO work from Step 4 must be preserved."""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the content formatting."

            return prompt

        elif step_name == "design_kit":
            article = context.get("article_generation", {})
            brief = context.get("marketing_brief", {})
            seo_keywords = context.get("seo_keywords", {})
            seo = context.get("seo_optimization", {})

            # Extract data safely
            article_title = article.get("article_title", title)
            target_audience = brief.get("target_audience", [])
            tone_and_voice = brief.get("tone_and_voice", "Professional and engaging")
            main_keyword = seo_keywords.get("main_keyword", "")
            primary_keywords = seo_keywords.get("primary_keywords", [])
            article_outline = article.get("outline", [])
            key_takeaways = article.get("key_takeaways", [])
            header_structure = seo.get("header_structure")
            readability_optimization = seo.get("readability_optimization")
            meta_title = seo.get("meta_title", "")
            meta_description = seo.get("meta_description", "")
            alt_texts = seo.get("alt_texts")

            prompt = f"""Recommend comprehensive design elements for this content:

CONTENT INFORMATION:
Title: {article_title}
Content Type: {category}

FROM STEP 1 (SEO Keywords):
- Main Keyword: {main_keyword}
- Primary Keywords: {', '.join(primary_keywords) if primary_keywords else 'None'}

FROM STEP 2 (Marketing Brief):
- Target Audience: {', '.join(target_audience) if target_audience else 'Not specified'}
- Tone and Voice: {tone_and_voice}

FROM STEP 3 (Article Generation):
- Article Outline: {', '.join(article_outline) if article_outline else 'None'}
- Key Takeaways: {', '.join(key_takeaways) if key_takeaways else 'None'}

FROM STEP 4 (SEO Optimization):
- Header Structure: {str(header_structure) if header_structure else 'Not available'}
- Readability Optimization: {str(readability_optimization) if readability_optimization else 'Not available'}
- Meta Title: {meta_title}
- Meta Description: {meta_description}
- Alt Texts: {str(alt_texts) if alt_texts else 'Not available'}

DESIGN RECOMMENDATIONS:

1. VISUAL HIERARCHY (Using Header Structure):
   - Use header_structure to determine H1 size, H2 spacing, H3 indentation
   - Create visual hierarchy that matches content structure
   - Ensure headings are visually distinct based on header_structure hierarchy
   - Use spacing and typography to reflect content organization

2. TYPOGRAPHY (Using Readability Optimization):
   - Use readability_optimization data for font size decisions
   - Adjust line heights based on readability grade level
   - Choose font sizes that support optimal reading experience
   - Ensure typography maintains readability score from optimization

3. ALT TEXT ENHANCEMENT (Using Keywords):
   - Build upon existing alt_texts from Step 4
   - Enhance alt text suggestions with primary keywords ({main_keyword})
   - Ensure all images have keyword-optimized, descriptive alt text
   - Improve accessibility while maintaining SEO benefits

4. DESIGN CHOICES (Using Target Audience):
   - Incorporate target audience preferences ({', '.join(target_audience) if target_audience else 'General audience'}) for color choices
   - Select layout style based on audience demographics and preferences
   - Choose visual elements that resonate with target audience
   - Ensure design aligns with audience expectations and consumption patterns

5. VISUAL EMPHASIS (Using Key Takeaways):
   - Use key takeaways to suggest callout boxes and highlights
   - Create visual emphasis for important takeaways
   - Design visual elements that draw attention to key messages
   - Use design to reinforce key takeaways visually

6. LAYOUT SUGGESTIONS (Using Article Outline):
   - Use article outline for section spacing recommendations
   - Suggest visual breaks at section boundaries
   - Create layout that reflects content structure
   - Ensure spacing matches content organization

7. SOCIAL SHARING DESIGN (Using Meta Data):
   - Consider meta_title and meta_description for social sharing card design
   - Suggest optimal image sizes for social previews
   - Ensure design works well for social media sharing
   - Create visual elements that enhance social sharing

Suggest:
- Visual components (images, charts, infographics) with keyword-optimized alt text
- Color scheme with hex codes (aligned with target audience preferences)
- Typography recommendations (optimized for readability from Step 4)
- Layout suggestions (based on article outline and header structure)
- Hero image concept (aligned with tone, audience, and keywords)
- Accessibility considerations (building upon existing alt texts and readability optimization)"""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the design recommendations."

            return prompt

        elif step_name == "transcript_preprocessing_approval":
            # Extract transcript data from input_data
            transcript_id = content.get("id", "N/A")
            transcript_title = content.get("title", "N/A")
            transcript_content = content.get("content", "")
            transcript_snippet = content.get("snippet", "")
            transcript_speakers = content.get("speakers", [])
            transcript_duration = content.get("duration")
            transcript_type = content.get("transcript_type")
            transcript_metadata = content.get("metadata", {})

            # Parsing information
            parsing_confidence = content.get("parsing_confidence")
            detected_format = content.get("detected_format")
            parsing_warnings = content.get("parsing_warnings", [])
            quality_metrics = content.get("quality_metrics", {})
            speaking_time_per_speaker = content.get("speaking_time_per_speaker", {})
            detected_language = content.get("detected_language")
            key_topics = content.get("key_topics", [])
            conversation_flow = content.get("conversation_flow", {})

            prompt = f"""Validate and approve the following transcript preprocessing data before proceeding to SEO keywords extraction.

TRANSCRIPT INFORMATION:
- Transcript ID: {transcript_id}
- Title: {transcript_title}
- Transcript Type: {transcript_type or 'Not specified'}
{f"- Detected Format: {detected_format}" if detected_format else ""}
{f"- Parsing Confidence: {parsing_confidence * 100:.1f}% ({'High' if parsing_confidence >= 0.8 else 'Medium' if parsing_confidence >= 0.5 else 'Low'})" if parsing_confidence is not None else ""}

SPEAKERS:
{f"- Number of Speakers: {len(transcript_speakers)}" if transcript_speakers else "- WARNING: No speakers found"}
{f"- Speaker List: {', '.join(transcript_speakers)}" if transcript_speakers else ""}

DURATION:
{f"- Duration: {transcript_duration} seconds ({transcript_duration / 60:.1f} minutes)" if transcript_duration else "- WARNING: Duration not specified"}

CONTENT:
- Content Length: {len(transcript_content)} characters
- Content Preview (first 500 chars): {transcript_content[:500] if transcript_content else 'No content available'}
- Snippet: {transcript_snippet or 'No snippet available'}

{f"PARSING WARNINGS ({len(parsing_warnings)}):" if parsing_warnings else ""}
{f"\n  - " + "\n  - ".join(parsing_warnings) if parsing_warnings else ""}

{f"\nQUALITY METRICS:" if quality_metrics else ""}
{f"\n  " + "\n  ".join([f"- {k}: {v * 100 if isinstance(v, float) and v <= 1 else v}%" for k, v in quality_metrics.items()]) if quality_metrics else ""}

{f"\nSPEAKING TIME PER SPEAKER:" if speaking_time_per_speaker else ""}
{f"\n  " + "\n  ".join([f"- {speaker}: {seconds / 60:.1f} minutes ({seconds} seconds)" for speaker, seconds in speaking_time_per_speaker.items()]) if speaking_time_per_speaker else ""}

{f"KEY TOPICS: {', '.join(key_topics)}" if key_topics else ""}

{f"METADATA: {json.dumps(transcript_metadata, indent=2)}" if transcript_metadata else "No additional metadata provided"}

VALIDATION REQUIREMENTS:
1. Validate speakers: At least one speaker must be present with reasonable names
2. Validate duration: Must be a positive integer (minimum 60 seconds)
3. Validate content: Must be non-empty and readable
4. Validate transcript_type: Should be specified (podcast, meeting, interview, video, etc.)
5. Check for parsing issues: Review parsing warnings and quality metrics
6. Determine if approval is required: Flag any issues that need human review

Output validation results with:
- is_valid: Overall validation status
- Individual validation flags (speakers_validated, duration_validated, content_validated, transcript_type_validated)
- validation_issues: List of any issues found
- requires_approval: Whether human approval is needed before proceeding"""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the transcript validation."

            return prompt

        elif step_name == "blog_post_preprocessing_approval":
            # Extract blog post data from input_data
            blog_post_id = content.get("id", "N/A")
            blog_post_title = content.get("title", "N/A")
            blog_post_content = content.get("content", "")
            blog_post_snippet = content.get("snippet", "")
            blog_post_author = content.get("author")
            blog_post_category = content.get("category")
            blog_post_tags = content.get("tags", [])
            blog_post_word_count = content.get("word_count")
            blog_post_reading_time = content.get("reading_time")
            blog_post_metadata = content.get("metadata", {})

            # Parsing information
            parsing_confidence = content.get("parsing_confidence")
            detected_format = content.get("detected_format")
            parsing_warnings = content.get("parsing_warnings", [])
            quality_metrics = content.get("quality_metrics", {})

            prompt = f"""Validate and approve the following blog post preprocessing data before proceeding to SEO keywords extraction.

BLOG POST INFORMATION:
- Blog Post ID: {blog_post_id}
- Title: {blog_post_title}
- Author: {blog_post_author or 'Not specified'}
- Category: {blog_post_category or 'Not specified'}
{f"- Detected Format: {detected_format}" if detected_format else ""}
{f"- Parsing Confidence: {parsing_confidence * 100:.1f}% ({'High' if parsing_confidence >= 0.8 else 'Medium' if parsing_confidence >= 0.5 else 'Low'})" if parsing_confidence is not None else ""}

TAGS:
{f"- Number of Tags: {len(blog_post_tags)}" if blog_post_tags else "- WARNING: No tags found"}
{f"- Tag List: {', '.join(blog_post_tags)}" if blog_post_tags else ""}

CONTENT METRICS:
{f"- Word Count: {blog_post_word_count} words" if blog_post_word_count else "- WARNING: Word count not specified"}
{f"- Reading Time: {blog_post_reading_time} minutes" if blog_post_reading_time else "- WARNING: Reading time not specified"}

CONTENT:
- Content Length: {len(blog_post_content)} characters
- Content Preview (first 500 chars): {blog_post_content[:500] if blog_post_content else 'No content available'}
- Snippet: {blog_post_snippet or 'No snippet available'}

{f"PARSING WARNINGS ({len(parsing_warnings)}):" if parsing_warnings else ""}
{f"\n  - " + "\n  - ".join(parsing_warnings) if parsing_warnings else ""}

{f"\nQUALITY METRICS:" if quality_metrics else ""}
{f"\n  " + "\n  ".join([f"- {k}: {v * 100 if isinstance(v, float) and v <= 1 else v}%" for k, v in quality_metrics.items()]) if quality_metrics else ""}

{f"METADATA: {json.dumps(blog_post_metadata, indent=2)}" if blog_post_metadata else "No additional metadata provided"}

VALIDATION AND EXTRACTION REQUIREMENTS:
1. Validate title: Must be present, non-empty, and reasonable length (10-200 chars)
2. Validate content: Must be non-empty, minimum 100 words, readable
3. Extract/validate author: Extract from content if missing (look for "by Author", "Author:", etc.)
4. Extract/validate category: Infer from content if missing
5. Extract/validate tags: Extract hashtags or generate from content if missing
6. Calculate word_count: Calculate from content if missing
7. Calculate reading_time: Calculate from word_count if missing (~200 words/minute)
8. Generate snippet: Generate from first paragraph if missing
9. Perform sentiment analysis: Analyze overall sentiment, sentiment score, emotional tone
10. Perform content analysis: Calculate readability, completeness, identify content type, target audience
11. Perform SEO/marketing analysis: Extract keywords, identify opportunities, assess engagement
12. Determine if approval is required: Flag any issues that need human review

Output validation results with:
- is_valid: Overall validation status
- Individual validation flags (title_validated, content_validated, author_validated, category_validated, tags_validated)
- Extracted data: author, category, tags, word_count, reading_time
- Sentiment analysis: overall_sentiment, sentiment_score, sentiment_confidence, emotional_tone
- Content analysis: readability_score, completeness_score, content_type, target_audience, headings, sections
- SEO/marketing analysis: potential_keywords, seo_opportunities, engagement_potential, shareability_score
- validation_issues: List of any issues found
- requires_approval: Whether human approval is needed before proceeding"""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the blog post validation."

            return prompt

        elif step_name == "social_media_marketing_brief":
            seo_keywords = context.get("seo_keywords", {})
            social_media_platform = context.get("social_media_platform", "linkedin")
            main_keyword = seo_keywords.get("main_keyword", "")
            primary_keywords = seo_keywords.get("primary_keywords", [])
            search_intent = seo_keywords.get("search_intent", "informational")
            long_tail_keywords = seo_keywords.get("long_tail_keywords", [])

            prompt = f"""Create a platform-specific marketing brief for {social_media_platform} focused on the keyword: {main_keyword}

KEYWORD CONTEXT:
- Main Keyword: {main_keyword}
- Primary Keywords: {', '.join(primary_keywords) if primary_keywords else 'None'}
- Search Intent: {search_intent}
{f"- Long-tail Opportunities: {', '.join(long_tail_keywords)}" if long_tail_keywords else ""}

PLATFORM: {social_media_platform.upper()}

Generate a marketing brief that includes:
- Target audience personas with demographics specific to {social_media_platform}
- Key messaging points (3-5 messages) aligned with search intent ({search_intent})
- Platform-specific tone and voice recommendations
- Content strategy tailored for {social_media_platform}
- Distribution and engagement strategy for {social_media_platform}
- Platform-specific notes and recommendations

Ensure the brief is optimized for {social_media_platform} audience and engagement patterns."""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the social media marketing brief."

            return prompt

        elif step_name == "social_media_angle_hook":
            brief = context.get("social_media_marketing_brief", {})
            social_media_platform = context.get("social_media_platform", "linkedin")
            platform = brief.get("platform", social_media_platform)
            target_audience = brief.get("target_audience", [])
            key_messages = brief.get("key_messages", [])
            tone_and_voice = brief.get("tone_and_voice", "Professional and engaging")

            prompt = f"""Generate engaging angles and hooks for {platform} social media content.

FROM MARKETING BRIEF:
- Platform: {platform}
- Target Audience: {', '.join(target_audience) if target_audience else 'Not specified'}
- Key Messages: {', '.join(key_messages) if key_messages else 'None'}
- Tone and Voice: {tone_and_voice}

Generate:
- Multiple angle options (3-5 angles) for approaching the content
- Hook variations (3-5 hooks) for capturing attention on {platform}
- Recommended angle: The best angle to use for this platform
- Recommended hook: The best hook to use for this platform
- Rationale: Explanation of why the recommended angle and hook were chosen

Ensure angles and hooks are:
- Tailored for {platform} audience and engagement patterns
- Aligned with the tone and voice: {tone_and_voice}
- Designed to capture attention and drive engagement
- Relevant to the target audience: {', '.join(target_audience) if target_audience else 'General audience'}"""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the angle and hook generation."

            return prompt

        elif step_name == "social_media_post_generation":
            angle_hook = context.get("social_media_angle_hook", {})
            brief = context.get("social_media_marketing_brief", {})
            social_media_platform = context.get("social_media_platform", "linkedin")
            platform = angle_hook.get(
                "platform", brief.get("platform", social_media_platform)
            )
            recommended_angle = angle_hook.get("recommended_angle", "")
            recommended_hook = angle_hook.get("recommended_hook", "")
            rationale = angle_hook.get("rationale", "")
            target_audience = brief.get("target_audience", [])
            key_messages = brief.get("key_messages", [])
            tone_and_voice = brief.get("tone_and_voice", "Professional and engaging")

            prompt = f"""Generate a complete social media post for {platform} using the recommended angle and hook.

FROM ANGLE & HOOK:
- Platform: {platform}
- Recommended Angle: {recommended_angle}
- Recommended Hook: {recommended_hook}
- Rationale: {rationale}

FROM MARKETING BRIEF:
- Target Audience: {', '.join(target_audience) if target_audience else 'Not specified'}
- Key Messages: {', '.join(key_messages) if key_messages else 'None'}
- Tone and Voice: {tone_and_voice}

Generate:
- Complete post content optimized for {platform}
{f"- Subject line (for email platform)" if platform == "email" else ""}
{f"- Hashtags (for LinkedIn platform)" if platform == "linkedin" else ""}
- Call-to-action (if appropriate)
- Platform-specific metadata

Ensure the post:
- Uses the recommended hook to capture attention
- Follows the recommended angle for content approach
- Matches the tone and voice: {tone_and_voice}
- Speaks directly to the target audience: {', '.join(target_audience) if target_audience else 'General audience'}
- Incorporates key messages naturally
- Is optimized for {platform} engagement patterns
- Includes appropriate formatting for {platform}"""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the social media post generation."

            return prompt

        else:
            # Fallback generic prompt
            prompt = f"""Process this content for the {step_name} step:

Title: {title}
Content: {content_text[:1000]}

Perform the required analysis and generate appropriate output."""

            # Add user guidance if provided
            if user_guidance:
                prompt += f"\n\nIMPORTANT: User feedback for improvement:\n{user_guidance}\n\nPlease incorporate this feedback into the output."

            return prompt


# Global service instance
_retry_service: Optional[StepRetryService] = None


def get_retry_service() -> StepRetryService:
    """Get or create the global retry service instance."""
    global _retry_service
    if _retry_service is None:
        _retry_service = StepRetryService()
    return _retry_service
