"""
SEO Keywords plugin tasks for Marketing Project.

This plugin handles SEO keyword extraction and analysis.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from marketing_project.models.pipeline_steps import EngineConfig, SEOKeywordsResult
from marketing_project.plugins.base import PipelineStepPlugin
from marketing_project.services.engines.composer import EngineComposer
from marketing_project.services.engines.registry import register_engine
from marketing_project.services.engines.seo_keywords.composer import SEOKeywordsComposer
from marketing_project.services.engines.seo_keywords.llm_engine import (
    LLMSEOKeywordsEngine,
)
from marketing_project.services.engines.seo_keywords.local_semantic_engine import (
    LocalSemanticSEOKeywordsEngine,
)

logger = logging.getLogger("marketing_project.plugins.seo_keywords")


class SEOKeywordsPlugin(PipelineStepPlugin):
    """Plugin for SEO Keywords extraction step."""

    @property
    def step_name(self) -> str:
        return "seo_keywords"

    @property
    def step_number(self) -> int:
        return 2

    @property
    def response_model(self) -> type[SEOKeywordsResult]:
        return SEOKeywordsResult

    def get_required_context_keys(self) -> list[str]:
        return ["input_content"]

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build prompt context for SEO keywords step.

        Handles content analysis and provides enhanced context for keyword extraction.
        """
        prompt_context = super()._build_prompt_context(context)

        # Handle content analysis for seo_keywords step
        if "content" in prompt_context:
            content = prompt_context.get("content", {})
            if isinstance(content, dict):
                content_str = content.get("content", "")
                # Increase truncation limit from 2000 to 8000 characters
                prompt_context["content_content_preview"] = (
                    content_str[:8000] if content_str else ""
                )

                # Add content structure analysis
                prompt_context["content_structure"] = self._analyze_content_structure(
                    content_str
                )

                # Extract key sections
                prompt_context["key_sections"] = self._extract_key_sections(content_str)

                # Add word count
                prompt_context["word_count"] = (
                    len(content_str.split()) if content_str else 0
                )
            else:
                prompt_context["content_content_preview"] = ""
                prompt_context["content_structure"] = {}
                prompt_context["key_sections"] = {}
                prompt_context["word_count"] = 0

        # Add content type
        prompt_context["content_type"] = context.get("content_type", "blog_post")

        return prompt_context

    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """
        Extract content structure for better keyword analysis.

        Args:
            content: Full content text

        Returns:
            Dict with structure analysis (headings, paragraphs, topics)
        """
        if not content:
            return {}

        # Extract headings
        h1_pattern = r"<h1[^>]*>(.*?)</h1>|^#\s+(.+)$"
        h2_pattern = r"<h2[^>]*>(.*?)</h2>|^##\s+(.+)$"
        h3_pattern = r"<h3[^>]*>(.*?)</h3>|^###\s+(.+)$"

        h1_matches = re.findall(h1_pattern, content, re.MULTILINE | re.IGNORECASE)
        h2_matches = re.findall(h2_pattern, content, re.MULTILINE | re.IGNORECASE)
        h3_matches = re.findall(h3_pattern, content, re.MULTILINE | re.IGNORECASE)

        # Clean heading text (handle tuple results from regex)
        h1_headings = [h[0] or h[1] for h in h1_matches if h[0] or h[1]]
        h2_headings = [h[0] or h[1] for h in h2_matches if h[0] or h[1]]
        h3_headings = [h[0] or h[1] for h in h3_matches if h[0] or h[1]]

        # Count paragraphs (approximate)
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        return {
            "h1_count": len(h1_headings),
            "h2_count": len(h2_headings),
            "h3_count": len(h3_headings),
            "h1_headings": h1_headings[:5],  # Limit to first 5
            "h2_headings": h2_headings[:10],  # Limit to first 10
            "h3_headings": h3_headings[:15],  # Limit to first 15
            "paragraph_count": len(paragraphs),
            "main_topics": (
                h2_headings[:5] if h2_headings else h1_headings[:3]
            ),  # Use H2s as main topics, fallback to H1s
        }

    def _extract_key_sections(self, content: str) -> Dict[str, str]:
        """
        Extract key content sections for targeted keyword placement.

        Args:
            content: Full content text

        Returns:
            Dict with introduction, body_topics, and conclusion
        """
        if not content:
            return {}

        # Split content into logical sections
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        # Introduction: first 2-3 paragraphs or first 500 chars
        intro_paragraphs = paragraphs[:3] if len(paragraphs) >= 3 else paragraphs[:2]
        introduction = " ".join(intro_paragraphs)[:500]

        # Body topics: extract from middle section (skip first 20% and last 20%)
        body_start = max(1, len(paragraphs) // 5)
        body_end = max(body_start + 1, len(paragraphs) - (len(paragraphs) // 5))
        body_paragraphs = (
            paragraphs[body_start:body_end]
            if body_end > body_start
            else paragraphs[1:-1]
        )
        body_text = " ".join(body_paragraphs[:5])[:1000] if body_paragraphs else ""

        # Extract topic keywords from body (simple approach: use first few significant words)
        body_words = re.findall(r"\b[a-zA-Z]{4,}\b", body_text.lower())
        body_topics = list(set(body_words))[:10]  # Unique significant words

        # Conclusion: last 1-2 paragraphs or last 300 chars
        conclusion_paragraphs = (
            paragraphs[-2:] if len(paragraphs) >= 2 else paragraphs[-1:]
        )
        conclusion = " ".join(conclusion_paragraphs)[:300]

        return {
            "introduction": introduction,
            "body_topics": body_topics,
            "conclusion": conclusion,
        }

    def _post_process_keywords(
        self, result: SEOKeywordsResult, context: Dict[str, Any]
    ) -> SEOKeywordsResult:
        """
        Post-process keywords to improve quality.

        Args:
            result: SEOKeywordsResult from LLM
            context: Execution context

        Returns:
            Post-processed SEOKeywordsResult
        """
        # Ensure main_keyword is in primary_keywords
        if result.main_keyword not in result.primary_keywords:
            result.primary_keywords.insert(0, result.main_keyword)

        # Deduplicate keywords across categories
        result = self._deduplicate_keywords(result)

        # Normalize keyword formatting
        result = self._normalize_keywords(result)

        # Validate keyword counts
        result = self._validate_keyword_counts(result)

        return result

    def _deduplicate_keywords(self, result: SEOKeywordsResult) -> SEOKeywordsResult:
        """Remove duplicate keywords across categories."""
        # Normalize all keywords for comparison
        primary_normalized = {
            self._normalize_keyword(kw): kw for kw in result.primary_keywords
        }
        secondary_normalized = {}
        lsi_normalized = {}
        long_tail_normalized = {}

        if result.secondary_keywords:
            secondary_normalized = {
                self._normalize_keyword(kw): kw for kw in result.secondary_keywords
            }

        if result.lsi_keywords:
            lsi_normalized = {
                self._normalize_keyword(kw): kw for kw in result.lsi_keywords
            }

        if result.long_tail_keywords:
            long_tail_normalized = {
                self._normalize_keyword(kw): kw for kw in result.long_tail_keywords
            }

        # Remove duplicates from secondary that are in primary
        if result.secondary_keywords:
            result.secondary_keywords = [
                kw
                for kw in result.secondary_keywords
                if self._normalize_keyword(kw) not in primary_normalized
            ]

        # Remove duplicates from LSI that are in primary or secondary
        if result.lsi_keywords:
            all_primary_secondary = set(primary_normalized.keys()) | set(
                secondary_normalized.keys()
            )
            result.lsi_keywords = [
                kw
                for kw in result.lsi_keywords
                if self._normalize_keyword(kw) not in all_primary_secondary
            ]

        # Remove duplicates from long_tail that are in other categories
        if result.long_tail_keywords:
            all_other = (
                set(primary_normalized.keys())
                | set(secondary_normalized.keys())
                | set(lsi_normalized.keys())
            )
            result.long_tail_keywords = [
                kw
                for kw in result.long_tail_keywords
                if self._normalize_keyword(kw) not in all_other
            ]

        return result

    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize keyword for comparison (lowercase, trimmed)."""
        return keyword.strip().lower()

    def _normalize_keywords(self, result: SEOKeywordsResult) -> SEOKeywordsResult:
        """Standardize keyword formatting (trim whitespace)."""
        result.main_keyword = result.main_keyword.strip()
        result.primary_keywords = [
            kw.strip() for kw in result.primary_keywords if kw.strip()
        ]

        if result.secondary_keywords:
            result.secondary_keywords = [
                kw.strip() for kw in result.secondary_keywords if kw.strip()
            ]

        if result.lsi_keywords:
            result.lsi_keywords = [
                kw.strip() for kw in result.lsi_keywords if kw.strip()
            ]

        if result.long_tail_keywords:
            result.long_tail_keywords = [
                kw.strip() for kw in result.long_tail_keywords if kw.strip()
            ]

        return result

    def _validate_keyword_counts(self, result: SEOKeywordsResult) -> SEOKeywordsResult:
        """Ensure keyword counts meet requirements."""
        # Primary keywords: 3-5
        if len(result.primary_keywords) < 3:
            logger.warning(
                f"Only {len(result.primary_keywords)} primary keywords, minimum is 3"
            )
        elif len(result.primary_keywords) > 5:
            # Keep main_keyword + top 4
            result.primary_keywords = [result.main_keyword] + [
                kw for kw in result.primary_keywords if kw != result.main_keyword
            ][:4]

        # Secondary keywords: 5-10
        if result.secondary_keywords:
            if len(result.secondary_keywords) < 5:
                logger.warning(
                    f"Only {len(result.secondary_keywords)} secondary keywords, minimum is 5"
                )
            elif len(result.secondary_keywords) > 10:
                result.secondary_keywords = result.secondary_keywords[:10]

        # LSI keywords: minimum 5
        if not result.lsi_keywords or len(result.lsi_keywords) < 5:
            logger.warning(
                f"Only {len(result.lsi_keywords) if result.lsi_keywords else 0} LSI keywords, minimum is 5"
            )

        return result

    def _validate_and_fix(
        self, result: SEOKeywordsResult, context: Dict[str, Any]
    ) -> SEOKeywordsResult:
        """
        Validate result and fix common issues.

        Args:
            result: SEOKeywordsResult to validate
            context: Execution context

        Returns:
            Fixed SEOKeywordsResult
        """
        # Ensure primary_keywords has 3-5 keywords
        if len(result.primary_keywords) < 3:
            # Try to promote from secondary
            needed = 3 - len(result.primary_keywords)
            if result.secondary_keywords and len(result.secondary_keywords) >= needed:
                promoted = result.secondary_keywords[:needed]
                result.primary_keywords.extend(promoted)
                result.secondary_keywords = result.secondary_keywords[needed:]
                logger.info(f"Promoted {needed} keywords from secondary to primary")

        if len(result.primary_keywords) > 5:
            # Keep main_keyword + top 4 by relevance
            result.primary_keywords = [result.main_keyword] + [
                kw for kw in result.primary_keywords if kw != result.main_keyword
            ][:4]

        # Ensure secondary_keywords has 5-10
        if result.secondary_keywords:
            if len(result.secondary_keywords) < 5:
                # Try to generate more or promote from LSI
                if result.lsi_keywords and len(result.lsi_keywords) >= (
                    5 - len(result.secondary_keywords)
                ):
                    needed = 5 - len(result.secondary_keywords)
                    promoted = result.lsi_keywords[:needed]
                    result.secondary_keywords.extend(promoted)
                    result.lsi_keywords = result.lsi_keywords[needed:]
            elif len(result.secondary_keywords) > 10:
                result.secondary_keywords = result.secondary_keywords[:10]
        else:
            # Initialize if missing
            result.secondary_keywords = []

        # Ensure LSI keywords exist (minimum 5)
        if not result.lsi_keywords or len(result.lsi_keywords) < 5:
            result.lsi_keywords = self._generate_lsi_keywords(result, context)

        return result

    def _generate_lsi_keywords(
        self, result: SEOKeywordsResult, context: Dict[str, Any]
    ) -> List[str]:
        """
        Generate LSI keywords as fallback if missing.

        Args:
            result: Current SEOKeywordsResult
            context: Execution context

        Returns:
            List of LSI keywords
        """
        # Simple fallback: extract related terms from primary keywords
        lsi_keywords = []

        # Use primary keywords to generate variations
        for keyword in result.primary_keywords[:3]:  # Use top 3 primary
            words = keyword.split()
            if len(words) > 1:
                # Add individual words as LSI
                lsi_keywords.extend([w for w in words if len(w) > 3])

            # Add common variations
            if " " in keyword:
                # Add without spaces, with hyphens
                lsi_keywords.append(keyword.replace(" ", "-"))
                lsi_keywords.append(keyword.replace(" ", ""))

        # Add from secondary if available
        if result.secondary_keywords:
            lsi_keywords.extend(result.secondary_keywords[:3])

        # Deduplicate and limit
        seen = set()
        unique_lsi = []
        for kw in lsi_keywords:
            normalized = self._normalize_keyword(kw)
            if normalized not in seen and normalized not in [
                self._normalize_keyword(k) for k in result.primary_keywords
            ]:
                seen.add(normalized)
                unique_lsi.append(kw)

        # Ensure we have at least 5
        while len(unique_lsi) < 5:
            unique_lsi.append(f"related term {len(unique_lsi) + 1}")

        return unique_lsi[:10]  # Max 10

    def _calculate_derived_metrics(
        self, result: SEOKeywordsResult, context: Dict[str, Any]
    ) -> SEOKeywordsResult:
        """
        Calculate metrics that weren't provided by LLM.

        Args:
            result: SEOKeywordsResult
            context: Execution context

        Returns:
            SEOKeywordsResult with calculated metrics
        """
        # Get content for density calculation
        content = context.get("input_content", {})
        content_str = content.get("content", "") if isinstance(content, dict) else ""

        # Calculate keyword density if missing
        if not result.keyword_density_analysis and content_str:
            result.keyword_density_analysis = self._calculate_keyword_density(
                content_str, result.primary_keywords + (result.secondary_keywords or [])
            )

        # Also populate legacy keyword_density for backward compatibility
        if not result.keyword_density and result.keyword_density_analysis:
            result.keyword_density = {
                analysis.keyword: analysis.current_density
                for analysis in result.keyword_density_analysis
            }

        # Calculate relevance scores if missing
        if not result.relevance_score:
            result.relevance_score = self._calculate_overall_relevance(
                result, content_str
            )

        # Convert keyword_difficulty from string to Dict if needed
        if isinstance(result.keyword_difficulty, str):
            result.keyword_difficulty = self._convert_difficulty_to_scores(
                result.keyword_difficulty, result.primary_keywords
            )

        return result

    def _calculate_keyword_density(self, content: str, keywords: List[str]) -> List:
        """
        Calculate keyword density from content.

        Args:
            content: Content text
            keywords: Keywords to analyze

        Returns:
            List of KeywordDensityAnalysis objects
        """
        from marketing_project.models.pipeline_steps import KeywordDensityAnalysis

        if not content or not keywords:
            return []

        content_lower = content.lower()
        total_words = len(content.split())
        density_analyses = []

        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Count occurrences
            occurrences = content_lower.count(keyword_lower)

            # Calculate density
            keyword_word_count = len(keyword.split())
            current_density = (
                (occurrences * keyword_word_count / total_words * 100)
                if total_words > 0
                else 0.0
            )

            # Optimal density: 1-3% for primary, 0.5-1% for secondary
            optimal_density = 2.0 if keyword in keywords[:5] else 0.75

            # Find placement locations
            placement_locations = []
            if keyword_lower in content_lower[:200]:  # First 200 chars (likely intro)
                placement_locations.append("introduction")

            # Check for headings
            if re.search(rf"<h[1-3][^>]*>{re.escape(keyword)}", content, re.IGNORECASE):
                placement_locations.append("heading")

            if not placement_locations:
                placement_locations.append("body")

            density_analyses.append(
                KeywordDensityAnalysis(
                    keyword=keyword,
                    current_density=round(current_density, 2),
                    optimal_density=optimal_density,
                    occurrences=occurrences,
                    placement_locations=placement_locations,
                )
            )

        return density_analyses

    def _calculate_overall_relevance(
        self, result: SEOKeywordsResult, content: str
    ) -> float:
        """
        Calculate overall relevance score from keyword quality.

        Args:
            result: SEOKeywordsResult
            content: Content text

        Returns:
            Relevance score (0-100)
        """
        if not content:
            return 50.0  # Default if no content

        score = 0.0
        max_score = 100.0

        # Check if main keyword appears in content (40 points)
        if result.main_keyword.lower() in content.lower():
            score += 40.0

        # Check primary keywords coverage (30 points)
        primary_in_content = sum(
            1 for kw in result.primary_keywords if kw.lower() in content.lower()
        )
        score += (primary_in_content / len(result.primary_keywords)) * 30.0

        # Check secondary keywords coverage (20 points)
        if result.secondary_keywords:
            secondary_in_content = sum(
                1 for kw in result.secondary_keywords if kw.lower() in content.lower()
            )
            score += (secondary_in_content / len(result.secondary_keywords)) * 20.0

        # Check LSI keywords coverage (10 points)
        if result.lsi_keywords:
            lsi_in_content = sum(
                1 for kw in result.lsi_keywords if kw.lower() in content.lower()
            )
            score += min((lsi_in_content / len(result.lsi_keywords)) * 10.0, 10.0)

        return round(score, 1)

    def _convert_difficulty_to_scores(
        self, difficulty_str: str, keywords: List[str]
    ) -> Dict[str, float]:
        """
        Convert string difficulty to numeric scores per keyword.

        Args:
            difficulty_str: String like "easy", "medium", "hard"
            keywords: List of keywords

        Returns:
            Dict mapping keywords to difficulty scores
        """
        # Map string to numeric range
        difficulty_map = {"easy": 30.0, "medium": 50.0, "hard": 75.0}

        base_score = difficulty_map.get(difficulty_str.lower(), 50.0)

        # Create scores for each keyword (slight variation)
        scores = {}
        for i, keyword in enumerate(keywords):
            # Add slight variation based on keyword position
            variation = (i % 3) * 5.0  # 0, 5, or 10 point variation
            scores[keyword] = min(100.0, max(0.0, base_score + variation - 5.0))

        return scores

    def _validate_result(self, result: SEOKeywordsResult) -> Tuple[bool, List[str]]:
        """
        Validate SEO keywords result and return (is_valid, errors).

        Args:
            result: SEOKeywordsResult to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # 1. Main keyword validation
        if not result.main_keyword or len(result.main_keyword.strip()) < 2:
            errors.append("Main keyword is required and must be at least 2 characters")

        # 2. Primary keywords validation
        if len(result.primary_keywords) < 3:
            errors.append(
                f"Need at least 3 primary keywords, got {len(result.primary_keywords)}"
            )
        if len(result.primary_keywords) > 5:
            errors.append(
                f"Should have max 5 primary keywords, got {len(result.primary_keywords)}"
            )
        if result.main_keyword not in result.primary_keywords:
            errors.append("Main keyword must be included in primary_keywords list")

        # 3. Secondary keywords validation
        if result.secondary_keywords:
            if len(result.secondary_keywords) < 5:
                errors.append(
                    f"Should have at least 5 secondary keywords, got {len(result.secondary_keywords)}"
                )
            if len(result.secondary_keywords) > 10:
                errors.append(
                    f"Should have max 10 secondary keywords, got {len(result.secondary_keywords)}"
                )

        # 4. LSI keywords validation
        if not result.lsi_keywords or len(result.lsi_keywords) < 5:
            errors.append(
                f"Should have at least 5 LSI keywords, got {len(result.lsi_keywords) if result.lsi_keywords else 0}"
            )

        # 5. Search intent validation
        valid_intents = ["informational", "transactional", "navigational", "commercial"]
        if result.search_intent not in valid_intents:
            errors.append(
                f"Search intent must be one of {valid_intents}, got '{result.search_intent}'"
            )

        # 6. Quality score validation
        if result.relevance_score and result.relevance_score < 60:
            errors.append(
                f"Relevance score is low ({result.relevance_score}), should be >= 60"
            )

        # 7. Keyword overlap validation
        if result.secondary_keywords:
            overlap = set(result.primary_keywords) & set(result.secondary_keywords)
            if overlap:
                errors.append(
                    f"Keywords appear in both primary and secondary: {overlap}"
                )

        return len(errors) == 0, errors

    async def execute(
        self, context: Dict[str, Any], pipeline: Any, job_id: Optional[str] = None
    ) -> SEOKeywordsResult:
        """
        Execute SEO keywords extraction step with post-processing and validation.

        Args:
            context: Context containing input_content
            pipeline: FunctionPipeline instance
            job_id: Optional job ID for tracking

        Returns:
            SEOKeywordsResult with extracted keywords
        """
        # Step 1: Get engine configuration
        engine_config = self._get_engine_config(context, pipeline)

        # Step 2: Register engines if not already registered
        self._ensure_engines_registered()

        # Step 3: Create LLM engine instance (needs plugin)
        llm_engine = LLMSEOKeywordsEngine(self)
        from marketing_project.services.engines.registry import register_engine

        register_engine("llm", llm_engine)

        # Step 4: Create composer and extract keywords
        composer = EngineComposer(
            default_engine_type=engine_config.default_engine,
            field_overrides=engine_config.field_overrides,
        )
        seo_composer = SEOKeywordsComposer(composer)

        # Step 5: Get content
        content = context.get("input_content", {})

        # Log input_content state for debugging
        if isinstance(content, dict):
            logger.info(
                f"SEO keywords step received input_content with keys: {list(content.keys())}, "
                f"author={content.get('author')}, category={content.get('category')}, "
                f"tags={content.get('tags')}, word_count={content.get('word_count')}, "
                f"has_content={'content' in content and bool(content.get('content'))}"
            )
        else:
            logger.warning(
                f"SEO keywords step received input_content that is not a dict: {type(content)}"
            )

        # Check if blog_post_preprocessing_approval result is in context
        if "blog_post_preprocessing_approval" in context:
            logger.info(
                f"blog_post_preprocessing_approval result found in context: "
                f"is_valid={context['blog_post_preprocessing_approval'].get('is_valid') if isinstance(context['blog_post_preprocessing_approval'], dict) else 'N/A'}"
            )

        # Step 6: Compose result from engines
        result = await seo_composer.compose_result(content, context, pipeline)

        # Step 7: Post-process keywords
        result = self._post_process_keywords(result, context)

        # Step 8: Validate and fix issues
        result = self._validate_and_fix(result, context)

        # Step 9: Calculate derived metrics
        result = self._calculate_derived_metrics(result, context)

        # Step 10: Validate final result
        is_valid, errors = self._validate_result(result)
        if not is_valid:
            logger.warning(f"SEO keywords validation issues: {errors}")
            # Try to fix errors automatically
            result = self._validate_and_fix(result, context)

        # Step 11: Check for approval requirement (since we bypassed _execute_step)
        # This ensures SEO keywords step respects approval settings
        if job_id and pipeline:
            try:
                from marketing_project.services.function_pipeline.approval import (
                    check_step_approval,
                )

                step_start_time = time.time()
                execution_step_number = context.get(
                    "_execution_step_number", self.step_number
                )

                # Get prompt and system instruction for approval context
                prompt_context = self._build_prompt_context(context)
                prompt = pipeline._get_user_prompt(self.step_name, prompt_context)
                system_instruction = pipeline._get_system_instruction(
                    self.step_name, context=prompt_context
                )

                # Check approval
                approval_result = await check_step_approval(
                    parsed_result=result,
                    step_name=self.step_name,
                    step_number=execution_step_number,
                    job_id=job_id,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    context=context,
                    start_time=step_start_time,
                    step_info_list=[],  # Will be handled by pipeline
                )

                # If approval is required, return sentinel to stop pipeline
                if approval_result.requires_approval:
                    from marketing_project.processors.approval_helper import (
                        ApprovalRequiredSentinel,
                    )

                    logger.info(
                        f"SEO keywords step requires approval (approval_id: {approval_result.approval_id}). "
                        f"Returning sentinel to stop pipeline."
                    )
                    return ApprovalRequiredSentinel(approval_result)
            except Exception as approval_error:
                from marketing_project.processors.approval_helper import (
                    ApprovalCheckFailedException,
                )

                if isinstance(approval_error, ApprovalCheckFailedException):
                    # Re-raise approval check failures - these are critical
                    raise
                else:
                    # Catch any other errors in approval check (e.g., bugs, import errors)
                    # Log the error but allow pipeline to continue to prevent blocking on approval bugs
                    logger.error(
                        f"[APPROVAL] Unexpected error in approval check for SEO keywords step: {approval_error}. "
                        f"Continuing pipeline execution. Error type: {type(approval_error).__name__}",
                        exc_info=True,
                    )
                    # Continue pipeline execution - approval check failed but we don't want to block

        return result

    def _get_engine_config(
        self, context: Dict[str, Any], pipeline: Any
    ) -> EngineConfig:
        """
        Get engine configuration from context or pipeline config.

        Args:
            context: Execution context
            pipeline: FunctionPipeline instance

        Returns:
            EngineConfig with default and overrides
        """
        # Check context first
        if "seo_keywords_engine_config" in context:
            config = context["seo_keywords_engine_config"]
            if isinstance(config, dict):
                return EngineConfig(**config)
            elif isinstance(config, EngineConfig):
                return config

        # Check pipeline config (top-level)
        if hasattr(pipeline, "pipeline_config") and pipeline.pipeline_config:
            # Check top-level seo_keywords_engine_config
            if hasattr(pipeline.pipeline_config, "seo_keywords_engine_config"):
                config = pipeline.pipeline_config.seo_keywords_engine_config
                if config:
                    return config

            # Check step config
            step_config = pipeline.pipeline_config.get_step_config("seo_keywords")
            if step_config and step_config.seo_keywords_engine_config:
                return step_config.seo_keywords_engine_config

        # Default: LLM only
        return EngineConfig(default_engine="llm", field_overrides=None)

    def _ensure_engines_registered(self) -> None:
        """Ensure engines are registered in the global registry."""
        from marketing_project.services.engines.registry import get_registry

        registry = get_registry()

        # Register LLM engine (create new instance each time since it needs plugin)
        # Note: We'll create it on-demand in the composer instead
        # Just ensure local semantic is registered
        if not registry.has("local_semantic"):
            local_engine = LocalSemanticSEOKeywordsEngine()
            register_engine("local_semantic", local_engine)
