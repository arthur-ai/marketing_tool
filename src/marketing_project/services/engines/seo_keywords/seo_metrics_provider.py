"""
SEO metrics provider for keyword difficulty and search volume.

Integrates:
- OpenPageRank API for domain authority
- LLM-based SERP analysis (alternative to scraping)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from pydantic import BaseModel, Field

from marketing_project.models.pipeline_steps import KeywordMetadata

logger = logging.getLogger(__name__)


class SERPAnalysisResult(BaseModel):
    """Pydantic model for SERP analysis results."""

    result_count_estimate: str = Field(
        description="Estimated number of search results (number or range, e.g., '1000000' or '500000-2000000')"
    )
    typical_domains: List[str] = Field(
        default_factory=list,
        description="Types of domains that would rank (e.g., 'wikipedia.org', 'example.com')",
    )
    content_types: List[str] = Field(
        default_factory=list,
        description="Content types in top results (e.g., 'blog_post', 'guide', 'documentation', 'commercial')",
    )
    avg_domain_authority_estimate: float = Field(
        default=50.0,
        description="Estimated average domain authority of ranking sites (0-100 scale)",
    )
    competition_level: str = Field(
        default="medium",
        description="Competition level: 'low', 'medium', or 'high'",
    )
    serp_characteristics: str = Field(
        default="",
        description="Brief description of what dominates the search results",
    )


class SEOMetricsProvider:
    """
    Provider for SEO metrics (difficulty, search volume).

    Integrates:
    - OpenPageRank API for domain authority
    - LLM-based SERP analysis for competition insights
    """

    def __init__(
        self,
        openpagerank_api_key: Optional[str] = None,
        pipeline: Optional[Any] = None,
    ):
        """
        Initialize the SEO metrics provider.

        Args:
            openpagerank_api_key: OpenPageRank API key
            pipeline: Optional pipeline instance for LLM-based SERP analysis
        """
        self.openpagerank_api_key = openpagerank_api_key or os.getenv(
            "OPENPAGERANK_API_KEY", "owg04c8wgckoo0gk0c0go84s8gw48g0cso04080k"
        )
        self.pipeline = pipeline
        # Cache for SERP analysis results (limited to prevent memory issues)
        self._serp_cache: Dict[str, Dict[str, Any]] = {}
        self._serp_cache_max_size = int(os.getenv("SEO_METRICS_CACHE_SIZE", "200"))

    async def get_keyword_difficulty(
        self, keywords: List[str], serp_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Get keyword difficulty scores (0-100) using OpenPageRank + SERP analysis.

        Args:
            keywords: List of keywords
            serp_data: Optional SERP analysis data from LLM

        Returns:
            Dict mapping keywords to difficulty scores
        """
        difficulty_scores = {}

        for keyword in keywords:
            # Base difficulty from keyword characteristics
            base_difficulty = self._calculate_base_difficulty(keyword)

            # Adjust based on SERP data if available
            if serp_data and keyword in serp_data:
                serp_info = serp_data[keyword]
                # More results + stronger domains = harder
                result_count = serp_info.get("result_count", 0)
                avg_domain_authority = serp_info.get("avg_domain_authority", 0)

                # Adjust difficulty based on competition
                if result_count > 1000000:  # Very competitive
                    base_difficulty += 20
                elif result_count > 100000:
                    base_difficulty += 10

                if avg_domain_authority > 70:  # Strong domains
                    base_difficulty += 15
                elif avg_domain_authority > 50:
                    base_difficulty += 8

            # Adjust based on OpenPageRank if we have domain data
            # (Would need to fetch top domains for keyword first)

            difficulty_scores[keyword] = max(0.0, min(100.0, base_difficulty))

        return difficulty_scores

    def _calculate_base_difficulty(self, keyword: str) -> float:
        """Calculate base difficulty from keyword characteristics."""
        word_count = len(keyword.split())
        # Shorter keywords (1-2 words) = harder (60-80)
        # Longer keywords (3+ words) = easier (20-40)
        if word_count <= 2:
            difficulty = 60.0 + (word_count - 1) * 10.0
        else:
            difficulty = 40.0 - (word_count - 3) * 5.0
        return max(0.0, min(100.0, difficulty))

    async def get_search_volume(self, keywords: List[str]) -> Dict[str, int]:
        """
        Get estimated monthly search volume.

        Args:
            keywords: List of keywords

        Returns:
            Dict mapping keywords to search volumes
        """
        # Stub implementation - in production, would integrate with SEO API
        # (e.g., Google Keyword Planner, Ahrefs, SEMrush)
        return self._stub_search_volume(keywords)

    async def get_keyword_metadata(
        self, keywords: List[str], serp_data: Optional[Dict[str, Any]] = None
    ) -> List[KeywordMetadata]:
        """
        Get complete metadata for keywords.

        Args:
            keywords: List of keywords
            serp_data: Optional SERP analysis data

        Returns:
            List of KeywordMetadata objects
        """
        difficulty = await self.get_keyword_difficulty(keywords, serp_data)
        volume = await self.get_search_volume(keywords)

        metadata = []
        for keyword in keywords:
            metadata.append(
                KeywordMetadata(
                    keyword=keyword,
                    search_volume=volume.get(keyword, 0),
                    difficulty_score=difficulty.get(keyword, 50.0),
                )
            )

        return metadata

    async def analyze_serp_with_llm(
        self,
        keyword: str,
        pipeline: Optional[Any] = None,
        job_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze SERP using LLM (alternative to scraping) - Phase 4.1.

        Uses LLM via pipeline to simulate SERP analysis by reasoning about:
        - Expected result count
        - Typical domain types in results
        - Content types (docs, guides, commercial, etc.)
        - Competition level

        Args:
            keyword: Keyword to analyze
            pipeline: Pipeline instance for LLM calls (uses self.pipeline if not provided)
            job_id: Optional job ID for tracking

        Returns:
            Dict with SERP analysis data
        """
        # Check cache
        if keyword in self._serp_cache:
            return self._serp_cache[keyword]

        # Use provided pipeline or fallback to instance pipeline
        pipeline_to_use = pipeline or self.pipeline
        if pipeline_to_use is None:
            logger.warning(
                f"No pipeline available for SERP analysis of '{keyword}', using fallback"
            )
            return self._get_fallback_serp_analysis(keyword)

        try:
            # Build prompt following plugin pattern
            prompt = f"""Analyze the search engine results page (SERP) for the keyword: "{keyword}"

Provide a realistic analysis of what would appear in Google search results for this keyword.

Consider:
1. Estimated number of search results (typical range)
2. Types of domains that would rank (e.g., .com, .org, .edu, .gov, Wikipedia, etc.)
3. Content types in top results (blog posts, guides, product pages, documentation, etc.)
4. Typical domain authority of ranking sites (estimate 0-100 scale)
5. Competition level (low/medium/high)
6. Whether results are dominated by specific content types"""

            system_instruction = (
                "You are an SEO expert analyzing search engine results. "
                "Provide realistic, data-driven estimates based on typical SERP patterns. "
                "Be specific and accurate in your analysis."
            )

            # Determine model to use
            # Priority: 1) provided model, 2) step config for "seo_keywords", 3) pipeline default
            serp_model = model
            if not serp_model and hasattr(pipeline_to_use, "pipeline_config"):
                # Try to get model from seo_keywords step config
                step_config = pipeline_to_use.pipeline_config.get_step_config(
                    "seo_keywords"
                )
                if step_config and hasattr(step_config, "seo_keywords_engine_config"):
                    engine_config = step_config.seo_keywords_engine_config
                    if engine_config and engine_config.serp_analysis_model:
                        serp_model = engine_config.serp_analysis_model

            # If a specific model is configured, temporarily add it to pipeline config
            original_step_configs = None
            if serp_model and hasattr(pipeline_to_use, "pipeline_config"):
                from marketing_project.models.pipeline_steps import PipelineStepConfig

                original_step_configs = (
                    pipeline_to_use.pipeline_config.step_configs.copy()
                    if pipeline_to_use.pipeline_config.step_configs
                    else {}
                )
                # Add temporary step config for serp_analysis
                pipeline_to_use.pipeline_config.step_configs = (
                    pipeline_to_use.pipeline_config.step_configs or {}
                )
                pipeline_to_use.pipeline_config.step_configs["serp_analysis"] = (
                    PipelineStepConfig(
                        step_name="serp_analysis",
                        model=serp_model,
                    )
                )

            try:
                # Call pipeline's _call_function following plugin pattern
                serp_analysis = await pipeline_to_use._call_function(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    response_model=SERPAnalysisResult,
                    step_name="serp_analysis",
                    step_number=999,  # High number to avoid conflicts
                    context={"keyword": keyword},
                    max_retries=2,
                    job_id=job_id,
                )
            finally:
                # Restore original step configs if we modified them
                if original_step_configs is not None and hasattr(
                    pipeline_to_use, "pipeline_config"
                ):
                    pipeline_to_use.pipeline_config.step_configs = original_step_configs

            # Normalize result count
            result_count_str = serp_analysis.result_count_estimate
            if "-" in result_count_str:
                # Range - take average
                parts = result_count_str.split("-")
                result_count = (int(parts[0].strip()) + int(parts[1].strip())) // 2
            else:
                result_count = int(result_count_str.replace(",", ""))

            # Get domain authority for typical domains
            domains = serp_analysis.typical_domains
            domain_authorities = await self._get_domain_authorities(domains)
            avg_domain_authority = (
                sum(domain_authorities.values()) / len(domain_authorities)
                if domain_authorities
                else serp_analysis.avg_domain_authority_estimate
            )

            result = {
                "keyword": keyword,
                "result_count": result_count,
                "typical_domains": domains,
                "content_types": serp_analysis.content_types,
                "avg_domain_authority": avg_domain_authority,
                "competition_level": serp_analysis.competition_level,
                "serp_characteristics": serp_analysis.serp_characteristics,
            }

            # Cache result (with size limit)
            if len(self._serp_cache) >= self._serp_cache_max_size:
                # Remove oldest entry (FIFO eviction)
                oldest_key = next(iter(self._serp_cache))
                del self._serp_cache[oldest_key]
            self._serp_cache[keyword] = result
            return result

        except Exception as e:
            logger.warning(f"LLM SERP analysis failed for '{keyword}': {e}")
            return self._get_fallback_serp_analysis(keyword)

    def _get_fallback_serp_analysis(self, keyword: str) -> Dict[str, Any]:
        """Get fallback SERP analysis when LLM call fails."""
        return {
            "keyword": keyword,
            "result_count": 1000000,
            "typical_domains": [],
            "content_types": [],
            "avg_domain_authority": 50.0,
            "competition_level": "medium",
            "serp_characteristics": "",
        }

    async def _get_domain_authorities(self, domains: List[str]) -> Dict[str, float]:
        """
        Get domain authorities using OpenPageRank API - Phase 4.2.

        Args:
            domains: List of domain names (without protocol)

        Returns:
            Dict mapping domains to authority scores (0-100)
        """
        if not self.openpagerank_api_key or not domains:
            return {}

        # OpenPageRank API expects domains without protocol
        clean_domains = [
            d.replace("http://", "").replace("https://", "").split("/")[0]
            for d in domains
        ]

        try:
            async with aiohttp.ClientSession() as session:
                # OpenPageRank API endpoint
                url = "https://openpagerank.com/api/v1.0/getPageRank"
                headers = {"API-OPR": self.openpagerank_api_key}

                # OpenPageRank accepts up to 100 domains per request
                results = {}
                for i in range(0, len(clean_domains), 100):
                    batch = clean_domains[i : i + 100]
                    params = {"domains[]": batch}

                    async with session.get(
                        url, headers=headers, params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "response" in data:
                                for item in data["response"]:
                                    domain = item.get("domain", "")
                                    page_rank_decimal = item.get("page_rank_decimal", 0)
                                    # Convert to 0-100 scale (OpenPageRank is 0-10 scale)
                                    authority = page_rank_decimal * 10.0
                                    results[domain] = authority
                        else:
                            logger.warning(
                                f"OpenPageRank API returned status {response.status}"
                            )

                return results

        except Exception as e:
            logger.warning(f"OpenPageRank API call failed: {e}")
            return {}

    def _stub_search_volume(self, keywords: List[str]) -> Dict[str, int]:
        """Stub implementation for search volume."""
        # Simple heuristic: shorter keywords = higher volume
        result = {}
        for keyword in keywords:
            word_count = len(keyword.split())
            # Shorter keywords = higher volume
            if word_count == 1:
                volume = 10000
            elif word_count == 2:
                volume = 5000
            elif word_count == 3:
                volume = 1000
            else:
                volume = 100
            result[keyword] = volume

        return result
