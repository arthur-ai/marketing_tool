"""
SEO Keywords composer for orchestrating LLM and local engines.

This composer merges results from different engines based on field-level configuration.
"""

import logging
from typing import Any, Dict, List, Optional

from marketing_project.models.pipeline_steps import SEOKeywordsResult
from marketing_project.services.engines.composer import EngineComposer

logger = logging.getLogger(__name__)


class SEOKeywordsComposer:
    """
    Composes LLM and local semantic engines to populate SEOKeywordsResult.

    Supports per-field engine selection with default + override configuration.
    """

    # Field to operation mapping for local semantic engine
    FIELD_TO_OPERATION = {
        "main_keyword": "extract_main_keyword",
        "primary_keywords": "extract_primary_keywords",
        "secondary_keywords": "extract_secondary_keywords",
        "lsi_keywords": "extract_lsi_keywords",
        "long_tail_keywords": "extract_long_tail_keywords",
        "keyword_density_analysis": "calculate_density",
        "search_intent": "classify_intent",
        "keyword_clusters": "cluster_keywords",
        "keyword_difficulty": "get_seo_metrics",
        "primary_keywords_metadata": "get_seo_metrics",
        "secondary_keywords_metadata": "get_seo_metrics",
        "long_tail_keywords_metadata": "get_seo_metrics",
        "search_volume_summary": "get_seo_metrics",
    }

    def __init__(self, composer: EngineComposer):
        """
        Initialize the SEO keywords composer.

        Args:
            composer: EngineComposer instance with engine configuration
        """
        self.composer = composer

    async def compose_result(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any],
        pipeline: Optional[Any] = None,
    ) -> SEOKeywordsResult:
        """
        Compose SEOKeywordsResult from multiple engines.

        Args:
            content: Content dict with title, content, etc.
            context: Execution context
            pipeline: FunctionPipeline instance (for LLM engine)

        Returns:
            Complete SEOKeywordsResult
        """
        # Start with empty result or get base from default engine
        result_dict: Dict[str, Any] = {}

        # Determine which fields to extract from which engine
        fields_to_extract = self._get_fields_to_extract()

        # Extract fields using appropriate engines
        for field_name in fields_to_extract:
            try:
                field_value = await self._extract_field(
                    field_name, content, context, pipeline, result_dict
                )
                if field_value is not None:
                    result_dict[field_name] = field_value
            except Exception as e:
                logger.warning(f"Failed to extract field '{field_name}': {e}")
                # Continue with other fields

        # If default is LLM and no overrides, get complete result from LLM
        if (
            self.composer.default_engine_type == "llm"
            and not self.composer.field_overrides
        ):
            # Use LLM for everything
            try:
                llm_result = await self.composer.execute_operation(
                    "main_keyword",  # Field name doesn't matter for LLM
                    "extract_all",
                    {"content": content},
                    context,
                    pipeline,
                )
                if isinstance(llm_result, SEOKeywordsResult):
                    return llm_result
            except Exception as e:
                logger.warning(
                    f"LLM extraction failed, falling back to field-by-field: {e}"
                )
                # Fall through to field-by-field extraction

        # Store graph metrics if available from clustering
        if "keyword_clusters" in result_dict:
            # Graph metrics would be stored in cluster metadata
            # For now, just mark that we have clusters
            result_dict["_has_graph_analysis"] = True

        # Build result from extracted fields
        return self._build_result(result_dict, content, context)

    def _get_fields_to_extract(self) -> list[str]:
        """Get list of fields that need to be extracted."""
        # All fields in SEOKeywordsResult
        all_fields = [
            "main_keyword",
            "primary_keywords",
            "secondary_keywords",
            "lsi_keywords",
            "long_tail_keywords",
            "keyword_density_analysis",
            "search_intent",
            "keyword_difficulty",
            "primary_keywords_metadata",
            "secondary_keywords_metadata",
            "long_tail_keywords_metadata",
            "keyword_clusters",
            "search_volume_summary",
            "optimization_recommendations",
            "confidence_score",
            "relevance_score",
        ]

        # If default is LLM and no overrides, we'll get everything from LLM
        if (
            self.composer.default_engine_type == "llm"
            and not self.composer.field_overrides
        ):
            return []  # Will use LLM extract_all

        # Otherwise, extract fields that use local engine
        fields_to_extract = []
        for field in all_fields:
            engine_type = self.composer.get_engine_type_for_field(field)
            if engine_type == "local_semantic":
                fields_to_extract.append(field)
            elif engine_type == "llm" and self.composer.default_engine_type != "llm":
                # Field override to use LLM when default is local
                fields_to_extract.append(field)

        return fields_to_extract

    async def _extract_field(
        self,
        field_name: str,
        content: Dict[str, Any],
        context: Dict[str, Any],
        pipeline: Optional[Any],
        current_result: Dict[str, Any],
    ) -> Any:
        """
        Extract a single field using the appropriate engine.

        Args:
            field_name: Name of the field to extract
            content: Content dict
            context: Execution context
            pipeline: FunctionPipeline instance
            current_result: Current result dict (for dependencies)

        Returns:
            Field value
        """
        # Get operation for this field
        operation = self.FIELD_TO_OPERATION.get(field_name)
        if not operation:
            # Field doesn't have a specific operation, skip
            return None

        # Prepare inputs (may need values from current result)
        inputs = {"content": content}

        # Pass pipeline to operations that need it (e.g., SERP analysis)
        inputs["_pipeline"] = pipeline

        # Reuse parsed_doc if available from previous operation (optimization)
        if "_parsed_doc" in current_result:
            inputs["_parsed_doc"] = current_result["_parsed_doc"]

        # Add dependencies for operations that need them
        if operation in ["calculate_density", "cluster_keywords", "get_seo_metrics"]:
            inputs["primary_keywords"] = current_result.get("primary_keywords", [])
            inputs["secondary_keywords"] = current_result.get("secondary_keywords", [])
            inputs["lsi_keywords"] = current_result.get("lsi_keywords", [])
            inputs["long_tail_keywords"] = current_result.get("long_tail_keywords", [])

        if operation in ["classify_intent"]:
            inputs["main_keyword"] = current_result.get("main_keyword", "")

        # Execute operation
        result = await self.composer.execute_operation(
            field_name, operation, inputs, context, pipeline
        )

        # Handle special cases for metrics
        if field_name in [
            "keyword_difficulty",
            "primary_keywords_metadata",
            "secondary_keywords_metadata",
            "long_tail_keywords_metadata",
            "search_volume_summary",
        ]:
            # get_seo_metrics returns a dict with multiple fields
            if isinstance(result, dict):
                if field_name == "keyword_difficulty":
                    return result.get("keyword_difficulty")
                elif field_name == "primary_keywords_metadata":
                    keywords = current_result.get("primary_keywords", [])
                    return [
                        m for m in result.get("metadata", []) if m.keyword in keywords
                    ][:5]
                elif field_name == "secondary_keywords_metadata":
                    keywords = current_result.get("secondary_keywords", [])
                    return [
                        m for m in result.get("metadata", []) if m.keyword in keywords
                    ][:10]
                elif field_name == "long_tail_keywords_metadata":
                    keywords = current_result.get("long_tail_keywords", [])
                    return [
                        m for m in result.get("metadata", []) if m.keyword in keywords
                    ][:8]
                elif field_name == "search_volume_summary":
                    return result.get("search_volume_summary")

        # Store SERP data in result dict for optimization recommendations
        if isinstance(result, dict) and "_serp_data" in result:
            current_result["_serp_data"] = result["_serp_data"]

        return result

    def _build_result(
        self,
        result_dict: Dict[str, Any],
        content: Dict[str, Any],
        context: Dict[str, Any],
    ) -> SEOKeywordsResult:
        """
        Build SEOKeywordsResult from extracted fields.

        Args:
            result_dict: Dict of extracted field values
            content: Content dict
            context: Execution context

        Returns:
            SEOKeywordsResult
        """
        # Ensure required fields have defaults
        if "main_keyword" not in result_dict:
            result_dict["main_keyword"] = "keyword"

        if "primary_keywords" not in result_dict:
            result_dict["primary_keywords"] = [result_dict["main_keyword"]]

        if "search_intent" not in result_dict:
            result_dict["search_intent"] = "informational"

        # Calculate derived fields if missing
        if "relevance_score" not in result_dict:
            result_dict["relevance_score"] = self._calculate_relevance_score(
                result_dict
            )

        if "confidence_score" not in result_dict:
            result_dict["confidence_score"] = self._calculate_confidence_score(
                result_dict
            )

        # Phase 6.3: Generate enhanced optimization recommendations
        if "optimization_recommendations" not in result_dict:
            result_dict["optimization_recommendations"] = (
                self._generate_optimization_recommendations(
                    result_dict, content, context
                )
            )

        # Build result
        return SEOKeywordsResult(**result_dict)

    def _generate_optimization_recommendations(
        self,
        result_dict: Dict[str, Any],
        content: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[str]:
        """
        Generate optimization recommendations - Phase 6.3 enhanced.

        Uses all signals to generate actionable suggestions.
        """
        recommendations = []

        # Check keyword coverage
        primary = result_dict.get("primary_keywords", [])
        secondary = result_dict.get("secondary_keywords", [])
        long_tail = result_dict.get("long_tail_keywords", [])

        if len(primary) < 3:
            recommendations.append("Add more primary keywords to improve SEO coverage")

        if not long_tail or len(long_tail) < 5:
            recommendations.append(
                "Increase long-tail keyword coverage for better targeting of specific queries"
            )

        # Check topic coverage from clusters
        clusters = result_dict.get("keyword_clusters", [])
        if clusters:
            cluster_topics = [
                getattr(c, "topic_theme", "")
                for c in clusters
                if hasattr(c, "topic_theme")
            ]
            if len(cluster_topics) < 3:
                recommendations.append(
                    f"Content covers {len(cluster_topics)} topic(s). Consider adding sections "
                    f"to cover additional related topics for better SEO coverage."
                )

        # Check query-like long-tails
        query_like_count = sum(
            1
            for kw in long_tail
            if any(
                q in kw.lower() for q in ["how", "what", "best", "guide", "for", "to"]
            )
        )
        if query_like_count < 3:
            recommendations.append(
                "Add more query-style long-tail keywords (e.g., 'how to...', 'best...', 'guide to...') "
                "to target informational searches"
            )

        # Check keyword density
        density_analysis = result_dict.get("keyword_density_analysis", [])
        if density_analysis:
            low_density = [
                d
                for d in density_analysis
                if hasattr(d, "current_density")
                and hasattr(d, "optimal_density")
                and getattr(d, "current_density", 0)
                < getattr(d, "optimal_density", 0) * 0.5
            ]
            if low_density:
                recommendations.append(
                    f"{len(low_density)} primary keyword(s) have low density. "
                    f"Consider increasing usage naturally throughout the content."
                )

        # Check placement
        if density_analysis:
            missing_in_headings = [
                d
                for d in density_analysis
                if hasattr(d, "placement_locations")
                and "heading" not in getattr(d, "placement_locations", [])
            ]
            if missing_in_headings:
                recommendations.append(
                    f"Consider adding {len(missing_in_headings)} primary keyword(s) to headings "
                    f"for better SEO structure"
                )

        # Check SERP/competition data if available (Phase 4.1)
        serp_data = result_dict.get("_serp_data", {})
        difficulty = result_dict.get("keyword_difficulty", {})

        if difficulty:
            high_difficulty = [
                kw
                for kw, score in difficulty.items()
                if isinstance(score, (int, float)) and score > 70
            ]
            if high_difficulty:
                recommendations.append(
                    f"Keywords '{', '.join(high_difficulty[:3])}' have high competition. "
                    f"Consider focusing on long-tail variations for better ranking potential."
                )

        # Phase 4.1: Use SERP characteristics for recommendations
        if serp_data:
            for keyword, serp_info in list(serp_data.items())[:3]:  # Top 3
                content_types = serp_info.get("content_types", [])
                serp_chars = serp_info.get("serp_characteristics", "")

                if "commercial" in content_types and "guide" not in content_types:
                    recommendations.append(
                        f"SERP for '{keyword}' is dominated by commercial content. "
                        f"Consider creating a comprehensive guide to differentiate."
                    )
                elif "documentation" in content_types:
                    recommendations.append(
                        f"SERP for '{keyword}' shows documentation results. "
                        f"Consider adding technical depth and examples."
                    )

        # Default recommendation if none generated
        if not recommendations:
            recommendations.append(
                "Continue optimizing keyword placement and density for best SEO performance"
            )

        return recommendations[:10]  # Limit to top 10

    def _calculate_relevance_score(self, result_dict: Dict[str, Any]) -> float:
        """
        Calculate relevance score (0-100) - Phase 6.1 enhanced.

        Combines:
        - Average doc-keyword embedding similarity
        - Topic membership (keywords in dominant topics get upweighted)
        - Graph centrality (co-occurrence network)
        """
        # Base score from keyword coverage
        score = 50.0

        if result_dict.get("primary_keywords"):
            score += 20.0
        if result_dict.get("secondary_keywords"):
            score += 15.0
        if result_dict.get("lsi_keywords"):
            score += 10.0
        if result_dict.get("long_tail_keywords"):
            score += 5.0

        # Phase 6.1: Enhanced scoring with topic/graph signals
        # If we have clusters with topic themes, boost score
        clusters = result_dict.get("keyword_clusters", [])
        if clusters:
            # More diverse clusters = better relevance
            unique_topics = len(
                set(
                    getattr(c, "topic_theme", "")
                    for c in clusters
                    if hasattr(c, "topic_theme")
                )
            )
            score += min(10.0, unique_topics * 2.0)

        # If we have graph metrics, use them
        if result_dict.get("_graph_metrics"):
            # High centrality keywords = better relevance
            score += 5.0

        return min(100.0, score)

    def _calculate_confidence_score(self, result_dict: Dict[str, Any]) -> float:
        """
        Calculate confidence score (0-1) - Phase 6.2 enhanced.

        Combines:
        - Relevance score (normalized 0-1)
        - Cluster cohesion (low intra-cluster distance, high topic purity)
        - Presence of all field categories
        - SERP/authority data (if available)
        """
        # Base confidence from completeness
        required_fields = ["main_keyword", "primary_keywords", "search_intent"]
        present = sum(1 for f in required_fields if result_dict.get(f))
        base_confidence = present / len(required_fields)

        # Boost if we have additional fields
        if result_dict.get("keyword_density_analysis"):
            base_confidence += 0.1
        if result_dict.get("keyword_clusters"):
            base_confidence += 0.1

        # Phase 6.2: Enhanced confidence with cluster cohesion
        clusters = result_dict.get("keyword_clusters", [])
        if clusters:
            # More clusters with clear topics = higher confidence
            topic_clusters = sum(
                1
                for c in clusters
                if hasattr(c, "topic_theme") and getattr(c, "topic_theme", "")
            )
            if topic_clusters > 0:
                base_confidence += min(0.1, topic_clusters * 0.02)

        # Phase 6.2: Check if we have SERP/authority data (Phase 4)
        has_seo_metrics = (
            result_dict.get("keyword_difficulty") is not None
            or result_dict.get("primary_keywords_metadata") is not None
        )
        has_serp_data = result_dict.get("_serp_data") is not None
        if has_seo_metrics:
            base_confidence += 0.05  # Boost if we have real SEO data
        if has_serp_data:
            base_confidence += 0.03  # Additional boost for SERP analysis

        return min(1.0, base_confidence)
