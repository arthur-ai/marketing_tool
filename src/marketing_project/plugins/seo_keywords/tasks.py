"""
SEO Keywords processing plugin tasks for Marketing Project.

This module provides functions to extract, analyze, and optimize SEO keywords
from content for better search engine visibility and content strategy.

Functions:
    extract_primary_keywords: Extracts main SEO keywords from content
    extract_secondary_keywords: Extracts supporting keywords and LSI terms
    analyze_keyword_density: Analyzes keyword density and distribution
    generate_keyword_suggestions: Generates additional keyword suggestions
    optimize_keyword_placement: Optimizes keyword placement in content
    calculate_keyword_scores: Calculates keyword relevance and difficulty scores
    extract_keywords_with_kwx: Extracts keywords using kwx library with advanced NLP
    extract_keywords_advanced: Combines multiple extraction methods for better results
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

from marketing_project.core.models import (
    AppContext,
    BlogPostContext,
    ContentContext,
    ReleaseNotesContext,
    TranscriptContext,
)
from marketing_project.core.utils import (
    create_standard_task_result,
    ensure_content_context,
    extract_content_metadata_for_pipeline,
    validate_content_for_processing,
)

try:
    from keybert import KeyBERT

    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False

logger = logging.getLogger("marketing_project.plugins.seo_keywords")


def extract_primary_keywords(
    content: Union[ContentContext, Dict[str, Any]], max_keywords: int = 5
) -> Dict[str, Any]:
    """
    Extracts primary SEO keywords from content based on frequency and relevance.

    Args:
        content: Content context object or dictionary
        max_keywords: Maximum number of primary keywords to extract

    Returns:
        Dict[str, Any]: Standardized task result with primary keywords
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Handle empty content gracefully (but not completely invalid content)
        if not content_obj.content and content_obj.title:
            # Content is empty but title exists - return empty results
            return create_standard_task_result(
                success=True,
                data={
                    "primary_keywords": [],
                    "total_keywords_found": 0,
                    "filtered_words_count": 0,
                },
                task_name="extract_primary_keywords",
                metadata=extract_content_metadata_for_pipeline(content_obj),
            )

        # Validate content
        validation = validate_content_for_processing(content_obj)
        if not validation["is_valid"]:
            return create_standard_task_result(
                success=False,
                error=f"Content validation failed: {', '.join(validation['issues'])}",
                task_name="extract_primary_keywords",
            )

        text = f"{content_obj.title} {content_obj.content}".lower()

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
        }

        # Extract words and filter
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
        filtered_words = [word for word in words if word not in stop_words]

        # Count word frequency
        word_counts = Counter(filtered_words)

        # Calculate keyword scores based on frequency and position
        keywords = []
        for word, count in word_counts.most_common(max_keywords * 2):
            # Calculate position score (words in title get higher score)
            title_words = content_obj.title.lower().split() if content_obj.title else []
            position_score = 2.0 if word in title_words else 1.0

            # Calculate final score
            score = count * position_score

            keywords.append(
                {
                    "keyword": word,
                    "frequency": count,
                    "score": score,
                    "in_title": word in title_words,
                    "density": count / len(filtered_words) * 100,
                }
            )

        # Sort by score and return top keywords
        keywords.sort(key=lambda x: x["score"], reverse=True)
        result_keywords = keywords[:max_keywords]

        return create_standard_task_result(
            success=True,
            data={
                "primary_keywords": result_keywords,
                "total_keywords_found": len(keywords),
                "filtered_words_count": len(filtered_words),
            },
            task_name="extract_primary_keywords",
            metadata=extract_content_metadata_for_pipeline(content_obj),
        )

    except Exception as e:
        logger.error(f"Error in extract_primary_keywords: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Keyword extraction failed: {str(e)}",
            task_name="extract_primary_keywords",
        )


def extract_secondary_keywords(
    content: Union[ContentContext, Dict[str, Any]],
    primary_keywords: List[str],
    max_keywords: int = 10,
) -> Dict[str, Any]:
    """
    Extracts secondary keywords and LSI (Latent Semantic Indexing) terms.

    Args:
        content: Content context object or dictionary
        primary_keywords: List of primary keywords to find related terms for
        max_keywords: Maximum number of secondary keywords to extract

    Returns:
        Dict[str, Any]: Standardized task result with secondary keywords
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Validate content
        validation = validate_content_for_processing(content_obj)
        if not validation["is_valid"]:
            return create_standard_task_result(
                success=False,
                error=f"Content validation failed: {', '.join(validation['issues'])}",
                task_name="extract_secondary_keywords",
            )

        text = content_obj.content.lower()

        # Find 2-3 word phrases that contain primary keywords
        secondary_keywords = []

        for primary in primary_keywords:
            # Find phrases containing the primary keyword
            pattern = rf"\b\w*\s*{re.escape(primary)}\s*\w*\b"
            phrases = re.findall(pattern, text)

            for phrase in phrases:
                if len(phrase.split()) <= 3 and phrase != primary:
                    secondary_keywords.append(
                        {
                            "keyword": phrase.strip(),
                            "related_to": primary,
                            "type": "phrase",
                            "frequency": text.count(phrase),
                        }
                    )

        # Find synonyms and related terms (simplified approach)
        synonym_patterns = {
            "best": ["top", "greatest", "excellent", "superior"],
            "guide": ["tutorial", "how-to", "instructions", "manual"],
            "review": ["analysis", "evaluation", "assessment", "critique"],
            "tips": ["advice", "recommendations", "suggestions", "hints"],
            "learn": ["understand", "discover", "explore", "study"],
        }

        for word in text.split():
            if word in synonym_patterns:
                for synonym in synonym_patterns[word]:
                    if synonym in text:
                        secondary_keywords.append(
                            {
                                "keyword": synonym,
                                "related_to": word,
                                "type": "synonym",
                                "frequency": text.count(synonym),
                            }
                        )

        # Remove duplicates and sort by frequency
        seen = set()
        unique_keywords = []
        for kw in secondary_keywords:
            if kw["keyword"] not in seen:
                seen.add(kw["keyword"])
                unique_keywords.append(kw)

        unique_keywords.sort(key=lambda x: x["frequency"], reverse=True)
        result_keywords = unique_keywords[:max_keywords]

        return create_standard_task_result(
            success=True,
            data={
                "secondary_keywords": result_keywords,
                "total_secondary_found": len(unique_keywords),
                "primary_keywords_used": primary_keywords,
            },
            task_name="extract_secondary_keywords",
            metadata=extract_content_metadata_for_pipeline(content_obj),
        )

    except Exception as e:
        logger.error(f"Error in extract_secondary_keywords: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Secondary keyword extraction failed: {str(e)}",
            task_name="extract_secondary_keywords",
        )


def analyze_keyword_density(
    content: Union[ContentContext, Dict[str, Any]], keywords: List[str]
) -> Dict[str, Any]:
    """
    Analyzes keyword density and distribution in content.

    Args:
        content: Content context object or dictionary
        keywords: List of keywords to analyze

    Returns:
        Dict[str, Any]: Standardized task result with density analysis
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Validate content
        validation = validate_content_for_processing(content_obj)
        if not validation["is_valid"]:
            return create_standard_task_result(
                success=False,
                error=f"Content validation failed: {', '.join(validation['issues'])}",
                task_name="analyze_keyword_density",
            )

        text = content_obj.content.lower()
        word_count = len(text.split())

        density_analysis = {
            "total_words": word_count,
            "keyword_densities": {},
            "keyword_positions": {},
            "recommendations": [],
        }

        for keyword in keywords:
            keyword_lower = keyword.lower()
            frequency = text.count(keyword_lower)
            density = (frequency / word_count) * 100 if word_count > 0 else 0

            density_analysis["keyword_densities"][keyword] = {
                "frequency": frequency,
                "density": density,
                "status": "optimal" if 1 <= density <= 3 else "needs_optimization",
            }

            # Find keyword positions
            positions = []
            start = 0
            while True:
                pos = text.find(keyword_lower, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1

            density_analysis["keyword_positions"][keyword] = positions

            # Generate recommendations
            if density < 1:
                density_analysis["recommendations"].append(
                    f"Increase '{keyword}' density (currently {density:.2f}%)"
                )
            elif density > 3:
                density_analysis["recommendations"].append(
                    f"Reduce '{keyword}' density (currently {density:.2f}%)"
                )

        return create_standard_task_result(
            success=True,
            data=density_analysis,
            task_name="analyze_keyword_density",
            metadata=extract_content_metadata_for_pipeline(content_obj),
        )

    except Exception as e:
        logger.error(f"Error in analyze_keyword_density: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Keyword density analysis failed: {str(e)}",
            task_name="analyze_keyword_density",
        )


def generate_keyword_suggestions(
    content: Union[ContentContext, Dict[str, Any]],
    industry: str = None,
    target_audience: str = None,
) -> Dict[str, Any]:
    """
    Generates additional keyword suggestions based on content analysis.

    Args:
        content: Content context object or dictionary
        industry: Industry context for keyword suggestions
        target_audience: Target audience for keyword suggestions

    Returns:
        Dict[str, Any]: Standardized task result with keyword suggestions
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Validate content
        validation = validate_content_for_processing(content_obj)
        if not validation["is_valid"]:
            return create_standard_task_result(
                success=False,
                error=f"Content validation failed: {', '.join(validation['issues'])}",
                task_name="generate_keyword_suggestions",
            )

        suggestions = []

        # Extract key topics from content
        text = content_obj.content.lower()
        title = content_obj.title.lower() if content_obj.title else ""

        # Industry-specific keyword suggestions
        industry_keywords = {
            "technology": [
                "innovation",
                "digital",
                "automation",
                "software",
                "platform",
            ],
            "marketing": ["strategy", "campaign", "brand", "engagement", "conversion"],
            "finance": ["investment", "portfolio", "revenue", "profit", "growth"],
            "healthcare": ["treatment", "patient", "medical", "health", "wellness"],
            "education": ["learning", "training", "course", "skill", "knowledge"],
        }

        if industry and industry.lower() in industry_keywords:
            for keyword in industry_keywords[industry.lower()]:
                if keyword not in text:
                    suggestions.append(
                        {
                            "keyword": keyword,
                            "type": "industry",
                            "relevance_score": 0.8,
                            "suggestion_reason": f"Industry-specific keyword for {industry}",
                        }
                    )
        elif industry and industry.lower() not in industry_keywords:
            # For unknown industry, generate general suggestions based on content
            general_keywords = [
                "strategy",
                "solution",
                "approach",
                "method",
                "technique",
                "process",
                "system",
                "platform",
                "tool",
                "service",
            ]
            for keyword in general_keywords:
                if keyword not in text:
                    suggestions.append(
                        {
                            "keyword": keyword,
                            "type": "general",
                            "relevance_score": 0.5,
                            "suggestion_reason": f"General keyword suggestion for {industry}",
                        }
                    )

        # Audience-specific keyword suggestions
        audience_keywords = {
            "beginners": [
                "introduction",
                "basics",
                "getting started",
                "tutorial",
                "guide",
            ],
            "professionals": [
                "advanced",
                "expert",
                "enterprise",
                "optimization",
                "strategy",
            ],
            "developers": ["code", "implementation", "technical", "api", "integration"],
            "managers": [
                "management",
                "leadership",
                "strategy",
                "planning",
                "decision",
            ],
        }

        if target_audience and target_audience.lower() in audience_keywords:
            for keyword in audience_keywords[target_audience.lower()]:
                if keyword not in text:
                    suggestions.append(
                        {
                            "keyword": keyword,
                            "type": "audience",
                            "relevance_score": 0.7,
                            "suggestion_reason": f"Audience-specific keyword for {target_audience}",
                        }
                    )

        # Long-tail keyword suggestions based on content
        words = text.split()
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        trigrams = [
            f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)
        ]

        # Find common phrases that could be keywords
        phrase_counts = Counter(bigrams + trigrams)
        for phrase, count in phrase_counts.most_common(5):
            if count >= 1 and len(phrase.split()) >= 2:
                suggestions.append(
                    {
                        "keyword": phrase,
                        "type": "long_tail",
                        "relevance_score": min(count / 10, 1.0),
                        "suggestion_reason": "Frequently used phrase in content",
                    }
                )

        # Sort by relevance score
        suggestions.sort(key=lambda x: x["relevance_score"], reverse=True)
        result_suggestions = suggestions[:10]

        return create_standard_task_result(
            success=True,
            data={
                "suggestions": result_suggestions,
                "total_suggestions": len(suggestions),
                "industry": industry,
                "target_audience": target_audience,
            },
            task_name="generate_keyword_suggestions",
            metadata=extract_content_metadata_for_pipeline(content_obj),
        )

    except Exception as e:
        logger.error(f"Error in generate_keyword_suggestions: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Keyword suggestion generation failed: {str(e)}",
            task_name="generate_keyword_suggestions",
        )


def optimize_keyword_placement(
    content: Union[ContentContext, Dict[str, Any]], keywords: List[str]
) -> Dict[str, Any]:
    """
    Provides recommendations for optimizing keyword placement in content.

    Args:
        content: Content context object or dictionary
        keywords: List of keywords to optimize placement for

    Returns:
        Dict[str, Any]: Standardized task result with placement optimization
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Validate content
        validation = validate_content_for_processing(content_obj)
        if not validation["is_valid"]:
            return create_standard_task_result(
                success=False,
                error=f"Content validation failed: {', '.join(validation['issues'])}",
                task_name="optimize_keyword_placement",
            )

        text = content_obj.content
        title = content_obj.title or ""

        optimization = {
            "title_optimization": {},
            "heading_optimization": {},
            "content_optimization": {},
            "meta_optimization": {},
            "recommendations": [],
        }

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Title optimization
            title_contains = keyword_lower in title.lower()
            optimization["title_optimization"][keyword] = {
                "in_title": title_contains,
                "recommendation": (
                    "Add to title" if not title_contains else "Already in title"
                ),
            }

            # Heading optimization (simplified - would need proper heading detection)
            headings = re.findall(r"^#+\s+(.+)$", text, re.MULTILINE)
            heading_contains = any(
                keyword_lower in heading.lower() for heading in headings
            )
            optimization["heading_optimization"][keyword] = {
                "in_headings": heading_contains,
                "recommendation": (
                    "Add to headings" if not heading_contains else "Already in headings"
                ),
            }

            # Content optimization
            first_paragraph = text.split("\n")[0] if text else ""
            first_100_words = " ".join(text.split()[:100])

            optimization["content_optimization"][keyword] = {
                "in_first_paragraph": keyword_lower in first_paragraph.lower(),
                "in_first_100_words": keyword_lower in first_100_words.lower(),
                "recommendation": (
                    "Add to first paragraph"
                    if keyword_lower not in first_100_words.lower()
                    else "Good placement"
                ),
            }

            # Generate overall recommendations
            if not title_contains:
                optimization["recommendations"].append(
                    f"Consider adding '{keyword}' to the title"
                )
            if not heading_contains:
                optimization["recommendations"].append(
                    f"Consider adding '{keyword}' to a heading"
                )
            if keyword_lower not in first_100_words.lower():
                optimization["recommendations"].append(
                    f"Consider adding '{keyword}' to the first paragraph"
                )

        return create_standard_task_result(
            success=True,
            data=optimization,
            task_name="optimize_keyword_placement",
            metadata=extract_content_metadata_for_pipeline(content_obj),
        )

    except Exception as e:
        logger.error(f"Error in optimize_keyword_placement: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Keyword placement optimization failed: {str(e)}",
            task_name="optimize_keyword_placement",
        )


def calculate_keyword_scores(
    keywords: List[Dict[str, Any]], content: Union[ContentContext, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculates comprehensive keyword scores including relevance and difficulty.

    Args:
        keywords: List of keyword dictionaries
        content: Content context object or dictionary

    Returns:
        Dict[str, Any]: Standardized task result with scored keywords
    """
    try:
        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Validate content
        validation = validate_content_for_processing(content_obj)
        if not validation["is_valid"]:
            return create_standard_task_result(
                success=False,
                error=f"Content validation failed: {', '.join(validation['issues'])}",
                task_name="calculate_keyword_scores",
            )

        scored_keywords = []

        for keyword_data in keywords:
            keyword = keyword_data["keyword"]

            # Calculate relevance score (0-1)
            frequency = keyword_data.get("frequency", 0)
            in_title = keyword_data.get("in_title", False)
            density = keyword_data.get("density", 0)

            relevance_score = min(frequency / 10, 1.0)  # Normalize frequency
            if in_title:
                relevance_score += 0.2
            if 1 <= density <= 3:
                relevance_score += 0.1

            # Calculate difficulty score (0-1, where 1 is most difficult)
            word_length = len(keyword.split())
            difficulty_score = min(word_length / 5, 1.0)  # Longer phrases are harder

            # Calculate overall score
            overall_score = (relevance_score * 0.7) + ((1 - difficulty_score) * 0.3)

            scored_keyword = keyword_data.copy()
            scored_keyword.update(
                {
                    "relevance_score": relevance_score,
                    "difficulty_score": difficulty_score,
                    "overall_score": overall_score,
                    "priority": (
                        "high"
                        if overall_score > 0.7
                        else "medium" if overall_score > 0.4 else "low"
                    ),
                }
            )

            scored_keywords.append(scored_keyword)

        # Sort by overall score
        scored_keywords.sort(key=lambda x: x["overall_score"], reverse=True)

        return create_standard_task_result(
            success=True,
            data={
                "scored_keywords": scored_keywords,
                "high_priority_count": sum(
                    1 for kw in scored_keywords if kw["priority"] == "high"
                ),
                "medium_priority_count": sum(
                    1 for kw in scored_keywords if kw["priority"] == "medium"
                ),
                "low_priority_count": sum(
                    1 for kw in scored_keywords if kw["priority"] == "low"
                ),
            },
            task_name="calculate_keyword_scores",
            metadata=extract_content_metadata_for_pipeline(content_obj),
        )

    except Exception as e:
        logger.error(f"Error in calculate_keyword_scores: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Keyword scoring failed: {str(e)}",
            task_name="calculate_keyword_scores",
        )


def extract_keywords_with_keybert(
    content: Union[ContentContext, Dict[str, Any]],
    max_keywords: int = 10,
    method: str = "mmr",
    ngram_range: Tuple[int, int] = (1, 3),
    language: str = "english",
) -> Dict[str, Any]:
    """
    Extracts keywords using the KeyBERT library with advanced NLP techniques.

    Args:
        content: Content context object or dictionary
        max_keywords: Maximum number of keywords to extract
        method: Extraction method ('mmr', 'maxsum', 'maximal_marginal_relevance')
        ngram_range: Range of n-grams to consider (min, max)
        language: Language for processing ('english', 'spanish', 'french', etc.)

    Returns:
        Dict[str, Any]: Standardized task result with KeyBERT-extracted keywords
    """
    try:
        if not KEYBERT_AVAILABLE:
            return create_standard_task_result(
                success=False,
                error="KeyBERT library not available. Please install with: pip install keybert",
                task_name="extract_keywords_with_keybert",
            )

        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Handle empty content gracefully
        if not content_obj.content and content_obj.title:
            return create_standard_task_result(
                success=True,
                data={
                    "keywords": [],
                    "total_keywords_found": 0,
                    "method": method,
                    "ngram_range": ngram_range,
                    "language": language,
                },
                task_name="extract_keywords_with_keybert",
                metadata=extract_content_metadata_for_pipeline(content_obj),
            )

        # Validate content
        validation = validate_content_for_processing(content_obj)
        if not validation["is_valid"]:
            return create_standard_task_result(
                success=False,
                error=f"Content validation failed: {', '.join(validation['issues'])}",
                task_name="extract_keywords_with_keybert",
            )

        # Prepare text for processing
        text = f"{content_obj.title} {content_obj.content}".strip()

        if not text:
            return create_standard_task_result(
                success=False,
                error="No text content available for keyword extraction",
                task_name="extract_keywords_with_keybert",
            )

        # Map method names to KeyBERT method names
        method_mapping = {
            "mmr": "mmr",
            "maxsum": "maxsum",
            "maximal_marginal_relevance": "mmr",
        }

        keybert_method = method_mapping.get(method, "mmr")

        # Extract keywords using KeyBERT
        try:
            # Initialize KeyBERT model
            kw_model = KeyBERT()

            # Extract keywords
            keywords_result = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=ngram_range,
                stop_words=language,
                use_mmr=True if keybert_method == "mmr" else False,
                use_maxsum=True if keybert_method == "maxsum" else False,
                diversity=0.5 if keybert_method == "mmr" else 0.0,
                top_n=max_keywords,
            )

            # Convert KeyBERT result to our format
            if keywords_result:
                keywords = [(kw, score) for kw, score in keywords_result]
            else:
                keywords = []
        except Exception as e:
            logger.error(
                f"KeyBERT extraction failed with method {keybert_method}: {str(e)}"
            )
            return create_standard_task_result(
                success=False,
                error=f"KeyBERT keyword extraction failed: {str(e)}",
                task_name="extract_keywords_with_keybert",
            )

        # Format keywords for consistency with other functions
        formatted_keywords = []
        for i, (keyword, score) in enumerate(keywords):
            formatted_keywords.append(
                {
                    "keyword": keyword,
                    "score": float(score) if isinstance(score, (int, float)) else 0.0,
                    "rank": i + 1,
                    "method": method,
                    "ngram_length": len(keyword.split()),
                    "in_title": keyword.lower() in (content_obj.title or "").lower(),
                }
            )

        return create_standard_task_result(
            success=True,
            data={
                "keywords": formatted_keywords,
                "total_keywords_found": len(formatted_keywords),
                "method": method,
                "ngram_range": ngram_range,
                "language": language,
                "extraction_library": "keybert",
            },
            task_name="extract_keywords_with_keybert",
            metadata=extract_content_metadata_for_pipeline(content_obj),
        )

    except Exception as e:
        logger.error(f"Error in extract_keywords_with_keybert: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"KeyBERT keyword extraction failed: {str(e)}",
            task_name="extract_keywords_with_keybert",
        )


def extract_keywords_advanced(
    content: Union[ContentContext, Dict[str, Any]],
    max_keywords: int = 10,
    methods: List[str] = None,
    combine_results: bool = True,
) -> Dict[str, Any]:
    """
    Extracts keywords using multiple methods and optionally combines results.

    Args:
        content: Content context object or dictionary
        max_keywords: Maximum number of keywords to extract per method
        methods: List of methods to use ['mmr', 'maxsum', 'maximal_marginal_relevance']
        combine_results: Whether to combine and deduplicate results from all methods

    Returns:
        Dict[str, Any]: Standardized task result with advanced keyword extraction
    """
    try:
        if not KEYBERT_AVAILABLE:
            return create_standard_task_result(
                success=False,
                error="KeyBERT library not available. Please install with: pip install keybert",
                task_name="extract_keywords_advanced",
            )

        # Default methods if none specified
        if methods is None:
            methods = ["mmr", "maxsum"]

        # Ensure content is a ContentContext object
        content_obj = ensure_content_context(content)

        # Handle empty content gracefully
        if not content_obj.content and content_obj.title:
            return create_standard_task_result(
                success=True,
                data={
                    "keywords": [],
                    "methods_used": methods,
                    "combined_results": combine_results,
                    "total_keywords_found": 0,
                },
                task_name="extract_keywords_advanced",
                metadata=extract_content_metadata_for_pipeline(content_obj),
            )

        # Validate content
        validation = validate_content_for_processing(content_obj)
        if not validation["is_valid"]:
            return create_standard_task_result(
                success=False,
                error=f"Content validation failed: {', '.join(validation['issues'])}",
                task_name="extract_keywords_advanced",
            )

        all_keywords = {}
        method_results = {}

        # Extract keywords using each method
        for method in methods:
            try:
                result = extract_keywords_with_keybert(
                    content_obj, max_keywords, method
                )
                if result["success"]:
                    method_results[method] = result["data"]["keywords"]
                    # Add to combined results
                    for kw in result["data"]["keywords"]:
                        keyword = kw["keyword"]
                        if keyword not in all_keywords:
                            all_keywords[keyword] = {
                                "keyword": keyword,
                                "scores": {},
                                "methods": [],
                                "total_score": 0.0,
                                "ngram_length": kw["ngram_length"],
                                "in_title": kw["in_title"],
                            }
                        all_keywords[keyword]["scores"][method] = kw["score"]
                        all_keywords[keyword]["methods"].append(method)
                else:
                    logger.warning(f"Method {method} failed: {result['error']}")
            except Exception as e:
                logger.warning(f"Method {method} failed with exception: {str(e)}")

        if not all_keywords:
            return create_standard_task_result(
                success=True,
                data={
                    "keywords": [],
                    "methods_used": methods,
                    "method_results": method_results,
                    "combined_results": combine_results,
                    "total_keywords_found": 0,
                    "extraction_library": "keybert",
                },
                task_name="extract_keywords_advanced",
                metadata=extract_content_metadata_for_pipeline(content_obj),
            )

        # Calculate combined scores if requested
        if combine_results:
            for keyword_data in all_keywords.values():
                scores = list(keyword_data["scores"].values())
                keyword_data["total_score"] = sum(scores) / len(scores)  # Average score
                keyword_data["confidence"] = len(keyword_data["methods"]) / len(
                    methods
                )  # Method coverage

        # Sort by total score and return top keywords
        sorted_keywords = sorted(
            all_keywords.values(), key=lambda x: x["total_score"], reverse=True
        )[:max_keywords]

        return create_standard_task_result(
            success=True,
            data={
                "keywords": sorted_keywords,
                "methods_used": methods,
                "method_results": method_results,
                "combined_results": combine_results,
                "total_keywords_found": len(sorted_keywords),
                "extraction_library": "keybert",
            },
            task_name="extract_keywords_advanced",
            metadata=extract_content_metadata_for_pipeline(content_obj),
        )

    except Exception as e:
        logger.error(f"Error in extract_keywords_advanced: {str(e)}")
        return create_standard_task_result(
            success=False,
            error=f"Advanced keyword extraction failed: {str(e)}",
            task_name="extract_keywords_advanced",
        )
