"""
Competitor Research Service.

Analyzes competitor blogs and social media posts to identify
what makes their content perform well and surface actionable insights.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from jinja2 import Environment
from sqlalchemy import select

from marketing_project.models.competitor_models import (
    CompetitorContentAnalysis,
    CompetitorResearchListItem,
    CompetitorResearchRequest,
    CompetitorResearchResult,
    CompetitorResearchSummary,
)
from marketing_project.models.db_models import (
    CompetitorResearchModel,
    CrawledUrlContentModel,
)
from marketing_project.prompts.prompts import get_template
from marketing_project.services.arthur_prompt_client import fetch_arthur_prompt
from marketing_project.services.database import get_database_manager

logger = logging.getLogger("marketing_project.services.competitor_research")

_LANG = "en"


class CompetitorResearchService:
    """
    Service for competitor content analysis.

    Uses LLM to deeply analyze competitor blogs and social media posts,
    identifying why they perform well and surfacing actionable insights.
    """

    async def fetch_url_content(self, url: str) -> Dict[str, Any]:
        """
        Fetch and extract readable content from a URL using aiohttp + BeautifulSoup.

        Uses the same fallback selector chain as BeautifulSoupScrapingSource:
        article → main → .content → .post-content → .entry-content → body.

        Returns a dict with keys: title, content, meta_description, word_count.
        Returns empty strings on failure (analysis proceeds with URL-only context).
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; MarketingToolBot/1.0; +https://example.com/bot)"
            ),
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        timeout = aiohttp.ClientTimeout(total=15)

        try:
            async with aiohttp.ClientSession(
                headers=headers, timeout=timeout
            ) as session:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status >= 400:
                        logger.warning(
                            f"[COMPETITOR] fetch {url} returned HTTP {response.status}"
                        )
                        return {
                            "title": "",
                            "content": "",
                            "meta_description": "",
                            "word_count": 0,
                        }

                    content_type = response.headers.get("Content-Type", "")
                    if "html" not in content_type:
                        logger.warning(
                            f"[COMPETITOR] {url} is not HTML ({content_type}), skipping fetch"
                        )
                        return {
                            "title": "",
                            "content": "",
                            "meta_description": "",
                            "word_count": 0,
                        }

                    html = await response.text(errors="replace")

            soup = BeautifulSoup(html, "html.parser")

            # Remove noise: scripts, styles, nav, footer, ads
            for tag in soup(
                ["script", "style", "nav", "footer", "header", "aside", "noscript"]
            ):
                tag.decompose()

            # Title — h1 first, fall back to <title>
            title = ""
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(separator=" ").strip()
            if not title:
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text().strip()

            # Main content — prioritise semantic/common CMS selectors
            content_elem = None
            for selector in [
                "article",
                "main",
                "[role='main']",
                ".post-content",
                ".entry-content",
                ".article-body",
                ".blog-post",
                ".content",
                "[class*='content']",
                "[class*='article']",
                "[class*='post']",
            ]:
                content_elem = soup.select_one(selector)
                if content_elem:
                    break
            if not content_elem:
                content_elem = soup.find("body")

            content_text = (
                content_elem.get_text(separator="\n", strip=True)
                if content_elem
                else ""
            )

            # Collapse excessive blank lines
            content_text = re.sub(r"\n{3,}", "\n\n", content_text).strip()

            # Meta description
            meta_desc = ""
            for meta in soup.find_all("meta"):
                name = (meta.get("name") or meta.get("property") or "").lower()
                if name in ("description", "og:description", "twitter:description"):
                    meta_desc = meta.get("content", "").strip()
                    if meta_desc:
                        break

            word_count = len(content_text.split())
            logger.info(f"[COMPETITOR] Fetched {url}: {word_count} words")

            return {
                "title": title,
                "content": content_text,
                "meta_description": meta_desc,
                "word_count": word_count,
            }

        except Exception as e:
            logger.warning(f"[COMPETITOR] Failed to fetch {url}: {e}")
            return {"title": "", "content": "", "meta_description": "", "word_count": 0}

    async def create_research_job(
        self,
        request: CompetitorResearchRequest,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Create a new competitor research job in the database.

        Returns the job_id.
        """
        job_id = str(uuid.uuid4())

        competitor_count = 0
        if request.competitor_urls:
            competitor_count += len(request.competitor_urls)
        if request.competitor_content:
            competitor_count += len(request.competitor_content)

        db_manager = get_database_manager()
        if db_manager.is_initialized:
            async with db_manager.get_session() as session:
                record = CompetitorResearchModel(
                    job_id=job_id,
                    user_id=user_id,
                    status="pending",
                    content_type=request.content_type,
                    your_niche=request.your_niche,
                    your_content_goals=request.your_content_goals,
                    request_data=request.model_dump(mode="json"),
                    competitor_count=competitor_count,
                )
                session.add(record)
                await session.commit()

        logger.info(
            f"[COMPETITOR] Created research job {job_id} with {competitor_count} items"
        )
        return job_id

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        result_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update job status in the database."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return

        async with db_manager.get_session() as session:
            result = await session.execute(
                select(CompetitorResearchModel).where(
                    CompetitorResearchModel.job_id == job_id
                )
            )
            record = result.scalar_one_or_none()
            if record:
                record.status = status
                if result_data:
                    record.result_data = result_data
                if error:
                    record.error = error
                if status in ("completed", "failed"):
                    record.completed_at = datetime.now(timezone.utc)
                await session.commit()

    async def get_research_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get research job result from database."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return None

        async with db_manager.get_session() as session:
            result = await session.execute(
                select(CompetitorResearchModel).where(
                    CompetitorResearchModel.job_id == job_id
                )
            )
            record = result.scalar_one_or_none()
            if record:
                return record.to_dict()
        return None

    async def list_research_jobs(
        self,
        user_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[CompetitorResearchListItem]:
        """List competitor research jobs."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return []

        async with db_manager.get_session() as session:
            query = select(CompetitorResearchModel).order_by(
                CompetitorResearchModel.created_at.desc()
            )
            if user_id:
                query = query.where(CompetitorResearchModel.user_id == user_id)
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            records = result.scalars().all()

            items = []
            for r in records:
                items.append(
                    CompetitorResearchListItem(
                        job_id=r.job_id,
                        status=r.status,
                        content_type=r.content_type,
                        competitor_count=r.competitor_count,
                        your_niche=r.your_niche,
                        created_at=r.created_at,
                        completed_at=r.completed_at,
                    )
                )
            return items

    async def save_crawled_url_content(
        self,
        research_job_id: str,
        url: str,
        fetched: Dict[str, Any],
    ) -> None:
        """Persist the raw fetched content for a URL to the database."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return

        async with db_manager.get_session() as session:
            record = CrawledUrlContentModel(
                research_job_id=research_job_id,
                url=url,
                title=fetched.get("title") or None,
                full_content=fetched.get("content") or None,
                meta_description=fetched.get("meta_description") or None,
                word_count=fetched.get("word_count") or 0,
            )
            session.add(record)
            await session.commit()

    async def get_crawled_url_content(
        self, research_job_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieve all crawled URL content records for a research job."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return []

        async with db_manager.get_session() as session:
            result = await session.execute(
                select(CrawledUrlContentModel).where(
                    CrawledUrlContentModel.research_job_id == research_job_id
                )
            )
            records = result.scalars().all()
            return [r.to_dict() for r in records]

    async def analyze_content_with_llm(
        self,
        content_item: Dict[str, Any],
        niche: Optional[str],
        goals: Optional[str],
    ) -> CompetitorContentAnalysis:
        """
        Use LLM to analyze a single piece of competitor content.

        Args:
            content_item: Dict with keys: title, content, url, platform, content_type
            niche: User's niche/industry for context
            goals: User's content goals

        Returns:
            CompetitorContentAnalysis with detailed breakdown
        """
        title = content_item.get("title", "Unknown")
        content = content_item.get("content", "")
        url = content_item.get("url", "")
        platform = content_item.get("platform", "")
        content_type = content_item.get("content_type", "blog")

        content_snippet = content[:3000] if content else "(no content provided)"

        template_vars = dict(
            url=url or "N/A",
            title=title,
            content_type=content_type,
            platform=platform or "",
            meta_description=content_item.get("meta_description") or "",
            content_snippet=content_snippet,
            niche=niche or "general",
            goals=goals or "increase engagement and organic traffic",
        )
        arthur_result = await fetch_arthur_prompt("competitor_research_analysis")
        arthur_model: Optional[str] = None
        arthur_provider: Optional[str] = None
        if arthur_result:
            system_prompt = arthur_result.system_content
            user_prompt = (
                Environment(autoescape=False)
                .from_string(arthur_result.user_template)
                .render(**template_vars)
            )
            arthur_model = arthur_result.model_name
            arthur_provider = arthur_result.model_provider
        else:
            system_prompt = get_template(
                _LANG, "competitor_research_analysis_agent_instructions"
            ).render()
            user_prompt = get_template(
                _LANG, "competitor_research_analysis_user_prompt"
            ).render(**template_vars)

        effective_model = arthur_model or "gpt-4o-mini"
        extra_config = arthur_result.model_config if arthur_result else None
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "assistant",
                "content": "I have reviewed the context from the previous pipeline steps and am ready to proceed.\n\n",
            },
            {"role": "user", "content": user_prompt},
        ]

        from marketing_project.services.function_pipeline.providers import (
            call_llm_structured,
        )

        try:
            parsed, _ = await call_llm_structured(
                messages=messages,
                response_model=CompetitorContentAnalysis,
                model=effective_model,
                temperature=0.3,
                provider=arthur_provider,
                model_config=extra_config,
            )
            return parsed
        except Exception as e:
            logger.error(
                f"[COMPETITOR] LLM analysis failed for '{title}': {e}", exc_info=True
            )
            return CompetitorContentAnalysis(
                url=url or None,
                title=title,
                content_type=(
                    content_type
                    if content_type in ("blog", "social_media")
                    else "unknown"
                ),
                platform=platform or None,
                content_snippet=content_snippet[:2000] if content_snippet else None,
                actionable_insights=[f"Analysis failed: {str(e)[:200]}"],
            )

    async def generate_summary(
        self,
        analyses: List[CompetitorContentAnalysis],
        niche: Optional[str],
        goals: Optional[str],
    ) -> CompetitorResearchSummary:
        """
        Generate cross-competitor strategic summary using LLM.

        Args:
            analyses: List of individual content analyses
            niche: User's niche
            goals: User's content goals

        Returns:
            CompetitorResearchSummary with strategic insights
        """
        if not analyses:
            return CompetitorResearchSummary(
                recommended_content_strategy="No competitor content was analyzed."
            )

        # Build a compact JSON of all analyses for the prompt
        analyses_compact = []
        for a in analyses:
            analyses_compact.append(
                {
                    "title": a.title,
                    "url": a.url,
                    "content_type": a.content_type,
                    "platform": a.platform,
                    "overall_quality_score": a.overall_quality_score,
                    "performance_tier": a.performance_tier,
                    "strength_factors": [f.factor for f in a.strength_factors],
                    "weakness_factors": [f.factor for f in a.weakness_factors],
                    "key_topics_covered": a.key_topics_covered[:5],
                    "tone_and_voice": a.tone_and_voice,
                    "unique_angle": a.unique_angle,
                    "actionable_insights": a.actionable_insights[:3],
                }
            )

        has_social = any(a.content_type == "social_media" for a in analyses)
        template_vars = dict(
            count=len(analyses),
            analyses_json=json.dumps(analyses_compact, indent=2),
            niche=niche or "general",
            goals=goals or "increase engagement and organic traffic",
            has_social=has_social,
        )
        arthur_result = await fetch_arthur_prompt("competitor_research_summary")
        arthur_model: Optional[str] = None
        arthur_provider: Optional[str] = None
        if arthur_result:
            system_prompt = arthur_result.system_content
            user_prompt = (
                Environment(autoescape=False)
                .from_string(arthur_result.user_template)
                .render(**template_vars)
            )
            arthur_model = arthur_result.model_name
            arthur_provider = arthur_result.model_provider
        else:
            system_prompt = get_template(
                _LANG, "competitor_research_summary_agent_instructions"
            ).render()
            user_prompt = get_template(
                _LANG, "competitor_research_summary_user_prompt"
            ).render(**template_vars)

        effective_model = arthur_model or "gpt-4o-mini"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "assistant",
                "content": "I have reviewed the context from the previous pipeline steps and am ready to proceed.\n\n",
            },
            {"role": "user", "content": user_prompt},
        ]

        extra_config = arthur_result.model_config if arthur_result else None

        from marketing_project.services.function_pipeline.providers import (
            call_llm_structured,
        )

        try:
            parsed, _ = await call_llm_structured(
                messages=messages,
                response_model=CompetitorResearchSummary,
                model=effective_model,
                temperature=0.3,
                provider=arthur_provider,
                model_config=extra_config,
            )
            return parsed
        except Exception as e:
            logger.error(f"[COMPETITOR] Summary generation failed: {e}", exc_info=True)
            return CompetitorResearchSummary(
                recommended_content_strategy=f"Summary generation failed: {str(e)[:200]}"
            )

    async def run_research(
        self,
        job_id: str,
        request: CompetitorResearchRequest,
    ) -> CompetitorResearchResult:
        """
        Execute a full competitor research job.

        Analyzes each content item individually then synthesizes a cross-competitor summary.

        Args:
            job_id: The job ID to update during processing
            request: The research request with content to analyze

        Returns:
            CompetitorResearchResult with all analyses and summary
        """
        await self.update_job_status(job_id, "processing")

        all_items: List[Dict[str, Any]] = []

        # Build the list of items to analyze
        if request.competitor_content:
            for item in request.competitor_content:
                all_items.append(item)

        # For URLs: fetch the actual page content before analysis.
        if request.competitor_urls:
            social_platforms = [
                "linkedin.com",
                "twitter.com",
                "instagram.com",
                "facebook.com",
                "x.com",
            ]
            for url in request.competitor_urls:
                detected_platform = next(
                    (p.split(".")[0] for p in social_platforms if p in url.lower()),
                    None,
                )
                content_type = "social_media" if detected_platform else "blog"

                fetched = await self.fetch_url_content(url)
                await self.save_crawled_url_content(job_id, url, fetched)
                all_items.append(
                    {
                        "url": url,
                        "title": fetched["title"]
                        or f"Content at {urlparse(url).netloc}",
                        "content": fetched["content"],
                        "meta_description": fetched["meta_description"],
                        "platform": detected_platform or "",
                        "content_type": content_type,
                    }
                )

        logger.info(
            f"[COMPETITOR] Job {job_id}: analyzing {len(all_items)} content items"
        )

        analyses: List[CompetitorContentAnalysis] = []
        for item in all_items:
            try:
                analysis = await self.analyze_content_with_llm(
                    content_item=item,
                    niche=request.your_niche,
                    goals=request.your_content_goals,
                )
                analyses.append(analysis)
                logger.info(
                    f"[COMPETITOR] Job {job_id}: analyzed '{item.get('title', 'unknown')}' "
                    f"(score={analysis.overall_quality_score})"
                )
            except Exception as e:
                logger.error(f"[COMPETITOR] Failed to analyze item: {e}", exc_info=True)

        # Generate cross-competitor summary
        summary = None
        if analyses:
            summary = await self.generate_summary(
                analyses=analyses,
                niche=request.your_niche,
                goals=request.your_content_goals,
            )

        result = CompetitorResearchResult(
            job_id=job_id,
            status="completed",
            request=request,
            analyses=analyses,
            summary=summary,
            completed_at=datetime.now(timezone.utc),
        )

        # Persist result
        await self.update_job_status(
            job_id,
            "completed",
            result_data=result.model_dump(mode="json"),
        )

        logger.info(
            f"[COMPETITOR] Job {job_id} completed: {len(analyses)} analyses, "
            f"summary={'yes' if summary else 'no'}"
        )

        return result


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_competitor_research_service: Optional[CompetitorResearchService] = None


def get_competitor_research_service() -> CompetitorResearchService:
    """Get or create the singleton CompetitorResearchService instance."""
    global _competitor_research_service
    if _competitor_research_service is None:
        _competitor_research_service = CompetitorResearchService()
    return _competitor_research_service
