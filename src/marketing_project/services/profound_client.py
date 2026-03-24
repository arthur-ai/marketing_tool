"""
Profound API client for fetching category personas.

API reference: https://docs.tryprofound.com/api-reference/organization/get-category-personas

Personas provide audience context (pain points, motivations, job roles) that
the SEO keywords step uses to generate keywords aligned with who will actually
read and search for the content.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

try:
    from marketing_project.services.profound_settings_manager import (
        get_profound_settings_manager,
    )
except ImportError:  # pragma: no cover
    get_profound_settings_manager = None  # type: ignore[assignment]

# Module-level in-process cache: { "personas:{category_id}": {"ts": float, "data": list} }
_CACHE: Dict[str, Any] = {}
_CACHE_TTL = 3600  # seconds — personas don't change often


class ProfoundPersona:
    """
    Structured representation of a single Profound persona.

    Wraps the raw API response into typed, named attributes used
    by the keyword prompt template.
    """

    def __init__(self, raw: Dict[str, Any]) -> None:
        self.id: str = raw.get("id", "")
        self.name: str = raw.get("name", "")
        persona = raw.get("persona", {})
        behavior = persona.get("behavior", {})
        employment = persona.get("employment", {})
        demographics = persona.get("demographics", {})

        self.pain_points: Optional[str] = behavior.get("painPoints")
        self.motivations: Optional[str] = behavior.get("motivations")
        self.industries: List[str] = employment.get("industry", [])
        self.job_titles: List[str] = employment.get("jobTitle", [])
        self.company_sizes: List[str] = employment.get("companySize", [])
        self.role_seniority: List[str] = employment.get("roleSeniority", [])
        self.age_ranges: List[str] = demographics.get("ageRange", [])

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Return a compact dict for use in Jinja prompt context."""
        return {
            "name": self.name,
            "pain_points": self.pain_points,
            "motivations": self.motivations,
            "job_titles": self.job_titles,
            "industries": self.industries,
            "role_seniority": self.role_seniority,
        }


class ProfoundClient:
    """
    Async HTTP client for the Profound API.

    Caches persona responses per category for CACHE_TTL seconds.
    All errors are non-fatal — the caller receives an empty list
    and pipeline execution continues normally.
    """

    BASE_URL = "https://api.tryprofound.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_category_id: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("PROFOUND_API_KEY", "")
        self.default_category_id = default_category_id or os.getenv(
            "PROFOUND_CATEGORY_ID", ""
        )

    def is_configured(self) -> bool:
        """Return True if an API key is available."""
        return bool(self.api_key)

    async def get_category_personas(self, category_id: str) -> List[ProfoundPersona]:
        """
        Fetch personas for a Profound category UUID.

        Results are cached in-process for CACHE_TTL seconds so repeated
        pipeline runs don't hammer the API.

        Args:
            category_id: UUID of the Profound category.

        Returns:
            List of ProfoundPersona objects, or [] on any error / missing config.
        """
        if not self.is_configured():
            logger.debug("PROFOUND_API_KEY not set — skipping persona fetch")
            return []

        cache_key = f"personas:{category_id}"
        cached = _CACHE.get(cache_key)
        if cached and (time.time() - cached["ts"]) < _CACHE_TTL:
            logger.debug(
                "Profound personas for %s served from cache (%d personas)",
                category_id,
                len(cached["data"]),
            )
            return cached["data"]

        try:
            if aiohttp is None:
                logger.warning("aiohttp not installed — cannot fetch Profound personas")
                return []

            url = f"{self.BASE_URL}/v1/org/categories/{category_id}/personas"
            headers = {
                "X-API-Key": self.api_key,
                "Accept": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "Profound API returned HTTP %d for category %s — skipping personas",
                            resp.status,
                            category_id,
                        )
                        return []
                    data = await resp.json()

            personas = [ProfoundPersona(p) for p in data.get("data", [])]
            _CACHE[cache_key] = {"ts": time.time(), "data": personas}
            logger.info(
                "Fetched %d Profound personas for category %s",
                len(personas),
                category_id,
            )
            return personas

        except Exception as e:
            logger.warning(
                "Failed to fetch Profound personas for category %s: %s", category_id, e
            )
            return []


async def get_profound_client() -> Tuple[ProfoundClient, Optional[str]]:
    """
    Return a (ProfoundClient, default_category_id) tuple, loading credentials
    from the database first (falling back to environment variables).

    If no API key is configured in either source, the client's is_configured()
    will return False and the SEO step will run without persona context.
    """
    try:
        mgr = get_profound_settings_manager()
        api_key, default_category_id = await mgr.get_credentials()
    except Exception as exc:
        logger.warning(
            "Could not load Profound settings from DB — falling back to env vars: %s",
            exc,
        )
        api_key = os.getenv("PROFOUND_API_KEY", "") or None
        default_category_id = os.getenv("PROFOUND_CATEGORY_ID", "") or None

    client = ProfoundClient(
        api_key=api_key,
        default_category_id=default_category_id,
    )
    return client, default_category_id
