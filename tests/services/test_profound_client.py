"""
Unit tests for ProfoundClient and SEOKeywordsPlugin persona injection.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.profound_client import (
    _CACHE,
    ProfoundClient,
    ProfoundPersona,
)

# ---------------------------------------------------------------------------
# ProfoundPersona
# ---------------------------------------------------------------------------


def test_profound_persona_parses_full_response():
    raw = {
        "id": "abc-123",
        "name": "Enterprise Buyer",
        "persona": {
            "behavior": {
                "painPoints": "Scaling ML ops without growing headcount",
                "motivations": "Reduce time-to-production for models",
            },
            "employment": {
                "industry": ["Financial Services", "Insurance"],
                "jobTitle": ["Head of Data Science", "VP of Analytics"],
                "companySize": ["1001-5000", "5001+"],
                "roleSeniority": ["Director", "VP"],
            },
            "demographics": {"ageRange": ["35-44", "45-54"]},
        },
    }
    p = ProfoundPersona(raw)
    assert p.name == "Enterprise Buyer"
    assert p.pain_points == "Scaling ML ops without growing headcount"
    assert "Financial Services" in p.industries
    assert "Head of Data Science" in p.job_titles
    assert p.role_seniority == ["Director", "VP"]

    d = p.to_prompt_dict()
    assert d["name"] == "Enterprise Buyer"
    assert d["job_titles"] == ["Head of Data Science", "VP of Analytics"]


def test_profound_persona_handles_missing_fields():
    p = ProfoundPersona({"id": "x", "name": "Minimal"})
    assert p.pain_points is None
    assert p.motivations is None
    assert p.industries == []
    d = p.to_prompt_dict()
    assert d["pain_points"] is None


# ---------------------------------------------------------------------------
# ProfoundClient
# ---------------------------------------------------------------------------


def test_profound_client_not_configured_without_key():
    client = ProfoundClient(api_key="")
    assert not client.is_configured()


def test_profound_client_configured_with_key():
    client = ProfoundClient(api_key="test-key-abc")
    assert client.is_configured()


@pytest.mark.asyncio
async def test_get_category_personas_returns_empty_when_not_configured():
    client = ProfoundClient(api_key="")
    result = await client.get_category_personas("some-uuid")
    assert result == []


@pytest.mark.asyncio
async def test_get_category_personas_uses_cache():
    import time

    category_id = "cached-category-uuid"
    cache_key = f"personas:{category_id}"
    persona = ProfoundPersona({"id": "1", "name": "Cached Persona"})
    _CACHE[cache_key] = {"ts": time.time(), "data": [persona]}

    client = ProfoundClient(api_key="key")
    result = await client.get_category_personas(category_id)
    assert len(result) == 1
    assert result[0].name == "Cached Persona"

    # Cleanup
    del _CACHE[cache_key]


@pytest.mark.asyncio
async def test_get_category_personas_fetches_from_api():
    category_id = "fresh-uuid"
    api_response = {
        "data": [
            {
                "id": "p1",
                "name": "Data Scientist",
                "persona": {
                    "behavior": {
                        "painPoints": "Too much manual work",
                        "motivations": "Automation",
                    },
                    "employment": {
                        "industry": ["Tech"],
                        "jobTitle": ["Data Scientist"],
                        "companySize": ["100-500"],
                        "roleSeniority": ["IC"],
                    },
                    "demographics": {"ageRange": ["25-34"]},
                },
            }
        ]
    }

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=api_response)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    # Remove any stale cache entry
    _CACHE.pop(f"personas:{category_id}", None)

    with patch(
        "marketing_project.services.profound_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = ProfoundClient(api_key="test-key")
        result = await client.get_category_personas(category_id)

    assert len(result) == 1
    assert result[0].name == "Data Scientist"
    assert result[0].pain_points == "Too much manual work"

    # Cleanup
    _CACHE.pop(f"personas:{category_id}", None)


@pytest.mark.asyncio
async def test_get_category_personas_returns_empty_on_api_error():
    category_id = "error-uuid"
    _CACHE.pop(f"personas:{category_id}", None)

    with patch(
        "marketing_project.services.profound_client.aiohttp.ClientSession",
        side_effect=Exception("network error"),
    ):
        client = ProfoundClient(api_key="test-key")
        result = await client.get_category_personas(category_id)

    assert result == []


@pytest.mark.asyncio
async def test_get_category_personas_returns_empty_on_non_200():
    category_id = "not-found-uuid"
    _CACHE.pop(f"personas:{category_id}", None)

    mock_resp = AsyncMock()
    mock_resp.status = 404
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.profound_client.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        client = ProfoundClient(api_key="test-key")
        result = await client.get_category_personas(category_id)

    assert result == []
