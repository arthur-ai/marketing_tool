"""
Tests for quality score persistence (_persist_quality_score) and analytics endpoint.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.models.user_context import UserContext
from marketing_project.server import app

client = TestClient(app)


def _admin_user():
    return UserContext(
        user_id="admin-1",
        email="admin@example.com",
        username="admin",
        roles=["admin"],
        realm_roles=["admin"],
        client_roles=[],
    )


@pytest.fixture
def override_auth_admin():
    from marketing_project.middleware.rbac import require_roles

    admin = _admin_user()

    def _require_admin_override():
        return admin

    app.dependency_overrides[require_roles(["admin"])] = _require_admin_override
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# _persist_quality_score helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_quality_score_writes_row():
    """Happy path: _persist_quality_score executes an upsert statement."""
    from marketing_project.api.jobs import _persist_quality_score

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()

    mock_db = MagicMock()
    mock_db.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_db.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.database.get_database_manager", return_value=mock_db
    ):
        await _persist_quality_score(
            job_id="job-123",
            word_count=500,
            flesch_kincaid_grade=8.5,
            has_headings=True,
            keyword_match_pct=0.75,
            profound_personas_used=["Persona A"],
        )

    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_persist_quality_score_swallows_exception():
    """DB failure must NOT propagate — the helper is fire-and-forget."""
    from marketing_project.api.jobs import _persist_quality_score

    mock_db = MagicMock()
    mock_db.get_session.return_value.__aenter__ = AsyncMock(
        side_effect=RuntimeError("DB down")
    )
    mock_db.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.database.get_database_manager", return_value=mock_db
    ):
        # Should not raise
        await _persist_quality_score(
            job_id="job-fail",
            word_count=None,
            flesch_kincaid_grade=None,
            has_headings=None,
            keyword_match_pct=None,
            profound_personas_used=None,
        )


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/quality-scores endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_quality_scores_db():
    """Mock the DB session used by get_quality_scores."""
    row1 = MagicMock()
    row1.to_dict.return_value = {
        "id": 1,
        "job_id": "job-1",
        "word_count": 400,
        "flesch_kincaid_grade": 7.5,
        "has_headings": True,
        "keyword_match_pct": 0.8,
        "profound_personas_used": [],
        "computed_at": "2026-01-01T00:00:00",
    }
    row2 = MagicMock()
    row2.to_dict.return_value = {
        "id": 2,
        "job_id": "job-2",
        "word_count": 600,
        "flesch_kincaid_grade": 9.0,
        "has_headings": False,
        "keyword_match_pct": 0.5,
        "profound_personas_used": ["P1"],
        "computed_at": "2026-01-02T00:00:00",
    }

    count_result = MagicMock()
    count_result.scalar_one.return_value = 2

    rows_result = MagicMock()
    rows_result.scalars.return_value.all.return_value = [row1, row2]

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=[count_result, rows_result])

    mock_db = MagicMock()
    mock_db.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_db.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    return mock_db


@pytest.mark.asyncio
async def test_get_quality_scores_returns_paginated_result(
    mock_quality_scores_db, override_auth_admin
):
    # Lazy imports in get_quality_scores resolve at the services.database module level
    with patch(
        "marketing_project.services.database.get_database_manager",
        return_value=mock_quality_scores_db,
    ):
        response = client.get("/api/v1/analytics/quality-scores")

    assert response.status_code == 200
    data = response.json()
    assert "scores" in data
    assert "total" in data
    assert data["total"] == 2
    assert len(data["scores"]) == 2


@pytest.mark.asyncio
async def test_get_quality_scores_total_from_count_not_len(override_auth_admin):
    """
    total must reflect COUNT(*), not len(scores).
    Simulate: COUNT returns 50, but only 10 rows returned (due to limit).
    """
    count_result = MagicMock()
    count_result.scalar_one.return_value = 50

    ten_rows = []
    for i in range(10):
        r = MagicMock()
        r.to_dict.return_value = {
            "id": i,
            "job_id": f"job-{i}",
            "word_count": 100,
            "flesch_kincaid_grade": 7.0,
            "has_headings": True,
            "keyword_match_pct": 0.5,
            "profound_personas_used": [],
            "computed_at": None,
        }
        ten_rows.append(r)

    rows_result = MagicMock()
    rows_result.scalars.return_value.all.return_value = ten_rows

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=[count_result, rows_result])

    mock_db = MagicMock()
    mock_db.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_db.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.database.get_database_manager",
        return_value=mock_db,
    ):
        response = client.get("/api/v1/analytics/quality-scores?limit=10&offset=0")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 50  # COUNT(*), not len(scores)
    assert len(data["scores"]) == 10


@pytest.mark.asyncio
async def test_get_quality_scores_filter_by_job_id(
    mock_quality_scores_db, override_auth_admin
):
    with patch(
        "marketing_project.services.database.get_database_manager",
        return_value=mock_quality_scores_db,
    ):
        response = client.get("/api/v1/analytics/quality-scores?job_id=job-1")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_quality_scores_requires_admin():
    """Without admin auth, endpoint should return 401, 403, or 500 (Keycloak unreachable)."""
    response = client.get("/api/v1/analytics/quality-scores")
    assert response.status_code in (401, 403, 422, 500)


@pytest.mark.asyncio
async def test_get_quality_scores_pagination_params(override_auth_admin):
    """limit and offset query params are validated (ge/le constraints)."""
    # limit=0 should fail validation (ge=1)
    response = client.get("/api/v1/analytics/quality-scores?limit=0")
    assert response.status_code == 422

    # limit=501 should fail (le=500)
    response = client.get("/api/v1/analytics/quality-scores?limit=501")
    assert response.status_code == 422

    # offset=-1 should fail (ge=0)
    response = client.get("/api/v1/analytics/quality-scores?offset=-1")
    assert response.status_code == 422
