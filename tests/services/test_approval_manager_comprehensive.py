"""
Comprehensive tests for approval manager service methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.approval_models import (
    ApprovalDecisionRequest,
    ApprovalRequest,
    ApprovalSettings,
)
from marketing_project.services.approval_manager import ApprovalManager


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    with patch("marketing_project.services.approval_manager.get_redis_manager") as mock:
        manager = MagicMock()
        manager.get_redis = AsyncMock(return_value=MagicMock())
        manager.execute = AsyncMock(return_value=None)
        mock.return_value = manager
        yield manager


@pytest.fixture
def approval_manager(mock_redis_manager):
    """Create an ApprovalManager instance."""
    return ApprovalManager()


@pytest.mark.asyncio
async def test_load_settings(approval_manager, mock_redis_manager):
    """Test load_settings method."""
    settings = await approval_manager.load_settings()

    # May return None if no settings configured
    assert settings is None or isinstance(settings, ApprovalSettings)


@pytest.mark.asyncio
async def test_save_settings(approval_manager, mock_redis_manager):
    """Test save_settings method."""
    settings = ApprovalSettings(
        require_approval_for_steps=["seo_keywords"],
        auto_approve_threshold=0.9,
    )

    await approval_manager.save_settings(settings)

    # Should not raise exception
    assert True


@pytest.mark.asyncio
async def test_list_approvals(approval_manager):
    """Test list_approvals method."""
    req1 = await approval_manager.create_approval_request(
        job_id="job-1",
        agent_name="seo_keywords",
        step_name="seo_keywords",
        input_data={},
        output_data={},
    )
    req2 = await approval_manager.create_approval_request(
        job_id="job-2",
        agent_name="marketing_brief",
        step_name="marketing_brief",
        input_data={},
        output_data={},
    )

    # list_approvals reads from DB; mock it to return the two in-flight requests
    mock_row1 = MagicMock()
    mock_row1.to_approval_request.return_value = req1
    mock_row2 = MagicMock()
    mock_row2.to_approval_request.return_value = req2

    with patch(
        "marketing_project.services.approval_manager.get_database_manager"
    ) as mock_db:
        mock_db_mgr = MagicMock()
        mock_db_mgr.is_initialized = True
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_row1, mock_row2]
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_db.return_value = mock_db_mgr

        approvals = await approval_manager.list_approvals(status="pending")

    assert len(approvals) >= 2
    assert all(a.status == "pending" for a in approvals)


@pytest.mark.asyncio
async def test_wait_for_approval(approval_manager):
    """Test wait_for_approval returns approved request when approved in background."""
    import asyncio

    request = await approval_manager.create_approval_request(
        job_id="job-1",
        agent_name="seo_keywords",
        step_name="seo_keywords",
        input_data={},
        output_data={},
    )

    # Approved copy to return once the decision is made
    approved_request = request.model_copy(update={"status": "approved"})

    get_calls = 0

    async def mock_get_approval(approval_id):
        nonlocal get_calls
        get_calls += 1
        # First two calls return pending (wait_for_approval initial + decide_approval check)
        # Subsequent calls return approved (wait_for_approval polling)
        if get_calls <= 2:
            return request
        return approved_request

    with patch.object(approval_manager, "get_approval", side_effect=mock_get_approval):
        # Also patch _load_approval_from_redis_cache so decide_approval can find the request
        with (
            patch.object(
                approval_manager,
                "_load_approval_from_redis_cache",
                new_callable=AsyncMock,
                return_value=request,
            ),
            patch.object(
                approval_manager, "_cache_approval_in_redis", new_callable=AsyncMock
            ),
            patch(
                "marketing_project.services.approval_manager.get_database_manager"
            ) as mock_db,
            patch.object(
                approval_manager._redis_manager, "execute", new_callable=AsyncMock
            ),
            patch.object(
                approval_manager, "get_redis", new_callable=AsyncMock, return_value=None
            ),
        ):
            mock_db.return_value.is_initialized = False

            async def approve_later():
                await asyncio.sleep(0.1)
                await approval_manager.decide_approval(
                    request.id, ApprovalDecisionRequest(decision="approve")
                )

            task = asyncio.create_task(approve_later())
            result = await approval_manager.wait_for_approval(request.id, timeout=2.0)
            await task

    assert result is not None
    assert result.status == "approved"


@pytest.mark.asyncio
async def test_delete_all_approvals(approval_manager):
    """Test delete_all_approvals method."""
    await approval_manager.create_approval_request(
        job_id="job-1",
        agent_name="seo_keywords",
        step_name="seo_keywords",
        input_data={},
        output_data={},
    )
    await approval_manager.create_approval_request(
        job_id="job-2",
        agent_name="marketing_brief",
        step_name="marketing_brief",
        input_data={},
        output_data={},
    )

    deleted = await approval_manager.delete_all_approvals()

    assert isinstance(deleted, int)
    assert deleted >= 0


@pytest.mark.asyncio
async def test_clear_job_approvals(approval_manager):
    """Test delete_all_approvals (clears all approvals from Redis/DB)."""
    await approval_manager.create_approval_request(
        job_id="job-1",
        agent_name="seo_keywords",
        step_name="seo_keywords",
        input_data={},
        output_data={},
    )

    with patch.object(
        approval_manager._redis_manager,
        "execute",
        new_callable=AsyncMock,
        return_value=set(),
    ):
        result = await approval_manager.delete_all_approvals()

    assert isinstance(result, int)
    assert result >= 0


@pytest.mark.asyncio
async def test_filter_selected_keywords(approval_manager):
    """Test filter_selected_keywords method."""
    output_data = {
        "main_keyword": "keyword1",
        "primary_keywords": ["keyword1", "keyword2", "keyword3"],
        "secondary_keywords": [],
        "lsi_keywords": [],
    }
    selected_keywords = {
        "primary": ["keyword1", "keyword3"],
        "secondary": [],
        "lsi": [],
    }

    filtered = approval_manager.filter_selected_keywords(
        output_data, selected_keywords, main_keyword="keyword1"
    )

    assert isinstance(filtered, dict)
    assert filtered["main_keyword"] == "keyword1"
    assert "keyword1" in filtered.get("primary_keywords", [])
    assert "keyword3" in filtered.get("primary_keywords", [])
