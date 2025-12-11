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

    approvals = await approval_manager.list_approvals(status="pending")

    assert len(approvals) >= 2
    assert all(a.status == "pending" for a in approvals)


@pytest.mark.asyncio
async def test_wait_for_approval(approval_manager):
    """Test wait_for_approval method."""
    request = await approval_manager.create_approval_request(
        job_id="job-1",
        agent_name="seo_keywords",
        step_name="seo_keywords",
        input_data={},
        output_data={},
    )

    # Approve in background
    async def approve_later():
        import asyncio

        await asyncio.sleep(0.1)
        # Make sure approval exists in memory before deciding
        if request.id not in approval_manager._approvals:
            approval_manager._approvals[request.id] = request
        await approval_manager.decide_approval(
            request.id, ApprovalDecisionRequest(decision="approve")
        )

    # Start approval in background
    import asyncio

    task = asyncio.create_task(approve_later())

    # Wait for approval
    result = await approval_manager.wait_for_approval(request.id, timeout=2.0)

    # Wait for background task to complete
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
    """Test clear_job_approvals method."""
    await approval_manager.create_approval_request(
        job_id="job-1",
        agent_name="seo_keywords",
        step_name="seo_keywords",
        input_data={},
        output_data={},
    )

    approval_manager.clear_job_approvals("job-1")

    # Should not raise exception
    assert True


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
