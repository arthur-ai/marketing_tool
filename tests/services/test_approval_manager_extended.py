"""
Extended tests for approval manager service.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.approval_models import (
    ApprovalDecisionRequest,
    ApprovalRequest,
    ApprovalSettings,
)
from marketing_project.services.approval_manager import (
    ApprovalManager,
    get_approval_manager,
)


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
async def test_create_approval_request(approval_manager):
    """Test creating an approval request."""
    request = await approval_manager.create_approval_request(
        job_id="test-job-1",
        step_name="seo_keywords",
        step_number=1,
        input_data={"test": "input"},
        output_data={"test": "output"},
    )

    assert request is not None
    assert request.job_id == "test-job-1"
    assert request.step_name == "seo_keywords"
    assert request.status == "pending"


@pytest.mark.asyncio
async def test_get_approval_request(approval_manager):
    """Test getting an approval request."""
    request = await approval_manager.create_approval_request(
        "test-job-1", "seo_keywords", 1, {}, {}
    )

    retrieved = await approval_manager.get_approval_request(request.id)

    assert retrieved is not None
    assert retrieved.id == request.id


@pytest.mark.asyncio
async def test_decide_approval(approval_manager):
    """Test deciding on an approval."""
    request = await approval_manager.create_approval_request(
        "test-job-1", "seo_keywords", 1, {}, {}
    )

    decision = ApprovalDecisionRequest(decision="approved", comments="Looks good")

    result = await approval_manager.decide_approval(request.id, decision)

    assert result is not None
    assert result.status == "approved"


@pytest.mark.asyncio
async def test_get_approvals_for_job(approval_manager):
    """Test getting approvals for a job."""
    await approval_manager.create_approval_request(
        "test-job-1", "seo_keywords", 1, {}, {}
    )
    await approval_manager.create_approval_request(
        "test-job-1", "marketing_brief", 2, {}, {}
    )

    approvals = await approval_manager.get_approvals_for_job("test-job-1")

    assert len(approvals) >= 2


@pytest.mark.asyncio
async def test_save_pipeline_context(approval_manager):
    """Test saving pipeline context."""
    context = {"seo_keywords": {}, "input_content": {}}

    success = await approval_manager.save_pipeline_context("test-job-1", context)

    assert success is True


@pytest.mark.asyncio
async def test_get_pipeline_context(approval_manager):
    """Test getting pipeline context."""
    context = {"seo_keywords": {}, "input_content": {}}
    await approval_manager.save_pipeline_context("test-job-1", context)

    retrieved = await approval_manager.get_pipeline_context("test-job-1")

    assert retrieved is not None
    assert "seo_keywords" in retrieved


@pytest.mark.asyncio
async def test_get_approval_stats(approval_manager):
    """Test getting approval statistics."""
    request1 = await approval_manager.create_approval_request(
        "job-1", "seo_keywords", 1, {}, {}
    )
    request2 = await approval_manager.create_approval_request(
        "job-2", "marketing_brief", 2, {}, {}
    )

    await approval_manager.decide_approval(
        request1.id, ApprovalDecisionRequest(decision="approved")
    )

    stats = await approval_manager.get_approval_stats()

    assert stats is not None
    assert "total_approvals" in stats or "approval_rate" in stats


def test_get_approval_manager_singleton():
    """Test that get_approval_manager returns a singleton."""
    manager1 = get_approval_manager()
    manager2 = get_approval_manager()

    assert manager1 is manager2
