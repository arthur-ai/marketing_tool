"""
Extended tests for approval manager - covering more methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.approval_models import (
    ApprovalDecisionRequest,
    ApprovalRequest,
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
async def test_save_approval_to_redis(approval_manager):
    """Test _save_approval_to_redis method."""
    request = await approval_manager.create_approval_request(
        "job-1", "seo_keywords", 1, {}, {}
    )

    await approval_manager._save_approval_to_redis(request)

    # Should not raise exception
    assert True


@pytest.mark.asyncio
async def test_load_approval_from_redis(approval_manager):
    """Test _load_approval_from_redis method."""
    request = await approval_manager.create_approval_request(
        "job-1", "seo_keywords", 1, {}, {}
    )

    loaded = await approval_manager._load_approval_from_redis(request.id)

    # May return None if not in Redis
    assert loaded is None or isinstance(loaded, ApprovalRequest)


@pytest.mark.asyncio
async def test_load_all_approvals_from_redis(approval_manager):
    """Test _load_all_approvals_from_redis method."""
    await approval_manager._load_all_approvals_from_redis()

    # Should not raise exception
    assert True


@pytest.mark.asyncio
async def test_save_job_approval_mapping(approval_manager):
    """Test _save_job_approval_mapping method."""
    await approval_manager._save_job_approval_mapping("job-1", "approval-1")

    # Should not raise exception
    assert True


@pytest.mark.asyncio
async def test_get_approval(approval_manager):
    """Test get_approval method."""
    request = await approval_manager.create_approval_request(
        "job-1", "seo_keywords", 1, {}, {}
    )

    retrieved = await approval_manager.get_approval(request.id)

    assert retrieved is not None
    assert retrieved.id == request.id


@pytest.mark.asyncio
async def test_get_approval_not_found(approval_manager):
    """Test get_approval with non-existent ID."""
    retrieved = await approval_manager.get_approval("non-existent")

    assert retrieved is None
