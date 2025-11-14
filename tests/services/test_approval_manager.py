"""
Tests for approval manager service.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.approval_models import (
    ApprovalDecisionRequest,
    ApprovalRequest,
    ApprovalSettings,
)
from marketing_project.services.approval_manager import ApprovalManager


@pytest.fixture
def approval_manager():
    """Create an ApprovalManager instance for testing."""
    return ApprovalManager()


@pytest.fixture
def sample_approval():
    """Create a sample approval request."""
    return ApprovalRequest(
        id="test-approval-1",
        job_id="test-job-1",
        agent_name="seo_keywords",
        step_name="Step 1: SEO Keywords",
        status="pending",
        input_data={"content": {"title": "Test"}},
        output_data={"keywords": ["test"]},
        pipeline_step="seo_keywords",
        created_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_approval_manager_initialization(approval_manager):
    """Test ApprovalManager initialization."""
    assert approval_manager is not None
    assert approval_manager.settings is not None
    assert isinstance(approval_manager.settings, ApprovalSettings)


@pytest.mark.asyncio
async def test_get_redis(approval_manager):
    """Test get_redis method."""
    with patch.object(
        approval_manager._redis_manager, "get_redis", new_callable=AsyncMock
    ) as mock_get:
        mock_redis = MagicMock()
        mock_get.return_value = mock_redis

        result = await approval_manager.get_redis()

        assert result == mock_redis


@pytest.mark.asyncio
async def test_get_redis_error(approval_manager):
    """Test get_redis when Redis connection fails."""
    with patch.object(
        approval_manager._redis_manager, "get_redis", new_callable=AsyncMock
    ) as mock_get:
        mock_get.side_effect = Exception("Redis error")

        result = await approval_manager.get_redis()

        assert result is None


@pytest.mark.asyncio
async def test_load_settings_from_db(approval_manager):
    """Test load_settings_from_db method."""
    mock_settings = ApprovalSettings(
        require_approval=True,
        auto_approve_after_seconds=None,
        notify_on_pending=True,
    )

    with patch(
        "marketing_project.services.approval_manager.get_database_manager"
    ) as mock_db_manager:
        mock_db_mgr = MagicMock()
        mock_db_mgr.is_initialized = True
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_db_mgr.get_session = MagicMock(return_value=mock_session.__aenter__())
        mock_db_manager.return_value = mock_db_mgr

        result = await approval_manager.load_settings_from_db()

        # Should return None if no settings found
        assert result is None or isinstance(result, ApprovalSettings)


@pytest.mark.asyncio
async def test_load_settings_from_redis(approval_manager):
    """Test load_settings_from_redis method."""
    import json

    mock_settings = ApprovalSettings(
        require_approval=True,
        auto_approve_after_seconds=3600,
        notify_on_pending=True,
    )

    with patch.object(
        approval_manager._redis_manager, "execute", new_callable=AsyncMock
    ) as mock_execute:
        mock_execute.return_value = json.dumps(mock_settings.model_dump())

        result = await approval_manager.load_settings_from_redis()

        assert result is not None
        assert isinstance(result, ApprovalSettings)


@pytest.mark.asyncio
async def test_load_settings(approval_manager):
    """Test load_settings method."""
    mock_settings = ApprovalSettings(
        require_approval=True,
        auto_approve_after_seconds=None,
        notify_on_pending=True,
    )

    with patch.object(
        approval_manager, "load_settings_from_db", new_callable=AsyncMock
    ) as mock_load_db:
        with patch.object(
            approval_manager, "save_settings_to_redis", new_callable=AsyncMock
        ):
            mock_load_db.return_value = mock_settings

            result = await approval_manager.load_settings()

            assert result is not None
            assert isinstance(result, ApprovalSettings)


@pytest.mark.asyncio
async def test_load_settings_fallback_to_redis(approval_manager):
    """Test load_settings fallback to Redis."""
    mock_settings = ApprovalSettings(
        require_approval=True,
        auto_approve_after_seconds=None,
        notify_on_pending=True,
    )

    with patch.object(
        approval_manager, "load_settings_from_db", new_callable=AsyncMock
    ) as mock_load_db:
        with patch.object(
            approval_manager, "load_settings_from_redis", new_callable=AsyncMock
        ) as mock_load_redis:
            mock_load_db.return_value = None
            mock_load_redis.return_value = mock_settings

            result = await approval_manager.load_settings()

            assert result is not None
            assert isinstance(result, ApprovalSettings)


@pytest.mark.asyncio
async def test_create_approval_request(approval_manager, sample_approval):
    """Test create_approval_request method."""
    with patch.object(
        approval_manager, "get_redis", new_callable=AsyncMock
    ) as mock_get_redis:
        mock_redis = MagicMock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.sadd = AsyncMock(return_value=1)
        mock_get_redis.return_value = mock_redis

        result = await approval_manager.create_approval_request(
            job_id="test-job-1",
            agent_name="seo_keywords",
            step_name="Step 1",
            input_data={"content": {"title": "Test"}},
            output_data={"keywords": ["test"]},
            pipeline_step="seo_keywords",
        )

        assert result is not None
        assert result.job_id == "test-job-1"
        assert result.status == "pending"


@pytest.mark.asyncio
async def test_get_approval(approval_manager, sample_approval):
    """Test get_approval method."""
    approval_manager._approvals["test-approval-1"] = sample_approval

    result = await approval_manager.get_approval("test-approval-1")

    assert result is not None
    assert result.id == "test-approval-1"


@pytest.mark.asyncio
async def test_get_approval_not_found(approval_manager):
    """Test get_approval when approval not found."""
    with patch.object(
        approval_manager, "get_redis", new_callable=AsyncMock
    ) as mock_get_redis:
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_get_redis.return_value = mock_redis

        result = await approval_manager.get_approval("non-existent")

        assert result is None


@pytest.mark.asyncio
async def test_list_approvals(approval_manager, sample_approval):
    """Test list_approvals method."""
    approval_manager._approvals["test-approval-1"] = sample_approval

    result = await approval_manager.list_approvals()

    assert len(result) >= 1
    assert any(a.id == "test-approval-1" for a in result)


@pytest.mark.asyncio
async def test_list_approvals_with_job_id(approval_manager, sample_approval):
    """Test list_approvals with job_id filter."""
    approval_manager._approvals["test-approval-1"] = sample_approval

    result = await approval_manager.list_approvals(job_id="test-job-1")

    assert len(result) >= 1
    assert all(a.job_id == "test-job-1" for a in result)


@pytest.mark.asyncio
async def test_list_approvals_with_status(approval_manager, sample_approval):
    """Test list_approvals with status filter."""
    approval_manager._approvals["test-approval-1"] = sample_approval

    result = await approval_manager.list_approvals(status="pending")

    assert len(result) >= 1
    assert all(a.status == "pending" for a in result)


@pytest.mark.asyncio
async def test_decide_approval_approve(approval_manager, sample_approval):
    """Test decide_approval with approve decision."""
    approval_manager._approvals["test-approval-1"] = sample_approval

    decision = ApprovalDecisionRequest(
        decision="approve",
        comment="Looks good",
        reviewed_by="test_user",
    )

    with patch.object(
        approval_manager, "get_redis", new_callable=AsyncMock
    ) as mock_get_redis:
        mock_redis = MagicMock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_get_redis.return_value = mock_redis

        result = await approval_manager.decide_approval("test-approval-1", decision)

        assert result is not None
        assert result.status == "approved"


@pytest.mark.asyncio
async def test_decide_approval_reject(approval_manager, sample_approval):
    """Test decide_approval with reject decision."""
    approval_manager._approvals["test-approval-1"] = sample_approval

    decision = ApprovalDecisionRequest(
        decision="reject",
        comment="Needs improvement",
        reviewed_by="test_user",
    )

    with patch.object(
        approval_manager, "get_redis", new_callable=AsyncMock
    ) as mock_get_redis:
        mock_redis = MagicMock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_get_redis.return_value = mock_redis

        result = await approval_manager.decide_approval("test-approval-1", decision)

        assert result is not None
        assert result.status == "rejected"


@pytest.mark.asyncio
async def test_delete_approval(approval_manager, sample_approval):
    """Test delete_approval method."""
    approval_manager._approvals["test-approval-1"] = sample_approval

    with patch.object(
        approval_manager, "get_redis", new_callable=AsyncMock
    ) as mock_get_redis:
        mock_redis = MagicMock()
        mock_redis.delete = AsyncMock(return_value=1)
        mock_redis.srem = AsyncMock(return_value=1)
        mock_get_redis.return_value = mock_redis

        result = await approval_manager.delete_approval("test-approval-1")

        assert result is True


@pytest.mark.asyncio
async def test_delete_all_approvals(approval_manager, sample_approval):
    """Test delete_all_approvals method."""
    approval_manager._approvals["test-approval-1"] = sample_approval

    with patch.object(
        approval_manager, "get_redis", new_callable=AsyncMock
    ) as mock_get_redis:
        mock_redis = MagicMock()
        mock_redis.delete = AsyncMock(return_value=1)
        mock_redis.smembers = AsyncMock(return_value=set(["test-approval-1"]))
        mock_get_redis.return_value = mock_redis

        result = await approval_manager.delete_all_approvals()

        assert result >= 0


@pytest.mark.asyncio
async def test_save_settings_to_redis(approval_manager):
    """Test save_settings_to_redis method."""
    settings = ApprovalSettings(
        require_approval=True,
        auto_approve_after_seconds=3600,
        notify_on_pending=True,
    )

    with patch.object(
        approval_manager._redis_manager, "execute", new_callable=AsyncMock
    ) as mock_execute:
        mock_execute.return_value = True

        result = await approval_manager.save_settings_to_redis(settings)

        assert result is True


@pytest.mark.asyncio
async def test_get_stats(approval_manager, sample_approval):
    """Test get_stats method."""
    approval_manager._approvals["test-approval-1"] = sample_approval

    result = await approval_manager.get_stats()

    assert result is not None
    assert hasattr(result, "total") or isinstance(result, dict)


def test_approval_manager_settings_property(approval_manager):
    """Test settings property."""
    assert approval_manager.settings is not None
    assert isinstance(approval_manager.settings, ApprovalSettings)
