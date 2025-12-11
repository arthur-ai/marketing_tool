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
        redis_store = {}  # In-memory store for testing

        async def execute_operation(operation):
            mock_redis = MagicMock()

            async def setex(key, ttl, value):
                # Store as string, Redis will return bytes
                redis_store[key] = value

            async def get(key):
                result = redis_store.get(key)
                if result is None:
                    return None
                # Redis returns bytes, json.loads can handle bytes in Python 3.6+
                if isinstance(result, str):
                    return result.encode("utf-8")
                return result

            async def scan(cursor=0, match=None, count=None):
                # Mock scan for delete_all_approvals
                if match:
                    pattern = match.replace("*", "")
                    keys = [
                        k.encode("utf-8") if isinstance(k, str) else k
                        for k in redis_store.keys()
                        if pattern in k
                    ]
                else:
                    keys = [
                        k.encode("utf-8") if isinstance(k, str) else k
                        for k in redis_store.keys()
                    ]
                return (0, keys)  # Return (next_cursor, keys)

            async def delete(*keys):
                count = 0
                for key in keys:
                    # Handle both str and bytes keys
                    key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                    if key_str in redis_store:
                        del redis_store[key_str]
                        count += 1
                return count

            async def smembers(key):
                # Mock smembers for delete_all_approvals
                return set()

            mock_redis.setex = setex
            mock_redis.get = get
            mock_redis.scan = scan
            mock_redis.delete = delete
            mock_redis.smembers = smembers
            return await operation(mock_redis)

        manager.get_redis = AsyncMock(return_value=MagicMock())
        manager.execute = execute_operation
        mock.return_value = manager
        yield manager


@pytest.fixture
def approval_manager(mock_redis_manager):
    """Create an ApprovalManager instance."""
    # Ensure the manager uses the mocked redis_manager
    manager = ApprovalManager()
    # The mock_redis_manager fixture should already be patched
    return manager


@pytest.mark.asyncio
async def test_create_approval_request(approval_manager):
    """Test creating an approval request."""
    request = await approval_manager.create_approval_request(
        job_id="test-job-1",
        agent_name="test-agent",
        step_name="seo_keywords",
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
        "test-job-1", "test-agent", "seo_keywords", {}, {}
    )

    retrieved = await approval_manager.get_approval(request.id)

    assert retrieved is not None
    assert retrieved.id == request.id


@pytest.mark.asyncio
async def test_decide_approval(approval_manager):
    """Test deciding on an approval."""
    request = await approval_manager.create_approval_request(
        "test-job-1", "test-agent", "seo_keywords", {}, {}
    )

    decision = ApprovalDecisionRequest(decision="approve", comment="Looks good")

    result = await approval_manager.decide_approval(request.id, decision)

    assert result is not None
    assert result.status == "approved"


@pytest.mark.asyncio
async def test_get_approvals_for_job(approval_manager):
    """Test getting approvals for a job."""
    await approval_manager.create_approval_request(
        "test-job-1", "test-agent", "seo_keywords", {}, {}
    )
    await approval_manager.create_approval_request(
        "test-job-1", "test-agent", "marketing_brief", {}, {}
    )

    approvals = await approval_manager.list_approvals(job_id="test-job-1")

    assert len(approvals) >= 2


@pytest.mark.asyncio
async def test_save_pipeline_context(approval_manager):
    """Test saving pipeline context."""
    context = {"seo_keywords": {}, "input_content": {}}

    await approval_manager.save_pipeline_context(
        "test-job-1", context, "seo_keywords", 1, {}
    )

    # Should not raise exception
    assert True


@pytest.mark.asyncio
async def test_get_pipeline_context(approval_manager):
    """Test getting pipeline context."""
    context = {"seo_keywords": {}, "input_content": {}}
    await approval_manager.save_pipeline_context(
        "test-job-1", context, "seo_keywords", 1, {}
    )

    retrieved = await approval_manager.load_pipeline_context("test-job-1")

    assert retrieved is not None
    assert "context" in retrieved
    assert "seo_keywords" in retrieved.get("context", {})


@pytest.mark.asyncio
async def test_get_approval_stats(approval_manager):
    """Test getting approval statistics."""
    request1 = await approval_manager.create_approval_request(
        "job-1", "test-agent", "seo_keywords", {}, {}
    )
    request2 = await approval_manager.create_approval_request(
        "job-2", "test-agent", "marketing_brief", {}, {}
    )

    await approval_manager.decide_approval(
        request1.id, ApprovalDecisionRequest(decision="approve")
    )

    stats = await approval_manager.get_stats()

    assert stats is not None
    assert hasattr(stats, "total_requests") or hasattr(stats, "approved")


@pytest.mark.asyncio
async def test_get_approval_manager_singleton():
    """Test that get_approval_manager returns a singleton."""
    manager1 = await get_approval_manager()
    manager2 = await get_approval_manager()

    assert manager1 is manager2
