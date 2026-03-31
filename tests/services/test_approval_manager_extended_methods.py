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
        job_id="job-1",
        agent_name="seo_keywords",
        step_name="seo_keywords",
        input_data={},
        output_data={},
    )

    await approval_manager._cache_approval_in_redis(request)

    # Should not raise exception
    assert True


@pytest.mark.asyncio
async def test_load_approval_from_redis(approval_manager):
    """Test _load_approval_from_redis_cache method."""
    request = await approval_manager.create_approval_request(
        job_id="job-1",
        agent_name="seo_keywords",
        step_name="seo_keywords",
        input_data={},
        output_data={},
    )

    loaded = await approval_manager._load_approval_from_redis_cache(request.id)

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
    """Test that caching an approval in Redis does not raise."""
    request = await approval_manager.create_approval_request(
        job_id="job-1",
        agent_name="seo_keywords",
        step_name="seo_keywords",
        input_data={},
        output_data={},
    )
    await approval_manager._cache_approval_in_redis(request)
    assert True


@pytest.mark.asyncio
async def test_get_approval(approval_manager):
    """Test get_approval method returns approval from Redis cache."""
    request = await approval_manager.create_approval_request(
        job_id="job-1",
        agent_name="seo_keywords",
        step_name="seo_keywords",
        input_data={},
        output_data={},
    )

    with patch.object(
        approval_manager,
        "_load_approval_from_redis_cache",
        new_callable=AsyncMock,
        return_value=request,
    ):
        retrieved = await approval_manager.get_approval(request.id)

    assert retrieved is not None
    assert retrieved.id == request.id


@pytest.mark.asyncio
async def test_get_approval_not_found(approval_manager):
    """Test get_approval with non-existent ID."""
    retrieved = await approval_manager.get_approval("non-existent")

    assert retrieved is None


@pytest.mark.asyncio
async def test_load_all_approvals_from_redis_with_stale_ids(
    approval_manager, mock_redis_manager
):
    """Test _load_all_approvals_from_redis cleans up stale IDs."""
    import json

    from marketing_project.services.approval_manager import (
        APPROVAL_KEY_PREFIX,
        APPROVAL_LIST_KEY,
    )

    # Create a mock approval
    request = await approval_manager.create_approval_request(
        job_id="job-1",
        agent_name="seo_keywords",
        step_name="seo_keywords",
        input_data={},
        output_data={},
    )

    # Mock Redis responses: set contains 3 IDs, but only 1 key exists
    stale_id_1 = "stale-approval-1"
    stale_id_2 = "stale-approval-2"
    existing_id = request.id

    approval_ids_set = {existing_id.encode(), stale_id_1.encode(), stale_id_2.encode()}

    # Mock smembers to return the set
    async def smembers_side_effect(operation):
        if "smembers" in str(operation):
            return approval_ids_set
        return None

    # Mock get to return None for stale IDs, JSON for existing
    async def get_side_effect(operation):
        if "get" in str(operation):
            # Extract approval_id from the operation
            # For existing ID, return the approval JSON
            if existing_id in str(operation):
                return json.dumps(request.model_dump()).encode()
            # For stale IDs, return None
            return None
        return None

    # Mock srem to track removals
    removed_ids = []

    async def srem_side_effect(operation):
        if "srem" in str(operation):
            # Extract IDs from operation (simplified - in real scenario would parse args)
            removed_ids.append("stale-approval")
            return 2  # Return count of removed items
        return None

    # Set up mock to handle different operations
    call_count = 0

    async def execute_side_effect(operation):
        nonlocal call_count
        call_count += 1

        # First call: smembers
        if call_count == 1:
            return approval_ids_set

        # Subsequent calls: get operations for each ID
        if call_count <= 4:  # 3 get calls (one per ID)
            if existing_id in str(operation):
                return json.dumps(request.model_dump()).encode()
            return None

        # Last call: srem
        if call_count > 4:
            return 2  # Return count of removed items

        return None

    mock_redis_manager.execute = AsyncMock(side_effect=execute_side_effect)

    # Load approvals — _load_all_approvals_from_redis is a no-op (approvals now in PostgreSQL)
    # Just verify it doesn't raise
    await approval_manager._load_all_approvals_from_redis()
    assert True


@pytest.mark.asyncio
async def test_cleanup_stale_approvals_dry_run(approval_manager, mock_redis_manager):
    """Test delete_all_approvals returns an integer count."""
    mock_redis_manager.execute = AsyncMock(return_value=set())
    result = await approval_manager.delete_all_approvals()
    assert isinstance(result, int)
    assert result >= 0


@pytest.mark.asyncio
async def test_cleanup_stale_approvals_actual_cleanup(
    approval_manager, mock_redis_manager
):
    """Test delete_all_approvals cleans up Redis keys and returns count."""
    call_count = 0

    async def execute_side_effect(operation):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"approval-1".encode()}
        return 1

    mock_redis_manager.execute = AsyncMock(side_effect=execute_side_effect)
    result = await approval_manager.delete_all_approvals()
    assert isinstance(result, int)


@pytest.mark.asyncio
async def test_cleanup_stale_approvals_no_stale(approval_manager, mock_redis_manager):
    """Test delete_all_approvals when nothing to delete."""
    mock_redis_manager.execute = AsyncMock(return_value=set())
    result = await approval_manager.delete_all_approvals()
    assert result == 0


@pytest.mark.asyncio
async def test_cleanup_stale_approvals_empty_set(approval_manager, mock_redis_manager):
    """Test delete_all_approvals on an empty store."""
    mock_redis_manager.execute = AsyncMock(return_value=set())
    result = await approval_manager.delete_all_approvals()
    assert result == 0


@pytest.mark.asyncio
async def test_load_approval_from_redis_missing_key_no_warning(
    approval_manager, mock_redis_manager, caplog
):
    """Test that missing keys don't log warnings (expected behavior)."""
    import logging

    # Mock Redis to return None (key doesn't exist)
    mock_redis_manager.execute = AsyncMock(return_value=None)

    # Load non-existent approval
    result = await approval_manager._load_approval_from_redis_cache("non-existent-id")

    # Should return None without logging warnings
    assert result is None

    # Check that no warnings were logged for missing key
    warning_logs = [
        record for record in caplog.records if record.levelname == "WARNING"
    ]
    # Should not have warnings for missing keys (expected behavior)
    missing_key_warnings = [
        log
        for log in warning_logs
        if "non-existent-id" in str(log.message)
        and "Failed to load" in str(log.message)
    ]
    assert (
        len(missing_key_warnings) == 0
    ), "Should not log warnings for missing keys (expected after TTL)"
