"""
Tests for approval manager edge cases and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.approval_models import ApprovalDecisionRequest
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
async def test_create_approval_request_auto_approve(approval_manager):
    """Test create_approval_request with auto-approval threshold."""
    approval_manager.settings.auto_approve_threshold = 0.9

    request = await approval_manager.create_approval_request(
        "job-1",
        "seo_keywords",
        "Step 1: SEO Keywords",
        {},
        {},
        confidence_score=0.95,  # Above threshold
    )

    assert request.status == "approved"
    assert request.reviewed_at is not None


@pytest.mark.asyncio
async def test_create_approval_request_below_threshold(approval_manager):
    """Test create_approval_request below auto-approval threshold."""
    approval_manager.settings.auto_approve_threshold = 0.9

    request = await approval_manager.create_approval_request(
        "job-1",
        "seo_keywords",
        "Step 1: SEO Keywords",
        {},
        {},
        confidence_score=0.5,  # Below threshold
    )

    assert request.status == "pending"


@pytest.mark.asyncio
async def test_decide_approval_already_decided(approval_manager):
    """Test decide_approval on already decided request."""
    request = await approval_manager.create_approval_request(
        "job-1", "seo_keywords", "Step 1", {}, {}
    )
    await approval_manager.decide_approval(
        request.id, ApprovalDecisionRequest(decision="approved")
    )

    # Try to decide again
    with pytest.raises(ValueError, match="already decided"):
        await approval_manager.decide_approval(
            request.id, ApprovalDecisionRequest(decision="rejected")
        )


@pytest.mark.asyncio
async def test_decide_approval_modify_without_output(approval_manager):
    """Test decide_approval with modify decision but no modified_output."""
    request = await approval_manager.create_approval_request(
        "job-1", "marketing_brief", "Step 2", {}, {}
    )

    with pytest.raises(ValueError, match="Modified output required"):
        await approval_manager.decide_approval(
            request.id,
            ApprovalDecisionRequest(
                decision="modify",
                comment="Needs changes",
            ),
        )


@pytest.mark.asyncio
async def test_decide_approval_modify_with_output(approval_manager):
    """Test decide_approval with modify decision and modified_output."""
    request = await approval_manager.create_approval_request(
        "job-1", "marketing_brief", "Step 2", {}, {}
    )

    modified = await approval_manager.decide_approval(
        request.id,
        ApprovalDecisionRequest(
            decision="modify",
            comment="Updated",
            modified_output={"target_audience": "updated"},
        ),
    )

    assert modified.status == "modified"
    assert modified.modified_output == {"target_audience": "updated"}


@pytest.mark.asyncio
async def test_decide_approval_rerun(approval_manager):
    """Test decide_approval with rerun decision."""
    request = await approval_manager.create_approval_request(
        "job-1", "seo_keywords", "Step 1", {}, {}
    )

    rerun = await approval_manager.decide_approval(
        request.id,
        ApprovalDecisionRequest(
            decision="rerun",
            comment="Please retry with different approach",
        ),
    )

    assert rerun.status == "rerun"


@pytest.mark.asyncio
async def test_wait_for_approval_timeout(approval_manager):
    """Test wait_for_approval with timeout."""
    request = await approval_manager.create_approval_request(
        "job-1", "seo_keywords", "Step 1", {}, {}
    )

    # Wait with short timeout
    result = await approval_manager.wait_for_approval(request.id, timeout=0.1)

    # Should auto-reject on timeout
    assert result.status == "rejected"
    assert (
        "timeout" in result.user_comment.lower()
        or "auto-rejected" in result.user_comment.lower()
    )


@pytest.mark.asyncio
async def test_wait_for_approval_already_decided(approval_manager):
    """Test wait_for_approval on already decided request."""
    request = await approval_manager.create_approval_request(
        "job-1", "seo_keywords", "Step 1", {}, {}
    )
    await approval_manager.decide_approval(
        request.id, ApprovalDecisionRequest(decision="approved")
    )

    # Should return immediately
    result = await approval_manager.wait_for_approval(request.id)

    assert result.status == "approved"


@pytest.mark.asyncio
async def test_list_approvals_with_filters(approval_manager):
    """Test list_approvals with various filters."""
    await approval_manager.create_approval_request(
        "job-1", "seo_keywords", "Step 1", {}, {}
    )
    await approval_manager.create_approval_request(
        "job-2", "marketing_brief", "Step 2", {}, {}
    )

    # Filter by job_id
    approvals = await approval_manager.list_approvals(job_id="job-1")

    assert len(approvals) >= 1
    assert all(a.job_id == "job-1" for a in approvals)

    # Filter by status
    pending = await approval_manager.list_approvals(status="pending")

    assert len(pending) >= 2
    assert all(a.status == "pending" for a in pending)


@pytest.mark.asyncio
async def test_get_approvals_for_job(approval_manager):
    """Test get_approvals_for_job method."""
    await approval_manager.create_approval_request(
        "job-1", "seo_keywords", "Step 1", {}, {}
    )
    await approval_manager.create_approval_request(
        "job-1", "marketing_brief", "Step 2", {}, {}
    )

    approvals = await approval_manager.get_approvals_for_job("job-1")

    assert len(approvals) >= 2
    assert all(a.job_id == "job-1" for a in approvals)


@pytest.mark.asyncio
async def test_get_approvals_for_job_not_found(approval_manager):
    """Test get_approvals_for_job with non-existent job."""
    approvals = await approval_manager.get_approvals_for_job("non-existent-job")

    assert len(approvals) == 0


@pytest.mark.asyncio
async def test_get_stats_empty(approval_manager):
    """Test get_stats with no approvals."""
    stats = await approval_manager.get_stats()

    assert stats.total_requests == 0
    assert stats.pending == 0
    assert stats.approved == 0


@pytest.mark.asyncio
async def test_get_stats_with_approvals(approval_manager):
    """Test get_stats with various approval statuses."""
    req1 = await approval_manager.create_approval_request(
        "job-1", "seo_keywords", "Step 1", {}, {}
    )
    req2 = await approval_manager.create_approval_request(
        "job-2", "marketing_brief", "Step 2", {}, {}
    )

    await approval_manager.decide_approval(
        req1.id, ApprovalDecisionRequest(decision="approved")
    )
    await approval_manager.decide_approval(
        req2.id, ApprovalDecisionRequest(decision="rejected")
    )

    stats = await approval_manager.get_stats()

    assert stats.total_requests == 2
    assert stats.approved == 1
    assert stats.rejected == 1
    assert stats.pending == 0


@pytest.mark.asyncio
async def test_delete_all_approvals_empty(approval_manager):
    """Test delete_all_approvals with no approvals."""
    deleted = await approval_manager.delete_all_approvals()

    assert deleted == 0


@pytest.mark.asyncio
async def test_clear_job_approvals(approval_manager):
    """Test clear_job_approvals method."""
    await approval_manager.create_approval_request(
        "job-1", "seo_keywords", "Step 1", {}, {}
    )

    approval_manager.clear_job_approvals("job-1")

    # Should not raise exception
    assert True


@pytest.mark.asyncio
async def test_save_pipeline_context(approval_manager):
    """Test save_pipeline_context method."""
    context = {
        "seo_keywords": {"main_keyword": "test"},
        "marketing_brief": {"target_audience": "developers"},
    }

    await approval_manager.save_pipeline_context("job-1", context)

    # Should not raise exception
    assert True


@pytest.mark.asyncio
async def test_load_pipeline_context(approval_manager):
    """Test load_pipeline_context method."""
    context = {
        "seo_keywords": {"main_keyword": "test"},
    }
    await approval_manager.save_pipeline_context("job-1", context)

    loaded = await approval_manager.load_pipeline_context("job-1")

    assert loaded is not None
    assert isinstance(loaded, dict)


@pytest.mark.asyncio
async def test_load_pipeline_context_not_found(approval_manager):
    """Test load_pipeline_context with non-existent job."""
    loaded = await approval_manager.load_pipeline_context("non-existent-job")

    assert loaded is None
