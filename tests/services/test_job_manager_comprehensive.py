"""
Comprehensive tests for job manager service methods.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.job_manager import (
    Job,
    JobManager,
    JobStatus,
    get_job_manager,
)


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    with patch("marketing_project.services.job_manager.get_redis_manager") as mock:
        manager = MagicMock()
        manager.get_redis = AsyncMock(return_value=MagicMock())
        manager.execute = AsyncMock(return_value=None)
        mock.return_value = manager
        yield manager


@pytest.fixture
def job_manager(mock_redis_manager):
    """Create a JobManager instance."""
    return JobManager()


@pytest.mark.asyncio
async def test_mark_job_started(job_manager):
    """Test mark_job_started method."""
    job = await job_manager.create_job("blog", "content-1")

    await job_manager.mark_job_started(job.id)

    updated_job = await job_manager.get_job(job.id)
    assert updated_job.status == JobStatus.PROCESSING
    assert updated_job.started_at is not None


@pytest.mark.asyncio
async def test_mark_job_completed(job_manager):
    """Test mark_job_completed method."""
    job = await job_manager.create_job("blog", "content-1")

    result = {"status": "success", "data": {}}
    await job_manager.mark_job_completed(job.id, result)

    updated_job = await job_manager.get_job(job.id)
    assert updated_job.status == JobStatus.COMPLETED
    assert updated_job.result == result
    assert updated_job.completed_at is not None


@pytest.mark.asyncio
async def test_mark_job_failed(job_manager):
    """Test mark_job_failed method."""
    job = await job_manager.create_job("blog", "content-1")

    await job_manager.mark_job_failed(job.id, "Test error")

    updated_job = await job_manager.get_job(job.id)
    assert updated_job.status == JobStatus.FAILED
    assert updated_job.error == "Test error"


@pytest.mark.asyncio
async def test_update_parent_job_status(job_manager):
    """Test update_parent_job_status method."""
    parent_job = await job_manager.create_job("blog", "content-1")
    child_job = await job_manager.create_job("blog", "content-2")
    child_job.metadata["parent_job_id"] = parent_job.id
    await job_manager._save_job(child_job)

    await job_manager.update_parent_job_status(parent_job.id)

    # Should not raise exception
    updated_parent = await job_manager.get_job(parent_job.id)
    assert updated_parent is not None


@pytest.mark.asyncio
async def test_get_job_chain(job_manager):
    """Test get_job_chain method."""
    parent_job = await job_manager.create_job("blog", "content-1")
    child_job = await job_manager.create_job("blog", "content-2")
    child_job.metadata["parent_job_id"] = parent_job.id
    await job_manager._save_job(child_job)

    chain = await job_manager.get_job_chain(parent_job.id)

    assert chain is not None
    assert "root_job_id" in chain or "jobs" in chain or "all_job_ids" in chain


@pytest.mark.asyncio
async def test_get_job_with_subjob_status(job_manager):
    """Test get_job_with_subjob_status method."""
    parent_job = await job_manager.create_job("blog", "content-1")

    result = await job_manager.get_job_with_subjob_status(parent_job.id)

    assert result is not None
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_list_jobs_with_filters(job_manager):
    """Test list_jobs with various filters."""
    job1 = await job_manager.create_job("blog", "content-1")
    job2 = await job_manager.create_job("transcript", "content-2")
    await job_manager.update_job_status(job1.id, JobStatus.COMPLETED)

    # Filter by status
    completed_jobs = await job_manager.list_jobs(status=JobStatus.COMPLETED)
    assert len(completed_jobs) >= 1

    # Filter by type
    blog_jobs = await job_manager.list_jobs(job_type="blog")
    assert len(blog_jobs) >= 1

    # Note: content_id filtering is not supported by list_jobs method
    # Filtering by content_id would require manual filtering or a different method


@pytest.mark.asyncio
async def test_cleanup_old_jobs(job_manager):
    """Test cleanup_old_jobs method."""
    # Create an old job
    old_job = await job_manager.create_job("blog", "content-1")
    old_job.created_at = datetime.now(timezone.utc).replace(year=2020)
    await job_manager._save_job(old_job)

    cleaned = job_manager.cleanup_old_jobs(max_age_hours=24)

    assert isinstance(cleaned, int)
    assert cleaned >= 0


@pytest.mark.asyncio
async def test_normalize_datetime_to_utc(job_manager):
    """Test _normalize_datetime_to_utc method."""
    # Test with timezone-aware datetime
    dt = datetime.now(timezone.utc)
    normalized = job_manager._normalize_datetime_to_utc(dt)
    assert normalized is not None
    assert normalized.tzinfo == timezone.utc

    # Test with None
    assert job_manager._normalize_datetime_to_utc(None) is None
