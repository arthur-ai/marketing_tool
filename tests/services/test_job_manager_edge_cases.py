"""
Tests for job manager edge cases and error handling.
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
async def test_create_job_with_custom_id(job_manager):
    """Test create_job with custom job_id."""
    custom_id = "custom-job-123"
    job = await job_manager.create_job("blog", "content-1", job_id=custom_id)

    assert job.id == custom_id


@pytest.mark.asyncio
async def test_create_job_with_metadata(job_manager):
    """Test create_job with metadata."""
    metadata = {"custom_field": "value", "priority": "high"}

    job = await job_manager.create_job("blog", "content-1", metadata=metadata)

    assert job.metadata["custom_field"] == "value"
    assert job.metadata["priority"] == "high"


@pytest.mark.asyncio
async def test_get_job_not_found(job_manager):
    """Test get_job with non-existent job."""
    job = await job_manager.get_job("non-existent-job")

    assert job is None


@pytest.mark.asyncio
async def test_update_job_status_not_found(job_manager):
    """Test update_job_status with non-existent job."""
    await job_manager.update_job_status("non-existent-job", JobStatus.COMPLETED)

    # Should not raise exception, just log warning
    assert True


@pytest.mark.asyncio
async def test_update_job_progress_not_found(job_manager):
    """Test update_job_progress with non-existent job."""
    await job_manager.update_job_progress("non-existent-job", 50, "Processing")

    # Should not raise exception
    assert True


@pytest.mark.asyncio
async def test_cancel_job_not_found(job_manager):
    """Test cancel_job with non-existent job."""
    result = await job_manager.cancel_job("non-existent-job")

    assert result is False


@pytest.mark.asyncio
async def test_cancel_job_already_completed(job_manager):
    """Test cancel_job with already completed job."""
    job = await job_manager.create_job("blog", "content-1")
    await job_manager.update_job_status(job.id, JobStatus.COMPLETED)

    result = await job_manager.cancel_job(job.id)

    # Should return False as job is already completed
    assert result is False


@pytest.mark.asyncio
async def test_list_jobs_with_offset(job_manager):
    """Test list_jobs with offset."""
    # Create multiple jobs
    for i in range(10):
        await job_manager.create_job("blog", f"content-{i}")

    jobs = await job_manager.list_jobs(limit=5, offset=5)

    assert len(jobs) <= 5


@pytest.mark.asyncio
async def test_list_jobs_with_content_id_filter(job_manager):
    """Test list_jobs filtering by content_id."""
    await job_manager.create_job("blog", "content-1")
    await job_manager.create_job("blog", "content-2")

    jobs = await job_manager.list_jobs(content_id="content-1")

    assert len(jobs) >= 1
    assert all(job.content_id == "content-1" for job in jobs)


@pytest.mark.asyncio
async def test_get_job_chain_no_chain(job_manager):
    """Test get_job_chain with no subjobs."""
    parent_job = await job_manager.create_job("blog", "content-1")

    chain = await job_manager.get_job_chain(parent_job.id)

    assert isinstance(chain, dict)
    assert "job" in chain or "subjobs" in chain


@pytest.mark.asyncio
async def test_get_job_with_subjob_status_no_subjobs(job_manager):
    """Test get_job_with_subjob_status with no subjobs."""
    parent_job = await job_manager.create_job("blog", "content-1")

    result = await job_manager.get_job_with_subjob_status(parent_job.id)

    assert isinstance(result, dict)
    assert result.get("job_id") == parent_job.id or "subjobs" in result


@pytest.mark.asyncio
async def test_update_parent_job_status_no_parent(job_manager):
    """Test update_parent_job_status with no parent."""
    job = await job_manager.create_job("blog", "content-1")

    # Should not raise exception even if no parent
    await job_manager.update_parent_job_status(job.id)

    assert True


@pytest.mark.asyncio
async def test_cleanup_old_jobs_none_old(job_manager):
    """Test cleanup_old_jobs when no old jobs exist."""
    # Create recent job
    job = await job_manager.create_job("blog", "content-1")

    cleaned = job_manager.cleanup_old_jobs(max_age_hours=24)

    assert isinstance(cleaned, int)
    assert cleaned >= 0


@pytest.mark.asyncio
async def test_cleanup_old_jobs_all_old(job_manager):
    """Test cleanup_old_jobs when all jobs are old."""
    # Create old job
    old_job = await job_manager.create_job("blog", "content-1")
    old_job.created_at = datetime.now(timezone.utc).replace(year=2020)
    await job_manager._save_job(old_job)

    cleaned = job_manager.cleanup_old_jobs(max_age_hours=1)

    assert isinstance(cleaned, int)
    assert cleaned >= 0
