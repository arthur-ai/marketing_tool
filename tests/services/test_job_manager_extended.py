"""
Extended tests for job manager service.
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
async def test_create_job(job_manager):
    """Test creating a job."""
    job = await job_manager.create_job(
        job_type="blog",
        content_id="test-content-1",
        metadata={"test": "data"},
    )

    assert job is not None
    assert job.type == "blog"
    assert job.content_id == "test-content-1"
    assert job.status == JobStatus.PENDING


@pytest.mark.asyncio
async def test_get_job(job_manager):
    """Test getting a job."""
    job = await job_manager.create_job("blog", "test-content-1")

    retrieved = await job_manager.get_job(job.id)

    assert retrieved is not None
    assert retrieved.id == job.id


@pytest.mark.asyncio
async def test_get_job_not_found(job_manager):
    """Test getting a non-existent job."""
    job = await job_manager.get_job("non-existent")

    assert job is None


@pytest.mark.asyncio
async def test_update_job_status(job_manager):
    """Test updating job status."""
    job = await job_manager.create_job("blog", "test-content-1")

    updated = await job_manager.update_job_status(job.id, JobStatus.PROCESSING)

    assert updated is True
    retrieved = await job_manager.get_job(job.id)
    assert retrieved.status == JobStatus.PROCESSING


@pytest.mark.asyncio
async def test_update_job_progress(job_manager):
    """Test updating job progress."""
    job = await job_manager.create_job("blog", "test-content-1")

    updated = await job_manager.update_job_progress(job.id, 50, "Processing step 2")

    assert updated is True
    retrieved = await job_manager.get_job(job.id)
    assert retrieved.progress == 50
    assert retrieved.current_step == "Processing step 2"


@pytest.mark.asyncio
async def test_list_jobs(job_manager):
    """Test listing jobs."""
    await job_manager.create_job("blog", "content-1")
    await job_manager.create_job("transcript", "content-2")

    jobs = await job_manager.list_jobs(limit=10)

    assert len(jobs) >= 2


@pytest.mark.asyncio
async def test_list_jobs_with_status_filter(job_manager):
    """Test listing jobs with status filter."""
    job = await job_manager.create_job("blog", "content-1")
    await job_manager.update_job_status(job.id, JobStatus.COMPLETED)

    jobs = await job_manager.list_jobs(status=JobStatus.COMPLETED)

    assert all(j.status == JobStatus.COMPLETED for j in jobs)


@pytest.mark.asyncio
async def test_cancel_job(job_manager):
    """Test canceling a job."""
    job = await job_manager.create_job("blog", "content-1")

    success = await job_manager.cancel_job(job.id)

    assert success is True
    retrieved = await job_manager.get_job(job.id)
    assert retrieved.status == JobStatus.CANCELLED


def test_get_job_manager_singleton():
    """Test that get_job_manager returns a singleton."""
    manager1 = get_job_manager()
    manager2 = get_job_manager()

    assert manager1 is manager2
