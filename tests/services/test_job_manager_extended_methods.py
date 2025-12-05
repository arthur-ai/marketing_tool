"""
Extended tests for job manager - covering more methods.
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
async def test_submit_to_arq(job_manager):
    """Test submit_to_arq method."""
    job = await job_manager.create_job("blog", "content-1")

    with patch.object(job_manager, "get_arq_pool") as mock_pool:
        mock_arq_pool = MagicMock()
        mock_arq_job = MagicMock()
        mock_arq_job.job_id = "arq-job-1"
        mock_arq_pool.enqueue_job = AsyncMock(return_value=mock_arq_job)
        mock_pool.return_value = mock_arq_pool

        arq_job_id = await job_manager.submit_to_arq(
            "process_blog_job", content_json='{"id": "test"}', job_id=job.id
        )

        assert arq_job_id is not None


@pytest.mark.asyncio
async def test_update_job_chain_metadata(job_manager):
    """Test update_job_chain_metadata method."""
    parent_job = await job_manager.create_job("blog", "content-1")
    child_job = await job_manager.create_job("blog", "content-2")
    child_job.metadata["parent_job_id"] = parent_job.id
    await job_manager._save_job(child_job)

    await job_manager.update_job_chain_metadata(parent_job.id)

    # Should not raise exception
    updated = await job_manager.get_job(parent_job.id)
    assert updated is not None


@pytest.mark.asyncio
async def test_list_jobs_with_pagination(job_manager):
    """Test list_jobs with pagination."""
    # Create multiple jobs
    for i in range(5):
        await job_manager.create_job("blog", f"content-{i}")

    jobs = await job_manager.list_jobs(limit=3, offset=0)

    assert len(jobs) <= 3

    # Test offset
    jobs_page2 = await job_manager.list_jobs(limit=3, offset=3)

    assert len(jobs_page2) <= 3


@pytest.mark.asyncio
async def test_clear_all_arq_jobs(job_manager):
    """Test clear_all_arq_jobs method."""
    with patch.object(job_manager, "get_arq_pool") as mock_pool:
        mock_arq_pool = MagicMock()
        mock_arq_pool.delete_job = AsyncMock(return_value=True)
        mock_arq_pool.all_job_ids = AsyncMock(return_value=["job1", "job2"])
        mock_pool.return_value = mock_arq_pool

        cleared = await job_manager.clear_all_arq_jobs()

        assert isinstance(cleared, int)
        assert cleared >= 0


@pytest.mark.asyncio
async def test_delete_all_jobs(job_manager):
    """Test delete_all_jobs method."""
    # Create some jobs
    await job_manager.create_job("blog", "content-1")
    await job_manager.create_job("transcript", "content-2")

    deleted = await job_manager.delete_all_jobs()

    assert isinstance(deleted, int)
    assert deleted >= 0
