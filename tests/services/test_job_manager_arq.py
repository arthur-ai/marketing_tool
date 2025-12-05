"""
Tests for job manager ARQ integration methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.job_manager import JobManager


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
async def test_get_arq_pool(job_manager):
    """Test get_arq_pool method."""
    with patch("marketing_project.worker.get_arq_pool") as mock_get:
        mock_pool = MagicMock()
        mock_get.return_value = mock_pool

        pool = await job_manager.get_arq_pool()

        assert pool is not None


@pytest.mark.asyncio
async def test_submit_to_arq_with_retry(job_manager):
    """Test submit_to_arq with retry configuration."""
    job = await job_manager.create_job("blog", "content-1")

    with patch.object(job_manager, "get_arq_pool") as mock_pool:
        mock_arq_pool = MagicMock()
        mock_arq_job = MagicMock()
        mock_arq_job.job_id = "arq-job-1"
        mock_arq_pool.enqueue_job = AsyncMock(return_value=mock_arq_job)
        mock_pool.return_value = mock_arq_pool

        arq_job_id = await job_manager.submit_to_arq(
            "process_blog_job",
            content_json='{"id": "test"}',
            job_id=job.id,
            max_retries=5,
            retry_delay=60,
        )

        assert arq_job_id is not None


@pytest.mark.asyncio
async def test_submit_to_arq_job_not_found(job_manager):
    """Test submit_to_arq with non-existent job."""
    with pytest.raises(ValueError, match="not found"):
        await job_manager.submit_to_arq(
            "process_blog_job", content_json='{"id": "test"}', job_id="non-existent"
        )
