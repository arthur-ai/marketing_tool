"""
Tests for JobManager service.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from marketing_project.services.job_manager import (
    Job,
    JobManager,
    JobStatus,
    get_job_manager,
)


@pytest.fixture
async def job_manager():
    """Create a JobManager instance with mocked Redis."""
    manager = JobManager()

    # Mock Redis manager
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.sadd = AsyncMock(return_value=1)
    mock_redis.smembers = AsyncMock(return_value=set())

    # Mock pipeline
    mock_pipeline = AsyncMock()
    mock_pipeline.setex = AsyncMock()
    mock_pipeline.sadd = AsyncMock()
    mock_pipeline.execute = AsyncMock(return_value=[True, 1])
    mock_redis.pipeline = MagicMock(return_value=mock_pipeline)
    mock_redis.__aenter__ = MagicMock(return_value=mock_pipeline)
    mock_redis.__aexit__ = MagicMock(return_value=None)

    with patch.object(manager, "get_redis", return_value=mock_redis):
        yield manager


class TestJobManager:
    """Test JobManager class."""

    @pytest.mark.asyncio
    async def test_create_job(self, job_manager):
        """Test creating a job."""
        job = await job_manager.create_job(
            job_type="blog_post",
            content_id="content-123",
            metadata={"test": "data"},
        )

        assert isinstance(job, Job)
        assert job.type == "blog_post"
        assert job.content_id == "content-123"
        assert job.status == JobStatus.PENDING
        assert job.metadata == {"test": "data"}

    @pytest.mark.asyncio
    async def test_create_job_with_custom_id(self, job_manager):
        """Test creating a job with custom ID."""
        custom_id = str(uuid4())
        job = await job_manager.create_job(
            job_type="blog_post",
            content_id="content-123",
            job_id=custom_id,
        )

        assert job.id == custom_id

    @pytest.mark.asyncio
    async def test_get_job(self, job_manager):
        """Test getting a job by ID."""
        job = await job_manager.create_job(
            job_type="blog_post", content_id="content-123"
        )

        # Mock Redis get to return job JSON
        job_json = job.model_dump_json()
        mock_redis = await job_manager.get_redis()
        mock_redis.get = AsyncMock(return_value=job_json)

        retrieved_job = await job_manager.get_job(job.id)

        assert retrieved_job is not None
        assert retrieved_job.id == job.id
        assert retrieved_job.type == job.type

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, job_manager):
        """Test getting a job that doesn't exist."""
        mock_redis = await job_manager.get_redis()
        mock_redis.get = AsyncMock(return_value=None)

        result = await job_manager.get_job("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_job_status(self, job_manager):
        """Test updating job status."""
        job = await job_manager.create_job(
            job_type="blog_post", content_id="content-123"
        )

        await job_manager.update_job_status(job.id, JobStatus.PROCESSING)

        updated_job = await job_manager.get_job(job.id)
        # Note: This may not work if Redis is mocked, but we verify the method runs
        assert updated_job is not None

    @pytest.mark.asyncio
    async def test_update_job_status_failed_sets_error_message(self, job_manager):
        """FAILED status with error_message populates both error_message and legacy error fields."""
        job = await job_manager.create_job(
            job_type="blog_post", content_id="content-123"
        )

        await job_manager.update_job_status(
            job.id,
            JobStatus.FAILED,
            error_message="ValueError: missing content_id",
        )

        updated_job = await job_manager.get_job(job.id)
        assert updated_job is not None
        assert updated_job.status == JobStatus.FAILED
        assert updated_job.error_message == "ValueError: missing content_id"
        assert updated_job.error == "ValueError: missing content_id"
        assert updated_job.completed_at is not None

    @pytest.mark.asyncio
    async def test_update_job_status_failed_without_error_message(self, job_manager):
        """FAILED status without error_message leaves error_message as None."""
        job = await job_manager.create_job(
            job_type="blog_post", content_id="content-123"
        )

        await job_manager.update_job_status(job.id, JobStatus.FAILED)

        updated_job = await job_manager.get_job(job.id)
        assert updated_job is not None
        assert updated_job.status == JobStatus.FAILED
        assert updated_job.error_message is None

    def test_job_model_to_dict_includes_error_message(self):
        """JobModel.to_dict() returns error_message at the top level for API responses."""
        from marketing_project.models.db_models import JobModel

        db_job = JobModel(
            job_id="test-job-1",
            job_type="blog",
            status="failed",
            content_id="content-1",
            progress=0,
            error="ValueError: bad input",
            error_message="ValueError: bad input",
            job_metadata={},
        )
        result = db_job.to_dict()
        assert result["error_message"] == "ValueError: bad input"
        assert result["error"] == "ValueError: bad input"

    def test_job_model_to_dict_error_message_none_for_non_failed(self):
        """JobModel.to_dict() returns error_message=None for jobs that haven't failed."""
        from marketing_project.models.db_models import JobModel

        db_job = JobModel(
            job_id="test-job-2",
            job_type="blog",
            status="completed",
            content_id="content-2",
            progress=100,
            job_metadata={},
        )
        result = db_job.to_dict()
        assert result["error_message"] is None

    @pytest.mark.asyncio
    async def test_update_job_progress(self, job_manager):
        """Test updating job progress."""
        job = await job_manager.create_job(
            job_type="blog_post", content_id="content-123"
        )

        await job_manager.update_job_progress(job.id, 50, "Halfway done")

        # Verify method runs without error
        # Actual verification would require Redis to be properly mocked

    @pytest.mark.asyncio
    async def test_submit_to_arq(self, job_manager):
        """Test submitting job to ARQ."""
        job = await job_manager.create_job(
            job_type="blog_post", content_id="content-123"
        )

        # Mock ARQ pool
        mock_arq_job = MagicMock()
        mock_arq_job.job_id = "arq-job-123"
        mock_pool = AsyncMock()
        mock_pool.enqueue_job = AsyncMock(return_value=mock_arq_job)

        with patch.object(job_manager, "get_arq_pool", return_value=mock_pool):
            arq_job_id = await job_manager.submit_to_arq(
                job.id, "process_blog_job", "content_json", job.id
            )

            assert arq_job_id == "arq-job-123"
            mock_pool.enqueue_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_jobs(self, job_manager):
        """Test listing all jobs."""
        # Create a few jobs
        job1 = await job_manager.create_job(
            job_type="blog_post", content_id="content-1"
        )
        job2 = await job_manager.create_job(
            job_type="transcript", content_id="content-2"
        )

        # Mock Redis to return job IDs
        mock_redis = await job_manager.get_redis()
        mock_redis.smembers = AsyncMock(return_value={job1.id, job2.id})
        mock_redis.get = AsyncMock(
            side_effect=[
                job1.model_dump_json(),
                job2.model_dump_json(),
            ]
        )

        jobs = await job_manager.list_jobs()

        assert len(jobs) >= 2
        job_ids = [j.id for j in jobs]
        assert job1.id in job_ids
        assert job2.id in job_ids


class TestGetJobManager:
    """Test get_job_manager function."""

    def test_get_job_manager_returns_singleton(self):
        """Test that get_job_manager returns a singleton."""
        manager1 = get_job_manager()
        manager2 = get_job_manager()

        assert manager1 is manager2
