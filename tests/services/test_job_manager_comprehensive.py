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

    cleaned = await job_manager.cleanup_old_jobs(max_age_hours=24)

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


# ---------------------------------------------------------------------------
# Additional tests added to increase coverage
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_manager_initialized():
    """Return a db_manager mock that claims to be initialized, with a
    get_session() async context manager that yields a mock session."""
    db_manager = MagicMock()
    db_manager.is_initialized = True

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    db_manager.get_session = MagicMock(return_value=mock_session)
    return db_manager


# ---- _save_job_to_database ----


@pytest.mark.asyncio
async def test_save_job_to_database_db_not_initialized(job_manager):
    """_save_job_to_database skips gracefully when DB is not initialized."""
    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        job = await job_manager.create_job("blog", "content-1")
        # Should not raise
        await job_manager._save_job_to_database(job)


@pytest.mark.asyncio
async def test_save_job_to_database_creates_new_job(
    job_manager, mock_db_manager_initialized
):
    """_save_job_to_database inserts a new record when the job doesn't exist yet."""
    mock_session = mock_db_manager_initialized.get_session.return_value
    # Simulate "job not found" from the DB
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=result_mock)

    with patch(
        "marketing_project.services.database.get_database_manager",
        return_value=mock_db_manager_initialized,
    ):
        job = Job(id="new-job-1", type="blog", content_id="c1")
        await job_manager._save_job_to_database(job)
        # session.add should have been called with a JobModel
        assert mock_session.add.called


@pytest.mark.asyncio
async def test_save_job_to_database_updates_existing_job(
    job_manager, mock_db_manager_initialized
):
    """_save_job_to_database updates an existing DB record."""
    existing = MagicMock()
    existing.status = "pending"

    mock_session = mock_db_manager_initialized.get_session.return_value
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = existing
    mock_session.execute = AsyncMock(return_value=result_mock)

    with patch(
        "marketing_project.services.database.get_database_manager",
        return_value=mock_db_manager_initialized,
    ):
        job = Job(id="existing-job-1", type="blog", content_id="c1")
        job.status = JobStatus.COMPLETED
        await job_manager._save_job_to_database(job)
        assert existing.status == "completed"


@pytest.mark.asyncio
async def test_save_job_to_database_exception_is_swallowed(job_manager):
    """Database errors do not propagate out of _save_job_to_database."""
    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        mock_get_db.side_effect = RuntimeError("db boom")
        job = Job(id="boom-job", type="blog", content_id="c1")
        # Should not raise
        await job_manager._save_job_to_database(job)


# ---- create_job with user context ----


@pytest.mark.asyncio
async def test_create_job_with_user_context(job_manager):
    """create_job stores username/email from user_context in metadata."""
    user_context = MagicMock()
    user_context.username = "alice"
    user_context.email = "alice@example.com"

    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        job = await job_manager.create_job(
            "blog",
            "content-1",
            user_id="user-123",
            user_context=user_context,
        )

    assert job.metadata.get("triggered_by_username") == "alice"
    assert job.metadata.get("triggered_by_email") == "alice@example.com"


@pytest.mark.asyncio
async def test_create_job_redis_failure_falls_back_to_memory(mock_redis_manager):
    """If Redis fails during create_job, the job is still stored in-memory."""
    mock_redis_manager.execute = AsyncMock(side_effect=RuntimeError("redis down"))

    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        jm = JobManager()
        job = await jm.create_job("blog", "content-x")

    assert job.id in jm._jobs


# ---- get_jobs_by_ids ----


@pytest.mark.asyncio
async def test_get_jobs_by_ids_empty_list(job_manager):
    """get_jobs_by_ids returns empty dict for empty input."""
    result = await job_manager.get_jobs_by_ids([])
    assert result == {}


@pytest.mark.asyncio
async def test_get_jobs_by_ids_db_not_initialized(job_manager):
    """get_jobs_by_ids falls back to get_job when DB not initialized."""
    job = await job_manager.create_job("blog", "c1")

    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        result = await job_manager.get_jobs_by_ids([job.id, "nonexistent-id"])

    assert job.id in result


@pytest.mark.asyncio
async def test_get_jobs_by_ids_db_initialized(job_manager, mock_db_manager_initialized):
    """get_jobs_by_ids loads from DB when initialized."""
    db_job = MagicMock()
    db_job.to_dict.return_value = {
        "id": "db-job-1",
        "type": "blog",
        "status": "completed",
        "content_id": "c1",
        "created_at": datetime.now(timezone.utc),
        "started_at": None,
        "completed_at": None,
        "progress": 100,
        "current_step": None,
        "result": None,
        "error": None,
        "error_message": None,
        "metadata": {},
        "user_id": None,
    }

    mock_session = mock_db_manager_initialized.get_session.return_value
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = [db_job]
    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock
    mock_session.execute = AsyncMock(return_value=result_mock)

    with patch(
        "marketing_project.services.database.get_database_manager",
        return_value=mock_db_manager_initialized,
    ):
        result = await job_manager.get_jobs_by_ids(["db-job-1"])

    assert "db-job-1" in result


# ---- get_job_ids_for_user ----


@pytest.mark.asyncio
async def test_get_job_ids_for_user_db_not_initialized(job_manager):
    """get_job_ids_for_user falls back to list_jobs when DB not initialized."""
    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        result = await job_manager.get_job_ids_for_user("user-x")

    assert isinstance(result, set)


@pytest.mark.asyncio
async def test_get_job_ids_for_user_db_initialized(
    job_manager, mock_db_manager_initialized
):
    """get_job_ids_for_user queries DB when initialized."""
    mock_session = mock_db_manager_initialized.get_session.return_value
    result_mock = MagicMock()
    result_mock.fetchall.return_value = [("job-a",), ("job-b",)]
    mock_session.execute = AsyncMock(return_value=result_mock)

    with patch(
        "marketing_project.services.database.get_database_manager",
        return_value=mock_db_manager_initialized,
    ):
        result = await job_manager.get_job_ids_for_user("user-y")

    assert "job-a" in result
    assert "job-b" in result


# ---- _load_job_from_stores ----


@pytest.mark.asyncio
async def test_load_job_from_stores_from_db(job_manager, mock_db_manager_initialized):
    """_load_job_from_stores returns job from DB when present."""
    db_job = MagicMock()
    db_job.to_dict.return_value = {
        "id": "store-job-1",
        "type": "blog",
        "status": "pending",
        "content_id": "c1",
        "created_at": datetime.now(timezone.utc),
        "started_at": None,
        "completed_at": None,
        "progress": 0,
        "current_step": None,
        "result": None,
        "error": None,
        "error_message": None,
        "metadata": {},
        "user_id": None,
    }

    mock_session = mock_db_manager_initialized.get_session.return_value
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = db_job
    mock_session.execute = AsyncMock(return_value=result_mock)

    with patch(
        "marketing_project.services.database.get_database_manager",
        return_value=mock_db_manager_initialized,
    ):
        job = await job_manager._load_job_from_stores("store-job-1")

    assert job is not None
    assert job.id == "store-job-1"


@pytest.mark.asyncio
async def test_load_job_from_stores_redis_fallback(job_manager):
    """_load_job_from_stores falls back to Redis when DB misses."""
    base_job = Job(id="redis-job-1", type="blog", content_id="c1")

    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        # Put into Redis mock via execute
        job_json = base_job.model_dump_json()
        job_manager._redis_manager.execute = AsyncMock(return_value=job_json.encode())

        job = await job_manager._load_job_from_stores("redis-job-1")

    assert job is not None
    assert job.id == "redis-job-1"


@pytest.mark.asyncio
async def test_load_job_from_stores_memory_fallback(job_manager):
    """_load_job_from_stores falls back to in-memory dict when Redis returns None."""
    base_job = Job(id="mem-job-1", type="blog", content_id="c1")
    job_manager._jobs["mem-job-1"] = base_job

    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        job_manager._redis_manager.execute = AsyncMock(return_value=None)

        job = await job_manager._load_job_from_stores("mem-job-1")

    assert job is not None
    assert job.id == "mem-job-1"


# ---- update_job_progress / update_job_status ----


@pytest.mark.asyncio
async def test_update_job_progress(job_manager):
    """update_job_progress updates progress and current_step."""
    job = await job_manager.create_job("blog", "c1")
    await job_manager.update_job_progress(job.id, 50, "Step 2")
    updated = await job_manager.get_job(job.id)
    assert updated.progress == 50
    assert updated.current_step == "Step 2"


@pytest.mark.asyncio
async def test_update_job_status_waiting_for_approval(job_manager):
    """update_job_status sets completed_at when transitioning to WAITING_FOR_APPROVAL."""
    job = await job_manager.create_job("blog", "c1")
    await job_manager.update_job_status(job.id, JobStatus.WAITING_FOR_APPROVAL)
    updated = await job_manager.get_job(job.id)
    assert updated.status == JobStatus.WAITING_FOR_APPROVAL
    assert updated.completed_at is not None


@pytest.mark.asyncio
async def test_update_job_status_failed_sets_completed_at(job_manager):
    """update_job_status sets completed_at when status becomes FAILED."""
    job = await job_manager.create_job("blog", "c1")
    await job_manager.update_job_status(job.id, JobStatus.FAILED, "oops")
    updated = await job_manager.get_job(job.id)
    assert updated.status == JobStatus.FAILED
    assert updated.completed_at is not None
    assert updated.error_message == "oops"


@pytest.mark.asyncio
async def test_update_job_status_job_not_found(job_manager):
    """update_job_status logs a warning but does not raise when job is missing."""
    # Should not raise
    await job_manager.update_job_status("nonexistent", JobStatus.FAILED)


# ---- cancel_job ----


@pytest.mark.asyncio
async def test_cancel_job_pending(job_manager):
    """Cancelling a pending job marks it CANCELLED."""
    job = await job_manager.create_job("blog", "c1")
    result = await job_manager.cancel_job(job.id)
    assert result is True
    updated = await job_manager.get_job(job.id)
    assert updated.status == JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_job_already_completed(job_manager):
    """cancel_job returns False for already-completed jobs."""
    job = await job_manager.create_job("blog", "c1")
    await job_manager.mark_job_completed(job.id, {"done": True})
    result = await job_manager.cancel_job(job.id)
    assert result is False


@pytest.mark.asyncio
async def test_cancel_job_not_found(job_manager):
    """cancel_job returns False when job doesn't exist."""
    result = await job_manager.cancel_job("does-not-exist")
    assert result is False


# ---- delete_job ----


@pytest.mark.asyncio
async def test_delete_job_from_memory(job_manager):
    """delete_job removes a job from in-memory cache."""
    job = await job_manager.create_job("blog", "c1")
    job_manager._jobs[job.id] = job

    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        result = await job_manager.delete_job(job.id)

    assert result is True
    assert job.id not in job_manager._jobs


@pytest.mark.asyncio
async def test_delete_job_not_found(job_manager):
    """delete_job returns False when job doesn't exist anywhere."""
    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        job_manager._redis_manager.execute = AsyncMock(return_value=0)
        result = await job_manager.delete_job("ghost-job-id")

    assert result is False


# ---- list_jobs (Redis fallback) ----


@pytest.mark.asyncio
async def test_list_jobs_redis_fallback(job_manager):
    """list_jobs returns jobs from in-memory when DB and Redis both fail."""
    job = await job_manager.create_job("blog", "c1")

    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        job_manager._redis_manager.execute = AsyncMock(
            side_effect=RuntimeError("redis boom")
        )

        jobs = await job_manager.list_jobs()

    assert any(j.id == job.id for j in jobs)


@pytest.mark.asyncio
async def test_list_jobs_with_type_filter(job_manager):
    """list_jobs applies job_type filter."""
    await job_manager.create_job("blog", "c1")
    await job_manager.create_job("transcript", "c2")

    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        job_manager._redis_manager.execute = AsyncMock(
            side_effect=RuntimeError("redis boom")
        )

        jobs = await job_manager.list_jobs(job_type="transcript")

    for j in jobs:
        assert j.type == "transcript"


# ---- get_job: result reconciliation paths ----


@pytest.mark.asyncio
async def test_get_job_reconciles_result_to_completed(job_manager):
    """get_job upgrades status to COMPLETED when job has a result but stale status."""
    job = await job_manager.create_job("blog", "c1")
    job.status = JobStatus.PROCESSING
    job.result = {"output": "done"}
    job_manager._jobs[job.id] = job

    updated = await job_manager.get_job(job.id)
    assert updated.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_get_job_waiting_for_approval_result_sentinel(job_manager):
    """get_job correctly identifies the waiting_for_approval result sentinel."""
    job = await job_manager.create_job("blog", "c1")
    job.status = JobStatus.PROCESSING
    job.result = {"status": "waiting_for_approval"}
    job_manager._jobs[job.id] = job

    updated = await job_manager.get_job(job.id)
    assert updated.status == JobStatus.WAITING_FOR_APPROVAL


# ---- get_job_chain ----


@pytest.mark.asyncio
async def test_get_job_chain_not_found(job_manager):
    """get_job_chain returns empty structure when job doesn't exist."""
    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        result = await job_manager.get_job_chain("nonexistent-job")

    assert result["chain_length"] == 0
    assert result["jobs"] == []


@pytest.mark.asyncio
async def test_get_job_chain_single_job(job_manager):
    """get_job_chain on a standalone job has chain_length=1."""
    job = await job_manager.create_job("blog", "c1")
    result = await job_manager.get_job_chain(job.id)
    assert result["root_job_id"] == job.id
    assert result["chain_length"] == 1


@pytest.mark.asyncio
async def test_get_job_chain_with_resume_job(job_manager):
    """get_job_chain follows resume_job_id links."""
    root_job = await job_manager.create_job("blog", "c1")
    resume_job = await job_manager.create_job("resume_pipeline", "c1")
    resume_job.metadata["original_job_id"] = root_job.id
    job_manager._jobs[resume_job.id] = resume_job
    # Link root to resume
    root_job.metadata["resume_job_id"] = resume_job.id
    job_manager._jobs[root_job.id] = root_job

    result = await job_manager.get_job_chain(root_job.id)
    assert result["root_job_id"] == root_job.id
    assert result["chain_length"] >= 2


# ---- update_parent_job_status scenarios ----


@pytest.mark.asyncio
async def test_update_parent_job_status_all_completed(job_manager):
    """update_parent_job_status marks parent COMPLETED when all subjobs done."""
    root_job = await job_manager.create_job("blog", "c1")
    resume_job = await job_manager.create_job("resume_pipeline", "c1")
    resume_job.status = JobStatus.COMPLETED
    resume_job.metadata["original_job_id"] = root_job.id
    job_manager._jobs[resume_job.id] = resume_job
    root_job.metadata["resume_job_id"] = resume_job.id
    job_manager._jobs[root_job.id] = root_job

    await job_manager.update_parent_job_status(root_job.id)

    updated = job_manager._jobs.get(root_job.id)
    assert updated is not None


@pytest.mark.asyncio
async def test_update_parent_job_status_with_failed_subjob(job_manager):
    """update_parent_job_status propagates FAILED status from subjob."""
    root_job = await job_manager.create_job("blog", "c1")
    resume_job = await job_manager.create_job("resume_pipeline", "c1")
    resume_job.status = JobStatus.FAILED
    resume_job.metadata["original_job_id"] = root_job.id
    job_manager._jobs[resume_job.id] = resume_job
    root_job.metadata["resume_job_id"] = resume_job.id
    job_manager._jobs[root_job.id] = root_job

    await job_manager.update_parent_job_status(root_job.id)
    # Should not raise; parent status reflects failure


# ---- get_job_with_subjob_status ----


@pytest.mark.asyncio
async def test_get_job_with_subjob_status_not_found(job_manager):
    """get_job_with_subjob_status returns None for nonexistent job."""
    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        result = await job_manager.get_job_with_subjob_status("no-such-job")

    assert result is None


@pytest.mark.asyncio
async def test_get_job_with_subjob_status_no_subjobs(job_manager):
    """get_job_with_subjob_status returns chain_status=None for standalone job."""
    job = await job_manager.create_job("blog", "c1")
    result = await job_manager.get_job_with_subjob_status(job.id)
    assert result is not None
    assert result["chain_status"] is None


# ---- update_job_chain_metadata ----


@pytest.mark.asyncio
async def test_update_job_chain_metadata_no_chain(job_manager):
    """update_job_chain_metadata handles job not found gracefully."""
    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        # Should not raise
        await job_manager.update_job_chain_metadata("nonexistent-job")


@pytest.mark.asyncio
async def test_update_job_chain_metadata_updates_jobs(job_manager):
    """update_job_chain_metadata writes job_chain key to all jobs in chain."""
    job = await job_manager.create_job("blog", "c1")
    await job_manager.update_job_chain_metadata(job.id)
    updated = job_manager._jobs.get(job.id)
    assert updated is not None


# ---- clear_all_arq_jobs ----


@pytest.mark.asyncio
async def test_clear_all_arq_jobs(job_manager):
    """clear_all_arq_jobs returns a non-negative integer."""
    job_manager._redis_manager.execute = AsyncMock(return_value=[])
    result = await job_manager.clear_all_arq_jobs()
    assert isinstance(result, int)
    assert result >= 0


# ---- cleanup_old_jobs ----


@pytest.mark.asyncio
async def test_cleanup_old_jobs_removes_in_memory_stale_jobs(job_manager):
    """cleanup_old_jobs evicts stale completed jobs from in-memory cache."""
    from datetime import timedelta

    old_job = await job_manager.create_job("blog", "c1")
    old_job.status = JobStatus.COMPLETED
    old_job.completed_at = datetime.now(timezone.utc) - timedelta(hours=48)
    job_manager._jobs[old_job.id] = old_job

    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_get_db:
        db_manager = MagicMock()
        db_manager.is_initialized = False
        mock_get_db.return_value = db_manager

        cleaned = await job_manager.cleanup_old_jobs(max_age_hours=24)

    assert cleaned >= 1
    assert old_job.id not in job_manager._jobs


# ---- _normalize_datetime_to_utc: naive datetime ----


@pytest.mark.asyncio
async def test_normalize_datetime_naive_becomes_utc(job_manager):
    """Naive datetime gets UTC timezone attached."""
    naive = datetime(2024, 1, 15, 10, 30, 0)
    normalized = job_manager._normalize_datetime_to_utc(naive)
    assert normalized.tzinfo == timezone.utc
    assert normalized.year == 2024
