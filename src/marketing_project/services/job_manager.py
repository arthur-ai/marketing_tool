"""
Job Manager Service.

Manages background job execution and status tracking for long-running pipeline operations.
Uses ARQ (Async Redis Queue) for distributed job execution and Redis for job state persistence.

Job state is stored in Redis for:
- Persistence across API restarts
- Horizontal scaling support (multiple API instances)
- Shared state between API and worker processes
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from arq.jobs import Job as ArqJob
from arq.jobs import JobStatus as ArqJobStatus
from pydantic import BaseModel, Field

from marketing_project.services.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_FOR_APPROVAL = "waiting_for_approval"


# Constants
ARQ_RESULT_TTL = 3600  # ARQ keeps results for 1 hour
JOB_MAX_AGE = 3600  # Consider jobs older than 1 hour as expired
# Redis TTL for active job tracking (jobs are primarily stored in PostgreSQL)
# Redis is used for ARQ queue coordination and active job status
JOB_REDIS_TTL = 86400  # 24 hours - only for active jobs in Redis cache
JOB_KEY_PREFIX = "job:"  # Redis key prefix for jobs
JOB_INDEX_KEY = "jobs:index"  # Redis set for job indexing


class JobMetadataKeys:
    """Known keys for Job.metadata dict. Use these constants to avoid typos."""

    ARQ_JOB_ID = "arq_job_id"
    ORIGINAL_JOB_ID = "original_job_id"
    RESUME_JOB_ID = "resume_job_id"
    RETRY_JOB_ID = "retry_job_id"
    JOB_CHAIN = "job_chain"
    APPROVED_AT = "approved_at"
    SESSION_ID = "session_id"
    TRIGGERED_BY_USER_ID = "triggered_by_user_id"
    TRIGGERED_BY_USERNAME = "triggered_by_username"
    TRIGGERED_BY_EMAIL = "triggered_by_email"
    INPUT_CONTENT = "input_content"
    TITLE = "title"
    STATUS = "status"


class Job(BaseModel):
    """Job model for tracking background task execution."""

    id: str = Field(..., description="Unique job ID")
    type: str = Field(
        ..., description="Job type (blog, release_notes, transcript, pipeline)"
    )
    status: JobStatus = Field(
        default=JobStatus.PENDING, description="Current job status"
    )
    content_id: str = Field(..., description="Content ID being processed")
    user_id: Optional[str] = Field(None, description="User ID who triggered the job")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = Field(default=0, description="Job progress percentage (0-100)")
    current_step: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobManager:
    """
    Manages background job status tracking with Redis persistence.

    Integrates with ARQ for distributed job execution and uses Redis for job state storage.
    This allows:
    - Job state to survive API restarts
    - Multiple API instances to share job state
    - Worker and API to have consistent view of jobs

    NOTE: ARQ handles job execution, this class tracks job metadata and status.
    """

    def __init__(self):
        # Legacy in-memory storage (kept for backward compatibility during migration)
        self._jobs: Dict[str, Job] = {}

        # Redis manager (shared instance)
        self._redis_manager = get_redis_manager()

        # ARQ pool (lazy init)
        self._arq_pool = None
        self._pool_lock = asyncio.Lock()

    async def get_redis(self) -> redis.Redis:
        """Get Redis client from RedisManager."""
        return await self._redis_manager.get_redis()

    def _normalize_datetime_to_utc(self, dt: Optional[datetime]) -> Optional[datetime]:
        """
        Normalize a datetime to UTC timezone-aware.

        Args:
            dt: Datetime to normalize (can be None, naive, or timezone-aware)

        Returns:
            UTC timezone-aware datetime, or None if input was None
        """
        if dt is None:
            return None
        if dt.tzinfo is None:
            # Naive datetime - assume UTC
            return dt.replace(tzinfo=timezone.utc)
        else:
            # Timezone-aware datetime - convert to UTC
            return dt.astimezone(timezone.utc)

    async def _save_job_to_database(self, job: Job) -> None:
        """Save job to PostgreSQL database for permanent storage."""
        try:
            from marketing_project.models.db_models import JobModel
            from marketing_project.services.database import get_database_manager

            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                logger.warning("Database not initialized, skipping database save")
                return

            async with db_manager.get_session() as session:
                from sqlalchemy import select

                # Check if job exists
                stmt = select(JobModel).where(JobModel.job_id == job.id)
                result = await session.execute(stmt)
                existing_job = result.scalar_one_or_none()

                # Convert job to dict for JSONB storage
                job_data = {
                    "id": job.id,
                    "type": job.type,
                    "status": job.status.value,
                    "content_id": job.content_id,
                    "created_at": (
                        job.created_at.isoformat() if job.created_at else None
                    ),
                    "started_at": (
                        job.started_at.isoformat() if job.started_at else None
                    ),
                    "completed_at": (
                        job.completed_at.isoformat() if job.completed_at else None
                    ),
                    "progress": job.progress,
                    "current_step": job.current_step,
                    "result": job.result,
                    "error": job.error,
                    "metadata": job.metadata,
                }

                if existing_job:
                    # Update existing job
                    existing_job.status = job.status.value
                    existing_job.content_id = job.content_id
                    existing_job.user_id = job.user_id
                    existing_job.progress = job.progress
                    existing_job.current_step = job.current_step
                    existing_job.result = job.result
                    existing_job.error = job.error
                    existing_job.job_metadata = job.metadata
                    if job.started_at:
                        existing_job.started_at = job.started_at
                    if job.completed_at:
                        existing_job.completed_at = job.completed_at
                    logger.debug(f"Updated job {job.id} in database")
                else:
                    # Create new job
                    db_job = JobModel(
                        job_id=job.id,
                        job_type=job.type,
                        status=job.status.value,
                        content_id=job.content_id,
                        user_id=job.user_id,
                        progress=job.progress,
                        current_step=job.current_step,
                        result=job.result,
                        error=job.error,
                        job_metadata=job.metadata,
                        created_at=job.created_at,
                        started_at=job.started_at,
                        completed_at=job.completed_at,
                    )
                    session.add(db_job)
                    logger.debug(f"Saved job {job.id} to database")

        except Exception as e:
            logger.error(
                f"Failed to save job {job.id} to database: {type(e).__name__}: {e}",
                exc_info=True,
            )
            # Don't fail if database save fails - Redis is still available

    async def _save_job(self, job: Job) -> None:
        """Save job to both Redis (cache) and database (permanent storage)."""
        # Save to database first (permanent storage)
        await self._save_job_to_database(job)
        # Then save to Redis (for active job tracking)
        await self._save_job_to_redis(job)

    async def _save_job_to_redis(self, job: Job) -> None:
        """Helper method to save job state to Redis (for active job tracking and ARQ coordination)."""
        try:
            job_key = f"{JOB_KEY_PREFIX}{job.id}"
            job_json = job.model_dump_json()

            # Use RedisManager with retry and circuit breaker
            async def save_operation(redis_client: redis.Redis):
                await redis_client.setex(job_key, JOB_REDIS_TTL, job_json)

            await self._redis_manager.execute(save_operation)

            # Also update in-memory cache
            self._jobs[job.id] = job

            logger.debug(
                f"Job {job.id} saved to Redis (key: {job_key}, TTL: {JOB_REDIS_TTL}s)"
            )

        except Exception as e:
            logger.error(
                f"Failed to save job {job.id} to Redis (key: {job_key}): {type(e).__name__}: {e}",
                extra={
                    "job_id": job.id,
                    "job_type": job.type,
                    "job_status": job.status.value,
                    "redis_key": job_key,
                },
            )
            # Still update in-memory
            self._jobs[job.id] = job

    async def create_job(
        self,
        job_type: str,
        content_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_context: Optional[Any] = None,
    ) -> Job:
        """
        Create a new job and store in Redis.

        Args:
            job_type: Type of job (blog, release_notes, transcript, pipeline)
            content_id: ID of content being processed
            metadata: Optional metadata dictionary
            job_id: Optional job ID (if not provided, a UUID will be generated)
            user_id: Optional user ID who triggered the job
            user_context: Optional UserContext object for storing username/email in metadata

        Returns:
            Created job object
        """
        if job_id is None:
            job_id = str(uuid.uuid4())

        # Prepare metadata with user information
        job_metadata = metadata.copy() if metadata else {}

        # Store user information in metadata for display purposes
        if user_id:
            job_metadata[JobMetadataKeys.TRIGGERED_BY_USER_ID] = user_id
            if user_context:
                if hasattr(user_context, "username") and user_context.username:
                    job_metadata[JobMetadataKeys.TRIGGERED_BY_USERNAME] = (
                        user_context.username
                    )
                if hasattr(user_context, "email") and user_context.email:
                    job_metadata[JobMetadataKeys.TRIGGERED_BY_EMAIL] = (
                        user_context.email
                    )

        job = Job(
            id=job_id,
            type=job_type,
            content_id=content_id,
            user_id=user_id,
            metadata=job_metadata,
        )

        try:
            # Store in Redis with TTL using pipeline for batch operations
            job_key = f"{JOB_KEY_PREFIX}{job_id}"
            job_json = job.model_dump_json()

            # Use pipeline for batch operations (SETEX + SADD)
            async def create_operation(redis_client: redis.Redis):
                async with redis_client.pipeline() as pipe:
                    pipe.setex(job_key, JOB_REDIS_TTL, job_json)
                    pipe.sadd(JOB_INDEX_KEY, job_id)
                    await pipe.execute()

            await self._redis_manager.execute(create_operation)

            # Also keep in memory for backward compatibility
            self._jobs[job_id] = job

            # Save to database for permanent storage
            await self._save_job_to_database(job)

            logger.info(
                f"Created job {job_id} for {job_type} processing of content {content_id} "
                f"(stored in Redis and database)"
            )

        except Exception as e:
            logger.error(f"Failed to store job {job_id} in Redis: {e}")
            # Fall back to in-memory only
            self._jobs[job_id] = job
            logger.warning(f"Job {job_id} stored in-memory only (Redis unavailable)")

        return job

    async def get_jobs_by_ids(self, job_ids: List[str]) -> Dict[str, "Job"]:
        """
        Fetch multiple jobs by ID in a single DB query.

        Returns a dict mapping job_id → Job. Jobs not found are omitted.
        Falls back to individual get_job() calls if DB is unavailable.
        """
        if not job_ids:
            return {}

        result: Dict[str, Job] = {}
        found_in_db: set = set()

        try:
            from sqlalchemy import select

            from marketing_project.models.db_models import JobModel
            from marketing_project.services.database import get_database_manager

            db_manager = get_database_manager()
            if db_manager.is_initialized:
                async with db_manager.get_session() as session:
                    stmt = select(JobModel).where(JobModel.job_id.in_(job_ids))
                    db_result = await session.execute(stmt)
                    db_jobs = db_result.scalars().all()

                    for db_job in db_jobs:
                        job_dict = db_job.to_dict()
                        job_dict["created_at"] = self._normalize_datetime_to_utc(
                            job_dict.get("created_at")
                        )
                        job_dict["started_at"] = self._normalize_datetime_to_utc(
                            job_dict.get("started_at")
                        )
                        job_dict["completed_at"] = self._normalize_datetime_to_utc(
                            job_dict.get("completed_at")
                        )
                        job = Job(**job_dict)
                        self._jobs[job.id] = job
                        result[job.id] = job
                        found_in_db.add(job.id)

        except Exception as e:
            logger.debug(f"Error bulk-fetching jobs from DB: {type(e).__name__}: {e}")

        # Fall back to individual get_job() for any IDs not found in DB
        missing = [jid for jid in job_ids if jid not in found_in_db]
        for jid in missing:
            job = await self.get_job(jid)
            if job:
                result[jid] = job

        return result

    async def get_job_ids_for_user(self, user_id: str) -> set:
        """
        Return the set of job_ids owned by a user. Uses a narrow SELECT (job_id only).
        Falls back to list_jobs() if DB is unavailable.
        """
        try:
            from sqlalchemy import select

            from marketing_project.models.db_models import JobModel
            from marketing_project.services.database import get_database_manager

            db_manager = get_database_manager()
            if db_manager.is_initialized:
                async with db_manager.get_session() as session:
                    stmt = select(JobModel.job_id).where(JobModel.user_id == user_id)
                    db_result = await session.execute(stmt)
                    return {row[0] for row in db_result.fetchall()}
        except Exception as e:
            logger.debug(
                f"Error fetching job IDs for user {user_id}: {type(e).__name__}: {e}"
            )

        # Fallback
        jobs = await self.list_jobs(user_id=user_id, limit=10000)
        return {j.id for j in jobs}

    async def get_arq_pool(self):
        """Get or create ARQ Redis pool with proper locking."""
        async with self._pool_lock:
            if self._arq_pool is None:
                from marketing_project.worker import get_arq_pool

                self._arq_pool = await get_arq_pool()
                logger.debug("ARQ pool initialized successfully")
            return self._arq_pool

    async def submit_to_arq(
        self,
        job_id: str,
        function_name: str,
        *args,
        **kwargs,
    ) -> str:
        """
        Submit a job to ARQ for background execution.

        Args:
            job_id: ID of the job to track
            function_name: Name of ARQ function to call (e.g., 'process_blog')
            *args: Arguments to pass to ARQ function
            **kwargs: Keyword arguments to pass to ARQ function

        Returns:
            ARQ job ID
        """
        # Load from DB/Redis — not in-memory only, so this works across processes
        job = await self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        try:
            # Get ARQ pool
            pool = await self.get_arq_pool()

            # Enqueue job in ARQ immediately with no artificial delays
            arq_job = await pool.enqueue_job(
                function_name,
                *args,
                **kwargs,
            )

            # Update job status
            job.status = JobStatus.QUEUED
            job.metadata[JobMetadataKeys.ARQ_JOB_ID] = arq_job.job_id

            # Save updated job status to Redis and database
            await self._save_job(job)

            logger.info(f"Submitted job {job_id} to ARQ as {arq_job.job_id}")
            return arq_job.job_id

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = f"Failed to submit to ARQ: {str(e)}"
            # Save failed status to Redis
            await self._save_job(job)
            logger.error(f"Failed to submit job {job_id} to ARQ: {e}")
            raise

    async def mark_job_started(self, job_id: str) -> None:
        """Mark a job as started (called by ARQ worker)."""
        job = await self.get_job(job_id)
        if job:
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)
            await self._save_job(job)
            logger.debug(f"Job {job_id} marked as started")

    async def mark_job_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        """Mark a job as completed (called by ARQ worker)."""
        job = await self.get_job(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.progress = 100
            job.result = result
            await self._save_job(job)
            logger.info(f"Job {job_id} marked as completed")

    async def mark_job_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed (called by ARQ worker)."""
        job = await self.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error = error
            await self._save_job(job)
            logger.error(f"Job {job_id} marked as failed: {error}")

    async def _load_job_from_stores(self, job_id: str) -> Optional[Job]:
        """
        Load a job from persistent stores only: PostgreSQL → Redis → in-memory.

        No ARQ polling. Pure storage read.
        """
        job = None

        # Try database first (permanent storage)
        try:
            from marketing_project.models.db_models import JobModel
            from marketing_project.services.database import get_database_manager

            db_manager = get_database_manager()
            if db_manager.is_initialized:
                async with db_manager.get_session() as session:
                    from sqlalchemy import select

                    stmt = select(JobModel).where(JobModel.job_id == job_id)
                    result = await session.execute(stmt)
                    db_job = result.scalar_one_or_none()

                    if db_job:
                        job_dict = db_job.to_dict()
                        job_dict["created_at"] = self._normalize_datetime_to_utc(
                            job_dict.get("created_at")
                        )
                        job_dict["started_at"] = self._normalize_datetime_to_utc(
                            job_dict.get("started_at")
                        )
                        job_dict["completed_at"] = self._normalize_datetime_to_utc(
                            job_dict.get("completed_at")
                        )
                        job = Job(**job_dict)
                        self._jobs[job_id] = job
                        logger.debug(f"Job {job_id} loaded from database")
        except Exception as e:
            logger.debug(
                f"Error reading job {job_id} from database: {type(e).__name__}: {e}"
            )

        # Redis fallback
        if not job:
            try:
                job_key = f"{JOB_KEY_PREFIX}{job_id}"

                async def get_operation(redis_client: redis.Redis):
                    return await redis_client.get(job_key)

                job_json = await self._redis_manager.execute(get_operation)

                if job_json:
                    job = Job.model_validate_json(job_json)
                    job.created_at = self._normalize_datetime_to_utc(job.created_at)
                    job.started_at = self._normalize_datetime_to_utc(job.started_at)
                    job.completed_at = self._normalize_datetime_to_utc(job.completed_at)
                    self._jobs[job_id] = job
                    logger.debug(f"Job {job_id} loaded from Redis")
                    await self._save_job_to_database(job)
                else:
                    # Last resort: in-memory
                    job = self._jobs.get(job_id)
                    if job:
                        job.created_at = self._normalize_datetime_to_utc(job.created_at)
                        job.started_at = self._normalize_datetime_to_utc(job.started_at)
                        job.completed_at = self._normalize_datetime_to_utc(
                            job.completed_at
                        )
                        logger.debug(f"Job {job_id} found in memory (not in Redis)")

            except Exception as e:
                logger.error(
                    f"Error reading job {job_id} from Redis: {type(e).__name__}: {e}",
                    extra={"job_id": job_id, "error_type": type(e).__name__},
                )
                job = self._jobs.get(job_id)
                if job:
                    job.created_at = self._normalize_datetime_to_utc(job.created_at)
                    job.started_at = self._normalize_datetime_to_utc(job.started_at)
                    job.completed_at = self._normalize_datetime_to_utc(job.completed_at)

        return job

    async def get_job(self, job_id: str, _skip_arq_poll: bool = False) -> Optional[Job]:
        """
        Get job by ID. Loads from stores (DB → Redis → memory), then optionally polls ARQ.

        ARQ polling is only done for QUEUED/PROCESSING jobs when _skip_arq_poll is False.
        Use _skip_arq_poll=True in the worker hot path to avoid redundant I/O.
        """
        job = await self._load_job_from_stores(job_id)

        if not job:
            return None

        # For terminal or waiting states, skip ARQ poll — status is already authoritative
        if job.status in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.WAITING_FOR_APPROVAL,
        ]:
            return job

        # If job has a result but status hasn't been updated yet (e.g. worker crashed before
        # marking complete), reconcile it now
        if job.result is not None and job.result != {}:
            # Don't upgrade waiting_for_approval to completed — the result dict
            # may be the approval sentinel ({"status": "waiting_for_approval"})
            # or a partial result, neither of which means the pipeline finished.
            if (
                isinstance(job.result, dict)
                and job.result.get("status") == "waiting_for_approval"
            ):
                # Ensure status is correctly set to WAITING_FOR_APPROVAL
                if job.status != JobStatus.WAITING_FOR_APPROVAL:
                    job.status = JobStatus.WAITING_FOR_APPROVAL
                    await self._save_job(job)
                return job
            if job.status not in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.WAITING_FOR_APPROVAL,
            ]:
                logger.info(
                    f"Job {job_id} has result in database but status is {job.status}, "
                    f"updating to COMPLETED"
                )
                job.status = JobStatus.COMPLETED
                if not job.completed_at:
                    job.completed_at = datetime.now(timezone.utc)
                job.progress = 100
                await self._save_job(job)
            return job

        # Check if job is too old and likely expired from ARQ
        # Only do this if job doesn't have a result (we already checked above)
        if job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
            # Ensure job.created_at is normalized (defensive check)
            job.created_at = self._normalize_datetime_to_utc(job.created_at)

            # Normalize datetimes to ensure both are UTC timezone-aware
            now = datetime.now(timezone.utc)
            created_at = job.created_at
            if created_at is None:
                # If created_at is None, skip age check
                logger.warning(f"Job {job_id} has no created_at timestamp")
            else:
                age_seconds = (now - created_at).total_seconds()

                if age_seconds > JOB_MAX_AGE:
                    logger.warning(
                        f"Job {job_id} is {age_seconds:.0f}s old (>{JOB_MAX_AGE}s), "
                        f"ARQ result likely expired"
                    )
                    job.error = f"Job exceeded maximum age ({JOB_MAX_AGE}s) - ARQ result expired"
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now(timezone.utc)
                    return job

        # If job is queued or processing, check ARQ for updates
        # Skip when called from the worker's own progress updates — the worker
        # already knows the job is running; polling ARQ is redundant and expensive.
        if not _skip_arq_poll and job.status in [
            JobStatus.QUEUED,
            JobStatus.PROCESSING,
        ]:
            arq_job_id = job.metadata.get(JobMetadataKeys.ARQ_JOB_ID)
            if arq_job_id:
                try:
                    pool = await self.get_arq_pool()
                    arq_job = ArqJob(arq_job_id, pool)

                    # Get ARQ job status
                    arq_status = await arq_job.status()

                    if arq_status == ArqJobStatus.complete:
                        # Job completed - get result.
                        # NOTE: WAITING_FOR_APPROVAL is set explicitly by the worker before
                        # returning; by the time ARQ shows "complete", the status is already
                        # persisted in DB/Redis and the early-return above will have caught it.
                        try:
                            result = await arq_job.result()
                            # Check if the pipeline stopped for an approval checkpoint
                            # (result dict contains {"status": "waiting_for_approval"})
                            if (
                                isinstance(result, dict)
                                and result.get("status") == "waiting_for_approval"
                            ):
                                job.status = JobStatus.WAITING_FOR_APPROVAL
                                if not job.completed_at:
                                    job.completed_at = datetime.now(timezone.utc)
                                job.result = result
                                logger.info(
                                    f"Job {job_id} ARQ task complete but pipeline is waiting_for_approval"
                                )
                            else:
                                job.status = JobStatus.COMPLETED
                                job.completed_at = datetime.now(timezone.utc)
                                job.progress = 100
                                job.result = result
                                job.current_step = "Completed"
                                logger.info(
                                    f"Job {job_id} completed with result from ARQ"
                                )
                            # Save updated status to Redis and database
                            await self._save_job(job)

                            # If this is a resume job and it's the final subjob, copy result to original job
                            if job.type == "resume_pipeline" and job.metadata.get(
                                "original_job_id"
                            ):
                                original_job_id = job.metadata.get(
                                    JobMetadataKeys.ORIGINAL_JOB_ID
                                )
                                # Check if this is the final subjob (no more resume_job_id)
                                if not job.metadata.get(JobMetadataKeys.RESUME_JOB_ID):
                                    # This is the final subjob - copy result to original job
                                    original_job = await self.get_job(original_job_id)
                                    if original_job:
                                        # Extract the actual pipeline result from the ARQ result
                                        pipeline_result = result
                                        if (
                                            isinstance(result, dict)
                                            and "result" in result
                                        ):
                                            pipeline_result = result["result"]

                                        original_job.result = pipeline_result
                                        original_job.status = JobStatus.COMPLETED
                                        original_job.completed_at = datetime.now(
                                            timezone.utc
                                        )
                                        original_job.progress = 100
                                        original_job.current_step = "Completed"
                                        # Update metadata to indicate all subjobs are complete
                                        original_job.metadata[
                                            "all_subjobs_completed"
                                        ] = True
                                        original_job.metadata["final_subjob_id"] = (
                                            job_id
                                        )
                                        await self._save_job(original_job)
                                        logger.info(
                                            f"Copied final result from subjob {job_id} to original job {original_job_id}"
                                        )
                        except (TypeError, KeyError, ValueError) as e:
                            # ARQ job failed due to execution error (wrong args, missing keys, etc.)
                            error_msg = str(e)
                            logger.error(
                                f"ARQ job {arq_job_id} for {job_id} failed with error: {error_msg}",
                                exc_info=True,
                            )
                            job.error = f"Job execution failed: {error_msg}"
                            job.status = JobStatus.FAILED
                            job.completed_at = datetime.now(timezone.utc)
                            await self._save_job(job)
                        except Exception as e:
                            # Other errors when getting result
                            logger.error(
                                f"Error getting ARQ result for job {job_id}: {e}",
                                exc_info=True,
                            )
                            job.error = f"Error retrieving job result: {str(e)}"
                            job.status = JobStatus.FAILED
                            job.completed_at = datetime.now(timezone.utc)
                            await self._save_job(job)

                    elif arq_status == ArqJobStatus.in_progress:
                        job.status = JobStatus.PROCESSING
                        if not job.started_at:
                            job.started_at = datetime.now(timezone.utc)
                        logger.debug(f"Job {job_id} is in progress")
                        await self._save_job(job)

                    elif arq_status == ArqJobStatus.queued:
                        # Job is queued and waiting for a worker
                        job.status = JobStatus.QUEUED
                        if (
                            not job.current_step
                            or job.current_step == "Retrying after failure"
                        ):
                            job.current_step = "Waiting for worker"
                        logger.debug(f"Job {job_id} is queued in ARQ")
                        await self._save_job(job)

                    elif arq_status == ArqJobStatus.deferred:
                        # Job is waiting to retry after failure
                        job.status = JobStatus.QUEUED
                        job.current_step = "Retrying after failure"
                        logger.info(f"Job {job_id} is deferred (will retry)")
                        await self._save_job(job)

                    elif arq_status == ArqJobStatus.not_found:
                        # ARQ job not found - might have expired or been cleaned up
                        logger.warning(
                            f"ARQ job {arq_job_id} for {job_id} not found - "
                            f"result may have expired (ARQ TTL: {ARQ_RESULT_TTL}s)"
                        )
                        job.error = "Job not found in ARQ - result may have expired"
                        job.status = JobStatus.FAILED
                        job.completed_at = datetime.now(timezone.utc)
                        await self._save_job(job)

                    else:
                        # Catch-all for any other status
                        logger.warning(
                            f"Job {job_id} has unknown ARQ status: {arq_status}"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to query ARQ for job {job_id}: {e}", exc_info=True
                    )
                    # Don't fail the job on transient ARQ query errors
                    # Just return current state and client can retry

        return job

    async def update_job_progress(
        self, job_id: str, progress: int, current_step: Optional[str] = None
    ) -> None:
        """
        Update job progress and save to Redis.

        Args:
            job_id: ID of the job to update
            progress: Progress percentage (0-100)
            current_step: Optional current step description
        """
        # Skip ARQ poll — this is called from the worker hot path where we know
        # the job is already processing; re-polling ARQ on every progress tick
        # adds unnecessary I/O (DB + Redis + ARQ per call).
        job = await self.get_job(job_id, _skip_arq_poll=True)
        if job:
            job.progress = progress
            if current_step:
                job.current_step = current_step
            await self._save_job(job)

    async def update_job_status(self, job_id: str, status: JobStatus) -> None:
        """
        Update job status and save to Redis.

        Args:
            job_id: ID of the job to update
            status: New job status
        """
        job = await self.get_job(job_id, _skip_arq_poll=True)
        if job:
            job.status = status
            # Keep metadata.status in sync with job.status for frontend compatibility
            job.metadata["status"] = status.value
            if status == JobStatus.WAITING_FOR_APPROVAL:
                # Mark as completed time if waiting for approval (job is "done" but waiting)
                job.completed_at = datetime.now(timezone.utc)
            await self._save_job(job)
            logger.info(f"Updated job {job_id} status to {status.value}")

            # If this is a subjob, update parent job status
            if job.metadata.get(JobMetadataKeys.ORIGINAL_JOB_ID):
                await self.update_parent_job_status(
                    job.metadata.get(JobMetadataKeys.ORIGINAL_JOB_ID)
                )

    async def update_parent_job_status(self, parent_job_id: str) -> None:
        """
        Update parent job status based on all subjob statuses.

        Args:
            parent_job_id: ID of the parent job to update
        """
        try:
            parent_job = await self.get_job(parent_job_id)
            if not parent_job:
                return

            # Get chain data to find all subjobs
            chain_data = await self.get_job_chain(parent_job_id)
            if chain_data["root_job_id"] != parent_job_id:
                # This is not the root job, don't update
                return

            if chain_data["chain_length"] <= 1:
                # No subjobs, nothing to update
                return

            subjob_ids = chain_data["chain_order"][1:]  # All except root
            subjob_statuses = []

            for subjob_id in subjob_ids:
                subjob = await self.get_job(subjob_id)
                if subjob:
                    subjob_statuses.append(subjob.status)

            # Determine aggregated status
            if all(s == JobStatus.COMPLETED for s in subjob_statuses):
                # All subjobs completed
                if parent_job.status != JobStatus.COMPLETED:
                    parent_job.status = JobStatus.COMPLETED
                    parent_job.completed_at = datetime.now(timezone.utc)
                    parent_job.current_step = "All subjobs completed"
                    parent_job.progress = 100
                    parent_job.metadata["status"] = "all_subjobs_completed"
                    await self._save_job(parent_job)
                    logger.info(
                        f"Updated parent job {parent_job_id} status to completed (all subjobs done)"
                    )
            elif any(s == JobStatus.WAITING_FOR_APPROVAL for s in subjob_statuses):
                # At least one subjob is waiting for approval — takes priority over processing
                waiting_count = sum(
                    1 for s in subjob_statuses if s == JobStatus.WAITING_FOR_APPROVAL
                )
                parent_job.current_step = (
                    f"Waiting for Approval (Subjob {waiting_count})"
                )
                if parent_job.status != JobStatus.WAITING_FOR_APPROVAL:
                    parent_job.status = JobStatus.WAITING_FOR_APPROVAL
                    parent_job.metadata["status"] = "waiting_for_subjob_approval"
                    await self._save_job(parent_job)
                    logger.info(
                        f"Updated parent job {parent_job_id} status to waiting_for_approval (subjob {waiting_count})"
                    )
            elif any(
                s == JobStatus.PROCESSING or s == JobStatus.QUEUED
                for s in subjob_statuses
            ):
                # At least one subjob is processing
                processing_count = sum(
                    1
                    for s in subjob_statuses
                    if s in [JobStatus.PROCESSING, JobStatus.QUEUED]
                )
                total_count = len(subjob_statuses)
                parent_job.current_step = (
                    f"In Progress (Subjob {processing_count}/{total_count} running)"
                )
                if parent_job.status != JobStatus.PROCESSING:
                    parent_job.status = JobStatus.PROCESSING
                    parent_job.metadata["status"] = "processing_with_subjobs"
                    await self._save_job(parent_job)
                    logger.info(
                        f"Updated parent job {parent_job_id} status to processing ({processing_count}/{total_count} subjobs)"
                    )
            elif any(s == JobStatus.FAILED for s in subjob_statuses):
                # At least one subjob failed
                failed_count = sum(1 for s in subjob_statuses if s == JobStatus.FAILED)
                parent_job.current_step = f"Failed (Subjob {failed_count} failed)"
                if parent_job.status != JobStatus.FAILED:
                    parent_job.status = JobStatus.FAILED
                    parent_job.metadata["status"] = "failed_with_subjobs"
                    await self._save_job(parent_job)
                    logger.info(
                        f"Updated parent job {parent_job_id} status to failed ({failed_count} subjobs failed)"
                    )
        except Exception as e:
            logger.error(f"Failed to update parent job status for {parent_job_id}: {e}")

    async def get_job_chain(self, job_id: str) -> Dict[str, Any]:
        """
        Get complete job chain hierarchy for a given job.

        Returns the root job, all subjobs in order (resume + retry), and chain metadata.
        Uses stored chain_order metadata when available to avoid full re-traversal;
        falls back to live traversal if metadata is absent.

        Args:
            job_id: Job identifier (can be root or any job in chain)

        Returns:
            Dictionary with chain structure:
            {
                "root_job_id": str,
                "chain_length": int,
                "chain_order": List[str],
                "all_job_ids": List[str],
                "chain_status": str,
                "jobs": List[Job]  # All jobs in chain
            }
        """
        _empty = {
            "root_job_id": None,
            "chain_length": 0,
            "chain_order": [],
            "all_job_ids": [],
            "chain_status": "unknown",
            "jobs": [],
        }
        try:
            # --- Step 1: find root job ---
            root_job_id = job_id
            current_job = await self.get_job(job_id)
            if not current_job:
                return _empty

            while current_job.metadata.get(JobMetadataKeys.ORIGINAL_JOB_ID):
                root_job_id = current_job.metadata[JobMetadataKeys.ORIGINAL_JOB_ID]
                current_job = await self.get_job(root_job_id)
                if not current_job:
                    break

            root_job = await self.get_job(root_job_id)
            if not root_job:
                return _empty

            # --- Step 2: build chain_order via live traversal ---
            # Always follow RESUME_JOB_ID links from root so we pick up newly
            # created resume jobs that haven't had their chain metadata synced
            # yet.  The stored chain_order can lag behind and cause incomplete
            # trees, so we never rely on it here.
            chain_order = [root_job_id]
            visited: set = {root_job_id}
            cur_id = root_job_id
            while cur_id:
                cur = await self.get_job(cur_id)
                if not cur:
                    break
                nxt = cur.metadata.get(JobMetadataKeys.RESUME_JOB_ID)
                if nxt and nxt not in visited:
                    visited.add(nxt)
                    chain_order.append(nxt)
                    cur_id = nxt
                else:
                    break

            # --- Step 3: include active retry jobs (Bug 5 fix) ---
            # A retry job is linked via job.metadata[JobMetadataKeys.RETRY_JOB_ID] on the waiting
            # job.  These are not resume steps so they don't appear in resume_job_id
            # links, but the frontend needs to see them to track retry progress.
            chain_set = set(chain_order)
            extra_retry_ids: list[str] = []
            for cid in list(chain_order):  # iterate snapshot to avoid mutation issues
                cjob = await self.get_job(cid)
                if cjob:
                    retry_id = cjob.metadata.get(JobMetadataKeys.RETRY_JOB_ID)
                    if retry_id and retry_id not in chain_set:
                        chain_set.add(retry_id)
                        extra_retry_ids.append(retry_id)

            all_job_ids = chain_order + extra_retry_ids

            # --- Step 4: load job objects ---
            jobs: list[Job] = []
            for jid in all_job_ids:
                j = await self.get_job(jid)
                if j:
                    jobs.append(j)

            # --- Step 5: determine aggregate chain status ---
            chain_status = "completed"
            for j in jobs:
                if j.status in (JobStatus.PROCESSING, JobStatus.QUEUED):
                    chain_status = "in_progress"
                    break
                elif j.status == JobStatus.WAITING_FOR_APPROVAL:
                    chain_status = "waiting_for_approval"
                    break
                elif j.status == JobStatus.FAILED:
                    chain_status = "failed"
                    break

            return {
                "root_job_id": root_job_id,
                "chain_length": len(all_job_ids),
                "chain_order": all_job_ids,
                "all_job_ids": all_job_ids,
                "chain_status": chain_status,
                "jobs": jobs,
            }
        except Exception as e:
            logger.error(f"Failed to get job chain for {job_id}: {e}")
            return _empty

    async def update_job_chain_metadata(self, job_id: str) -> None:
        """
        Update job chain metadata for a job and all jobs in its chain.

        Args:
            job_id: Job identifier
        """
        try:
            chain_data = await self.get_job_chain(job_id)
            if not chain_data["root_job_id"]:
                return

            # Update metadata for all jobs in chain
            for i, job_id_in_chain in enumerate(chain_data["chain_order"]):
                job = await self.get_job(job_id_in_chain)
                if job:
                    job.metadata[JobMetadataKeys.JOB_CHAIN] = {
                        "root_job_id": chain_data["root_job_id"],
                        "chain_length": chain_data["chain_length"],
                        "chain_order": chain_data["chain_order"],
                        "current_position": i + 1,
                        "all_job_ids": chain_data["all_job_ids"],
                        "chain_status": chain_data["chain_status"],
                    }
                    await self._save_job(job)
        except Exception as e:
            logger.error(f"Failed to update job chain metadata for {job_id}: {e}")

    async def get_job_with_subjob_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job with aggregated subjob status information.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with job and subjob status summary, or None if job not found
        """
        try:
            job = await self.get_job(job_id)
            if not job:
                return None

            # Get chain data to find all subjobs
            chain_data = await self.get_job_chain(job_id)

            # Only include subjob status if this is the root job
            if chain_data["root_job_id"] == job_id and chain_data["chain_length"] > 1:
                subjob_ids = chain_data["chain_order"][1:]  # All except root
                subjob_statuses = []

                for subjob_id in subjob_ids:
                    subjob = await self.get_job(subjob_id)
                    if subjob:
                        subjob_statuses.append(subjob.status)

                # Count statuses
                status_counts = {
                    "total": len(subjob_statuses),
                    "completed": sum(
                        1 for s in subjob_statuses if s == JobStatus.COMPLETED
                    ),
                    "pending": sum(
                        1 for s in subjob_statuses if s == JobStatus.PENDING
                    ),
                    "processing": sum(
                        1
                        for s in subjob_statuses
                        if s in [JobStatus.PROCESSING, JobStatus.QUEUED]
                    ),
                    "waiting_for_approval": sum(
                        1
                        for s in subjob_statuses
                        if s == JobStatus.WAITING_FOR_APPROVAL
                    ),
                    "failed": sum(1 for s in subjob_statuses if s == JobStatus.FAILED),
                }

                # Determine chain status
                chain_status = "all_completed"
                if status_counts["processing"] > 0:
                    chain_status = "in_progress"
                elif status_counts["waiting_for_approval"] > 0:
                    chain_status = "blocked"
                elif status_counts["failed"] > 0:
                    chain_status = "failed"
                elif status_counts["pending"] > 0:
                    chain_status = "in_progress"

                return {
                    "job": job,
                    "subjob_status": status_counts,
                    "chain_status": chain_status,
                }
            else:
                return {"job": job, "subjob_status": None, "chain_status": None}
        except Exception as e:
            logger.error(f"Failed to get job with subjob status for {job_id}: {e}")
            return None

    async def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        user_id: Optional[str] = None,
    ) -> List[Job]:
        """
        List jobs from database (primary) or Redis (fallback) with optional filters.

        Args:
            job_type: Filter by job type
            status: Filter by status
            limit: Maximum number of jobs to return
            user_id: If provided, only return jobs belonging to this user

        Returns:
            List of jobs
        """
        jobs = []

        # Try to get from database first
        try:
            from marketing_project.models.db_models import JobModel
            from marketing_project.services.database import get_database_manager

            db_manager = get_database_manager()
            if db_manager.is_initialized:
                async with db_manager.get_session() as session:
                    from sqlalchemy import desc, select

                    # Build query
                    stmt = select(JobModel)

                    # Apply filters
                    if job_type:
                        stmt = stmt.where(JobModel.job_type == job_type)
                    if status:
                        stmt = stmt.where(JobModel.status == status.value)
                    if user_id:
                        stmt = stmt.where(JobModel.user_id == user_id)

                    # Order by created_at descending and limit
                    stmt = stmt.order_by(desc(JobModel.created_at)).limit(limit)

                    result = await session.execute(stmt)
                    db_jobs = result.scalars().all()

                    # Convert to Pydantic Job models
                    for db_job in db_jobs:
                        job_dict = db_job.to_dict()
                        # Normalize datetime fields to ensure they're UTC timezone-aware
                        job_dict["created_at"] = self._normalize_datetime_to_utc(
                            job_dict.get("created_at")
                        )
                        job_dict["started_at"] = self._normalize_datetime_to_utc(
                            job_dict.get("started_at")
                        )
                        job_dict["completed_at"] = self._normalize_datetime_to_utc(
                            job_dict.get("completed_at")
                        )
                        job = Job(**job_dict)
                        jobs.append(job)
                        # Also update in-memory cache
                        self._jobs[job.id] = job

                    logger.debug(f"Loaded {len(jobs)} jobs from database")
        except Exception as e:
            logger.debug(f"Error listing jobs from database: {type(e).__name__}: {e}")

        # If no jobs from database, fall back to Redis
        if not jobs:
            try:

                async def list_operation(redis_client: redis.Redis):
                    return await redis_client.smembers(JOB_INDEX_KEY)

                job_ids = await self._redis_manager.execute(list_operation)

                # Fetch each job
                for job_id in job_ids:
                    job = await self.get_job(job_id)
                    if job:
                        jobs.append(job)

                logger.debug(f"Loaded {len(jobs)} jobs from Redis")

            except Exception as e:
                logger.error(f"Error listing jobs from Redis: {e}")
                # Fall back to in-memory
                jobs = list(self._jobs.values())
                logger.debug(f"Loaded {len(jobs)} jobs from memory (Redis unavailable)")

            # Apply filters (if not already applied from database query)
            if job_type:
                jobs = [j for j in jobs if j.type == job_type]
            if status:
                jobs = [j for j in jobs if j.status == status]
            if user_id:
                jobs = [j for j in jobs if j.user_id == user_id]

            # Sort by creation time (newest first)
            jobs.sort(key=lambda j: j.created_at, reverse=True)

            # Apply limit
            jobs = jobs[:limit]

        return jobs

    async def clear_all_arq_jobs(self) -> int:
        """
        Clear all jobs from ARQ queue by deleting ARQ keys from Redis.

        This will:
        - Clear all queued jobs
        - Clear all completed/failed job results
        - Clear deferred jobs

        Note: This does NOT affect jobs tracked in our JobManager (those are separate).
        This only clears the ARQ queue itself.

        Returns:
            Number of ARQ keys deleted
        """
        try:
            deleted_count = 0

            # ARQ stores keys with patterns like:
            # - arq:job:{job_id}
            # - arq:result:{job_id}
            # - arq:queued (list/set of queued jobs)
            # - arq:deferred (list/set of deferred jobs)
            # - arq:in_progress (list/set of in-progress jobs)
            arq_patterns = [
                "arq:job:*",
                "arq:result:*",
                "arq:queued",
                "arq:deferred",
                "arq:in_progress",
            ]

            for pattern in arq_patterns:
                try:
                    # Find all keys matching pattern
                    async def scan_operation(redis_client: redis.Redis):
                        keys = []
                        async for key in redis_client.scan_iter(match=pattern):
                            keys.append(key)
                        return keys

                    keys = await self._redis_manager.execute(scan_operation)

                    if keys:
                        # Delete in batches if there are many keys
                        batch_size = 100
                        for i in range(0, len(keys), batch_size):
                            batch = keys[i : i + batch_size]

                            async def delete_batch_operation(redis_client: redis.Redis):
                                return await redis_client.delete(*batch)

                            deleted = await self._redis_manager.execute(
                                delete_batch_operation
                            )
                            deleted_count += deleted
                        logger.info(
                            f"Deleted {len(keys)} ARQ keys matching pattern {pattern}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to delete ARQ keys matching {pattern}: {e}")

            logger.info(f"Cleared {deleted_count} ARQ jobs/keys from Redis")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to clear ARQ jobs: {e}", exc_info=True)
            raise

    async def delete_job(self, job_id: str) -> bool:
        """
        Hard-delete a single job from all storage layers.

        Removes the job from PostgreSQL, Redis, and in-memory storage.
        Unlike cancel_job, this works on jobs in any status including completed/failed.

        Args:
            job_id: ID of the job to delete

        Returns:
            True if the job was found and deleted, False if not found
        """
        found = False

        # Delete from PostgreSQL
        try:
            from marketing_project.models.db_models import JobModel
            from marketing_project.services.database import get_database_manager

            db_manager = get_database_manager()
            if db_manager.is_initialized:
                async with db_manager.get_session() as session:
                    from sqlalchemy import delete, select

                    result = await session.execute(
                        select(JobModel).where(JobModel.job_id == job_id)
                    )
                    db_job = result.scalar_one_or_none()
                    if db_job:
                        await session.execute(
                            delete(JobModel).where(JobModel.job_id == job_id)
                        )
                        await session.commit()
                        found = True
                        logger.info(f"Deleted job {job_id} from PostgreSQL")
        except Exception as e:
            logger.warning(f"Failed to delete job {job_id} from PostgreSQL: {e}")

        # Delete from Redis
        try:
            job_key = f"{JOB_KEY_PREFIX}{job_id}"

            async def delete_key_operation(redis_client: redis.Redis):
                deleted = await redis_client.delete(job_key)
                await redis_client.srem(JOB_INDEX_KEY, job_id)
                return deleted

            redis_deleted = await self._redis_manager.execute(delete_key_operation)
            if redis_deleted:
                found = True
                logger.info(f"Deleted job {job_id} from Redis")
        except Exception as e:
            logger.warning(f"Failed to delete job {job_id} from Redis: {e}")

        # Remove from in-memory cache
        if job_id in self._jobs:
            del self._jobs[job_id]
            found = True

        return found

    async def delete_all_jobs(self) -> int:
        """
        Delete all jobs from Redis and in-memory storage.

        This will:
        - Delete all job keys from Redis (job:*)
        - Clear the jobs index (jobs:index)
        - Clear in-memory job storage
        - Also clear ARQ jobs

        Returns:
            Number of jobs deleted
        """
        try:
            deleted_count = 0

            # Get all job IDs from index
            async def get_ids_operation(redis_client: redis.Redis):
                return await redis_client.smembers(JOB_INDEX_KEY)

            job_ids = await self._redis_manager.execute(get_ids_operation)
            job_ids_list = list(job_ids) if job_ids else []

            # Delete all job keys using pipeline for batch operations
            if job_ids_list:
                job_keys = [f"{JOB_KEY_PREFIX}{job_id}" for job_id in job_ids_list]
                # Delete in batches using pipeline
                batch_size = 100
                for i in range(0, len(job_keys), batch_size):
                    batch = job_keys[i : i + batch_size]

                    async def delete_batch_operation(redis_client: redis.Redis):
                        return await redis_client.delete(*batch)

                    deleted = await self._redis_manager.execute(delete_batch_operation)
                    deleted_count += deleted
                logger.info(f"Deleted {deleted_count} job keys from Redis")

            # Clear the index
            async def clear_index_operation(redis_client: redis.Redis):
                return await redis_client.delete(JOB_INDEX_KEY)

            await self._redis_manager.execute(clear_index_operation)
            logger.info("Cleared jobs index from Redis")

            # Clear in-memory storage
            in_memory_count = len(self._jobs)
            self._jobs.clear()
            logger.info(f"Cleared {in_memory_count} jobs from in-memory storage")

            # Also clear ARQ jobs
            try:
                arq_deleted = await self.clear_all_arq_jobs()
                logger.info(f"Also cleared {arq_deleted} ARQ jobs")
            except Exception as e:
                logger.warning(f"Failed to clear ARQ jobs: {e}")

            # Delete all jobs from PostgreSQL
            try:
                from sqlalchemy import delete as sa_delete

                from marketing_project.models.db_models import JobModel
                from marketing_project.services.database import get_database_manager

                db_manager = get_database_manager()
                if db_manager.is_initialized:
                    async with db_manager.get_session() as session:
                        await session.execute(sa_delete(JobModel))
                    logger.info("Deleted all jobs from PostgreSQL")
            except Exception as e:
                logger.warning(f"Failed to delete jobs from PostgreSQL: {e}")

            # Also clear approvals
            try:
                from ..services.approval_manager import get_approval_manager

                approval_manager = await get_approval_manager()
                approval_deleted = await approval_manager.delete_all_approvals()
                logger.info(f"Also cleared {approval_deleted} approvals")
            except Exception as e:
                logger.warning(f"Failed to clear approvals: {e}")

            total_deleted = deleted_count + in_memory_count
            logger.info(f"Total jobs deleted: {total_deleted}")
            return total_deleted

        except Exception as e:
            logger.error(f"Failed to delete all jobs: {e}", exc_info=True)
            raise

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: ID of the job to cancel

        Returns:
            True if job was cancelled, False if not found or already completed

        NOTE: ARQ doesn't support job cancellation once queued.
        This only updates the local status.
        """
        # Get job from Redis or memory
        job = await self.get_job(job_id)
        if not job:
            return False

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False

        pre_cancel_status = job.status
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        await self._save_job(job)

        # Attempt to abort the ARQ task so the worker doesn't continue executing
        # a job the user already cancelled.
        arq_job_id = job.metadata.get(JobMetadataKeys.ARQ_JOB_ID)
        if arq_job_id and pre_cancel_status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
            try:
                pool = await self.get_arq_pool()
                arq_job = ArqJob(arq_job_id, pool)
                await arq_job.abort(timeout=5)
                logger.info(f"Aborted ARQ job {arq_job_id} for cancelled job {job_id}")
            except Exception as e:
                # Non-fatal — job is already marked cancelled in our DB;
                # the worker will notice and stop after its next status check.
                logger.warning(
                    f"Could not abort ARQ job {arq_job_id} for {job_id}: {e}"
                )

        logger.info(f"Cancelled job {job_id} (was {pre_cancel_status.value})")
        return True

    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Purge old completed/failed/cancelled jobs from DB, Redis, and in-memory cache.

        Args:
            max_age_hours: Jobs completed more than this many hours ago are deleted.

        Returns:
            Number of jobs deleted.
        """
        from datetime import timedelta

        from marketing_project.models.db_models import JobModel
        from marketing_project.services.database import get_database_manager

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        terminal_statuses = [
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ]
        deleted_count = 0
        job_ids_to_purge: list[str] = []

        # --- PostgreSQL: find and delete old terminal jobs ---
        try:
            db_manager = get_database_manager()
            if db_manager.is_initialized:
                from sqlalchemy import delete, select

                async with db_manager.get_session() as session:
                    stmt = select(JobModel.job_id).where(
                        JobModel.status.in_(terminal_statuses),
                        JobModel.completed_at < cutoff,
                    )
                    result = await session.execute(stmt)
                    job_ids_to_purge = [row[0] for row in result.fetchall()]

                    if job_ids_to_purge:
                        await session.execute(
                            delete(JobModel).where(
                                JobModel.job_id.in_(job_ids_to_purge)
                            )
                        )
                        await session.commit()
                        deleted_count = len(job_ids_to_purge)
                        logger.info(f"Deleted {deleted_count} old jobs from PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to purge old jobs from PostgreSQL: {e}")

        # --- Redis: remove job keys and index entries ---
        if job_ids_to_purge:
            try:
                job_keys = [f"{JOB_KEY_PREFIX}{jid}" for jid in job_ids_to_purge]

                async def purge_redis(redis_client: redis.Redis):
                    async with redis_client.pipeline() as pipe:
                        pipe.delete(*job_keys)
                        pipe.srem(JOB_INDEX_KEY, *job_ids_to_purge)
                        await pipe.execute()

                await self._redis_manager.execute(purge_redis)
                logger.info(f"Removed {len(job_ids_to_purge)} old job keys from Redis")
            except Exception as e:
                logger.warning(f"Failed to purge old jobs from Redis: {e}")

        # --- In-memory cache: evict purged jobs ---
        for jid in job_ids_to_purge:
            self._jobs.pop(jid, None)

        # Also evict in-memory entries that were never in DB (edge case)
        now = datetime.now(timezone.utc)
        extra = [
            jid
            for jid, j in self._jobs.items()
            if j.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
            and j.completed_at
            and self._normalize_datetime_to_utc(j.completed_at) < cutoff
        ]
        for jid in extra:
            del self._jobs[jid]

        total = deleted_count + len(extra)
        if total:
            logger.info(f"cleanup_old_jobs: purged {total} jobs total")
        return total

    async def cleanup(self):
        """Cleanup Redis connections."""
        # RedisManager cleanup is handled globally
        # This method is here for consistency with other managers
        pass


# Global job manager instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get or create the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
