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
from datetime import datetime
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
JOB_REDIS_TTL = 7200  # Keep job state in Redis for 2 hours (longer than ARQ)
JOB_KEY_PREFIX = "job:"  # Redis key prefix for jobs
JOB_INDEX_KEY = "jobs:index"  # Redis set for job indexing


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
    created_at: datetime = Field(default_factory=datetime.utcnow)
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

    async def _save_job_to_redis(self, job: Job) -> None:
        """Helper method to save job state to Redis."""
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
    ) -> Job:
        """
        Create a new job and store in Redis.

        Args:
            job_type: Type of job (blog, release_notes, transcript, pipeline)
            content_id: ID of content being processed
            metadata: Optional metadata dictionary
            job_id: Optional job ID (if not provided, a UUID will be generated)

        Returns:
            Created job object
        """
        if job_id is None:
            job_id = str(uuid.uuid4())

        job = Job(
            id=job_id,
            type=job_type,
            content_id=content_id,
            metadata=metadata or {},
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

            logger.info(
                f"Created job {job_id} for {job_type} processing of content {content_id} "
                f"(stored in Redis with {JOB_REDIS_TTL}s TTL)"
            )

        except Exception as e:
            logger.error(f"Failed to store job {job_id} in Redis: {e}")
            # Fall back to in-memory only
            self._jobs[job_id] = job
            logger.warning(f"Job {job_id} stored in-memory only (Redis unavailable)")

        return job

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
        max_retries: int = 3,
        retry_delay: int = 30,
        **kwargs,
    ) -> str:
        """
        Submit a job to ARQ for background execution with retry support.

        Args:
            job_id: ID of the job to track
            function_name: Name of ARQ function to call (e.g., 'process_blog')
            *args: Arguments to pass to ARQ function
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay in seconds between retries (default: 30)
            **kwargs: Keyword arguments to pass to ARQ function

        Returns:
            ARQ job ID
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        try:
            # Get ARQ pool
            pool = await self.get_arq_pool()

            # Enqueue job in ARQ with retry configuration
            arq_job = await pool.enqueue_job(
                function_name,
                *args,
                _job_try=max_retries,  # Max retry attempts
                _defer_by=retry_delay,  # Delay between retries
                **kwargs,
            )

            # Update job status
            job.status = JobStatus.QUEUED
            job.metadata["arq_job_id"] = arq_job.job_id
            job.metadata["max_retries"] = max_retries
            job.metadata["retry_delay"] = retry_delay

            # Save updated job status to Redis
            await self._save_job_to_redis(job)

            logger.info(
                f"Submitted job {job_id} to ARQ as {arq_job.job_id} "
                f"(max_retries={max_retries}, retry_delay={retry_delay}s)"
            )
            return arq_job.job_id

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = f"Failed to submit to ARQ: {str(e)}"
            # Save failed status to Redis
            await self._save_job_to_redis(job)
            logger.error(f"Failed to submit job {job_id} to ARQ: {e}")
            raise

    def mark_job_started(self, job_id: str) -> None:
        """Mark a job as started (called by ARQ worker)."""
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow()
            logger.debug(f"Job {job_id} marked as started")

    def mark_job_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        """Mark a job as completed (called by ARQ worker)."""
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100
            job.result = result
            logger.info(f"Job {job_id} marked as completed")

    def mark_job_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed (called by ARQ worker)."""
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error = error
            logger.error(f"Job {job_id} marked as failed: {error}")

    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID from Redis, then poll ARQ for latest status.

        Queries ARQ for the latest status and result if job is queued/processing.
        Handles job expiration and all ARQ job states.
        """
        job = None

        # Try to get from Redis first
        try:
            job_key = f"{JOB_KEY_PREFIX}{job_id}"

            async def get_operation(redis_client: redis.Redis):
                return await redis_client.get(job_key)

            job_json = await self._redis_manager.execute(get_operation)

            if job_json:
                # Parse job from Redis
                job = Job.model_validate_json(job_json)
                # Also update in-memory cache
                self._jobs[job_id] = job
                logger.debug(f"Job {job_id} loaded from Redis")
            else:
                # Not in Redis, try memory fallback
                job = self._jobs.get(job_id)
                if job:
                    logger.debug(f"Job {job_id} found in memory (not in Redis)")

        except Exception as e:
            logger.error(
                f"Error reading job {job_id} from Redis (key: {job_key}): {type(e).__name__}: {e}",
                extra={
                    "job_id": job_id,
                    "redis_key": job_key,
                    "error_type": type(e).__name__,
                },
            )
            # Fall back to in-memory
            job = self._jobs.get(job_id)

        if not job:
            return None

        # Check if job is too old and likely expired from ARQ
        if job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
            age_seconds = (datetime.utcnow() - job.created_at).total_seconds()

            if age_seconds > JOB_MAX_AGE:
                logger.warning(
                    f"Job {job_id} is {age_seconds:.0f}s old (>{JOB_MAX_AGE}s), "
                    f"ARQ result likely expired"
                )
                job.error = (
                    f"Job exceeded maximum age ({JOB_MAX_AGE}s) - ARQ result expired"
                )
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                return job

        # If job is queued or processing, check ARQ for updates
        if job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]:
            arq_job_id = job.metadata.get("arq_job_id")
            if arq_job_id:
                try:
                    pool = await self.get_arq_pool()
                    arq_job = ArqJob(arq_job_id, pool)

                    # Get ARQ job status
                    arq_status = await arq_job.status()

                    if arq_status == ArqJobStatus.complete:
                        # Job completed - get result
                        try:
                            result = await arq_job.result()
                            job.status = JobStatus.COMPLETED
                            job.completed_at = datetime.utcnow()
                            job.progress = 100
                            job.result = result
                            job.current_step = "Completed"
                            logger.info(f"Job {job_id} completed with result from ARQ")
                            # Save updated status to Redis
                            await self._save_job_to_redis(job)

                            # If this is a resume job and it's the final subjob, copy result to original job
                            if job.type == "resume_pipeline" and job.metadata.get(
                                "original_job_id"
                            ):
                                original_job_id = job.metadata.get("original_job_id")
                                # Check if this is the final subjob (no more resume_job_id)
                                if not job.metadata.get("resume_job_id"):
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
                                        original_job.completed_at = datetime.utcnow()
                                        original_job.progress = 100
                                        original_job.current_step = "Completed"
                                        # Update metadata to indicate all subjobs are complete
                                        original_job.metadata[
                                            "all_subjobs_completed"
                                        ] = True
                                        original_job.metadata["final_subjob_id"] = (
                                            job_id
                                        )
                                        await self._save_job_to_redis(original_job)
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
                            job.completed_at = datetime.utcnow()
                            await self._save_job_to_redis(job)
                        except Exception as e:
                            # Other errors when getting result
                            logger.error(
                                f"Error getting ARQ result for job {job_id}: {e}",
                                exc_info=True,
                            )
                            job.error = f"Error retrieving job result: {str(e)}"
                            job.status = JobStatus.FAILED
                            job.completed_at = datetime.utcnow()
                            await self._save_job_to_redis(job)

                    elif arq_status == ArqJobStatus.in_progress:
                        job.status = JobStatus.PROCESSING
                        if not job.started_at:
                            job.started_at = datetime.utcnow()
                        logger.debug(f"Job {job_id} is in progress")
                        await self._save_job_to_redis(job)

                    elif arq_status == ArqJobStatus.queued:
                        # Job is queued and waiting for a worker
                        job.status = JobStatus.QUEUED
                        if (
                            not job.current_step
                            or job.current_step == "Retrying after failure"
                        ):
                            job.current_step = "Waiting for worker"
                        logger.debug(f"Job {job_id} is queued in ARQ")
                        await self._save_job_to_redis(job)

                    elif arq_status == ArqJobStatus.deferred:
                        # Job is waiting to retry after failure
                        job.status = JobStatus.QUEUED
                        job.current_step = "Retrying after failure"
                        logger.info(f"Job {job_id} is deferred (will retry)")
                        await self._save_job_to_redis(job)

                    elif arq_status == ArqJobStatus.not_found:
                        # ARQ job not found - might have expired or been cleaned up
                        logger.warning(
                            f"ARQ job {arq_job_id} for {job_id} not found - "
                            f"result may have expired (ARQ TTL: {ARQ_RESULT_TTL}s)"
                        )
                        job.error = "Job not found in ARQ - result may have expired"
                        job.status = JobStatus.FAILED
                        job.completed_at = datetime.utcnow()
                        await self._save_job_to_redis(job)

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
        # Get job (from Redis if available)
        job = await self.get_job(job_id)
        if job:
            job.progress = progress
            if current_step:
                job.current_step = current_step
            await self._save_job_to_redis(job)

    async def update_job_status(self, job_id: str, status: JobStatus) -> None:
        """
        Update job status and save to Redis.

        Args:
            job_id: ID of the job to update
            status: New job status
        """
        job = await self.get_job(job_id)
        if job:
            job.status = status
            # Keep metadata.status in sync with job.status for frontend compatibility
            job.metadata["status"] = status.value
            if status == JobStatus.WAITING_FOR_APPROVAL:
                # Mark as completed time if waiting for approval (job is "done" but waiting)
                job.completed_at = datetime.utcnow()
            await self._save_job_to_redis(job)
            logger.info(f"Updated job {job_id} status to {status.value}")

            # If this is a subjob, update parent job status
            if job.metadata.get("original_job_id"):
                await self.update_parent_job_status(job.metadata.get("original_job_id"))

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
                    parent_job.completed_at = datetime.utcnow()
                    parent_job.current_step = "All subjobs completed"
                    parent_job.progress = 100
                    parent_job.metadata["status"] = "all_subjobs_completed"
                    await self._save_job_to_redis(parent_job)
                    logger.info(
                        f"Updated parent job {parent_job_id} status to completed (all subjobs done)"
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
                    await self._save_job_to_redis(parent_job)
                    logger.info(
                        f"Updated parent job {parent_job_id} status to processing ({processing_count}/{total_count} subjobs)"
                    )
            elif any(s == JobStatus.WAITING_FOR_APPROVAL for s in subjob_statuses):
                # At least one subjob is waiting for approval
                waiting_count = sum(
                    1 for s in subjob_statuses if s == JobStatus.WAITING_FOR_APPROVAL
                )
                parent_job.current_step = (
                    f"Waiting for Approval (Subjob {waiting_count})"
                )
                if parent_job.status != JobStatus.WAITING_FOR_APPROVAL:
                    parent_job.status = JobStatus.WAITING_FOR_APPROVAL
                    parent_job.metadata["status"] = "waiting_for_subjob_approval"
                    await self._save_job_to_redis(parent_job)
                    logger.info(
                        f"Updated parent job {parent_job_id} status to waiting_for_approval (subjob {waiting_count})"
                    )
            elif any(s == JobStatus.FAILED for s in subjob_statuses):
                # At least one subjob failed
                failed_count = sum(1 for s in subjob_statuses if s == JobStatus.FAILED)
                parent_job.current_step = f"Failed (Subjob {failed_count} failed)"
                if parent_job.status != JobStatus.FAILED:
                    parent_job.status = JobStatus.FAILED
                    parent_job.metadata["status"] = "failed_with_subjobs"
                    await self._save_job_to_redis(parent_job)
                    logger.info(
                        f"Updated parent job {parent_job_id} status to failed ({failed_count} subjobs failed)"
                    )
        except Exception as e:
            logger.error(f"Failed to update parent job status for {parent_job_id}: {e}")

    async def get_job_chain(self, job_id: str) -> Dict[str, Any]:
        """
        Get complete job chain hierarchy for a given job.

        Returns the root job, all subjobs in order, and chain metadata.

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
        try:
            # Find root job
            root_job_id = job_id
            current_job = await self.get_job(job_id)
            if not current_job:
                return {
                    "root_job_id": None,
                    "chain_length": 0,
                    "chain_order": [],
                    "all_job_ids": [],
                    "chain_status": "unknown",
                    "jobs": [],
                }

            # Traverse up to find root
            while current_job.metadata.get("original_job_id"):
                root_job_id = current_job.metadata.get("original_job_id")
                current_job = await self.get_job(root_job_id)
                if not current_job:
                    break

            # Build chain order by following resume_job_id links
            chain_order = [root_job_id]
            all_job_ids = [root_job_id]
            jobs = [await self.get_job(root_job_id)]

            current_job_id = root_job_id
            visited = set([root_job_id])

            while current_job_id:
                current_job = await self.get_job(current_job_id)
                if not current_job:
                    break

                resume_job_id = current_job.metadata.get("resume_job_id")
                if resume_job_id and resume_job_id not in visited:
                    visited.add(resume_job_id)
                    chain_order.append(resume_job_id)
                    all_job_ids.append(resume_job_id)
                    resume_job = await self.get_job(resume_job_id)
                    if resume_job:
                        jobs.append(resume_job)
                    current_job_id = resume_job_id
                else:
                    break

            # Determine chain status
            chain_status = "completed"
            for job in jobs:
                if job.status == JobStatus.PROCESSING or job.status == JobStatus.QUEUED:
                    chain_status = "in_progress"
                    break
                elif job.status == JobStatus.WAITING_FOR_APPROVAL:
                    chain_status = "waiting_for_approval"
                    break
                elif job.status == JobStatus.FAILED:
                    chain_status = "failed"
                    break

            return {
                "root_job_id": root_job_id,
                "chain_length": len(chain_order),
                "chain_order": chain_order,
                "all_job_ids": all_job_ids,
                "chain_status": chain_status,
                "jobs": jobs,
            }
        except Exception as e:
            logger.error(f"Failed to get job chain for {job_id}: {e}")
            return {
                "root_job_id": None,
                "chain_length": 0,
                "chain_order": [],
                "all_job_ids": [],
                "chain_status": "unknown",
                "jobs": [],
            }

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
                    job.metadata["job_chain"] = {
                        "root_job_id": chain_data["root_job_id"],
                        "chain_length": chain_data["chain_length"],
                        "chain_order": chain_data["chain_order"],
                        "current_position": i + 1,
                        "all_job_ids": chain_data["all_job_ids"],
                        "chain_status": chain_data["chain_status"],
                    }
                    await self._save_job_to_redis(job)
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
    ) -> List[Job]:
        """
        List jobs from Redis with optional filters.

        Args:
            job_type: Filter by job type
            status: Filter by status
            limit: Maximum number of jobs to return

        Returns:
            List of jobs
        """
        jobs = []

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

        # Apply filters
        if job_type:
            jobs = [j for j in jobs if j.type == job_type]
        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Apply limit
        return jobs[:limit]

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

        # Note: ARQ doesn't support cancelling queued jobs
        # We can only mark it as cancelled in our tracking
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()

        # Save to Redis
        await self._save_job_to_redis(job)

        logger.warning(
            f"Marked job {job_id} as cancelled (ARQ worker may still process it)"
        )
        return True

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed/failed jobs.

        Args:
            max_age_hours: Maximum age in hours for completed jobs

        Returns:
            Number of jobs cleaned up
        """
        from datetime import timedelta

        now = datetime.utcnow()
        cutoff = now - timedelta(hours=max_age_hours)

        jobs_to_remove = []
        for job_id, job in self._jobs.items():
            if job.status in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ]:
                if job.completed_at and job.completed_at < cutoff:
                    jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self._jobs[job_id]

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")

        return len(jobs_to_remove)

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
