"""
Step Result Manager for storing and retrieving pipeline step results.

This service handles:
- Saving step results to disk organized by job_id
- Retrieving step results for display
- Managing file downloads
- Cleaning up old results
"""

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for datetime and other non-serializable objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # Handle Pydantic BaseModel instances
    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump(mode="json")
        except (TypeError, ValueError):
            # Fallback to regular model_dump if mode='json' fails
            return obj.model_dump()
    raise TypeError(f"Type {type(obj)} not serializable")


def _step_model_to_step_dict(s: Any) -> Dict[str, Any]:
    """Convert a StepResultModel ORM row to the step-dict format used by get_job_results()."""
    result_json = s.result if isinstance(s.result, (dict, list)) else {}
    file_size = len(json.dumps(result_json))
    filename = f"{s.step_number:02d}_{s.step_name.lower().replace(' ', '_')}.json"
    return {
        "job_id": s.job_id,
        "root_job_id": s.root_job_id,
        "execution_context_id": s.execution_context_id,
        "filename": filename,
        "step_number": s.step_number,
        "step_name": s.step_name,
        "timestamp": s.created_at.isoformat() if s.created_at else None,
        "has_result": s.result is not None,
        "file_size": file_size,
        "execution_time": float(s.execution_time) if s.execution_time else None,
        "tokens_used": s.tokens_used,
        "status": s.status,
        "error_message": s.error_message,
    }


class StepResultManager:
    """
    Manages storage and retrieval of pipeline step results.

    Directory structure (new with execution contexts):
    results/
      {root_job_id}/
        context_0/              # Initial execution
          00_input.json
          01_seo_keywords.json
          ...
        context_1/              # First resume after approval
          02_marketing_brief.json
          03_article_generation.json
          ...
        context_2/              # Second resume after approval
          ...
        metadata.json           # Job metadata

    Old structure (backward compatible):
    results/
      {job_id}/
        00_input.json
        01_seo_keywords.json
        ...
        metadata.json
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the step result manager.

        Args:
            base_dir: Base directory for storing results (default: ./results)
        """
        self.base_dir = Path(base_dir or os.getenv("STEP_RESULTS_DIR", "results"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Check if S3 should be used
        s3_bucket = os.getenv("AWS_S3_BUCKET")
        s3_prefix = os.getenv("STEP_RESULTS_S3_PREFIX", "results/")
        self._use_s3 = s3_bucket is not None

        if self._use_s3:
            try:
                from marketing_project.services.s3_storage import S3Storage

                self.s3_storage = S3Storage(bucket_name=s3_bucket, prefix=s3_prefix)
                if not self.s3_storage.is_available():
                    logger.warning(
                        "S3 bucket configured but not available, falling back to local filesystem"
                    )
                    self._use_s3 = False
                    self.s3_storage = None
                else:
                    logger.info(
                        f"Step Result Manager using S3: s3://{s3_bucket}/{s3_prefix}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize S3 storage, falling back to local: {e}"
                )
                self._use_s3 = False
                self.s3_storage = None
        else:
            self.s3_storage = None

        storage_type = "S3" if self._use_s3 else "local filesystem"
        logger.info(
            f"Step Result Manager initialized with {storage_type}, base_dir: {self.base_dir}"
        )

    def _get_job_dir(self, job_id: str) -> Path:
        """Get the directory for a specific job."""
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _get_step_filename(self, step_number: int, step_name: str) -> str:
        """
        Generate a filename for a step result.

        Args:
            step_number: Step sequence number (0-8)
            step_name: Human-readable step name

        Returns:
            Filename like "01_seo_keywords.json"
        """
        # Sanitize step name for filename
        safe_name = step_name.lower().replace(" ", "_").replace("-", "_")
        return f"{step_number:02d}_{safe_name}.json"

    def _get_s3_key(
        self, root_job_id: str, execution_context_id: str, filename: str
    ) -> str:
        """Get the S3 key for a step result file."""
        return f"{root_job_id}/context_{execution_context_id}/{filename}"

    def _get_metadata_s3_key(self, job_id: str) -> str:
        """Get the S3 key for job metadata."""
        return f"{job_id}/metadata.json"

    async def save_step_result(
        self,
        job_id: str,
        step_number: int,
        step_name: str,
        result_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        execution_context_id: Optional[str] = None,
        root_job_id: Optional[str] = None,
        input_snapshot: Optional[Dict[str, Any]] = None,
        context_keys_used: Optional[List[str]] = None,
        relative_step_number: Optional[int] = None,
    ) -> str:
        """
        Save a step result to disk.

        Args:
            job_id: Job identifier (current job executing the step)
            step_number: Step sequence number (0 for input, 1-8 for pipeline steps, 9 for final)
            step_name: Human-readable step name
            result_data: The result data to save (will be JSON serialized)
            metadata: Optional metadata to include in the file
            execution_context_id: Optional execution context ID (for tracking resume cycles)
            root_job_id: Optional root job ID (if different from job_id, saves to root directory)
            input_snapshot: Optional snapshot of input context used for this step
            context_keys_used: Optional list of context keys consumed by this step
            relative_step_number: Optional relative step number within execution context (1-indexed)

        Returns:
            Path to the saved file
        """
        try:
            from marketing_project.services.job_manager import get_job_manager

            # Determine where to save: root job directory if provided, otherwise current job
            target_job_id = root_job_id or job_id

            # Get job to determine parentid and other metadata
            job_manager = get_job_manager()
            job = None
            parent_job_id = None

            # If no root_job_id provided, try to find it from job metadata
            if not root_job_id:
                job = await job_manager.get_job(job_id)
                if job:
                    # Check if this is a subjob and find root
                    original_job_id = job.metadata.get("original_job_id")
                    if original_job_id:
                        # Find root by traversing up
                        current = job
                        while current.metadata.get("original_job_id"):
                            parent_id = current.metadata.get("original_job_id")
                            current = await job_manager.get_job(parent_id)
                            if not current:
                                break
                            if not current.metadata.get("original_job_id"):
                                target_job_id = current.id
                                break
                        else:
                            target_job_id = original_job_id
            else:
                # Get job if we already have root_job_id
                job = await job_manager.get_job(job_id)

            # Determine parentid for resume_pipeline jobs
            if job and job.type == "resume_pipeline":
                parent_job_id = job.metadata.get("original_job_id")

            # Get or create execution context ID
            if not execution_context_id:
                execution_context_id = await self._get_or_create_execution_context(
                    target_job_id, job_id, metadata
                )

            # Create context directory structure: results/{root_job_id}/context_{n}/
            root_job_dir = self._get_job_dir(target_job_id)
            context_dir = root_job_dir / f"context_{execution_context_id}"
            context_dir.mkdir(parents=True, exist_ok=True)

            filename = self._get_step_filename(step_number, step_name)
            file_path = context_dir / filename

            # Prepare data structure
            data = {
                "job_id": job_id,  # Job that executed this step
                "root_job_id": target_job_id,  # Root job this belongs to
                "parentid": parent_job_id,  # Parent job ID for resume steps
                "execution_context_id": execution_context_id,
                "step_number": step_number,  # Absolute step number (continues sequence)
                "relative_step_number": relative_step_number,  # Relative to context (1-indexed)
                "step_name": step_name,
                "timestamp": datetime.utcnow().isoformat(),
                "input_snapshot": input_snapshot,  # What was used as input
                "context_keys_used": context_keys_used or [],  # Which keys consumed
                "result": result_data,
                "metadata": metadata or {},
            }

            # Save to S3 or local filesystem
            if self._use_s3 and self.s3_storage:
                try:
                    s3_key = self._get_s3_key(
                        target_job_id, execution_context_id, filename
                    )
                    await self.s3_storage.upload_json(data, s3_key)
                    logger.debug(f"Uploaded step result to S3: {s3_key}")
                except Exception as e:
                    logger.warning(
                        f"Failed to upload step result to S3, falling back to local: {e}"
                    )
                    # Fallback to local
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(
                            data,
                            f,
                            indent=2,
                            ensure_ascii=False,
                            default=_json_serializer,
                        )
            else:
                # Save to local filesystem
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(
                        data, f, indent=2, ensure_ascii=False, default=_json_serializer
                    )

            # Also register with context registry for zero data loss context management
            try:
                from marketing_project.services.context_registry import (
                    get_context_registry,
                )

                context_registry = get_context_registry()
                await context_registry.register_step_output(
                    job_id=job_id,
                    step_name=step_name,
                    step_number=step_number,
                    output_data=result_data,
                    input_snapshot=input_snapshot,
                    context_keys_used=context_keys_used,
                    execution_context_id=execution_context_id,
                    root_job_id=target_job_id,
                )
                logger.debug(f"Registered step output in context registry: {step_name}")
            except Exception as e:
                # Log but don't fail if context registry registration fails
                logger.warning(f"Failed to register in context registry: {e}")

            # DB write (upsert — handles retries gracefully)
            try:
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                from marketing_project.models.db_models import StepResultModel
                from marketing_project.services.database import get_database_manager

                db_manager = get_database_manager()
                if db_manager.is_initialized:
                    result_for_db = result_data
                    if isinstance(result_data, BaseModel):
                        try:
                            result_for_db = result_data.model_dump(mode="json")
                        except Exception:
                            result_for_db = result_data.model_dump()
                    elif not isinstance(result_data, (dict, list, type(None))):
                        result_for_db = None

                    _meta = metadata or {}
                    _exec_time = _meta.get("execution_time")
                    values = dict(
                        job_id=job_id,
                        root_job_id=target_job_id,
                        execution_context_id=(
                            str(execution_context_id)
                            if execution_context_id is not None
                            else None
                        ),
                        step_number=step_number,
                        relative_step_number=relative_step_number,
                        step_name=step_name,
                        status=_meta.get("status", "success"),
                        result=result_for_db,
                        input_snapshot=input_snapshot,
                        context_keys_used=context_keys_used,
                        execution_time=(
                            str(_exec_time) if _exec_time is not None else None
                        ),
                        tokens_used=_meta.get("tokens_used"),
                        error_message=_meta.get("error_message"),
                    )
                    stmt = (
                        pg_insert(StepResultModel)
                        .values(**values)
                        .on_conflict_do_update(
                            constraint="uq_step_results_job_step",
                            set_={
                                k: v
                                for k, v in values.items()
                                if k not in ("job_id", "step_name")
                            },
                        )
                    )
                    async with db_manager.get_session() as session:
                        await session.execute(stmt)
                    logger.debug(
                        f"Wrote step result to DB: {step_name} for job {job_id}"
                    )
            except Exception as e:
                logger.warning(f"Failed to write step result to DB (non-fatal): {e}")

            logger.info(
                f"Saved step result: {filename} for job {job_id} (context: {execution_context_id}, root: {target_job_id})"
            )
            return str(file_path)

        except Exception as e:
            logger.error(
                f"Failed to save step result for job {job_id}, step {step_number}: {e}"
            )
            raise

    async def save_job_metadata(
        self,
        job_id: str,
        content_type: str,
        content_id: str,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save job metadata.

        Args:
            job_id: Job identifier
            content_type: Type of content (blog_post, release_notes, transcript)
            content_id: Content identifier
            started_at: Job start timestamp
            completed_at: Job completion timestamp
            additional_metadata: Additional metadata to store

        Returns:
            Path to the metadata file
        """
        try:
            job_dir = self._get_job_dir(job_id)
            metadata_path = job_dir / "metadata.json"

            metadata = {
                "job_id": job_id,
                "content_type": content_type,
                "content_id": content_id,
                "started_at": started_at.isoformat() if started_at else None,
                "completed_at": completed_at.isoformat() if completed_at else None,
                "created_at": datetime.utcnow().isoformat(),
                **(additional_metadata or {}),
            }

            # Save to S3 or local filesystem
            if self._use_s3 and self.s3_storage:
                try:
                    s3_key = self._get_metadata_s3_key(job_id)
                    await self.s3_storage.upload_json(metadata, s3_key)
                    logger.debug(f"Uploaded job metadata to S3: {s3_key}")
                except Exception as e:
                    logger.warning(
                        f"Failed to upload metadata to S3, falling back to local: {e}"
                    )
                    # Fallback to local
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
            else:
                # Save to local filesystem
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved job metadata for job {job_id}")
            return str(metadata_path)

        except Exception as e:
            logger.error(f"Failed to save job metadata for {job_id}: {e}")
            raise

    async def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """
        Get all results for a specific job, including subjobs and performance metrics.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with job metadata, all step results, subjobs, and performance metrics
        """
        try:
            from marketing_project.services.job_manager import get_job_manager

            job_manager = get_job_manager()
            job = await job_manager.get_job(job_id)

            if not job:
                raise FileNotFoundError(f"Job {job_id} not found")

            # Find related jobs (parent and subjobs)
            related_jobs = await self.find_related_jobs(job_id)
            parent_job_id = related_jobs.get("parent_job_id")
            subjob_ids = related_jobs.get("subjob_ids", [])

            # Collect all job IDs to aggregate steps from
            job_ids_to_aggregate = [job_id]
            if subjob_ids:
                job_ids_to_aggregate.extend(subjob_ids)

            # Find root job for loading steps from context directories
            root_job_id = job_id
            if job.metadata.get("original_job_id"):
                # Find root by traversing up
                current = job
                while current.metadata.get("original_job_id"):
                    parent_id = current.metadata.get("original_job_id")
                    current = await job_manager.get_job(parent_id)
                    if not current:
                        break
                    if not current.metadata.get("original_job_id"):
                        root_job_id = current.id
                        break
                else:
                    root_job_id = job.metadata.get("original_job_id")

            # Try to get results from step files first
            # IMPORTANT: Load ALL steps from root job's context directories
            # All steps from all execution contexts are saved to the root job directory
            steps = []
            metadata = {}

            # DB-first: load steps from step_results table
            try:
                from sqlalchemy import select as sa_select

                from marketing_project.models.db_models import StepResultModel
                from marketing_project.services.database import get_database_manager

                _db_manager = get_database_manager()
                if _db_manager.is_initialized:
                    async with _db_manager.get_session() as _session:
                        _q = (
                            sa_select(StepResultModel)
                            .where(StepResultModel.root_job_id == root_job_id)
                            .order_by(StepResultModel.step_number)
                        )
                        _result = await _session.execute(_q)
                        _db_steps = _result.scalars().all()
                    if _db_steps:
                        steps = [_step_model_to_step_dict(s) for s in _db_steps]
                        logger.info(
                            f"Loaded {len(steps)} steps from DB for root job {root_job_id}"
                        )
            except Exception as _e:
                logger.warning(f"DB step load failed, falling back to filesystem: {_e}")

            # Try S3 first if enabled (skipped when DB already populated steps)
            if not steps and self._use_s3 and self.s3_storage:
                try:
                    # Load metadata from S3
                    metadata_s3_key = self._get_metadata_s3_key(root_job_id)
                    full_metadata_key = self.s3_storage._get_s3_key(metadata_s3_key)
                    if await self.s3_storage.file_exists(full_metadata_key):
                        content = await self.s3_storage.get_file_content(
                            full_metadata_key
                        )
                        if content:
                            metadata = json.loads(content.decode("utf-8"))

                    # List all step files for this job
                    search_prefix = f"{root_job_id}/context_"
                    keys = await self.s3_storage.list_files(prefix=search_prefix)

                    # Group by context directory
                    context_files = {}
                    for key in keys:
                        if not key.endswith(".json") or key.endswith("metadata.json"):
                            continue

                        # Remove S3Storage prefix to get relative path
                        relative_key = key
                        if self.s3_storage.prefix and key.startswith(
                            self.s3_storage.prefix
                        ):
                            relative_key = key[len(self.s3_storage.prefix) :]

                        # Extract context_id from key: {root_job_id}/context_{id}/{filename}
                        if "/context_" in relative_key:
                            parts = relative_key.split("/context_")
                            if len(parts) > 1:
                                context_id_part = parts[1]  # {id}/{filename}
                                context_parts = context_id_part.split("/")
                                if len(context_parts) >= 2:
                                    context_id = context_parts[0]
                                    filename = context_parts[1]
                                    if context_id not in context_files:
                                        context_files[context_id] = []
                                    context_files[context_id].append(
                                        (relative_key, filename)
                                    )

                    # Load all steps from all contexts
                    for context_id in sorted(
                        context_files.keys(),
                        key=lambda x: int(x) if x.isdigit() else -1,
                    ):
                        for relative_key, filename in sorted(context_files[context_id]):
                            try:
                                full_s3_key = self.s3_storage._get_s3_key(relative_key)
                                content = await self.s3_storage.get_file_content(
                                    full_s3_key
                                )
                                if content:
                                    step_data = json.loads(content.decode("utf-8"))
                                    steps.append(
                                        {
                                            "job_id": step_data.get("job_id", job_id),
                                            "root_job_id": step_data.get(
                                                "root_job_id", root_job_id
                                            ),
                                            "execution_context_id": step_data.get(
                                                "execution_context_id", context_id
                                            ),
                                            "filename": filename,
                                            "step_number": step_data.get("step_number"),
                                            "step_name": step_data.get("step_name"),
                                            "timestamp": step_data.get("timestamp"),
                                            "has_result": "result" in step_data,
                                            "file_size": len(content),
                                            "execution_time": step_data.get(
                                                "metadata", {}
                                            ).get("execution_time"),
                                            "tokens_used": step_data.get(
                                                "metadata", {}
                                            ).get("tokens_used"),
                                            "status": step_data.get("metadata", {}).get(
                                                "status", "success"
                                            ),
                                            "error_message": step_data.get(
                                                "metadata", {}
                                            ).get("error_message"),
                                        }
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to load step from S3 {s3_key}: {e}"
                                )

                    if steps:
                        logger.info(
                            f"Loaded {len(steps)} steps from S3 for root job {root_job_id}"
                        )
                except Exception as e:
                    logger.warning(f"Error loading from S3, trying local: {e}")

            root_job_dir = self._get_job_dir(root_job_id)

            if root_job_dir.exists() and not steps:
                # Load metadata from file
                metadata_path = root_job_dir / "metadata.json"
                if metadata_path.exists() and not metadata:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                # Check for context directories (new structure)
                context_dirs = sorted(
                    [
                        d
                        for d in root_job_dir.iterdir()
                        if d.is_dir() and d.name.startswith("context_")
                    ]
                )

                if context_dirs:
                    # New structure: load ALL steps from ALL context directories
                    # This gives us all steps from all execution contexts (initial + all resumes)
                    for context_dir in context_dirs:
                        context_id = context_dir.name.replace("context_", "")
                        step_files = sorted([f for f in context_dir.glob("*.json")])

                        for step_file in step_files:
                            with open(step_file, "r", encoding="utf-8") as f:
                                step_data = json.load(f)
                                steps.append(
                                    {
                                        "job_id": step_data.get(
                                            "job_id", job_id
                                        ),  # The job that executed this step
                                        "root_job_id": step_data.get(
                                            "root_job_id", root_job_id
                                        ),
                                        "execution_context_id": step_data.get(
                                            "execution_context_id", context_id
                                        ),
                                        "filename": step_file.name,
                                        "step_number": step_data.get("step_number"),
                                        "step_name": step_data.get("step_name"),
                                        "timestamp": step_data.get("timestamp"),
                                        "has_result": "result" in step_data,
                                        "file_size": step_file.stat().st_size,
                                        "execution_time": step_data.get(
                                            "metadata", {}
                                        ).get("execution_time"),
                                        "tokens_used": step_data.get(
                                            "metadata", {}
                                        ).get("tokens_used"),
                                        "status": step_data.get("metadata", {}).get(
                                            "status", "success"
                                        ),
                                        "error_message": step_data.get(
                                            "metadata", {}
                                        ).get("error_message"),
                                    }
                                )
                    logger.info(
                        f"Loaded {len(steps)} steps from {len(context_dirs)} context directories for root job {root_job_id}"
                    )
                else:
                    # Old structure: load from job directory directly (backward compatibility)
                    step_files = sorted(
                        [
                            f
                            for f in root_job_dir.glob("*.json")
                            if f.name != "metadata.json"
                        ]
                    )
                    for step_file in step_files:
                        with open(step_file, "r", encoding="utf-8") as f:
                            step_data = json.load(f)
                            steps.append(
                                {
                                    "job_id": step_data.get("job_id", job_id),
                                    "root_job_id": root_job_id,
                                    "execution_context_id": step_data.get(
                                        "execution_context_id", "0"
                                    ),
                                    "filename": step_file.name,
                                    "step_number": step_data.get("step_number"),
                                    "step_name": step_data.get("step_name"),
                                    "timestamp": step_data.get("timestamp"),
                                    "has_result": "result" in step_data,
                                    "file_size": step_file.stat().st_size,
                                    "execution_time": step_data.get("metadata", {}).get(
                                        "execution_time"
                                    )
                                    or step_data.get("execution_time"),
                                    "tokens_used": step_data.get("metadata", {}).get(
                                        "tokens_used"
                                    )
                                    or step_data.get("tokens_used"),
                                    "status": step_data.get("metadata", {}).get(
                                        "status"
                                    )
                                    or step_data.get("status", "success"),
                                    "error_message": step_data.get("metadata", {}).get(
                                        "error_message"
                                    )
                                    or step_data.get("error_message"),
                                }
                            )

            # Always try to extract from job.result if it exists
            # Note: For original jobs with subjobs, we'll aggregate steps from all subjobs below
            # This extraction is mainly for jobs without subjobs or for the original job's initial steps
            if job.result and not subjob_ids:
                # Only extract from result if there are no subjobs (subjobs will be aggregated separately)
                steps_from_result = await self.extract_step_info_from_job_result(job_id)
                if steps_from_result:
                    if not steps:
                        # No step files, use result steps
                        steps = steps_from_result
                    else:
                        # Merge: add result steps that aren't already in file steps
                        existing_step_names = {
                            (s.get("step_name"), s.get("job_id")) for s in steps
                        }
                        for step in steps_from_result:
                            step_key = (step.get("step_name"), step.get("job_id"))
                            if step_key not in existing_step_names:
                                steps.append(step)
            elif job.result and subjob_ids:
                # For jobs with subjobs, we still want to extract initial steps from the original job
                # but we'll prioritize steps from subjobs (which contain the complete pipeline)
                steps_from_result = await self.extract_step_info_from_job_result(job_id)
                if steps_from_result and not steps:
                    # Only use result steps if we have no step files
                    # The subjobs will provide the complete step list
                    steps = steps_from_result

            # Aggregate steps from subjobs
            # Since all steps are saved to root job's context directories, we should already have them from above
            # But if we don't have enough steps (steps not saved to disk yet), try extracting from subjob results
            if subjob_ids and len(steps) < 3:
                # Steps might not be saved to disk yet, try extracting from job.result
                logger.info(
                    f"Only found {len(steps)} steps in context directories, trying to extract from subjob results"
                )
                for subjob_id in subjob_ids:
                    try:
                        subjob = await job_manager.get_job(subjob_id)
                        if subjob and subjob.result:
                            subjob_steps_from_result = (
                                await self.extract_step_info_from_job_result(subjob_id)
                            )
                            # Add subjob steps, avoiding duplicates (use execution_context_id in key)
                            existing_step_keys = {
                                (
                                    s.get("step_name"),
                                    s.get("job_id"),
                                    s.get("step_number"),
                                    s.get("execution_context_id"),
                                )
                                for s in steps
                            }
                            for step in subjob_steps_from_result:
                                step_key = (
                                    step.get("step_name"),
                                    step.get("job_id"),
                                    step.get("step_number"),
                                    step.get("execution_context_id"),
                                )
                                if step_key not in existing_step_keys:
                                    steps.append(step)
                                    existing_step_keys.add(step_key)
                                    logger.debug(
                                        f"Added step {step.get('step_name')} from subjob {subjob_id} (context {step.get('execution_context_id')})"
                                    )
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract steps from subjob {subjob_id}: {e}"
                        )

                # Also try aggregate_steps_from_jobs as a fallback (it will also look at root job's context dirs)
                if len(steps) < 3:
                    logger.info("Trying aggregate_steps_from_jobs as fallback")
                    subjob_steps = await self.aggregate_steps_from_jobs(subjob_ids)
                    # Add subjob steps, avoiding duplicates (use execution_context_id in key to allow same step in different contexts)
                    existing_step_keys = {
                        (
                            s.get("step_name"),
                            s.get("job_id"),
                            s.get("step_number"),
                            s.get("execution_context_id"),
                        )
                        for s in steps
                    }
                    for step in subjob_steps:
                        step_key = (
                            step.get("step_name"),
                            step.get("job_id"),
                            step.get("step_number"),
                            step.get("execution_context_id"),
                        )
                        if step_key not in existing_step_keys:
                            steps.append(step)
                            existing_step_keys.add(step_key)
                            logger.debug(
                                f"Added step {step.get('step_name')} from aggregate_steps_from_jobs (context {step.get('execution_context_id')})"
                            )

                logger.info(
                    f"After aggregating from subjobs, total steps: {len(steps)}"
                )

            # Sort steps by timestamp
            steps.sort(key=lambda x: x.get("timestamp", ""))

            # Extract performance metrics from job.result
            performance_metrics = {}
            quality_warnings = []

            if job.result and isinstance(job.result, dict):
                result_metadata = job.result.get("metadata", {})
                performance_metrics = {
                    "execution_time_seconds": result_metadata.get(
                        "execution_time_seconds"
                    ),
                    "total_tokens_used": result_metadata.get("total_tokens_used"),
                    "step_info": result_metadata.get("step_info", []),
                }
                quality_warnings = job.result.get("quality_warnings", [])

            # Also aggregate performance metrics from subjobs
            chain_metrics = {
                "total_execution_time_seconds": performance_metrics.get(
                    "execution_time_seconds", 0
                )
                or 0,
                "total_tokens_used": performance_metrics.get("total_tokens_used", 0)
                or 0,
                "total_steps": len(steps),
                "jobs_in_chain": 1 + len(subjob_ids),
                "average_time_per_step": 0,
                "estimated_cost_usd": 0,
            }

            if subjob_ids:
                for subjob_id in subjob_ids:
                    subjob_job = await job_manager.get_job(subjob_id)
                    if (
                        subjob_job
                        and subjob_job.result
                        and isinstance(subjob_job.result, dict)
                    ):
                        subjob_metadata = subjob_job.result.get("metadata", {})
                        subjob_exec_time = subjob_metadata.get("execution_time_seconds")
                        subjob_tokens = subjob_metadata.get("total_tokens_used")

                        if subjob_exec_time:
                            performance_metrics["execution_time_seconds"] = (
                                performance_metrics.get("execution_time_seconds") or 0
                            ) + subjob_exec_time
                            chain_metrics["total_execution_time_seconds"] = (
                                chain_metrics.get("total_execution_time_seconds") or 0
                            ) + subjob_exec_time
                        if subjob_tokens:
                            performance_metrics["total_tokens_used"] = (
                                performance_metrics.get("total_tokens_used") or 0
                            ) + subjob_tokens
                            chain_metrics["total_tokens_used"] = (
                                chain_metrics.get("total_tokens_used") or 0
                            ) + subjob_tokens

                        subjob_warnings = subjob_job.result.get("quality_warnings", [])
                        if subjob_warnings:
                            quality_warnings.extend(subjob_warnings)

            # Calculate chain-level metrics
            if chain_metrics["total_steps"] > 0:
                total_exec_time = chain_metrics.get("total_execution_time_seconds") or 0
                chain_metrics["average_time_per_step"] = (
                    total_exec_time / chain_metrics["total_steps"]
                )

            # Estimate cost (rough calculation: $0.01 per 1K tokens for gpt-5.1)
            # This is a simplified estimate - actual costs vary by model
            total_tokens = chain_metrics.get("total_tokens_used") or 0
            chain_metrics["estimated_cost_usd"] = (total_tokens / 1000) * 0.01

            # Add chain metrics to performance_metrics
            performance_metrics["chain_metrics"] = chain_metrics

            # Update metadata with job information
            if not metadata:
                metadata = {}

            # Get content type - use original_content_type for resume_pipeline jobs
            content_type = job.type
            if job.type == "resume_pipeline" and job.metadata.get(
                "original_content_type"
            ):
                content_type = job.metadata.get("original_content_type")

            metadata.update(
                {
                    "job_id": job_id,
                    "content_type": content_type,
                    "content_id": job.content_id,
                    "started_at": (
                        job.started_at.isoformat() if job.started_at else None
                    ),
                    "completed_at": (
                        job.completed_at.isoformat() if job.completed_at else None
                    ),
                    "status": (
                        job.status.value
                        if hasattr(job.status, "value")
                        else str(job.status)
                    ),
                    "parent_job_id": parent_job_id,
                    "subjob_ids": subjob_ids,
                    "resume_job_id": job.metadata.get("resume_job_id"),
                    "original_job_id": job.metadata.get("original_job_id"),
                }
            )

            return {
                "job_id": job_id,
                "metadata": metadata,
                "steps": steps,
                "total_steps": len(steps),
                "subjobs": subjob_ids,
                "parent_job_id": parent_job_id,
                "performance_metrics": performance_metrics,
                "quality_warnings": quality_warnings,
            }

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get job results for {job_id}: {e}")
            raise

    async def get_step_result(
        self,
        job_id: str,
        step_filename: str,
        execution_context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a specific step result.

        Args:
            job_id: Job identifier (may be subjob, will find root)
            step_filename: Step filename (e.g., "01_seo_keywords.json")
            execution_context_id: Optional execution context ID to search in specific context

        Returns:
            Step result data
        """
        try:
            from marketing_project.services.job_manager import get_job_manager

            job_manager = get_job_manager()
            job = await job_manager.get_job(job_id)

            if not job:
                raise FileNotFoundError(f"Job {job_id} not found")

            # Find root job
            root_job_id = job_id
            if job.metadata.get("original_job_id"):
                current = job
                while current.metadata.get("original_job_id"):
                    parent_id = current.metadata.get("original_job_id")
                    current = await job_manager.get_job(parent_id)
                    if not current:
                        break
                    if not current.metadata.get("original_job_id"):
                        root_job_id = current.id
                        break
                else:
                    root_job_id = job.metadata.get("original_job_id")

            # Try S3 first if enabled
            if self._use_s3 and self.s3_storage:
                try:
                    if execution_context_id:
                        # Search in specific context
                        s3_key = self._get_s3_key(
                            root_job_id, execution_context_id, step_filename
                        )
                        full_s3_key = self.s3_storage._get_s3_key(s3_key)
                        if await self.s3_storage.file_exists(full_s3_key):
                            content = await self.s3_storage.get_file_content(
                                full_s3_key
                            )
                            if content:
                                return json.loads(content.decode("utf-8"))
                    else:
                        # Search all contexts (list and find most recent)
                        search_prefix = f"{root_job_id}/context_"
                        keys = await self.s3_storage.list_files(prefix=search_prefix)
                        # Filter for this step filename and sort by context ID
                        matching_keys = []
                        for key in keys:
                            # Remove S3Storage prefix
                            relative_key = key
                            if self.s3_storage.prefix and key.startswith(
                                self.s3_storage.prefix
                            ):
                                relative_key = key[len(self.s3_storage.prefix) :]
                            if relative_key.endswith(f"/{step_filename}"):
                                matching_keys.append(relative_key)

                        if matching_keys:
                            # Sort by context ID (most recent first)
                            matching_keys.sort(
                                key=lambda k: (
                                    int(k.split("/context_")[1].split("/")[0])
                                    if "/context_" in k
                                    and k.split("/context_")[1].split("/")[0].isdigit()
                                    else -1
                                ),
                                reverse=True,
                            )
                            full_s3_key = self.s3_storage._get_s3_key(matching_keys[0])
                            content = await self.s3_storage.get_file_content(
                                full_s3_key
                            )
                            if content:
                                return json.loads(content.decode("utf-8"))
                except Exception as e:
                    logger.warning(f"Error loading from S3, trying local: {e}")

            root_job_dir = self._get_job_dir(root_job_id)

            # Try context directories first (new structure)
            context_dirs = (
                sorted(
                    [
                        d
                        for d in root_job_dir.iterdir()
                        if d.is_dir() and d.name.startswith("context_")
                    ]
                )
                if root_job_dir.exists()
                else []
            )

            if context_dirs:
                # Search in context directories
                if execution_context_id:
                    # Search in specific context
                    context_dir = root_job_dir / f"context_{execution_context_id}"
                    file_path = context_dir / step_filename
                    if file_path.exists():
                        with open(file_path, "r", encoding="utf-8") as f:
                            return json.load(f)
                else:
                    # Search all contexts (most recent first)
                    for context_dir in reversed(context_dirs):
                        file_path = context_dir / step_filename
                        if file_path.exists():
                            with open(file_path, "r", encoding="utf-8") as f:
                                return json.load(f)

            # Fallback to old structure (backward compatibility)
            file_path = root_job_dir / step_filename
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            # Step result not found - raise FileNotFoundError
            raise FileNotFoundError(
                f"Step result not found: {step_filename} for job {job_id}"
            )

        except FileNotFoundError:
            raise  # Re-raise FileNotFoundError
        except Exception as e:
            logger.error(
                f"Failed to get step result {step_filename} for job {job_id}: {e}"
            )
            raise FileNotFoundError(
                f"Step result not found: {step_filename} for job {job_id}"
            ) from e

    async def get_step_result_by_name(
        self,
        job_id: str,
        step_name: str,
        execution_context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a specific step result by step name.

        Args:
            job_id: Job identifier (may be subjob, will find root)
            step_name: Step name (e.g., "seo_keywords")
            execution_context_id: Optional execution context ID to search in specific context

        Returns:
            Step result data
        """
        try:
            from marketing_project.services.job_manager import get_job_manager

            job_manager = get_job_manager()
            job = await job_manager.get_job(job_id)

            if not job:
                raise FileNotFoundError(f"Job {job_id} not found")

            # Find root job
            root_job_id = job_id
            if job.metadata.get("original_job_id"):
                current = job
                while current.metadata.get("original_job_id"):
                    parent_id = current.metadata.get("original_job_id")
                    current = await job_manager.get_job(parent_id)
                    if not current:
                        break
                    if not current.metadata.get("original_job_id"):
                        root_job_id = current.id
                        break
                else:
                    root_job_id = job.metadata.get("original_job_id")

            # Normalize step name for matching
            normalized_step_name = step_name.lower().replace(" ", "_").replace("-", "_")

            # DB-first: look up by job_id + step_name
            try:
                from sqlalchemy import select as sa_select

                from marketing_project.models.db_models import StepResultModel
                from marketing_project.services.database import get_database_manager

                _db_manager = get_database_manager()
                if _db_manager.is_initialized:
                    # Try exact match on job_id first, then root_job_id
                    for _lookup_job_id in (job_id, root_job_id):
                        async with _db_manager.get_session() as _session:
                            _q = sa_select(StepResultModel).where(
                                StepResultModel.job_id == _lookup_job_id,
                                StepResultModel.step_name == normalized_step_name,
                            )
                            _result = await _session.execute(_q)
                            _row = _result.scalar_one_or_none()
                        if _row is not None:
                            logger.debug(
                                f"Loaded step '{step_name}' from DB for job {_lookup_job_id}"
                            )
                            return {
                                "job_id": _row.job_id,
                                "root_job_id": _row.root_job_id,
                                "execution_context_id": _row.execution_context_id,
                                "step_number": _row.step_number,
                                "step_name": _row.step_name,
                                "timestamp": (
                                    _row.created_at.isoformat()
                                    if _row.created_at
                                    else None
                                ),
                                "result": _row.result,
                                "input_snapshot": _row.input_snapshot,
                                "context_keys_used": _row.context_keys_used,
                                "metadata": {
                                    "status": _row.status,
                                    "execution_time": (
                                        float(_row.execution_time)
                                        if _row.execution_time
                                        else None
                                    ),
                                    "tokens_used": _row.tokens_used,
                                    "error_message": _row.error_message,
                                },
                            }
            except Exception as _e:
                logger.warning(
                    f"DB get_step_result_by_name failed, falling back to filesystem: {_e}"
                )

            # Try S3 first if enabled
            if self._use_s3 and self.s3_storage:
                try:
                    search_prefix = f"{root_job_id}/context_"
                    keys = await self.s3_storage.list_files(prefix=search_prefix)

                    # Filter by execution context if specified and load each file
                    for key in keys:
                        if not key.endswith(".json"):
                            continue

                        # Remove S3Storage prefix
                        relative_key = key
                        if self.s3_storage.prefix and key.startswith(
                            self.s3_storage.prefix
                        ):
                            relative_key = key[len(self.s3_storage.prefix) :]

                        # Filter by execution context if specified
                        if (
                            execution_context_id
                            and f"/context_{execution_context_id}/" not in relative_key
                        ):
                            continue

                        full_s3_key = self.s3_storage._get_s3_key(relative_key)
                        content = await self.s3_storage.get_file_content(full_s3_key)
                        if content:
                            step_data = json.loads(content.decode("utf-8"))
                            if (
                                step_data.get("step_name", "")
                                .lower()
                                .replace(" ", "_")
                                .replace("-", "_")
                                == normalized_step_name
                            ):
                                return step_data
                except Exception as e:
                    logger.warning(f"Error loading from S3, trying local: {e}")

            root_job_dir = self._get_job_dir(root_job_id)

            # Try context directories first (new structure)
            context_dirs = (
                sorted(
                    [
                        d
                        for d in root_job_dir.iterdir()
                        if d.is_dir() and d.name.startswith("context_")
                    ]
                )
                if root_job_dir.exists()
                else []
            )

            if context_dirs:
                # Search in context directories
                if execution_context_id:
                    # Search in specific context
                    context_dir = root_job_dir / f"context_{execution_context_id}"
                    for step_file in context_dir.glob("*.json"):
                        with open(step_file, "r", encoding="utf-8") as f:
                            step_data = json.load(f)
                            if (
                                step_data.get("step_name", "")
                                .lower()
                                .replace(" ", "_")
                                .replace("-", "_")
                                == normalized_step_name
                            ):
                                return step_data
                else:
                    # Search all contexts (most recent first)
                    for context_dir in reversed(context_dirs):
                        for step_file in context_dir.glob("*.json"):
                            with open(step_file, "r", encoding="utf-8") as f:
                                step_data = json.load(f)
                                if (
                                    step_data.get("step_name", "")
                                    .lower()
                                    .replace(" ", "_")
                                    .replace("-", "_")
                                    == normalized_step_name
                                ):
                                    return step_data

            # Fallback to old structure (backward compatibility)
            for step_file in root_job_dir.glob("*.json"):
                with open(step_file, "r", encoding="utf-8") as f:
                    step_data = json.load(f)
                    if (
                        step_data.get("step_name", "")
                        .lower()
                        .replace(" ", "_")
                        .replace("-", "_")
                        == normalized_step_name
                    ):
                        return step_data

            # Also try to get from job.result if available
            if job.result and isinstance(job.result, dict):
                step_results = job.result.get("result", {}).get("step_results", {})
                if step_results:
                    # Try exact match first
                    if step_name in step_results:
                        return step_results[step_name]
                    # Try normalized match
                    for key, value in step_results.items():
                        if (
                            key.lower().replace(" ", "_").replace("-", "_")
                            == normalized_step_name
                        ):
                            return value

            # Step result not found - raise FileNotFoundError
            raise FileNotFoundError(
                f"Step result not found: {step_name} for job {job_id}"
            )

        except FileNotFoundError:
            raise  # Re-raise FileNotFoundError
        except Exception as e:
            logger.error(f"Failed to get step result {step_name} for job {job_id}: {e}")
            raise FileNotFoundError(
                f"Step result not found: {step_name} for job {job_id}"
            ) from e

    async def get_step_file_path(
        self,
        job_id: str,
        step_filename: str,
        execution_context_id: Optional[str] = None,
    ) -> Path:
        """
        Get the file path for a step result (for download).

        Args:
            job_id: Job identifier (may be subjob, will find root)
            step_filename: Step filename
            execution_context_id: Optional execution context ID to search in specific context

        Returns:
            Path to the file
        """
        try:
            from marketing_project.services.job_manager import get_job_manager

            job_manager = get_job_manager()
            job = await job_manager.get_job(job_id)

            if not job:
                raise FileNotFoundError(f"Job {job_id} not found")

            # Find root job
            root_job_id = job_id
            if job.metadata.get("original_job_id"):
                current = job
                while current.metadata.get("original_job_id"):
                    parent_id = current.metadata.get("original_job_id")
                    current = await job_manager.get_job(parent_id)
                    if not current:
                        break
                    if not current.metadata.get("original_job_id"):
                        root_job_id = current.id
                        break
                else:
                    root_job_id = job.metadata.get("original_job_id")

            root_job_dir = self._get_job_dir(root_job_id)

            # Try context directories first (new structure)
            context_dirs = sorted(
                [
                    d
                    for d in root_job_dir.iterdir()
                    if d.is_dir() and d.name.startswith("context_")
                ]
            )

            if context_dirs:
                # Search in context directories
                if execution_context_id:
                    # Search in specific context
                    context_dir = root_job_dir / f"context_{execution_context_id}"
                    file_path = context_dir / step_filename
                    if file_path.exists():
                        return file_path
                else:
                    # Search all contexts (most recent first)
                    for context_dir in reversed(context_dirs):
                        file_path = context_dir / step_filename
                        if file_path.exists():
                            return file_path

            # Fallback to old structure (backward compatibility)
            file_path = root_job_dir / step_filename
            if file_path.exists():
                return file_path

            # Step file not found - raise FileNotFoundError
            raise FileNotFoundError(
                f"Step file not found: {step_filename} for job {job_id}"
            )

        except FileNotFoundError:
            raise  # Re-raise FileNotFoundError
        except Exception as e:
            logger.error(
                f"Failed to get step file path {step_filename} for job {job_id}: {e}"
            )
            raise FileNotFoundError(
                f"Step file not found: {step_filename} for job {job_id}"
            ) from e

    async def list_all_jobs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all jobs with results.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of job summaries
        """
        # DB-first: query the jobs table (always populated) and enrich with step counts
        try:
            from sqlalchemy import func as sqla_func
            from sqlalchemy import select

            from marketing_project.models.db_models import JobModel, StepResultModel
            from marketing_project.services.database import get_database_manager

            db_manager = get_database_manager()
            if db_manager.is_initialized:
                async with db_manager.get_session() as session:
                    # Step counts per root_job_id in one GROUP BY query
                    counts_q = select(
                        StepResultModel.root_job_id,
                        sqla_func.count(StepResultModel.id).label("cnt"),
                    ).group_by(StepResultModel.root_job_id)
                    counts_result = await session.execute(counts_q)
                    step_counts = {row.root_job_id: row.cnt for row in counts_result}

                    # Only show root jobs — exclude resume_pipeline subjobs which
                    # are internal and should be viewed via their parent job
                    q = (
                        select(JobModel)
                        .where(JobModel.job_type != "resume_pipeline")
                        .order_by(JobModel.created_at.desc())
                    )
                    if limit:
                        q = q.limit(limit)
                    jobs_result = await session.execute(q)
                    db_jobs = jobs_result.scalars().all()

                return [
                    {
                        "job_id": j.job_id,
                        "content_type": j.job_type,
                        "content_id": j.content_id,
                        "status": (
                            j.status.value
                            if hasattr(j.status, "value")
                            else str(j.status)
                        ),
                        "step_count": step_counts.get(j.job_id, 0),
                        "created_at": (
                            j.created_at.isoformat() if j.created_at else None
                        ),
                        "started_at": (
                            j.started_at.isoformat() if j.started_at else None
                        ),
                        "completed_at": (
                            j.completed_at.isoformat() if j.completed_at else None
                        ),
                        "user_id": j.user_id,
                    }
                    for j in db_jobs
                ]
        except Exception as e:
            logger.warning(f"DB list_all_jobs failed, falling back to filesystem: {e}")

        # Filesystem / S3 fallback (original implementation below)
        try:
            jobs = []
            job_ids_seen = set()

            # Try S3 first if enabled
            if self._use_s3 and self.s3_storage:
                try:
                    # List all keys with prefix
                    keys = await self.s3_storage.list_files(prefix="")
                    # Extract unique job IDs from keys
                    job_metadata_map = {}
                    job_step_counts = {}

                    for key in keys:
                        # Remove S3Storage prefix to get relative path
                        relative_key = key
                        if self.s3_storage.prefix and key.startswith(
                            self.s3_storage.prefix
                        ):
                            relative_key = key[len(self.s3_storage.prefix) :]

                        # Format: {job_id}/metadata.json or {job_id}/context_{id}/{filename}
                        parts = relative_key.split("/")
                        if parts:
                            job_id = parts[0]

                            # Track metadata files
                            if relative_key.endswith("metadata.json"):
                                if job_id not in job_metadata_map:
                                    job_metadata_map[job_id] = relative_key

                            # Count step files (all .json files except metadata.json)
                            if relative_key.endswith(
                                ".json"
                            ) and not relative_key.endswith("metadata.json"):
                                if job_id not in job_step_counts:
                                    job_step_counts[job_id] = 0
                                job_step_counts[job_id] += 1

                    # Load metadata and create job entries
                    for job_id in set(
                        list(job_metadata_map.keys()) + list(job_step_counts.keys())
                    ):
                        if job_id in job_ids_seen:
                            continue
                        job_ids_seen.add(job_id)

                        # Load metadata
                        metadata = {}
                        if job_id in job_metadata_map:
                            full_metadata_key = self.s3_storage._get_s3_key(
                                job_metadata_map[job_id]
                            )
                            content = await self.s3_storage.get_file_content(
                                full_metadata_key
                            )
                            if content:
                                metadata = json.loads(content.decode("utf-8"))

                        step_count = job_step_counts.get(job_id, 0)

                        jobs.append(
                            {
                                "job_id": job_id,
                                "content_type": metadata.get("content_type"),
                                "content_id": metadata.get("content_id"),
                                "started_at": metadata.get("started_at"),
                                "completed_at": metadata.get("completed_at"),
                                "step_count": step_count,
                                "created_at": metadata.get("created_at"),
                                "user_id": metadata.get("user_id"),
                            }
                        )

                        if limit and len(jobs) >= limit:
                            break
                except Exception as e:
                    logger.warning(f"Error listing jobs from S3, trying local: {e}")

            # Also check local filesystem
            if self.base_dir.exists():
                for job_dir in sorted(
                    self.base_dir.iterdir(),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True,
                ):
                    if not job_dir.is_dir():
                        continue

                    job_id = job_dir.name
                    if job_id in job_ids_seen:
                        continue  # Already loaded from S3
                    job_ids_seen.add(job_id)

                    # Load metadata
                    metadata_path = job_dir / "metadata.json"
                    metadata = {}
                    if metadata_path.exists():
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)

                    # Count steps
                    step_count = len(
                        [f for f in job_dir.glob("*.json") if f.name != "metadata.json"]
                    )
                    # Also count steps in context directories
                    context_dirs = [
                        d
                        for d in job_dir.iterdir()
                        if d.is_dir() and d.name.startswith("context_")
                    ]
                    for context_dir in context_dirs:
                        step_count += len([f for f in context_dir.glob("*.json")])

                    jobs.append(
                        {
                            "job_id": job_id,
                            "content_type": metadata.get("content_type"),
                            "content_id": metadata.get("content_id"),
                            "started_at": metadata.get("started_at"),
                            "completed_at": metadata.get("completed_at"),
                            "step_count": step_count,
                            "created_at": metadata.get("created_at"),
                            "user_id": metadata.get("user_id"),
                        }
                    )

                    if limit and len(jobs) >= limit:
                        break

            # Sort by created_at or started_at (most recent first)
            jobs.sort(
                key=lambda x: x.get("created_at") or x.get("started_at") or "",
                reverse=True,
            )

            return jobs[:limit] if limit else jobs

        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise

    async def find_related_jobs(self, job_id: str) -> Dict[str, Any]:
        """
        Find related jobs (parent and subjobs) for a given job.
        Recursively finds all subjobs in a chain of resume jobs.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with parent_job_id and subjob_ids
        """
        try:
            from marketing_project.services.job_manager import get_job_manager

            job_manager = get_job_manager()
            job = await job_manager.get_job(job_id)

            if not job:
                return {"parent_job_id": None, "subjob_ids": []}

            parent_job_id = None
            subjob_ids = []

            # Check if this job is a subjob (has original_job_id)
            if job.metadata.get("original_job_id"):
                parent_job_id = job.metadata.get("original_job_id")

            # Check if job.result has original_job_id (for resume jobs)
            if job.result and isinstance(job.result, dict):
                if job.result.get("original_job_id"):
                    parent_job_id = job.result.get("original_job_id")

            # Recursively find all subjobs by following the chain of resume_job_id
            current_job_id = job_id
            visited = set()  # Prevent infinite loops

            while current_job_id and current_job_id not in visited:
                visited.add(current_job_id)
                current_job = await job_manager.get_job(current_job_id)

                if not current_job:
                    break

                # Check if this job has a resume_job_id (subjob)
                resume_job_id = current_job.metadata.get("resume_job_id")
                if resume_job_id:
                    # Verify the resume job exists
                    resume_job = await job_manager.get_job(resume_job_id)
                    if resume_job:
                        if resume_job_id not in subjob_ids:
                            subjob_ids.append(resume_job_id)
                        # Continue following the chain
                        current_job_id = resume_job_id
                    else:
                        break
                else:
                    # No more resume jobs in the chain
                    break

            return {"parent_job_id": parent_job_id, "subjob_ids": subjob_ids}

        except Exception as e:
            logger.error(f"Failed to find related jobs for {job_id}: {e}")
            return {"parent_job_id": None, "subjob_ids": []}

    async def aggregate_steps_from_jobs(
        self, job_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Aggregate steps from multiple jobs, preserving chronological order.
        Now supports execution contexts - all steps from root job's contexts.

        Args:
            job_ids: List of job IDs to aggregate steps from

        Returns:
            List of aggregated steps with job_id and execution_context_id
        """
        from marketing_project.services.job_manager import get_job_manager

        job_manager = get_job_manager()
        all_steps = []

        for job_id in job_ids:
            try:
                # Find root job for this job_id
                job = await job_manager.get_job(job_id)
                if not job:
                    continue

                root_job_id = job_id
                if job.metadata.get("original_job_id"):
                    # Find root by traversing up
                    current = job
                    while current.metadata.get("original_job_id"):
                        parent_id = current.metadata.get("original_job_id")
                        current = await job_manager.get_job(parent_id)
                        if not current:
                            break
                        if not current.metadata.get("original_job_id"):
                            root_job_id = current.id
                            break
                    else:
                        root_job_id = job.metadata.get("original_job_id")

                # Load steps from all execution contexts in root job directory
                root_job_dir = self._get_job_dir(root_job_id)
                job_steps = []

                if root_job_dir.exists():
                    # Check for context directories (new structure)
                    context_dirs = sorted(
                        [
                            d
                            for d in root_job_dir.iterdir()
                            if d.is_dir() and d.name.startswith("context_")
                        ]
                    )

                    if context_dirs:
                        # New structure: load from context directories
                        for context_dir in context_dirs:
                            context_id = context_dir.name.replace("context_", "")
                            step_files = sorted([f for f in context_dir.glob("*.json")])

                            for step_file in step_files:
                                with open(step_file, "r", encoding="utf-8") as f:
                                    step_data = json.load(f)
                                    job_steps.append(
                                        {
                                            "job_id": step_data.get("job_id", job_id),
                                            "root_job_id": step_data.get(
                                                "root_job_id", root_job_id
                                            ),
                                            "execution_context_id": step_data.get(
                                                "execution_context_id", context_id
                                            ),
                                            "filename": step_file.name,
                                            "step_number": step_data.get("step_number"),
                                            "step_name": step_data.get("step_name"),
                                            "timestamp": step_data.get("timestamp"),
                                            "has_result": "result" in step_data,
                                            "file_size": step_file.stat().st_size,
                                            "execution_time": step_data.get(
                                                "metadata", {}
                                            ).get("execution_time"),
                                            "tokens_used": step_data.get(
                                                "metadata", {}
                                            ).get("tokens_used"),
                                            "status": step_data.get("metadata", {}).get(
                                                "status", "success"
                                            ),
                                            "error_message": step_data.get(
                                                "metadata", {}
                                            ).get("error_message"),
                                        }
                                    )
                    else:
                        # Old structure: load from job directory directly (backward compatibility)
                        step_files = sorted(
                            [
                                f
                                for f in root_job_dir.glob("*.json")
                                if f.name != "metadata.json"
                            ]
                        )

                        for step_file in step_files:
                            with open(step_file, "r", encoding="utf-8") as f:
                                step_data = json.load(f)
                                job_steps.append(
                                    {
                                        "job_id": step_data.get("job_id", job_id),
                                        "root_job_id": root_job_id,
                                        "execution_context_id": step_data.get(
                                            "execution_context_id", "0"
                                        ),
                                        "filename": step_file.name,
                                        "step_number": step_data.get("step_number"),
                                        "step_name": step_data.get("step_name"),
                                        "timestamp": step_data.get("timestamp"),
                                        "has_result": "result" in step_data,
                                        "file_size": step_file.stat().st_size,
                                        "execution_time": step_data.get(
                                            "metadata", {}
                                        ).get("execution_time"),
                                        "tokens_used": step_data.get(
                                            "metadata", {}
                                        ).get("tokens_used"),
                                        "status": step_data.get("metadata", {}).get(
                                            "status", "success"
                                        ),
                                        "error_message": step_data.get(
                                            "metadata", {}
                                        ).get("error_message"),
                                    }
                                )

                # If no step files, try to extract from job.result
                if not job_steps:
                    if job and job.result:
                        extracted_steps = await self.extract_step_info_from_job_result(
                            job_id
                        )
                        # Ensure job_id and execution_context_id are set
                        for step in extracted_steps:
                            if not step.get("job_id"):
                                step["job_id"] = job_id
                            if not step.get("root_job_id"):
                                step["root_job_id"] = root_job_id
                            if not step.get("execution_context_id"):
                                step["execution_context_id"] = "0"
                        job_steps = extracted_steps

                all_steps.extend(job_steps)
            except Exception as e:
                logger.warning(f"Failed to load steps from job {job_id}: {e}")
                continue

        # Sort by timestamp to maintain chronological order
        all_steps.sort(key=lambda x: x.get("timestamp", ""))

        return all_steps

    async def extract_step_info_from_job_result(
        self, job_id: str
    ) -> List[Dict[str, Any]]:
        """
        Extract step information from job.result when step files don't exist.

        Args:
            job_id: Job identifier

        Returns:
            List of step info dictionaries
        """
        try:
            from marketing_project.services.job_manager import get_job_manager

            job_manager = get_job_manager()
            job = await job_manager.get_job(job_id)

            if not job or not job.result:
                return []

            result = job.result
            if not isinstance(result, dict):
                return []

            # Extract step_info from metadata
            metadata = result.get("metadata", {})
            step_info_list = metadata.get("step_info", [])

            # Find root job for execution context
            root_job_id = job_id
            if job.metadata.get("original_job_id"):
                current = job
                while current.metadata.get("original_job_id"):
                    parent_id = current.metadata.get("original_job_id")
                    current = await job_manager.get_job(parent_id)
                    if not current:
                        break
                    if not current.metadata.get("original_job_id"):
                        root_job_id = current.id
                        break
                else:
                    root_job_id = job.metadata.get("original_job_id")

            # Determine execution context ID
            execution_context_id = "0"  # Default to initial execution
            if job_id != root_job_id:
                # This is a subjob, determine context from chain position
                root_job = await job_manager.get_job(root_job_id)
                if root_job:
                    chain_metadata = root_job.metadata.get("job_chain", {})
                    chain_order = chain_metadata.get("chain_order", [root_job_id])
                    try:
                        context_index = chain_order.index(job_id)
                        execution_context_id = str(context_index)
                    except ValueError:
                        execution_context_id = "0"

            steps = []
            for idx, step_info in enumerate(step_info_list):
                step_name = step_info.get("step_name", f"step_{idx}")
                step_number = step_info.get("step_number", idx)

                # Create a synthetic filename
                filename = self._get_step_filename(step_number, step_name)

                steps.append(
                    {
                        "job_id": job_id,
                        "root_job_id": root_job_id,
                        "execution_context_id": execution_context_id,
                        "filename": filename,
                        "step_number": step_number,
                        "step_name": step_name,
                        "timestamp": (
                            metadata.get("completed_at") or job.completed_at.isoformat()
                            if job.completed_at
                            else datetime.utcnow().isoformat()
                        ),
                        "has_result": True,
                        "file_size": 0,  # Unknown since it's from job.result
                        "execution_time": step_info.get("execution_time"),
                        "tokens_used": step_info.get("tokens_used"),
                        "status": step_info.get("status", "success"),
                        "error_message": step_info.get("error_message"),
                    }
                )

            return steps

        except Exception as e:
            logger.warning(
                f"Failed to extract step info from job.result for {job_id}: {e}"
            )
            return []

    async def cleanup_job(self, job_id: str) -> bool:
        """
        Delete all results for a specific job.

        Args:
            job_id: Job identifier

        Returns:
            True if successful
        """
        try:
            deleted_any = False

            # Delete from S3 if enabled
            if self._use_s3 and self.s3_storage:
                try:
                    # List all keys for this job
                    search_prefix = f"{job_id}/"
                    keys = await self.s3_storage.list_files(prefix=search_prefix)

                    # Delete all keys
                    deleted_count = 0
                    for key in keys:
                        # Key already includes S3Storage prefix from list_files
                        # But delete_file expects the full key, so use as-is
                        if await self.s3_storage.delete_file(key):
                            deleted_count += 1
                            deleted_any = True

                    if deleted_count > 0:
                        logger.info(
                            f"Deleted {deleted_count} files from S3 for job {job_id}"
                        )
                except Exception as e:
                    logger.warning(f"Error deleting from S3, trying local: {e}")

            # Also delete from local filesystem
            job_dir = self._get_job_dir(job_id)
            if job_dir.exists():
                # Delete all files in the directory
                for file_path in job_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                        deleted_any = True
                    elif file_path.is_dir():
                        # Delete files in subdirectories
                        for sub_file in file_path.rglob("*"):
                            if sub_file.is_file():
                                sub_file.unlink()
                                deleted_any = True
                        # Remove subdirectory
                        file_path.rmdir()

                # Remove the directory
                try:
                    job_dir.rmdir()
                except OSError:
                    pass  # Directory might not be empty yet

            if deleted_any:
                logger.info(f"Cleaned up results for job {job_id}")
            return deleted_any

        except Exception as e:
            logger.error(f"Failed to cleanup job {job_id}: {e}")
            raise

    async def _get_or_create_execution_context(
        self,
        root_job_id: str,
        current_job_id: str,
        step_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Get or create an execution context ID for a job.

        Execution contexts represent different execution cycles:
        - Context 0: Initial pipeline execution
        - Context 1: First resume after approval
        - Context 2: Second resume after approval
        - etc.

        Args:
            root_job_id: Root job ID
            current_job_id: Current job ID (may be subjob)
            step_metadata: Optional step metadata to determine context

        Returns:
            Execution context ID (string representation of context number)
        """
        try:
            from marketing_project.services.job_manager import get_job_manager

            job_manager = get_job_manager()
            root_job = await job_manager.get_job(root_job_id)

            if not root_job:
                # If root job doesn't exist, this is context 0
                return "0"

            # Check if current_job_id is the root job
            if current_job_id == root_job_id:
                # This is the initial execution - context 0
                return "0"

            # This is a subjob - determine context number from chain position
            chain_metadata = root_job.metadata.get("job_chain", {})
            chain_order = chain_metadata.get("chain_order", [root_job_id])

            # Find position of current_job_id in chain
            try:
                context_index = chain_order.index(current_job_id)
                # Context 0 is root, so context 1 is first subjob, etc.
                return str(context_index)
            except ValueError:
                # Current job not in chain, create new context
                # Count existing contexts
                root_job_dir = self._get_job_dir(root_job_id)
                if root_job_dir.exists():
                    context_dirs = [
                        d
                        for d in root_job_dir.iterdir()
                        if d.is_dir() and d.name.startswith("context_")
                    ]
                    return str(len(context_dirs))
                return "0"

        except Exception as e:
            logger.warning(
                f"Failed to determine execution context: {e}, defaulting to 0"
            )
            return "0"

    async def get_full_context_history(self, job_id: str) -> Dict[str, Any]:
        """
        Get full context history for a job from the context registry.

        This method retrieves all step outputs with their input snapshots and
        context keys used, preserving complete history for debugging and audit.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with full context history organized by execution context
        """
        try:
            from marketing_project.services.context_registry import get_context_registry

            context_registry = get_context_registry()
            history = await context_registry.get_full_history(job_id)
            return history
        except Exception as e:
            logger.warning(f"Failed to get full context history for {job_id}: {e}")
            # Fallback to empty history
            return {}

    async def get_pipeline_flow(self, job_id: str) -> Dict[str, Any]:
        """
        Get complete pipeline flow visualization data.

        Builds a structured flow showing:
        - Original input content
        - Each step with its input snapshot and output
        - Final output
        - Execution summary

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with pipeline flow data structured for visualization
        """
        try:
            from marketing_project.services.job_manager import get_job_manager

            job_manager = get_job_manager()
            job = await job_manager.get_job(job_id)

            if not job:
                raise FileNotFoundError(f"Job {job_id} not found")

            # Get input content from job metadata
            input_content = job.metadata.get("input_content", {})

            # Get all step results
            job_results = await self.get_job_results(job_id)
            # Ensure job_results is a dict (safeguard against unexpected return types)
            if not isinstance(job_results, dict):
                logger.error(
                    f"get_job_results returned non-dict type: {type(job_results)}"
                )
                raise ValueError(
                    f"get_job_results returned unexpected type: {type(job_results)}"
                )
            steps = job_results.get("steps", [])

            # Sort steps by step_number
            steps_sorted = sorted(steps, key=lambda x: x.get("step_number", 0) or 0)

            # Build step input/output list
            step_flow = []
            for step in steps_sorted:
                step_name = step.get("step_name")
                step_number = step.get("step_number")
                if not step_name:
                    continue

                try:
                    # Get full step result
                    step_result = await self.get_step_result_by_name(job_id, step_name)

                    # Extract input snapshot (with fallback to empty dict)
                    input_snapshot = step_result.get("input_snapshot", {})

                    # Extract output
                    output = step_result.get("result", {})

                    # Extract execution metadata
                    metadata = step_result.get("metadata", {})
                    execution_metadata = {
                        "execution_time": metadata.get("execution_time"),
                        "tokens_used": metadata.get("tokens_used"),
                        "status": metadata.get("status", "success"),
                        "error_message": metadata.get("error_message"),
                    }

                    # Get context keys used
                    context_keys_used = step_result.get("context_keys_used", [])

                    step_flow.append(
                        {
                            "step_name": step_name,
                            "step_number": step_number,
                            "input_snapshot": input_snapshot,
                            "output": output,
                            "context_keys_used": context_keys_used,
                            "execution_metadata": execution_metadata,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to get step result for {step_name} in job {job_id}: {e}"
                    )
                    # Add step with minimal info if we can't load full result
                    step_flow.append(
                        {
                            "step_name": step_name,
                            "step_number": step_number,
                            "input_snapshot": {},
                            "output": {},
                            "context_keys_used": [],
                            "execution_metadata": {
                                "status": step.get("status", "unknown"),
                                "error_message": str(e),
                            },
                        }
                    )

            # Get final output from job result
            final_output = {}
            if job.result and isinstance(job.result, dict):
                # Try to get final_content from various locations
                final_output = (
                    job.result.get("final_content")
                    or job.result.get("result", {}).get("final_content")
                    or {}
                )
                # If final_content is a string, wrap it
                if isinstance(final_output, str):
                    final_output = {"content": final_output}

            # Build execution summary
            performance_metrics = job_results.get("performance_metrics", {})
            execution_summary = {
                "total_execution_time_seconds": performance_metrics.get(
                    "execution_time_seconds"
                ),
                "total_tokens_used": performance_metrics.get("total_tokens_used"),
                "total_steps": len(step_flow),
                "quality_warnings": job_results.get("quality_warnings", []),
            }

            return {
                "job_id": job_id,
                "input_content": input_content,
                "steps": step_flow,
                "final_output": final_output,
                "execution_summary": execution_summary,
            }

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get pipeline flow for {job_id}: {e}")
            raise


# Global instance
_step_result_manager: Optional[StepResultManager] = None


def get_step_result_manager() -> StepResultManager:
    """Get or create the global step result manager instance."""
    global _step_result_manager
    if _step_result_manager is None:
        _step_result_manager = StepResultManager()
    return _step_result_manager
