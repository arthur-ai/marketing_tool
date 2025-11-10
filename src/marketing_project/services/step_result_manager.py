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

logger = logging.getLogger(__name__)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for datetime and other non-serializable objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


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
        logger.info(f"Step Result Manager initialized with base_dir: {self.base_dir}")

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

    async def save_step_result(
        self,
        job_id: str,
        step_number: int,
        step_name: str,
        result_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        execution_context_id: Optional[str] = None,
        root_job_id: Optional[str] = None,
    ) -> str:
        """
        Save a step result to disk.

        Args:
            job_id: Job identifier (current job executing the step)
            step_number: Step sequence number (0 for input, 1-7 for pipeline steps, 8 for final)
            step_name: Human-readable step name
            result_data: The result data to save (will be JSON serialized)
            metadata: Optional metadata to include in the file
            execution_context_id: Optional execution context ID (for tracking resume cycles)
            root_job_id: Optional root job ID (if different from job_id, saves to root directory)

        Returns:
            Path to the saved file
        """
        try:
            from marketing_project.services.job_manager import get_job_manager

            # Determine where to save: root job directory if provided, otherwise current job
            target_job_id = root_job_id or job_id

            # If no root_job_id provided, try to find it from job metadata
            if not root_job_id:
                job_manager = get_job_manager()
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
                "execution_context_id": execution_context_id,
                "step_number": step_number,
                "step_name": step_name,
                "timestamp": datetime.utcnow().isoformat(),
                "result": result_data,
                "metadata": metadata or {},
            }

            # Save to file with custom serializer for datetime objects
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    data, f, indent=2, ensure_ascii=False, default=_json_serializer
                )

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
            root_job_dir = self._get_job_dir(root_job_id)
            steps = []
            metadata = {}

            if root_job_dir.exists():
                # Load metadata from file
                metadata_path = root_job_dir / "metadata.json"
                if metadata_path.exists():
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
                ),
                "total_tokens_used": performance_metrics.get("total_tokens_used", 0),
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
                                performance_metrics.get("execution_time_seconds", 0)
                                + subjob_exec_time
                            )
                            chain_metrics[
                                "total_execution_time_seconds"
                            ] += subjob_exec_time
                        if subjob_tokens:
                            performance_metrics["total_tokens_used"] = (
                                performance_metrics.get("total_tokens_used", 0)
                                + subjob_tokens
                            )
                            chain_metrics["total_tokens_used"] += subjob_tokens

                        subjob_warnings = subjob_job.result.get("quality_warnings", [])
                        if subjob_warnings:
                            quality_warnings.extend(subjob_warnings)

            # Calculate chain-level metrics
            if chain_metrics["total_steps"] > 0:
                chain_metrics["average_time_per_step"] = (
                    chain_metrics["total_execution_time_seconds"]
                    / chain_metrics["total_steps"]
                )

            # Estimate cost (rough calculation: $0.01 per 1K tokens for GPT-4o-mini)
            # This is a simplified estimate - actual costs vary by model
            chain_metrics["estimated_cost_usd"] = (
                chain_metrics["total_tokens_used"] / 1000
            ) * 0.01

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

            raise FileNotFoundError(
                f"Step result not found: {step_filename} for job {job_id}"
            )

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to get step result {step_filename} for job {job_id}: {e}"
            )
            raise

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

            raise FileNotFoundError(
                f"Step result not found: {step_filename} for job {job_id}"
            )

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to get step file path {step_filename} for job {job_id}: {e}"
            )
            raise

    async def list_all_jobs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all jobs with results.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of job summaries
        """
        try:
            jobs = []

            for job_dir in sorted(
                self.base_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
            ):
                if not job_dir.is_dir():
                    continue

                job_id = job_dir.name

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

                jobs.append(
                    {
                        "job_id": job_id,
                        "content_type": metadata.get("content_type"),
                        "content_id": metadata.get("content_id"),
                        "started_at": metadata.get("started_at"),
                        "completed_at": metadata.get("completed_at"),
                        "step_count": step_count,
                        "created_at": metadata.get("created_at"),
                    }
                )

                if limit and len(jobs) >= limit:
                    break

            return jobs

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
            job_dir = self._get_job_dir(job_id)

            if not job_dir.exists():
                return False

            # Delete all files in the directory
            for file_path in job_dir.iterdir():
                file_path.unlink()

            # Remove the directory
            job_dir.rmdir()

            logger.info(f"Cleaned up results for job {job_id}")
            return True

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


# Global instance
_step_result_manager: Optional[StepResultManager] = None


def get_step_result_manager() -> StepResultManager:
    """Get or create the global step result manager instance."""
    global _step_result_manager
    if _step_result_manager is None:
        _step_result_manager = StepResultManager()
    return _step_result_manager
