"""
Context Registry Service for managing pipeline step context with zero data loss.

This service provides:
- Full step output history preservation
- Context versioning (step-level, pipeline-level, session-level)
- Context references/pointers for efficient access
- Lazy loading of context data
- Hierarchical storage with compression support
"""

import gzip
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ContextReference:
    """Reference to a context entry in the registry."""

    job_id: str
    step_name: str
    step_number: int
    execution_context_id: str
    timestamp: str
    compressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert reference to dictionary."""
        return {
            "job_id": self.job_id,
            "step_name": self.step_name,
            "step_number": self.step_number,
            "execution_context_id": self.execution_context_id,
            "timestamp": self.timestamp,
            "compressed": self.compressed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextReference":
        """Create reference from dictionary."""
        return cls(
            job_id=data["job_id"],
            step_name=data["step_name"],
            step_number=data["step_number"],
            execution_context_id=data["execution_context_id"],
            timestamp=data["timestamp"],
            compressed=data.get("compressed", False),
        )


class ContextRegistry:
    """
    Registry for managing full pipeline step context with zero data loss.

    Features:
    - Stores complete step outputs with versioning
    - Provides context references for efficient access
    - Implements lazy loading (load only when requested)
    - Supports hierarchical storage (step-level, pipeline-level, session-level)
    - Compression for storage while maintaining full retrievability
    """

    def __init__(self, base_dir: Optional[str] = None, enable_compression: bool = True):
        """
        Initialize the context registry.

        Args:
            base_dir: Base directory for storing context data (default: ./context_registry)
            enable_compression: Whether to compress stored context data
        """
        self.base_dir = Path(base_dir or "context_registry")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.enable_compression = enable_compression

        # Check if S3 should be used
        s3_bucket = os.getenv("AWS_S3_BUCKET")
        s3_prefix = os.getenv("CONTEXT_REGISTRY_S3_PREFIX", "context_registry/")
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
                        f"Context Registry using S3: s3://{s3_bucket}/{s3_prefix}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize S3 storage, falling back to local: {e}"
                )
                self._use_s3 = False
                self.s3_storage = None
        else:
            self.s3_storage = None

        # In-memory cache for frequently accessed contexts
        # Limited to prevent memory issues (LRU eviction when limit reached)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = int(os.getenv("CONTEXT_REGISTRY_CACHE_SIZE", "100"))
        self._reference_index: Dict[str, Dict[str, ContextReference]] = (
            {}
        )  # job_id -> step_name -> reference

        storage_type = "S3" if self._use_s3 else "local filesystem"
        logger.info(
            f"Context Registry initialized with {storage_type}, base_dir: {self.base_dir}, compression: {enable_compression}"
        )

    def _get_job_dir(self, job_id: str) -> Path:
        """Get the directory for a specific job's context."""
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _get_context_file_path(
        self,
        job_id: str,
        step_name: str,
        execution_context_id: str,
        compressed: bool = False,
    ) -> Path:
        """Get the file path for a context entry (local filesystem)."""
        job_dir = self._get_job_dir(job_id)
        context_dir = job_dir / f"context_{execution_context_id}"
        context_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{step_name}.json"
        if compressed:
            filename += ".gz"

        return context_dir / filename

    def _get_s3_key(
        self,
        job_id: str,
        step_name: str,
        execution_context_id: str,
        compressed: bool = False,
    ) -> str:
        """Get the S3 key for a context entry."""
        filename = f"{step_name}.json"
        if compressed:
            filename += ".gz"
        return f"{job_id}/context_{execution_context_id}/{filename}"

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        return gzip.compress(data)

    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress gzip-compressed data."""
        return gzip.decompress(compressed_data)

    async def register_step_output(
        self,
        job_id: str,
        step_name: str,
        step_number: int,
        output_data: Dict[str, Any],
        input_snapshot: Optional[Dict[str, Any]] = None,
        context_keys_used: Optional[List[str]] = None,
        execution_context_id: Optional[str] = None,
        root_job_id: Optional[str] = None,
    ) -> ContextReference:
        """
        Register a step output in the context registry.

        Args:
            job_id: Job identifier executing the step
            step_name: Name of the step
            step_number: Step sequence number
            output_data: Complete step output data
            input_snapshot: Snapshot of input context used for this step
            context_keys_used: List of context keys consumed by this step
            execution_context_id: Execution context ID (defaults to "0" if not provided)
            root_job_id: Root job ID if different from job_id

        Returns:
            ContextReference pointing to the registered context
        """
        try:
            from marketing_project.services.job_manager import get_job_manager

            # Determine root job ID
            target_job_id = root_job_id or job_id
            if not root_job_id:
                job_manager = get_job_manager()
                job = await job_manager.get_job(job_id)
                if job:
                    original_job_id = job.metadata.get("original_job_id")
                    if original_job_id:
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
                from marketing_project.services.step_result_manager import (
                    get_step_result_manager,
                )

                step_manager = get_step_result_manager()
                execution_context_id = (
                    await step_manager._get_or_create_execution_context(
                        target_job_id, job_id, None
                    )
                )

            # Prepare context data structure
            context_data = {
                "job_id": job_id,
                "root_job_id": target_job_id,
                "step_name": step_name,
                "step_number": step_number,
                "execution_context_id": execution_context_id,
                "timestamp": datetime.utcnow().isoformat(),
                "output_data": output_data,
                "input_snapshot": input_snapshot,
                "context_keys_used": context_keys_used or [],
            }

            # Serialize to JSON
            json_data = json.dumps(
                context_data, ensure_ascii=False, default=str
            ).encode("utf-8")

            # Compress if enabled
            if self.enable_compression:
                json_data = self._compress_data(json_data)

            # Save to S3 or local filesystem
            if self._use_s3 and self.s3_storage:
                try:
                    s3_key = self._get_s3_key(
                        target_job_id,
                        step_name,
                        execution_context_id,
                        compressed=self.enable_compression,
                    )
                    # Upload to S3 with gzip content encoding if compressed
                    extra_args = {}
                    if self.enable_compression:
                        extra_args["ContentEncoding"] = "gzip"
                    extra_args["ContentType"] = "application/json"

                    import io

                    file_obj = io.BytesIO(json_data)
                    self.s3_storage.s3_client.upload_fileobj(
                        file_obj,
                        self.s3_storage.bucket_name,
                        self.s3_storage._get_s3_key(s3_key),
                        ExtraArgs=extra_args,
                    )
                    logger.debug(f"Uploaded context to S3: {s3_key}")
                except Exception as e:
                    logger.warning(
                        f"Failed to upload context to S3, falling back to local: {e}"
                    )
                    # Fallback to local
                    file_path = self._get_context_file_path(
                        target_job_id,
                        step_name,
                        execution_context_id,
                        compressed=self.enable_compression,
                    )
                    file_path.write_bytes(json_data)
            else:
                # Save to local filesystem
                file_path = self._get_context_file_path(
                    target_job_id,
                    step_name,
                    execution_context_id,
                    compressed=self.enable_compression,
                )
                file_path.write_bytes(json_data)

            # Create reference
            reference = ContextReference(
                job_id=job_id,
                step_name=step_name,
                step_number=step_number,
                execution_context_id=execution_context_id,
                timestamp=context_data["timestamp"],
                compressed=self.enable_compression,
            )

            # Update reference index
            if target_job_id not in self._reference_index:
                self._reference_index[target_job_id] = {}
            self._reference_index[target_job_id][step_name] = reference

            # Cache the context data (with size limit)
            cache_key = f"{target_job_id}:{step_name}:{execution_context_id}"
            if len(self._cache) >= self._cache_max_size:
                # Remove oldest entry (FIFO eviction)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = context_data

            logger.debug(
                f"Registered context for step {step_name} (job: {job_id}, context: {execution_context_id})"
            )

            return reference

        except Exception as e:
            logger.error(
                f"Failed to register step output for {step_name} in job {job_id}: {e}"
            )
            raise

    async def get_context_reference(
        self, job_id: str, step_name: str, execution_context_id: Optional[str] = None
    ) -> Optional[ContextReference]:
        """
        Get a context reference for a specific step.

        Args:
            job_id: Job identifier
            step_name: Name of the step
            execution_context_id: Optional execution context ID (searches all if not provided)

        Returns:
            ContextReference if found, None otherwise
        """
        # Check in-memory index first
        if (
            job_id in self._reference_index
            and step_name in self._reference_index[job_id]
        ):
            return self._reference_index[job_id][step_name]

        if self._use_s3 and self.s3_storage:
            # Try to find from S3
            try:
                # Search context directories in S3
                if execution_context_id:
                    context_ids = [execution_context_id]
                else:
                    # List all context directories for this job
                    # S3Storage.list_files returns keys with prefix already applied
                    search_prefix = f"{job_id}/context_"
                    keys = await self.s3_storage.list_files(prefix=search_prefix)
                    # Extract unique context IDs from keys
                    # Keys format: {prefix}{job_id}/context_{id}/{filename}
                    context_ids = set()
                    for key in keys:
                        # Remove the S3Storage prefix to get relative path
                        relative_key = key
                        if self.s3_storage.prefix and key.startswith(
                            self.s3_storage.prefix
                        ):
                            relative_key = key[len(self.s3_storage.prefix) :]
                        # Extract context ID: {job_id}/context_{id}/...
                        if "/context_" in relative_key:
                            parts = relative_key.split("/context_")
                            if len(parts) > 1:
                                ctx_id = parts[1].split("/")[0]
                                context_ids.add(ctx_id)
                    context_ids = sorted(
                        context_ids,
                        key=lambda x: int(x) if x.isdigit() else -1,
                        reverse=True,
                    )

                for ctx_id in context_ids:
                    # Try both compressed and uncompressed
                    for compressed in [True, False]:
                        s3_key = self._get_s3_key(
                            job_id, step_name, ctx_id, compressed=compressed
                        )
                        full_s3_key = self.s3_storage._get_s3_key(s3_key)
                        if await self.s3_storage.file_exists(full_s3_key):
                            # Load reference metadata
                            try:
                                content = await self.s3_storage.get_file_content(
                                    full_s3_key
                                )
                                if content:
                                    data = self._load_context_data_from_bytes(
                                        content, compressed
                                    )
                                    reference = ContextReference(
                                        job_id=data.get("job_id", job_id),
                                        step_name=step_name,
                                        step_number=data.get("step_number", 0),
                                        execution_context_id=data.get(
                                            "execution_context_id", ctx_id
                                        ),
                                        timestamp=data.get(
                                            "timestamp", datetime.utcnow().isoformat()
                                        ),
                                        compressed=compressed,
                                    )
                                    # Cache in index
                                    if job_id not in self._reference_index:
                                        self._reference_index[job_id] = {}
                                    self._reference_index[job_id][step_name] = reference
                                    return reference
                            except Exception as e:
                                logger.warning(
                                    f"Failed to load reference from S3 {s3_key}: {e}"
                                )
                                continue
            except Exception as e:
                logger.warning(f"Error searching S3 for context reference: {e}")

        # Try to find from local disk
        job_dir = self._get_job_dir(job_id)
        if not job_dir.exists():
            return None

        # Search context directories
        if execution_context_id:
            context_dirs = [job_dir / f"context_{execution_context_id}"]
        else:
            # Search all context directories
            context_dirs = sorted(
                [
                    d
                    for d in job_dir.iterdir()
                    if d.is_dir() and d.name.startswith("context_")
                ],
                key=lambda x: (
                    int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else -1
                ),
                reverse=True,  # Most recent first
            )

        for context_dir in context_dirs:
            # Try both compressed and uncompressed
            for compressed in [True, False]:
                file_path = self._get_context_file_path(
                    job_id,
                    step_name,
                    context_dir.name.split("_")[1],
                    compressed=compressed,
                )
                if file_path.exists():
                    # Load reference metadata
                    try:
                        data = self._load_context_data(file_path, compressed)
                        reference = ContextReference(
                            job_id=data.get("job_id", job_id),
                            step_name=step_name,
                            step_number=data.get("step_number", 0),
                            execution_context_id=data.get("execution_context_id", "0"),
                            timestamp=data.get(
                                "timestamp", datetime.utcnow().isoformat()
                            ),
                            compressed=compressed,
                        )
                        # Cache in index
                        if job_id not in self._reference_index:
                            self._reference_index[job_id] = {}
                        self._reference_index[job_id][step_name] = reference
                        return reference
                    except Exception as e:
                        logger.warning(
                            f"Failed to load reference from {file_path}: {e}"
                        )
                        continue

        return None

    def _load_context_data(self, file_path: Path, compressed: bool) -> Dict[str, Any]:
        """Load context data from local file."""
        data = file_path.read_bytes()
        return self._load_context_data_from_bytes(data, compressed)

    def _load_context_data_from_bytes(
        self, data: bytes, compressed: bool
    ) -> Dict[str, Any]:
        """Load context data from bytes (works for both S3 and local)."""
        if compressed:
            data = self._decompress_data(data)
        return json.loads(data.decode("utf-8"))

    async def resolve_context(self, reference: ContextReference) -> Dict[str, Any]:
        """
        Resolve a context reference to actual context data (lazy loading).

        Args:
            reference: ContextReference to resolve

        Returns:
            Complete context data dictionary
        """
        cache_key = (
            f"{reference.job_id}:{reference.step_name}:{reference.execution_context_id}"
        )

        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load from S3 or local filesystem
        if self._use_s3 and self.s3_storage:
            try:
                s3_key = self._get_s3_key(
                    reference.job_id,
                    reference.step_name,
                    reference.execution_context_id,
                    compressed=reference.compressed,
                )
                full_s3_key = self.s3_storage._get_s3_key(s3_key)
                content = await self.s3_storage.get_file_content(full_s3_key)
                if content:
                    context_data = self._load_context_data_from_bytes(
                        content, reference.compressed
                    )
                    # Cache it (with size limit)
                    if len(self._cache) >= self._cache_max_size:
                        # Remove oldest entry (FIFO eviction)
                        oldest_key = next(iter(self._cache))
                        del self._cache[oldest_key]
                    self._cache[cache_key] = context_data
                    return context_data
                else:
                    raise FileNotFoundError(
                        f"Context not found in S3: {reference.step_name} in job {reference.job_id}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load from S3, trying local: {e}")
                # Fallback to local

        # Load from local disk
        file_path = self._get_context_file_path(
            reference.job_id,
            reference.step_name,
            reference.execution_context_id,
            compressed=reference.compressed,
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Context not found for reference: {reference.step_name} in job {reference.job_id}"
            )

        try:
            context_data = self._load_context_data(file_path, reference.compressed)
            # Cache it (with size limit)
            if len(self._cache) >= self._cache_max_size:
                # Remove oldest entry (FIFO eviction)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = context_data
            return context_data
        except Exception as e:
            logger.error(f"Failed to resolve context reference {reference}: {e}")
            raise

    async def get_full_history(self, job_id: str) -> Dict[str, Any]:
        """
        Get full context history for a job.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with all step outputs organized by execution context
        """
        history = {}

        if self._use_s3 and self.s3_storage:
            try:
                # List all files for this job in S3
                search_prefix = f"{job_id}/context_"
                keys = await self.s3_storage.list_files(prefix=search_prefix)

                # Organize by execution context
                for key in keys:
                    # Remove S3Storage prefix to get relative path
                    relative_key = key
                    if self.s3_storage.prefix and key.startswith(
                        self.s3_storage.prefix
                    ):
                        relative_key = key[len(self.s3_storage.prefix) :]

                    # Extract context_id and step_name from key
                    # Format: {job_id}/context_{context_id}/{step_name}.json[.gz]
                    if "/context_" in relative_key:
                        parts = relative_key.split("/context_")
                        if len(parts) > 1:
                            context_id_part = parts[1]  # {id}/{step_name}.json[.gz]
                            context_parts = context_id_part.split("/")
                            if len(context_parts) >= 2:
                                execution_context_id = context_parts[0]
                                filename = context_parts[1]
                                step_name = filename.replace(".json.gz", "").replace(
                                    ".json", ""
                                )
                                compressed = filename.endswith(".gz")

                                if execution_context_id not in history:
                                    history[execution_context_id] = {}

                                try:
                                    full_s3_key = self.s3_storage._get_s3_key(
                                        relative_key
                                    )
                                    content = await self.s3_storage.get_file_content(
                                        full_s3_key
                                    )
                                    if content:
                                        context_data = (
                                            self._load_context_data_from_bytes(
                                                content, compressed
                                            )
                                        )
                                        history[execution_context_id][
                                            step_name
                                        ] = context_data
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to load context from S3 {key}: {e}"
                                    )
            except Exception as e:
                logger.warning(f"Error loading history from S3, trying local: {e}")

        # Also check local filesystem
        job_dir = self._get_job_dir(job_id)
        if job_dir.exists():
            # Find all context directories
            context_dirs = sorted(
                [
                    d
                    for d in job_dir.iterdir()
                    if d.is_dir() and d.name.startswith("context_")
                ],
                key=lambda x: (
                    int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else -1
                ),
            )

            for context_dir in context_dirs:
                execution_context_id = context_dir.name.split("_")[1]
                if execution_context_id not in history:
                    history[execution_context_id] = {}

                # Load all step contexts in this execution context
                for file_path in context_dir.glob("*.json*"):
                    step_name = file_path.stem.replace(".json", "")
                    compressed = file_path.suffix == ".gz"

                    # Only load if not already loaded from S3
                    if step_name not in history[execution_context_id]:
                        try:
                            context_data = self._load_context_data(
                                file_path, compressed
                            )
                            history[execution_context_id][step_name] = context_data
                        except Exception as e:
                            logger.warning(
                                f"Failed to load context from {file_path}: {e}"
                            )

        return history

    async def query_context(
        self,
        job_id: str,
        keys: List[str],
        execution_context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query context for specific keys.

        Args:
            job_id: Job identifier
            keys: List of context keys to retrieve (e.g., ["seo_keywords", "marketing_brief"])
            execution_context_id: Optional execution context ID (searches all if not provided)

        Returns:
            Dictionary with requested context keys
        """
        result = {}

        for key in keys:
            reference = await self.get_context_reference(
                job_id, key, execution_context_id
            )
            if reference:
                context_data = await self.resolve_context(reference)
                result[key] = context_data.get("output_data", {})

        return result

    async def get_context(
        self,
        job_id: str,
        key: str,
        step_name: Optional[str] = None,
        execution_context_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get context for a specific key.

        Args:
            job_id: Job identifier
            key: Context key to retrieve
            step_name: Optional step name (if different from key)
            execution_context_id: Optional execution context ID

        Returns:
            Context data if found, None otherwise
        """
        target_step = step_name or key
        reference = await self.get_context_reference(
            job_id, target_step, execution_context_id
        )

        if not reference:
            return None

        context_data = await self.resolve_context(reference)
        return context_data.get("output_data")


# Singleton instance
_context_registry: Optional[ContextRegistry] = None


def get_context_registry() -> ContextRegistry:
    """Get the global context registry instance."""
    global _context_registry
    if _context_registry is None:
        _context_registry = ContextRegistry()
    return _context_registry
