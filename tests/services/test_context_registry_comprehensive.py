"""
Comprehensive tests for context registry service covering missed lines.
"""

import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.context_registry import (
    ContextReference,
    ContextRegistry,
    get_context_registry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for context files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def registry(temp_dir):
    """ContextRegistry using local filesystem without compression."""
    return ContextRegistry(base_dir=temp_dir, enable_compression=False)


@pytest.fixture
def compressed_registry(temp_dir):
    """ContextRegistry using local filesystem with compression enabled."""
    return ContextRegistry(base_dir=temp_dir, enable_compression=True)


def _make_mock_job_manager(job=None):
    """Return a mock job manager that returns the given job on get_job()."""
    mock_jm = MagicMock()
    mock_jm.get_job = AsyncMock(return_value=job)
    return mock_jm


def _patch_job_manager(job=None):
    """Return a context manager that patches get_job_manager in job_manager module."""
    mock_jm = _make_mock_job_manager(job)
    return patch(
        "marketing_project.services.job_manager.get_job_manager",
        return_value=mock_jm,
    )


# ---------------------------------------------------------------------------
# ContextReference tests
# ---------------------------------------------------------------------------


def test_context_reference_from_dict_missing_compressed_defaults_false():
    """from_dict uses False as default for missing 'compressed' key."""
    data = {
        "job_id": "j1",
        "step_name": "step_a",
        "step_number": 1,
        "execution_context_id": "0",
        "timestamp": "2024-01-01T00:00:00Z",
        # no 'compressed' key
    }
    ref = ContextReference.from_dict(data)
    assert ref.compressed is False


def test_context_reference_to_dict_round_trip():
    """to_dict / from_dict round-trip preserves all fields."""
    ref = ContextReference(
        job_id="job-99",
        step_name="my_step",
        step_number=5,
        execution_context_id="ctx-42",
        timestamp="2024-06-15T12:00:00+00:00",
        compressed=True,
    )
    restored = ContextReference.from_dict(ref.to_dict())
    assert restored.job_id == ref.job_id
    assert restored.step_name == ref.step_name
    assert restored.step_number == ref.step_number
    assert restored.execution_context_id == ref.execution_context_id
    assert restored.timestamp == ref.timestamp
    assert restored.compressed == ref.compressed


# ---------------------------------------------------------------------------
# ContextRegistry initialisation
# ---------------------------------------------------------------------------


def test_registry_creates_base_dir(temp_dir):
    """ContextRegistry creates the base directory on init."""
    new_path = Path(temp_dir) / "new_registry"
    assert not new_path.exists()
    ContextRegistry(base_dir=str(new_path), enable_compression=False)
    assert new_path.exists()


def test_registry_no_s3_when_env_not_set(temp_dir):
    """ContextRegistry._use_s3 is False when AWS_S3_BUCKET is not set."""
    with patch.dict("os.environ", {}, clear=False):
        # Remove the env var if present
        import os

        os.environ.pop("AWS_S3_BUCKET", None)
        reg = ContextRegistry(base_dir=temp_dir, enable_compression=False)
    assert reg._use_s3 is False
    assert reg.s3_storage is None


def test_registry_cache_max_size_from_env(temp_dir):
    """CONTEXT_REGISTRY_CACHE_SIZE env var controls cache max size."""
    with patch.dict("os.environ", {"CONTEXT_REGISTRY_CACHE_SIZE": "42"}):
        reg = ContextRegistry(base_dir=temp_dir, enable_compression=False)
    assert reg._cache_max_size == 42


# ---------------------------------------------------------------------------
# Internal helper methods
# ---------------------------------------------------------------------------


def test_get_job_dir_creates_directory(registry, temp_dir):
    """_get_job_dir creates job-specific subdirectory."""
    job_dir = registry._get_job_dir("my-job-123")
    assert job_dir.exists()
    assert job_dir.name == "my-job-123"


def test_get_context_file_path_uncompressed(registry, temp_dir):
    """_get_context_file_path returns correct .json path when not compressed."""
    path = registry._get_context_file_path("job1", "step_a", "0", compressed=False)
    assert path.suffix == ".json"
    assert "step_a" in path.name


def test_get_context_file_path_compressed(registry, temp_dir):
    """_get_context_file_path returns .json.gz path when compressed."""
    path = registry._get_context_file_path("job1", "step_a", "0", compressed=True)
    assert path.name.endswith(".json.gz")


def test_get_s3_key_uncompressed(registry):
    """_get_s3_key produces expected key without .gz extension."""
    key = registry._get_s3_key("job1", "step_a", "ctx0", compressed=False)
    assert key == "job1/context_ctx0/step_a.json"


def test_get_s3_key_compressed(registry):
    """_get_s3_key adds .gz to key when compressed."""
    key = registry._get_s3_key("job1", "step_a", "ctx0", compressed=True)
    assert key == "job1/context_ctx0/step_a.json.gz"


def test_compress_decompress_round_trip(registry):
    """_compress_data and _decompress_data are inverse operations."""
    original = b"hello world this is some test data"
    compressed = registry._compress_data(original)
    assert compressed != original
    result = registry._decompress_data(compressed)
    assert result == original


def test_load_context_data_from_bytes_uncompressed(registry):
    """_load_context_data_from_bytes handles plain JSON bytes."""
    data = json.dumps({"key": "value"}).encode("utf-8")
    result = registry._load_context_data_from_bytes(data, compressed=False)
    assert result == {"key": "value"}


def test_load_context_data_from_bytes_compressed(registry):
    """_load_context_data_from_bytes handles gzip-compressed JSON."""
    payload = json.dumps({"key": "compressed_value"}).encode("utf-8")
    compressed = gzip.compress(payload)
    result = registry._load_context_data_from_bytes(compressed, compressed=True)
    assert result == {"key": "compressed_value"}


def test_load_context_data_from_file(registry, temp_dir):
    """_load_context_data reads and deserializes a local JSON file."""
    file_path = Path(temp_dir) / "test_data.json"
    payload = {"step": "test", "value": 42}
    file_path.write_bytes(json.dumps(payload).encode("utf-8"))
    result = registry._load_context_data(file_path, compressed=False)
    assert result["step"] == "test"
    assert result["value"] == 42


# ---------------------------------------------------------------------------
# register_step_output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_step_output_creates_file(registry, temp_dir):
    """register_step_output writes a JSON file to disk."""
    # Always pass execution_context_id to bypass get_job_manager / get_step_result_manager lookups
    ref = await registry.register_step_output(
        job_id="job-file",
        step_name="step_one",
        step_number=1,
        output_data={"result": "ok"},
        execution_context_id="ctx0",
        root_job_id="job-file",
    )

    assert ref.job_id == "job-file"
    assert ref.step_name == "step_one"
    assert ref.execution_context_id == "ctx0"
    assert ref.compressed is False

    # File should exist
    file_path = registry._get_context_file_path(
        "job-file", "step_one", "ctx0", compressed=False
    )
    assert file_path.exists()


@pytest.mark.asyncio
async def test_register_step_output_compressed(compressed_registry, temp_dir):
    """register_step_output writes a .json.gz file when compression is on."""
    ref = await compressed_registry.register_step_output(
        job_id="job-gz",
        step_name="step_gz",
        step_number=2,
        output_data={"data": "compressed"},
        execution_context_id="ctx1",
        root_job_id="job-gz",
    )

    assert ref.compressed is True
    file_path = compressed_registry._get_context_file_path(
        "job-gz", "step_gz", "ctx1", compressed=True
    )
    assert file_path.exists()


@pytest.mark.asyncio
async def test_register_step_output_updates_cache(registry):
    """register_step_output puts context data into the in-memory cache."""
    ref = await registry.register_step_output(
        job_id="job-cache",
        step_name="step_cache",
        step_number=1,
        output_data={"cached": True},
        execution_context_id="ctx0",
        root_job_id="job-cache",
    )

    cache_key = "job-cache:step_cache:ctx0"
    assert cache_key in registry._cache
    assert registry._cache[cache_key]["output_data"] == {"cached": True}


@pytest.mark.asyncio
async def test_register_step_output_updates_reference_index(registry):
    """register_step_output populates the reference index."""
    await registry.register_step_output(
        job_id="job-idx",
        step_name="indexed_step",
        step_number=3,
        output_data={"x": 1},
        execution_context_id="ctx0",
        root_job_id="job-idx",
    )

    assert "job-idx" in registry._reference_index
    assert "indexed_step" in registry._reference_index["job-idx"]


@pytest.mark.asyncio
async def test_register_step_output_cache_eviction(registry):
    """register_step_output evicts oldest entry when cache is full."""
    registry._cache_max_size = 2

    for i in range(3):
        await registry.register_step_output(
            job_id=f"job-evict-{i}",
            step_name="step",
            step_number=i,
            output_data={"i": i},
            execution_context_id="ctx0",
            root_job_id=f"job-evict-{i}",
        )

    # Cache should not exceed max size
    assert len(registry._cache) <= registry._cache_max_size


@pytest.mark.asyncio
async def test_register_step_output_with_input_snapshot(registry):
    """register_step_output stores input_snapshot and context_keys_used."""
    ref = await registry.register_step_output(
        job_id="job-snap",
        step_name="snap_step",
        step_number=1,
        output_data={"result": "snap"},
        input_snapshot={"input_key": "input_val"},
        context_keys_used=["key1", "key2"],
        execution_context_id="ctx0",
        root_job_id="job-snap",
    )

    cache_key = "job-snap:snap_step:ctx0"
    cached = registry._cache[cache_key]
    assert cached["input_snapshot"] == {"input_key": "input_val"}
    assert cached["context_keys_used"] == ["key1", "key2"]


# ---------------------------------------------------------------------------
# get_context_reference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_context_reference_from_memory_index(registry):
    """get_context_reference returns from in-memory index if available."""
    await registry.register_step_output(
        job_id="job-ref",
        step_name="my_step",
        step_number=1,
        output_data={"val": 1},
        execution_context_id="ctx0",
        root_job_id="job-ref",
    )

    ref = await registry.get_context_reference("job-ref", "my_step")
    assert ref is not None
    assert ref.step_name == "my_step"


@pytest.mark.asyncio
async def test_get_context_reference_not_found_returns_none(registry):
    """get_context_reference returns None when no context exists."""
    ref = await registry.get_context_reference("nonexistent-job", "ghost_step")
    assert ref is None


@pytest.mark.asyncio
async def test_get_context_reference_from_disk(registry, temp_dir):
    """get_context_reference finds context from local disk when not in index."""
    # Manually write context file to disk (bypassing the index)
    job_id = "job-disk"
    step_name = "disk_step"
    execution_context_id = "42"
    context_data = {
        "job_id": job_id,
        "step_name": step_name,
        "step_number": 1,
        "execution_context_id": execution_context_id,
        "timestamp": "2024-01-01T00:00:00+00:00",
        "output_data": {"from": "disk"},
        "input_snapshot": None,
        "context_keys_used": [],
        "root_job_id": job_id,
    }
    file_path = registry._get_context_file_path(
        job_id, step_name, execution_context_id, compressed=False
    )
    file_path.write_bytes(json.dumps(context_data).encode("utf-8"))

    # Clear index to force disk lookup
    registry._reference_index.clear()

    ref = await registry.get_context_reference(job_id, step_name)
    assert ref is not None
    assert ref.step_name == step_name


@pytest.mark.asyncio
async def test_get_context_reference_specific_execution_context_id(registry, temp_dir):
    """get_context_reference uses specific execution_context_id when provided."""
    job_id = "job-specific"
    step_name = "specific_step"
    execution_context_id = "77"
    context_data = {
        "job_id": job_id,
        "step_name": step_name,
        "step_number": 2,
        "execution_context_id": execution_context_id,
        "timestamp": "2024-01-01T00:00:00+00:00",
        "output_data": {"specific": True},
        "input_snapshot": None,
        "context_keys_used": [],
        "root_job_id": job_id,
    }
    file_path = registry._get_context_file_path(
        job_id, step_name, execution_context_id, compressed=False
    )
    file_path.write_bytes(json.dumps(context_data).encode("utf-8"))
    registry._reference_index.clear()

    ref = await registry.get_context_reference(
        job_id, step_name, execution_context_id=execution_context_id
    )
    assert ref is not None
    assert ref.execution_context_id == execution_context_id


# ---------------------------------------------------------------------------
# resolve_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_context_from_cache(registry):
    """resolve_context returns data from cache without hitting disk."""
    ref = ContextReference(
        job_id="job-cached",
        step_name="cached_step",
        step_number=1,
        execution_context_id="ctx0",
        timestamp="2024-01-01T00:00:00+00:00",
        compressed=False,
    )
    cache_key = "job-cached:cached_step:ctx0"
    registry._cache[cache_key] = {"output_data": {"from": "cache"}}

    result = await registry.resolve_context(ref)

    assert result == {"output_data": {"from": "cache"}}


@pytest.mark.asyncio
async def test_resolve_context_from_disk(registry, temp_dir):
    """resolve_context loads data from disk when not in cache."""
    job_id = "job-disk-resolve"
    step_name = "resolve_step"
    execution_context_id = "ctx0"
    context_data = {
        "job_id": job_id,
        "step_name": step_name,
        "step_number": 1,
        "execution_context_id": execution_context_id,
        "timestamp": "2024-01-01T00:00:00+00:00",
        "output_data": {"loaded": "from_disk"},
        "input_snapshot": None,
        "context_keys_used": [],
        "root_job_id": job_id,
    }
    file_path = registry._get_context_file_path(
        job_id, step_name, execution_context_id, compressed=False
    )
    file_path.write_bytes(json.dumps(context_data).encode("utf-8"))

    ref = ContextReference(
        job_id=job_id,
        step_name=step_name,
        step_number=1,
        execution_context_id=execution_context_id,
        timestamp="2024-01-01T00:00:00+00:00",
        compressed=False,
    )

    result = await registry.resolve_context(ref)
    assert result["output_data"] == {"loaded": "from_disk"}


@pytest.mark.asyncio
async def test_resolve_context_file_not_found_raises(registry, temp_dir):
    """resolve_context raises FileNotFoundError when file is missing."""
    ref = ContextReference(
        job_id="no-job",
        step_name="no_step",
        step_number=1,
        execution_context_id="ctx0",
        timestamp="2024-01-01T00:00:00+00:00",
        compressed=False,
    )
    with pytest.raises(FileNotFoundError):
        await registry.resolve_context(ref)


@pytest.mark.asyncio
async def test_resolve_context_caches_loaded_data(registry, temp_dir):
    """resolve_context caches data after loading from disk."""
    job_id = "job-cache-load"
    step_name = "cache_step"
    execution_context_id = "ctx0"
    context_data = {
        "job_id": job_id,
        "step_name": step_name,
        "step_number": 1,
        "execution_context_id": execution_context_id,
        "timestamp": "2024-01-01T00:00:00+00:00",
        "output_data": {"cached_after_load": True},
        "input_snapshot": None,
        "context_keys_used": [],
        "root_job_id": job_id,
    }
    file_path = registry._get_context_file_path(
        job_id, step_name, execution_context_id, compressed=False
    )
    file_path.write_bytes(json.dumps(context_data).encode("utf-8"))

    ref = ContextReference(
        job_id=job_id,
        step_name=step_name,
        step_number=1,
        execution_context_id=execution_context_id,
        timestamp="2024-01-01T00:00:00+00:00",
        compressed=False,
    )

    await registry.resolve_context(ref)

    cache_key = f"{job_id}:{step_name}:{execution_context_id}"
    assert cache_key in registry._cache


@pytest.mark.asyncio
async def test_resolve_context_cache_eviction(registry, temp_dir):
    """resolve_context evicts oldest cache entry when cache is full."""
    registry._cache_max_size = 2

    # Pre-fill cache
    registry._cache["old-key-1"] = {"output_data": {}}
    registry._cache["old-key-2"] = {"output_data": {}}

    job_id = "job-evict-resolve"
    step_name = "evict_step"
    execution_context_id = "ctx0"
    context_data = {
        "job_id": job_id,
        "step_name": step_name,
        "step_number": 1,
        "execution_context_id": execution_context_id,
        "timestamp": "2024-01-01T00:00:00+00:00",
        "output_data": {"new": True},
        "input_snapshot": None,
        "context_keys_used": [],
        "root_job_id": job_id,
    }
    file_path = registry._get_context_file_path(
        job_id, step_name, execution_context_id, compressed=False
    )
    file_path.write_bytes(json.dumps(context_data).encode("utf-8"))

    ref = ContextReference(
        job_id=job_id,
        step_name=step_name,
        step_number=1,
        execution_context_id=execution_context_id,
        timestamp="2024-01-01T00:00:00+00:00",
        compressed=False,
    )

    await registry.resolve_context(ref)

    assert len(registry._cache) <= registry._cache_max_size


# ---------------------------------------------------------------------------
# get_full_history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_full_history_empty_for_unknown_job(registry):
    """get_full_history returns empty dict for unknown job."""
    history = await registry.get_full_history("nonexistent-job-xyz")
    assert history == {}


@pytest.mark.asyncio
async def test_get_full_history_returns_all_steps(registry, temp_dir):
    """get_full_history collects all step contexts for a job."""
    await registry.register_step_output(
        "job-hist",
        "step_a",
        1,
        {"a": 1},
        execution_context_id="ctx0",
        root_job_id="job-hist",
    )
    await registry.register_step_output(
        "job-hist",
        "step_b",
        2,
        {"b": 2},
        execution_context_id="ctx0",
        root_job_id="job-hist",
    )

    history = await registry.get_full_history("job-hist")

    assert isinstance(history, dict)
    assert "ctx0" in history
    assert "step_a" in history["ctx0"]
    assert "step_b" in history["ctx0"]


@pytest.mark.asyncio
async def test_get_full_history_multiple_contexts(registry, temp_dir):
    """get_full_history handles multiple execution contexts."""
    await registry.register_step_output(
        "job-multi",
        "step_a",
        1,
        {"a": 1},
        execution_context_id="0",
        root_job_id="job-multi",
    )
    await registry.register_step_output(
        "job-multi",
        "step_a",
        1,
        {"a": 2},
        execution_context_id="1",
        root_job_id="job-multi",
    )

    history = await registry.get_full_history("job-multi")

    assert "0" in history
    assert "1" in history


# ---------------------------------------------------------------------------
# query_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_context_returns_output_data(registry):
    """query_context retrieves output_data for requested keys."""
    await registry.register_step_output(
        "job-q",
        "seo_keywords",
        1,
        {"main_keyword": "ai"},
        execution_context_id="ctx0",
        root_job_id="job-q",
    )

    result = await registry.query_context("job-q", ["seo_keywords"])

    assert "seo_keywords" in result
    assert result["seo_keywords"] == {"main_keyword": "ai"}


@pytest.mark.asyncio
async def test_query_context_missing_key_not_in_result(registry):
    """query_context skips keys that don't exist in the registry."""
    result = await registry.query_context("no-job", ["nonexistent_key"])
    assert "nonexistent_key" not in result


@pytest.mark.asyncio
async def test_query_context_multiple_keys(registry):
    """query_context retrieves multiple keys at once."""
    await registry.register_step_output(
        "job-multi-q",
        "step_x",
        1,
        {"x_val": 1},
        execution_context_id="ctx0",
        root_job_id="job-multi-q",
    )
    await registry.register_step_output(
        "job-multi-q",
        "step_y",
        2,
        {"y_val": 2},
        execution_context_id="ctx0",
        root_job_id="job-multi-q",
    )

    result = await registry.query_context("job-multi-q", ["step_x", "step_y"])

    assert result["step_x"] == {"x_val": 1}
    assert result["step_y"] == {"y_val": 2}


# ---------------------------------------------------------------------------
# get_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_context_returns_output_data(registry):
    """get_context returns the output_data dict for a registered step."""
    await registry.register_step_output(
        "job-gc",
        "gc_step",
        1,
        {"gc_result": "hello"},
        execution_context_id="ctx0",
        root_job_id="job-gc",
    )

    result = await registry.get_context(
        "job-gc", "gc_step", execution_context_id="ctx0"
    )

    assert result == {"gc_result": "hello"}


@pytest.mark.asyncio
async def test_get_context_missing_returns_none(registry):
    """get_context returns None when step does not exist."""
    result = await registry.get_context("no-job", "no_step")
    assert result is None


@pytest.mark.asyncio
async def test_get_context_step_name_overrides_key(registry):
    """get_context uses step_name parameter when provided instead of key."""
    await registry.register_step_output(
        "job-alias",
        "actual_step",
        1,
        {"alias": True},
        execution_context_id="ctx0",
        root_job_id="job-alias",
    )

    result = await registry.get_context(
        "job-alias",
        key="alias_key",
        step_name="actual_step",
        execution_context_id="ctx0",
    )

    assert result == {"alias": True}


# ---------------------------------------------------------------------------
# S3 initialisation failure fallback
# ---------------------------------------------------------------------------


def test_registry_s3_init_exception_falls_back_to_local(temp_dir):
    """ContextRegistry falls back to local when S3Storage init raises."""
    mock_s3_module = MagicMock()
    # Make S3Storage constructor raise an exception
    mock_s3_module.S3Storage.side_effect = Exception("S3 init error")

    with patch.dict("os.environ", {"AWS_S3_BUCKET": "test-bucket"}):
        with patch.dict(
            "sys.modules", {"marketing_project.services.s3_storage": mock_s3_module}
        ):
            reg = ContextRegistry(base_dir=temp_dir, enable_compression=False)

    # Should have fallen back to local
    assert reg._use_s3 is False
    assert reg.s3_storage is None
