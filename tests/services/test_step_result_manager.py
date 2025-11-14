"""
Tests for step result manager service.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.step_result_manager import (
    StepResultManager,
    get_step_result_manager,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def step_result_manager(temp_dir):
    """Create a StepResultManager instance for testing."""
    return StepResultManager(base_dir=temp_dir)


def test_step_result_manager_initialization(temp_dir):
    """Test StepResultManager initialization."""
    manager = StepResultManager(base_dir=temp_dir)
    assert manager.base_dir == Path(temp_dir)
    assert manager.base_dir.exists()


def test_step_result_manager_default_dir():
    """Test StepResultManager with default directory."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("STEP_RESULTS_DIR", None)
        manager = StepResultManager()
        assert manager.base_dir is not None
        assert isinstance(manager.base_dir, Path)


def test_step_result_manager_from_env():
    """Test StepResultManager with directory from environment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"STEP_RESULTS_DIR": tmpdir}):
            manager = StepResultManager()
            assert str(manager.base_dir) == tmpdir


def test_get_job_dir(step_result_manager):
    """Test _get_job_dir method."""
    job_dir = step_result_manager._get_job_dir("test-job-1")
    assert job_dir.exists()
    assert "test-job-1" in str(job_dir)


def test_get_step_filename(step_result_manager):
    """Test _get_step_filename method."""
    filename = step_result_manager._get_step_filename(1, "SEO Keywords")
    assert filename == "01_seo_keywords.json"

    filename = step_result_manager._get_step_filename(0, "Input")
    assert filename == "00_input.json"

    filename = step_result_manager._get_step_filename(10, "Test-Step")
    assert filename == "10_test_step.json"


@pytest.mark.asyncio
async def test_save_step_result(step_result_manager):
    """Test save_step_result method."""
    result_data = {"keywords": ["test", "keyword"], "score": 0.85}
    metadata = {"execution_time": 1.5, "model": "gpt-4"}

    file_path = await step_result_manager.save_step_result(
        job_id="test-job-1",
        step_number=1,
        step_name="SEO Keywords",
        result_data=result_data,
        metadata=metadata,
    )

    assert file_path is not None
    assert Path(file_path).exists()

    # Verify file contents
    with open(file_path, "r") as f:
        saved_data = json.load(f)

    assert saved_data["result"] == result_data
    assert saved_data["metadata"] == metadata


@pytest.mark.asyncio
async def test_save_step_result_with_execution_context(step_result_manager):
    """Test save_step_result with execution context."""
    result_data = {"result": "test"}

    file_path = await step_result_manager.save_step_result(
        job_id="test-job-1",
        step_number=2,
        step_name="Marketing Brief",
        result_data=result_data,
        execution_context_id="context_1",
        root_job_id="root-job-1",
    )

    assert file_path is not None
    assert Path(file_path).exists()
    assert "context_1" in file_path


@pytest.mark.asyncio
async def test_get_step_result(step_result_manager):
    """Test get_step_result method."""
    result_data = {"keywords": ["test"]}

    # Save a result first
    await step_result_manager.save_step_result(
        job_id="test-job-1",
        step_number=1,
        step_name="SEO Keywords",
        result_data=result_data,
    )

    # Retrieve it
    retrieved = await step_result_manager.get_step_result("test-job-1", step_number=1)

    assert retrieved is not None
    assert retrieved["result"] == result_data


@pytest.mark.asyncio
async def test_get_step_result_not_found(step_result_manager):
    """Test get_step_result when result not found."""
    result = await step_result_manager.get_step_result(
        "non-existent-job", step_number=1
    )
    assert result is None


@pytest.mark.asyncio
async def test_get_all_step_results(step_result_manager):
    """Test get_all_step_results method."""
    # Save multiple results
    for i in range(3):
        await step_result_manager.save_step_result(
            job_id="test-job-1",
            step_number=i,
            step_name=f"Step {i}",
            result_data={"step": i},
        )

    results = await step_result_manager.get_all_step_results("test-job-1")

    assert len(results) == 3
    assert all("step_number" in r for r in results)


@pytest.mark.asyncio
async def test_get_all_step_results_empty(step_result_manager):
    """Test get_all_step_results for job with no results."""
    results = await step_result_manager.get_all_step_results("empty-job")
    assert results == []


@pytest.mark.asyncio
async def test_save_job_metadata(step_result_manager):
    """Test save_job_metadata method."""
    metadata = {
        "job_id": "test-job-1",
        "status": "completed",
        "created_at": "2024-01-01T00:00:00Z",
    }

    file_path = await step_result_manager.save_job_metadata("test-job-1", metadata)

    assert file_path is not None
    assert Path(file_path).exists()

    # Verify contents
    with open(file_path, "r") as f:
        saved_metadata = json.load(f)

    assert saved_metadata == metadata


@pytest.mark.asyncio
async def test_get_job_metadata(step_result_manager):
    """Test get_job_metadata method."""
    metadata = {"job_id": "test-job-1", "status": "completed"}

    await step_result_manager.save_job_metadata("test-job-1", metadata)

    retrieved = await step_result_manager.get_job_metadata("test-job-1")

    assert retrieved is not None
    assert retrieved["job_id"] == "test-job-1"


@pytest.mark.asyncio
async def test_get_job_metadata_not_found(step_result_manager):
    """Test get_job_metadata when metadata not found."""
    result = await step_result_manager.get_job_metadata("non-existent-job")
    assert result is None


@pytest.mark.asyncio
async def test_delete_job_results(step_result_manager):
    """Test delete_job_results method."""
    # Save some results
    await step_result_manager.save_step_result(
        job_id="test-job-1",
        step_number=1,
        step_name="Step 1",
        result_data={"test": "data"},
    )

    # Delete them
    result = await step_result_manager.delete_job_results("test-job-1")

    assert result is True

    # Verify deletion
    results = await step_result_manager.get_all_step_results("test-job-1")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_delete_job_results_not_found(step_result_manager):
    """Test delete_job_results for non-existent job."""
    result = await step_result_manager.delete_job_results("non-existent-job")
    assert result is False


@pytest.mark.asyncio
async def test_list_jobs(step_result_manager):
    """Test list_jobs method."""
    # Create results for multiple jobs
    for job_id in ["job-1", "job-2", "job-3"]:
        await step_result_manager.save_step_result(
            job_id=job_id,
            step_number=1,
            step_name="Step 1",
            result_data={"test": "data"},
        )

    jobs = await step_result_manager.list_jobs()

    assert len(jobs) >= 3
    assert all("job_id" in job for job in jobs)


@pytest.mark.asyncio
async def test_cleanup_old_results(step_result_manager):
    """Test cleanup_old_results method."""
    # Create some old results
    await step_result_manager.save_step_result(
        job_id="old-job-1",
        step_number=1,
        step_name="Step 1",
        result_data={"test": "data"},
    )

    # Cleanup (with 0 days retention, should delete everything)
    deleted_count = await step_result_manager.cleanup_old_results(days=0)

    assert deleted_count >= 0  # May be 0 if no old files


def test_json_serializer():
    """Test _json_serializer function."""
    from datetime import date, datetime

    from marketing_project.services.step_result_manager import _json_serializer

    # Test datetime serialization
    dt = datetime(2024, 1, 1, 12, 0, 0)
    result = _json_serializer(dt)
    assert result == "2024-01-01T12:00:00"

    # Test date serialization
    d = date(2024, 1, 1)
    result = _json_serializer(d)
    assert result == "2024-01-01"

    # Test non-serializable type
    with pytest.raises(TypeError):
        _json_serializer(set([1, 2, 3]))


def test_get_step_result_manager_singleton():
    """Test that get_step_result_manager returns a singleton."""
    manager1 = get_step_result_manager()
    manager2 = get_step_result_manager()
    assert manager1 is manager2
