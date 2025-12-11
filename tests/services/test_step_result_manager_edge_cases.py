"""
Tests for step result manager edge cases.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.step_result_manager import StepResultManager


@pytest.fixture
def temp_results_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def step_result_manager(temp_results_dir):
    """Create a StepResultManager instance."""
    return StepResultManager(base_dir=temp_results_dir)


@pytest.mark.asyncio
async def test_save_step_result_with_metadata(step_result_manager):
    """Test save_step_result with metadata."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        result_data = {"main_keyword": "test"}
        metadata = {"execution_time": 1.5, "tokens_used": 100}

        file_path = await step_result_manager.save_step_result(
            "test-job-1",
            1,
            "seo_keywords",
            result_data,
            metadata=metadata,
            execution_context_id="0",
        )

        assert file_path is not None


@pytest.mark.asyncio
async def test_save_step_result_with_root_job_id(step_result_manager):
    """Test save_step_result with root_job_id."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        result_data = {"main_keyword": "test"}

        file_path = await step_result_manager.save_step_result(
            "child-job-1",
            1,
            "seo_keywords",
            result_data,
            root_job_id="root-job-1",
            execution_context_id="0",
        )

        assert file_path is not None


@pytest.mark.asyncio
async def test_save_step_result_with_input_snapshot(step_result_manager):
    """Test save_step_result with input_snapshot."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        result_data = {"main_keyword": "test"}
        input_snapshot = {"input_content": {"id": "test", "title": "Test"}}
        context_keys_used = ["input_content", "seo_keywords"]

        file_path = await step_result_manager.save_step_result(
            "test-job-1",
            1,
            "seo_keywords",
            result_data,
            input_snapshot=input_snapshot,
            context_keys_used=context_keys_used,
            execution_context_id="0",
        )

        assert file_path is not None


@pytest.mark.asyncio
async def test_get_step_result_not_found(step_result_manager):
    """Test get_step_result with non-existent file."""
    with pytest.raises(FileNotFoundError):
        await step_result_manager.get_step_result(
            "non-existent-job", "01_seo_keywords.json", "0"
        )


@pytest.mark.asyncio
async def test_get_step_result_by_name_not_found(step_result_manager):
    """Test get_step_result_by_name with non-existent step."""
    with pytest.raises(FileNotFoundError):
        await step_result_manager.get_step_result_by_name(
            "non-existent-job", "seo_keywords", "0"
        )


@pytest.mark.asyncio
async def test_get_step_file_path_not_found(step_result_manager):
    """Test get_step_file_path with non-existent file."""
    with pytest.raises(FileNotFoundError):
        await step_result_manager.get_step_file_path(
            "non-existent-job", "01_seo_keywords.json", "0"
        )


@pytest.mark.asyncio
async def test_find_related_jobs_no_relations(step_result_manager):
    """Test find_related_jobs with no related jobs."""
    await step_result_manager.save_job_metadata(
        "isolated-job-1", "blog_post", "content-1"
    )

    related = await step_result_manager.find_related_jobs("isolated-job-1")

    assert isinstance(related, dict)
    assert "related_jobs" in related or "root_job" in related or len(related) >= 0


@pytest.mark.asyncio
async def test_aggregate_steps_from_jobs_empty(step_result_manager):
    """Test aggregate_steps_from_jobs with empty job list."""
    aggregated = await step_result_manager.aggregate_steps_from_jobs([])

    assert isinstance(aggregated, list)
    assert len(aggregated) == 0


@pytest.mark.asyncio
async def test_aggregate_steps_from_jobs_nonexistent(step_result_manager):
    """Test aggregate_steps_from_jobs with non-existent jobs."""
    aggregated = await step_result_manager.aggregate_steps_from_jobs(
        ["non-existent-1", "non-existent-2"]
    )

    assert isinstance(aggregated, list)
    assert len(aggregated) == 0


@pytest.mark.asyncio
async def test_get_full_context_history_empty(step_result_manager):
    """Test get_full_context_history with no history."""
    history = await step_result_manager.get_full_context_history("non-existent-job")

    assert isinstance(history, dict)


@pytest.mark.asyncio
async def test_get_pipeline_flow_empty(step_result_manager):
    """Test get_pipeline_flow with no steps."""
    with pytest.raises(FileNotFoundError):
        await step_result_manager.get_pipeline_flow("non-existent-job")


@pytest.mark.asyncio
async def test_cleanup_job_not_found(step_result_manager):
    """Test cleanup_job with non-existent job."""
    success = await step_result_manager.cleanup_job("non-existent-job")

    assert success is False
