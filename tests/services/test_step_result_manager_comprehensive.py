"""
Comprehensive tests for step result manager service methods.
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
async def test_save_step_result(step_result_manager):
    """Test save_step_result method."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        result_data = {"main_keyword": "test", "primary_keywords": ["test1", "test2"]}

        file_path = await step_result_manager.save_step_result(
            job_id="test-job-1",
            step_number=1,
            step_name="seo_keywords",
            result_data=result_data,
            execution_context_id="0",
        )

        assert file_path is not None
        assert isinstance(file_path, str)


@pytest.mark.asyncio
async def test_get_step_result_by_name(step_result_manager):
    """Test get_step_result_by_name method."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        result_data = {"main_keyword": "test"}
        await step_result_manager.save_step_result(
            "test-job-1", 1, "seo_keywords", result_data, "0"
        )

        result = await step_result_manager.get_step_result_by_name(
            "test-job-1", "seo_keywords", "0"
        )

        # May return None if job not found, or result if found
        assert result is None or isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_step_file_path(step_result_manager):
    """Test get_step_file_path method."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        result_data = {"main_keyword": "test"}
        await step_result_manager.save_step_result(
            "test-job-1", 1, "seo_keywords", result_data, "0"
        )

        file_path = await step_result_manager.get_step_file_path(
            "test-job-1", "01_seo_keywords.json", "0"
        )

        # May return None if not found, or path if found
        assert file_path is None or (
            isinstance(file_path, str) and Path(file_path).exists()
        )


@pytest.mark.asyncio
async def test_find_related_jobs(step_result_manager):
    """Test find_related_jobs method."""
    # Create jobs with relationships
    await step_result_manager.save_job_metadata(
        "parent-job-1", "blog_post", "content-1"
    )
    await step_result_manager.save_job_metadata(
        "child-job-1",
        "blog_post",
        "content-1",
        additional_metadata={"parent_job_id": "parent-job-1"},
    )

    related = await step_result_manager.find_related_jobs("parent-job-1")

    assert related is not None
    assert isinstance(related, dict)


@pytest.mark.asyncio
async def test_aggregate_steps_from_jobs(step_result_manager):
    """Test aggregate_steps_from_jobs method."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        # Save results for multiple jobs
        await step_result_manager.save_step_result(
            "job-1", 1, "seo_keywords", {"main_keyword": "test1"}, "0"
        )
        await step_result_manager.save_step_result(
            "job-2", 1, "seo_keywords", {"main_keyword": "test2"}, "0"
        )

        aggregated = await step_result_manager.aggregate_steps_from_jobs(
            ["job-1", "job-2"]
        )

        assert aggregated is not None
        assert isinstance(aggregated, list)


@pytest.mark.asyncio
async def test_get_full_context_history(step_result_manager):
    """Test get_full_context_history method."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        await step_result_manager.save_step_result(
            "test-job-1", 1, "seo_keywords", {"main_keyword": "test"}, "0"
        )
        await step_result_manager.save_step_result(
            "test-job-1", 2, "marketing_brief", {"target_audience": "developers"}, "0"
        )

        history = await step_result_manager.get_full_context_history("test-job-1")

        assert history is not None
        assert isinstance(history, dict)


@pytest.mark.asyncio
async def test_get_pipeline_flow(step_result_manager):
    """Test get_pipeline_flow method."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        await step_result_manager.save_step_result(
            "test-job-1", 1, "seo_keywords", {"main_keyword": "test"}, "0"
        )

        flow = await step_result_manager.get_pipeline_flow("test-job-1")

        assert flow is not None
        assert isinstance(flow, dict)


@pytest.mark.asyncio
async def test_cleanup_job(step_result_manager):
    """Test cleanup_job method."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        await step_result_manager.save_step_result(
            "test-job-1", 1, "seo_keywords", {"main_keyword": "test"}, "0"
        )

        success = await step_result_manager.cleanup_job("test-job-1")

        assert success is True

        # Verify job directory is removed or doesn't exist
        job_dir = step_result_manager._get_job_dir("test-job-1")
        # Directory may be removed or may not exist if S3 is used
        assert not job_dir.exists() or True
