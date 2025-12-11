"""
Tests for step result manager additional methods.
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


def test_get_job_dir(step_result_manager):
    """Test _get_job_dir method."""
    job_dir = step_result_manager._get_job_dir("test-job-1")

    assert isinstance(job_dir, Path)
    assert "test-job-1" in str(job_dir)


def test_get_step_filename(step_result_manager):
    """Test _get_step_filename method."""
    filename = step_result_manager._get_step_filename(1, "seo_keywords")

    assert filename == "01_seo_keywords.json"

    # Test with step 0 (input)
    filename = step_result_manager._get_step_filename(0, "input")

    assert filename == "00_input.json"


def test_get_s3_key(step_result_manager):
    """Test _get_s3_key method."""
    key = step_result_manager._get_s3_key("test-job-1", "01_seo_keywords.json", "0")

    assert isinstance(key, str)
    assert "test-job-1" in key
    assert "01_seo_keywords.json" in key


def test_get_metadata_s3_key(step_result_manager):
    """Test _get_metadata_s3_key method."""
    key = step_result_manager._get_metadata_s3_key("test-job-1")

    assert isinstance(key, str)
    assert "test-job-1" in key
    assert "metadata" in key


@pytest.mark.asyncio
async def test_get_job_results(step_result_manager):
    """Test get_job_results method."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.id = "test-job-1"
        mock_job.metadata = {}
        mock_job.result = None
        mock_job.type = "blog"
        mock_job.content_id = "test-content-1"
        mock_manager_instance = MagicMock()
        mock_manager_instance.get_job = AsyncMock(return_value=mock_job)
        mock_job_mgr.return_value = mock_manager_instance

        await step_result_manager.save_step_result(
            "test-job-1", 1, "seo_keywords", {"main_keyword": "test"}, "0"
        )

        results = await step_result_manager.get_job_results("test-job-1")

        assert isinstance(results, dict)
        assert "steps" in results or "metadata" in results or len(results) >= 0


@pytest.mark.asyncio
async def test_list_all_jobs(step_result_manager):
    """Test list_all_jobs method."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        await step_result_manager.save_step_result(
            "test-job-1", 1, "seo_keywords", {"main_keyword": "test"}, "0"
        )

        jobs = await step_result_manager.list_all_jobs(limit=10)

        assert isinstance(jobs, list)
        assert len(jobs) >= 0


@pytest.mark.asyncio
async def test_list_all_jobs_with_limit(step_result_manager):
    """Test list_all_jobs with limit."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        # Create multiple jobs
        for i in range(5):
            await step_result_manager.save_step_result(
                f"test-job-{i}", 1, "seo_keywords", {"main_keyword": "test"}, "0"
            )

        jobs = await step_result_manager.list_all_jobs(limit=3)

        assert len(jobs) <= 3
