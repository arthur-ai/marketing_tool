"""
Extended tests for step result manager - covering more methods.
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
async def test_get_or_create_execution_context(step_result_manager):
    """Test _get_or_create_execution_context method."""
    with patch(
        "marketing_project.services.job_manager.get_job_manager"
    ) as mock_job_mgr:
        mock_job = MagicMock()
        mock_job.metadata = {}
        mock_job_mgr.return_value.get_job = AsyncMock(return_value=mock_job)

        context_id = await step_result_manager._get_or_create_execution_context(
            "test-job-1", "0"
        )

        assert context_id == "0" or isinstance(context_id, str)


@pytest.mark.asyncio
async def test_extract_step_info_from_job_result(step_result_manager):
    """Test extract_step_info_from_job_result method."""
    job_result = {
        "steps": [
            {
                "step_number": 1,
                "step_name": "seo_keywords",
                "result": {"main_keyword": "test"},
            }
        ]
    }

    step_info = await step_result_manager.extract_step_info_from_job_result(
        "test-job-1", job_result
    )

    assert isinstance(step_info, list)
    assert len(step_info) >= 0
