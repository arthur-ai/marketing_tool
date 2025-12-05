"""
Tests for release notes processor.
"""

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from marketing_project.processors.releasenotes_processor import process_release_notes


@pytest.fixture
def sample_release_notes_data():
    """Sample release notes data for testing."""
    return {
        "id": "test-release-1",
        "title": "Version 1.0.0 Release Notes",
        "content": "This is a test release with new features.",
        "snippet": "A test release notes snippet",
        "version": "1.0.0",
        "features": ["New feature 1", "New feature 2"],
        "bug_fixes": ["Fixed bug 1"],
        "changes": ["Change 1", "Change 2"],
    }


@pytest.fixture
def sample_release_notes_json(sample_release_notes_data):
    """Sample release notes as JSON string."""
    return json.dumps(sample_release_notes_data)


class TestReleaseNotesProcessor:
    """Test the process_release_notes function."""

    @pytest.mark.asyncio
    async def test_process_release_notes_success(self, sample_release_notes_json):
        """Test successful release notes processing."""
        job_id = str(uuid4())
        mock_pipeline_result = {
            "seo_keywords": {"primary": ["release", "version"]},
            "marketing_brief": {"summary": "Test brief"},
            "article_generation": {"content": "Generated article"},
            "seo_optimization": {"optimized": True},
            "suggested_links": {"links": []},
            "content_formatting": {"formatted": True},
        }

        with patch(
            "marketing_project.processors.releasenotes_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_release_notes(
                sample_release_notes_json, job_id=job_id
            )
            result = json.loads(result_json)

            assert result["status"] == "success"
            assert result["content_type"] == "release_notes"
            assert "pipeline_result" in result

    @pytest.mark.asyncio
    async def test_process_release_notes_invalid_json(self):
        """Test release notes processing with invalid JSON."""
        invalid_json = "not valid json {"

        result_json = await process_release_notes(invalid_json)
        result = json.loads(result_json)

        assert result["status"] == "error"
        assert result["error"] == "invalid_json"

    @pytest.mark.asyncio
    async def test_process_release_notes_invalid_input(self):
        """Test release notes processing with invalid input."""
        invalid_data = {"title": "Test"}  # Missing required 'id' field
        invalid_json = json.dumps(invalid_data)

        result_json = await process_release_notes(invalid_json)
        result = json.loads(result_json)

        assert result["status"] == "error"
        assert result["error"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_process_release_notes_with_output_content_type(
        self, sample_release_notes_json
    ):
        """Test release notes processing with custom output_content_type."""
        job_id = str(uuid4())
        mock_pipeline_result = {"test": "result"}

        with patch(
            "marketing_project.processors.releasenotes_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_release_notes(
                sample_release_notes_json,
                job_id=job_id,
                output_content_type="press_release",
            )
            result = json.loads(result_json)

            assert result["status"] == "success"
            call_args = mock_pipeline.execute_pipeline.call_args
            assert call_args[1]["output_content_type"] == "press_release"

    @pytest.mark.asyncio
    async def test_process_release_notes_pipeline_failure(
        self, sample_release_notes_json
    ):
        """Test release notes processing when pipeline fails."""
        job_id = str(uuid4())

        with patch(
            "marketing_project.processors.releasenotes_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                side_effect=Exception("Pipeline error")
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_release_notes(
                sample_release_notes_json, job_id=job_id
            )
            result = json.loads(result_json)

            assert result["status"] == "error"
            assert result["error"] == "pipeline_failed"

    @pytest.mark.asyncio
    async def test_process_release_notes_approval_rejected(
        self, sample_release_notes_json
    ):
        """Test release notes processing when approval is rejected."""
        job_id = str(uuid4())

        with patch(
            "marketing_project.processors.releasenotes_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                side_effect=ValueError("Content rejected")
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_release_notes(
                sample_release_notes_json, job_id=job_id
            )
            result = json.loads(result_json)

            assert result["status"] == "error"
            assert result["error"] == "approval_rejected"
