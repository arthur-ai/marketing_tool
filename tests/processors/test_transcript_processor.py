"""
Tests for transcript processor.
"""

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from marketing_project.processors.transcript_processor import process_transcript


@pytest.fixture
def sample_transcript_data():
    """Sample transcript data for testing."""
    return {
        "id": "test-transcript-1",
        "title": "Test Transcript",
        "content": "Speaker 1: Hello. Speaker 2: Hi there.",
        "snippet": "A test transcript snippet",
        "speakers": ["Speaker 1", "Speaker 2"],
        "duration": "10:00",
        "transcript_type": "podcast",
    }


@pytest.fixture
def sample_transcript_json(sample_transcript_data):
    """Sample transcript as JSON string."""
    return json.dumps(sample_transcript_data)


class TestTranscriptProcessor:
    """Test the process_transcript function."""

    @pytest.mark.asyncio
    async def test_process_transcript_success(self, sample_transcript_json):
        """Test successful transcript processing."""
        job_id = str(uuid4())
        mock_pipeline_result = {
            "seo_keywords": {"primary": ["test", "transcript"]},
            "marketing_brief": {"summary": "Test brief"},
            "article_generation": {"content": "Generated article"},
            "seo_optimization": {"optimized": True},
            "suggested_links": {"links": []},
            "content_formatting": {"formatted": True},
        }

        with patch(
            "marketing_project.processors.transcript_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_transcript(
                sample_transcript_json, job_id=job_id
            )
            result = json.loads(result_json)

            assert result["status"] == "success"
            assert result["content_type"] == "transcript"
            assert "pipeline_result" in result

    @pytest.mark.asyncio
    async def test_process_transcript_invalid_json(self):
        """Test transcript processing with invalid JSON."""
        invalid_json = "not valid json {"

        result_json = await process_transcript(invalid_json)
        result = json.loads(result_json)

        assert result["status"] == "error"
        assert result["error"] == "invalid_json"

    @pytest.mark.asyncio
    async def test_process_transcript_invalid_input(self):
        """Test transcript processing with invalid input."""
        invalid_data = {"id": "test", "title": ""}  # Missing required fields
        invalid_json = json.dumps(invalid_data)

        result_json = await process_transcript(invalid_json)
        result = json.loads(result_json)

        assert result["status"] == "error"
        assert result["error"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_process_transcript_with_output_content_type(
        self, sample_transcript_json
    ):
        """Test transcript processing with custom output_content_type."""
        job_id = str(uuid4())
        mock_pipeline_result = {"test": "result"}

        with patch(
            "marketing_project.processors.transcript_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_transcript(
                sample_transcript_json, job_id=job_id, output_content_type="case_study"
            )
            result = json.loads(result_json)

            assert result["status"] == "success"
            call_args = mock_pipeline.execute_pipeline.call_args
            assert call_args[1]["output_content_type"] == "case_study"

    @pytest.mark.asyncio
    async def test_process_transcript_pipeline_failure(self, sample_transcript_json):
        """Test transcript processing when pipeline fails."""
        job_id = str(uuid4())

        with patch(
            "marketing_project.processors.transcript_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                side_effect=Exception("Pipeline error")
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_transcript(
                sample_transcript_json, job_id=job_id
            )
            result = json.loads(result_json)

            assert result["status"] == "error"
            assert result["error"] == "pipeline_failed"

    @pytest.mark.asyncio
    async def test_process_transcript_approval_rejected(self, sample_transcript_json):
        """Test transcript processing when approval is rejected."""
        job_id = str(uuid4())

        with patch(
            "marketing_project.processors.transcript_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                side_effect=ValueError("Content rejected")
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_transcript(
                sample_transcript_json, job_id=job_id
            )
            result = json.loads(result_json)

            assert result["status"] == "error"
            assert result["error"] == "approval_rejected"
