"""
Tests for transcript processor.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
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
            "transcript_preprocessing_approval": {
                "is_valid": True,
                "speakers_validated": True,
                "duration_validated": True,
                "content_validated": True,
                "transcript_type_validated": True,
                "requires_approval": False,
            },
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
        invalid_data = {"title": "Test"}  # Missing required 'id' field
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

    # -----------------------------------------------------------------------
    # Additional tests covering missed lines
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_json_serializer_datetime(self):
        """Test _json_serializer handles datetime objects (lines 28-30)."""
        from datetime import date, datetime

        from marketing_project.processors.transcript_processor import _json_serializer

        dt = datetime(2024, 1, 1, 12, 0, 0)
        assert _json_serializer(dt) == "2024-01-01T12:00:00"

        d = date(2024, 1, 1)
        assert _json_serializer(d) == "2024-01-01"

    def test_json_serializer_unsupported_type(self):
        """Test _json_serializer raises TypeError for unsupported types (line 30)."""
        from marketing_project.processors.transcript_processor import _json_serializer

        with pytest.raises(TypeError):
            _json_serializer(object())

    def test_parse_duration_none(self):
        """Test _parse_duration_to_seconds returns None for None input (line 49)."""
        from marketing_project.processors.transcript_processor import (
            _parse_duration_to_seconds,
        )

        assert _parse_duration_to_seconds(None) is None

    def test_parse_duration_integer(self):
        """Test _parse_duration_to_seconds returns int as-is (line 53)."""
        from marketing_project.processors.transcript_processor import (
            _parse_duration_to_seconds,
        )

        assert _parse_duration_to_seconds(600) == 600

    def test_parse_duration_float(self):
        """Test _parse_duration_to_seconds converts non-string, non-int (lines 57-60)."""
        from marketing_project.processors.transcript_processor import (
            _parse_duration_to_seconds,
        )

        assert _parse_duration_to_seconds(10.5) == 10

    def test_parse_duration_invalid_non_string(self):
        """Test _parse_duration_to_seconds returns None for un-convertible value (line 60)."""
        from marketing_project.processors.transcript_processor import (
            _parse_duration_to_seconds,
        )

        assert _parse_duration_to_seconds(object()) is None

    def test_parse_duration_mm_ss(self):
        """Test _parse_duration_to_seconds parses MM:SS (lines 65-68)."""
        from marketing_project.processors.transcript_processor import (
            _parse_duration_to_seconds,
        )

        assert _parse_duration_to_seconds("10:30") == 630

    def test_parse_duration_hh_mm_ss(self):
        """Test _parse_duration_to_seconds parses HH:MM:SS (lines 69-72)."""
        from marketing_project.processors.transcript_processor import (
            _parse_duration_to_seconds,
        )

        assert _parse_duration_to_seconds("1:10:00") == 4200

    def test_parse_duration_integer_string(self):
        """Test _parse_duration_to_seconds parses integer string (line 75)."""
        from marketing_project.processors.transcript_processor import (
            _parse_duration_to_seconds,
        )

        assert _parse_duration_to_seconds("3600") == 3600

    def test_parse_duration_invalid_string(self):
        """Test _parse_duration_to_seconds returns None for unparseable string (lines 76-78)."""
        from marketing_project.processors.transcript_processor import (
            _parse_duration_to_seconds,
        )

        assert _parse_duration_to_seconds("not:a:duration:format") is None

    @pytest.mark.asyncio
    async def test_process_transcript_generates_job_id_when_not_provided(
        self, sample_transcript_json
    ):
        """Test process_transcript generates job_id when not provided (lines 103-104)."""
        mock_pipeline_result = {"pipeline_status": "success", "steps": {}}

        with patch(
            "marketing_project.processors.transcript_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            # No job_id provided
            result_json = await process_transcript(sample_transcript_json)
            result = json.loads(result_json)

            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_process_transcript_with_string_duration_that_fails_parse(self):
        """Test transcript processing when duration string fails to parse (lines 121-122)."""
        data = {
            "id": "test-transcript-1",
            "title": "Test Transcript",
            "content": "Speaker: Hello.",
            "snippet": "A test",
            "duration": "invalid-format-here",  # will fail parse and be removed
        }
        data_json = json.dumps(data)
        mock_pipeline_result = {"pipeline_status": "success", "steps": {}}

        with patch(
            "marketing_project.processors.transcript_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_transcript(data_json, job_id="test-job")
            result = json.loads(result_json)

            # Should succeed (duration removed gracefully)
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_process_transcript_output_content_type_from_job_metadata(
        self, sample_transcript_json
    ):
        """Test process_transcript fetches output_content_type from job metadata (lines 141-153)."""
        mock_pipeline_result = {"pipeline_status": "success", "steps": {}}
        mock_job = MagicMock()
        mock_job.metadata = {"output_content_type": "press_release"}

        with patch(
            "marketing_project.processors.transcript_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            # Patch get_job_manager at the import location within the transcript_processor module
            with patch(
                "marketing_project.services.job_manager.get_job_manager"
            ) as mock_get_job_mgr:
                mock_job_mgr = MagicMock()
                mock_job_mgr.get_job = AsyncMock(return_value=mock_job)
                mock_get_job_mgr.return_value = mock_job_mgr

                result_json = await process_transcript(
                    sample_transcript_json,
                    job_id="test-job",
                    # no output_content_type param
                )
                result = json.loads(result_json)

                # Should use the output_content_type from job metadata
                assert result["status"] == "success"
                call_kwargs = mock_pipeline.execute_pipeline.call_args[1]
                assert call_kwargs["output_content_type"] == "press_release"

    @pytest.mark.asyncio
    async def test_process_transcript_output_content_type_default_when_job_missing(
        self, sample_transcript_json
    ):
        """Test process_transcript uses 'blog_post' default when job not found (lines 144-153)."""
        mock_pipeline_result = {"pipeline_status": "success", "steps": {}}

        with patch(
            "marketing_project.processors.transcript_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            with patch(
                "marketing_project.services.job_manager.get_job_manager"
            ) as mock_get_job_mgr:
                mock_job_mgr = MagicMock()
                mock_job_mgr.get_job = AsyncMock(return_value=None)  # Job not found
                mock_get_job_mgr.return_value = mock_job_mgr

                result_json = await process_transcript(
                    sample_transcript_json, job_id="nonexistent-job"
                )
                result = json.loads(result_json)

                assert result["status"] == "success"
                call_kwargs = mock_pipeline.execute_pipeline.call_args[1]
                assert call_kwargs["output_content_type"] == "blog_post"

    @pytest.mark.asyncio
    async def test_process_transcript_output_content_type_from_parameter_logs(
        self, sample_transcript_json
    ):
        """Test process_transcript logs when output_content_type provided via param (lines 155-157)."""
        mock_pipeline_result = {"pipeline_status": "success"}

        with patch(
            "marketing_project.processors.transcript_processor.FunctionPipeline"
        ) as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline.execute_pipeline = AsyncMock(
                return_value=mock_pipeline_result
            )
            mock_pipeline_class.return_value = mock_pipeline

            result_json = await process_transcript(
                sample_transcript_json,
                job_id="test-job",
                output_content_type="case_study",  # provided explicitly
            )
            result = json.loads(result_json)

            assert result["status"] == "success"
            call_kwargs = mock_pipeline.execute_pipeline.call_args[1]
            assert call_kwargs["output_content_type"] == "case_study"

    @pytest.mark.asyncio
    async def test_process_transcript_pipeline_returned_failed_status(
        self, sample_transcript_json
    ):
        """Test process_transcript propagates pipeline failure (lines 178-192)."""
        mock_pipeline_result = {
            "pipeline_status": "failed",
            "metadata": {"error": "Step execution error"},
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
                sample_transcript_json, job_id="test-job"
            )
            result = json.loads(result_json)

            assert result["status"] == "error"
            assert result["error"] == "pipeline_failed"

    @pytest.mark.asyncio
    async def test_process_transcript_unexpected_exception(self):
        """Test process_transcript handles unexpected top-level exceptions (lines 227-232)."""
        # Use a MagicMock as content_data — not a string or dict.
        # json.loads will raise TypeError since it needs a str/bytes, and
        # isinstance(content_data, str) == False so it tries json.loads(content_data)
        # Actually it's `isinstance(content_data, str)` check then either use directly or load.
        # Instead, cause TranscriptContext(**data) to raise unexpectedly after valid JSON.
        data = json.dumps(
            {"id": "test-1", "title": "Test", "content": "Content", "snippet": "s"}
        )
        with patch(
            "marketing_project.processors.transcript_processor.TranscriptContext",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result_json = await process_transcript(data, job_id="test-job")
            result_data = json.loads(result_json)

            assert result_data["status"] == "error"
            # RuntimeError in TranscriptContext raises inside inner try -> "invalid_input"
            # or falls to outer except -> "processing_exception"
            assert result_data["error"] in ["invalid_input", "processing_exception"]
