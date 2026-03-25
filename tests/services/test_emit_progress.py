"""
Unit tests for emit_progress() in orchestration.py.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.function_pipeline.orchestration import emit_progress


@pytest.fixture
def mock_redis():
    """Mock Redis client with pipeline support."""
    pipe = AsyncMock()
    pipe.publish = AsyncMock()
    pipe.hset = AsyncMock()
    pipe.expire = AsyncMock()
    pipe.execute = AsyncMock(return_value=[1, 1, 1])

    r = AsyncMock()
    r.pipeline = MagicMock(return_value=pipe)
    return r, pipe


@pytest.fixture
def mock_redis_manager(mock_redis):
    r, pipe = mock_redis
    mgr = AsyncMock()
    mgr.get_redis = AsyncMock(return_value=r)
    return mgr, r, pipe


@pytest.mark.asyncio
async def test_emit_progress_publishes_correct_payload(mock_redis_manager):
    """emit_progress publishes the correct JSON payload to the expected channel."""
    mgr, r, pipe = mock_redis_manager

    with patch(
        "marketing_project.services.function_pipeline.orchestration.get_redis_manager",
        return_value=mgr,
    ):
        await emit_progress(
            job_id="job-123",
            step_name="seo_keywords",
            step_number=3,
            total_steps=10,
        )

    pipe.publish.assert_called_once()
    channel, payload = pipe.publish.call_args[0]
    assert channel == "job:job-123:progress:events"

    data = json.loads(payload)
    assert data["type"] == "progress"
    assert data["step"] == "seo_keywords"
    assert data["step_number"] == 3
    assert data["total_steps"] == 10
    assert data["pct"] == 30  # round(3/10 * 100)


@pytest.mark.asyncio
async def test_emit_progress_updates_state_hash(mock_redis_manager):
    """emit_progress writes current state to the state hash for cold-connect clients."""
    mgr, r, pipe = mock_redis_manager

    with patch(
        "marketing_project.services.function_pipeline.orchestration.get_redis_manager",
        return_value=mgr,
    ):
        await emit_progress(
            job_id="job-456",
            step_name="content_formatting",
            step_number=10,
            total_steps=10,
        )

    pipe.hset.assert_called_once()
    state_key = pipe.hset.call_args[0][0]
    assert state_key == "job:job-456:progress:state"

    mapping = pipe.hset.call_args[1]["mapping"]
    assert mapping["step"] == "content_formatting"
    assert mapping["step_number"] == 10
    assert mapping["pct"] == 100


@pytest.mark.asyncio
async def test_emit_progress_sets_ttl(mock_redis_manager):
    """emit_progress sets a 1-hour TTL on the state hash."""
    mgr, r, pipe = mock_redis_manager

    with patch(
        "marketing_project.services.function_pipeline.orchestration.get_redis_manager",
        return_value=mgr,
    ):
        await emit_progress(
            job_id="job-789",
            step_name="transcript_preprocessing",
            step_number=1,
            total_steps=5,
        )

    pipe.expire.assert_called_once()
    key, ttl = pipe.expire.call_args[0]
    assert key == "job:job-789:progress:state"
    assert ttl == 3600


@pytest.mark.asyncio
async def test_emit_progress_pct_rounds_correctly():
    """Percentage is rounded correctly for non-integer values."""
    pipe = AsyncMock()
    pipe.publish = AsyncMock()
    pipe.hset = AsyncMock()
    pipe.expire = AsyncMock()
    pipe.execute = AsyncMock(return_value=[1, 1, 1])

    r = AsyncMock()
    r.pipeline = MagicMock(return_value=pipe)

    mgr = AsyncMock()
    mgr.get_redis = AsyncMock(return_value=r)

    with patch(
        "marketing_project.services.function_pipeline.orchestration.get_redis_manager",
        return_value=mgr,
    ):
        await emit_progress(
            job_id="job-round",
            step_name="step_a",
            step_number=1,
            total_steps=3,
        )

    _, payload = pipe.publish.call_args[0]
    data = json.loads(payload)
    # round(1/3 * 100) = round(33.33) = 33
    assert data["pct"] == 33


@pytest.mark.asyncio
async def test_emit_progress_swallows_redis_errors():
    """Failures in emit_progress are silently swallowed — never interrupt the pipeline."""
    mgr = AsyncMock()
    mgr.get_redis = AsyncMock(side_effect=ConnectionError("Redis down"))

    with patch(
        "marketing_project.services.function_pipeline.orchestration.get_redis_manager",
        return_value=mgr,
    ):
        # Should not raise
        await emit_progress(
            job_id="job-fail",
            step_name="step_x",
            step_number=1,
            total_steps=5,
        )


@pytest.mark.asyncio
async def test_emit_progress_swallows_pipeline_execute_errors():
    """Pipeline execute() failures are also silently swallowed."""
    pipe = AsyncMock()
    pipe.publish = AsyncMock()
    pipe.hset = AsyncMock()
    pipe.expire = AsyncMock()
    pipe.execute = AsyncMock(side_effect=RuntimeError("pipeline error"))

    r = AsyncMock()
    r.pipeline = MagicMock(return_value=pipe)

    mgr = AsyncMock()
    mgr.get_redis = AsyncMock(return_value=r)

    with patch(
        "marketing_project.services.function_pipeline.orchestration.get_redis_manager",
        return_value=mgr,
    ):
        # Should not raise
        await emit_progress(
            job_id="job-pipe-fail",
            step_name="step_y",
            step_number=2,
            total_steps=8,
        )


@pytest.mark.asyncio
async def test_emit_progress_total_steps_zero_guard():
    """total_steps=0 does not cause a ZeroDivisionError."""
    pipe = AsyncMock()
    pipe.publish = AsyncMock()
    pipe.hset = AsyncMock()
    pipe.expire = AsyncMock()
    pipe.execute = AsyncMock(return_value=[1, 1, 1])

    r = AsyncMock()
    r.pipeline = MagicMock(return_value=pipe)

    mgr = AsyncMock()
    mgr.get_redis = AsyncMock(return_value=r)

    with patch(
        "marketing_project.services.function_pipeline.orchestration.get_redis_manager",
        return_value=mgr,
    ):
        await emit_progress(
            job_id="job-zero",
            step_name="step_z",
            step_number=1,
            total_steps=0,  # edge case
        )

    _, payload = pipe.publish.call_args[0]
    data = json.loads(payload)
    # max(0, 1) = 1, so round(1/1 * 100) = 100
    assert data["pct"] == 100
