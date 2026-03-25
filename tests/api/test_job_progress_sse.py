"""
Unit tests for the SSE progress endpoint: GET /api/v1/jobs/{job_id}/progress
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marketing_project.api.jobs import router
from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.middleware.rbac import verify_job_ownership
from marketing_project.services.job_manager import Job, JobStatus
from tests.utils.keycloak_test_helpers import create_user_context

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user():
    return create_user_context(roles=["user"])


@pytest.fixture
def client(mock_user):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/jobs")
    app.dependency_overrides[get_current_user] = lambda: mock_user
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def completed_job():
    from datetime import datetime

    return Job(
        id="job-sse-1",
        type="blog_post",
        content_id="c-1",
        status=JobStatus.COMPLETED,
        created_at=datetime.utcnow(),
        progress=100,
    )


@pytest.fixture
def running_job():
    from datetime import datetime

    return Job(
        id="job-sse-1",
        type="blog_post",
        content_id="c-1",
        status=JobStatus.PROCESSING,
        created_at=datetime.utcnow(),
        progress=50,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse_events(text: str) -> list:
    """Parse raw SSE response body into a list of decoded JSON objects."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                events.append({"_raw": payload})
    return events


def _build_mock_redis(state_data: dict = None, pubsub_messages: list = None):
    """
    Build mock Redis + pubsub that emit the given pub/sub messages then stop.

    state_data: dict returned by hgetall (bytes keys/values)
    pubsub_messages: list of dicts like {"data": b"<json>"}.  None sentinel
                     is delivered after each real message so the loop exits.
    """
    r = AsyncMock()

    # hgetall mock
    if state_data:
        r.hgetall = AsyncMock(return_value=state_data)
    else:
        r.hgetall = AsyncMock(return_value={})

    # pubsub mock — get_message yields messages then None (heartbeat trigger)
    pubsub = AsyncMock()

    if pubsub_messages:
        # For each real message append a None so the loop gets a heartbeat after
        side_effects = []
        for msg in pubsub_messages:
            side_effects.append(msg)
        side_effects.append(None)  # triggers heartbeat + terminal-job check
    else:
        side_effects = [None]  # immediate heartbeat

    pubsub.get_message = AsyncMock(side_effect=side_effects)
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.close = AsyncMock()
    r.pubsub = MagicMock(return_value=pubsub)

    return r, pubsub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSSEProgressEndpoint:
    """Tests for GET /api/v1/jobs/{job_id}/progress"""

    # ------------------------------------------------------------------
    # Cold-connect: replay state hash
    # ------------------------------------------------------------------

    def test_cold_connect_replays_state_hash(self, client, completed_job):
        """When a state hash exists the first SSE event replicates it."""
        state_data = {
            b"step": b"seo_keywords",
            b"step_number": b"3",
            b"total_steps": b"10",
            b"pct": b"30",
        }
        r, pubsub = _build_mock_redis(
            state_data=state_data,
            # pct=100 message → stream closes
            pubsub_messages=[
                {
                    "data": json.dumps(
                        {
                            "type": "progress",
                            "step": "content_formatting",
                            "step_number": 10,
                            "total_steps": 10,
                            "pct": 100,
                        }
                    ).encode()
                }
            ],
        )
        mgr = AsyncMock()
        mgr.get_job = AsyncMock(return_value=completed_job)

        with (
            patch("marketing_project.api.jobs.get_job_manager", return_value=mgr),
            patch(
                "marketing_project.api.jobs.verify_job_ownership",
                return_value=completed_job,
            ),
            patch(
                "marketing_project.services.function_pipeline.orchestration.get_redis_manager"
            ),
            patch(
                "marketing_project.api.jobs.get_redis_manager"
                if hasattr(
                    __import__(
                        "marketing_project.api.jobs", fromlist=["get_redis_manager"]
                    ),
                    "get_redis_manager",
                )
                else "marketing_project.services.redis_manager.get_redis_manager"
            ),
        ):
            # Directly patch the import inside the endpoint closure
            with patch(
                "marketing_project.services.redis_manager.get_redis_manager"
            ) as mock_rm:
                mock_rm_instance = AsyncMock()
                mock_rm_instance.get_redis = AsyncMock(return_value=r)
                mock_rm.return_value = mock_rm_instance

                response = client.get("/api/v1/jobs/job-sse-1/progress")

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = _parse_sse_events(response.text)
        # First event should be the cold-connect state replay
        assert events[0]["type"] == "progress"
        assert events[0]["step"] == "seo_keywords"
        assert events[0]["pct"] == 30

    def test_no_state_hash_skips_cold_connect(self, client, completed_job):
        """When there is no state hash, stream starts directly with pub/sub."""
        r, pubsub = _build_mock_redis(
            state_data={},  # empty — no cold-connect event
            pubsub_messages=[
                {
                    "data": json.dumps(
                        {"type": "progress", "step": "seo_keywords", "pct": 100}
                    ).encode()
                }
            ],
        )
        mgr = AsyncMock()
        mgr.get_job = AsyncMock(return_value=completed_job)

        with (
            patch("marketing_project.api.jobs.get_job_manager", return_value=mgr),
            patch(
                "marketing_project.api.jobs.verify_job_ownership",
                return_value=completed_job,
            ),
            patch(
                "marketing_project.services.redis_manager.get_redis_manager"
            ) as mock_rm,
        ):
            mock_rm_instance = AsyncMock()
            mock_rm_instance.get_redis = AsyncMock(return_value=r)
            mock_rm.return_value = mock_rm_instance

            response = client.get("/api/v1/jobs/job-sse-1/progress")

        events = _parse_sse_events(response.text)
        # No cold-connect event — only the live pub/sub event + done
        progress_events = [e for e in events if e.get("type") == "progress"]
        assert len(progress_events) == 1
        assert progress_events[0]["step"] == "seo_keywords"

    # ------------------------------------------------------------------
    # Stream termination
    # ------------------------------------------------------------------

    def test_stream_closes_when_pct_100(self, client, completed_job):
        """Stream emits a 'done' event and closes when pct reaches 100."""
        r, pubsub = _build_mock_redis(
            pubsub_messages=[
                {
                    "data": json.dumps(
                        {"type": "progress", "step": "final_step", "pct": 100}
                    ).encode()
                }
            ]
        )
        mgr = AsyncMock()
        mgr.get_job = AsyncMock(return_value=completed_job)

        with (
            patch("marketing_project.api.jobs.get_job_manager", return_value=mgr),
            patch(
                "marketing_project.api.jobs.verify_job_ownership",
                return_value=completed_job,
            ),
            patch(
                "marketing_project.services.redis_manager.get_redis_manager"
            ) as mock_rm,
        ):
            mock_rm_instance = AsyncMock()
            mock_rm_instance.get_redis = AsyncMock(return_value=r)
            mock_rm.return_value = mock_rm_instance

            response = client.get("/api/v1/jobs/job-sse-1/progress")

        events = _parse_sse_events(response.text)
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1

    def test_stream_closes_on_terminal_job_status(self, client, completed_job):
        """On heartbeat timeout the stream closes if job is in a terminal state."""
        r, pubsub = _build_mock_redis(
            pubsub_messages=None  # immediate None → heartbeat → job check
        )
        mgr = AsyncMock()
        mgr.get_job = AsyncMock(return_value=completed_job)  # COMPLETED status

        with (
            patch("marketing_project.api.jobs.get_job_manager", return_value=mgr),
            patch(
                "marketing_project.api.jobs.verify_job_ownership",
                return_value=completed_job,
            ),
            patch(
                "marketing_project.services.redis_manager.get_redis_manager"
            ) as mock_rm,
            patch("asyncio.sleep", new_callable=AsyncMock),  # skip real sleep
        ):
            mock_rm_instance = AsyncMock()
            mock_rm_instance.get_redis = AsyncMock(return_value=r)
            mock_rm.return_value = mock_rm_instance

            response = client.get("/api/v1/jobs/job-sse-1/progress")

        events = _parse_sse_events(response.text)
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_redis_failure_yields_error_event(self, client, completed_job):
        """When Redis is unavailable the stream yields an error event."""
        mgr = AsyncMock()
        mgr.get_job = AsyncMock(return_value=completed_job)

        with (
            patch("marketing_project.api.jobs.get_job_manager", return_value=mgr),
            patch(
                "marketing_project.api.jobs.verify_job_ownership",
                return_value=completed_job,
            ),
            patch(
                "marketing_project.services.redis_manager.get_redis_manager"
            ) as mock_rm,
        ):
            mock_rm_instance = AsyncMock()
            mock_rm_instance.get_redis = AsyncMock(
                side_effect=ConnectionError("Redis down")
            )
            mock_rm.return_value = mock_rm_instance

            response = client.get("/api/v1/jobs/job-sse-1/progress")

        assert response.status_code == 200  # StreamingResponse headers already sent
        events = _parse_sse_events(response.text)
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1

    # ------------------------------------------------------------------
    # Response headers
    # ------------------------------------------------------------------

    def test_response_headers_set_correctly(self, client, completed_job):
        """SSE response must include correct headers for streaming."""
        r, _ = _build_mock_redis(
            pubsub_messages=[
                {"data": json.dumps({"type": "progress", "pct": 100}).encode()}
            ]
        )
        mgr = AsyncMock()
        mgr.get_job = AsyncMock(return_value=completed_job)

        with (
            patch("marketing_project.api.jobs.get_job_manager", return_value=mgr),
            patch(
                "marketing_project.api.jobs.verify_job_ownership",
                return_value=completed_job,
            ),
            patch(
                "marketing_project.services.redis_manager.get_redis_manager"
            ) as mock_rm,
        ):
            mock_rm_instance = AsyncMock()
            mock_rm_instance.get_redis = AsyncMock(return_value=r)
            mock_rm.return_value = mock_rm_instance

            response = client.get("/api/v1/jobs/job-sse-1/progress")

        assert "text/event-stream" in response.headers.get("content-type", "")
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("x-accel-buffering") == "no"

    # ------------------------------------------------------------------
    # Pub/sub cleanup
    # ------------------------------------------------------------------

    def test_pubsub_unsubscribed_after_stream(self, client, completed_job):
        """pubsub.unsubscribe() and pubsub.close() are always called."""
        r, pubsub = _build_mock_redis(
            pubsub_messages=[
                {"data": json.dumps({"type": "progress", "pct": 100}).encode()}
            ]
        )
        mgr = AsyncMock()
        mgr.get_job = AsyncMock(return_value=completed_job)

        with (
            patch("marketing_project.api.jobs.get_job_manager", return_value=mgr),
            patch(
                "marketing_project.api.jobs.verify_job_ownership",
                return_value=completed_job,
            ),
            patch(
                "marketing_project.services.redis_manager.get_redis_manager"
            ) as mock_rm,
        ):
            mock_rm_instance = AsyncMock()
            mock_rm_instance.get_redis = AsyncMock(return_value=r)
            mock_rm.return_value = mock_rm_instance

            client.get("/api/v1/jobs/job-sse-1/progress")

        pubsub.unsubscribe.assert_called_once()
        pubsub.close.assert_called_once()
