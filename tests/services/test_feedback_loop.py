"""
Tests for feedback loop service.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from marketing_project.services.feedback_loop import (
    FeedbackLoopService,
    get_feedback_service,
)


@pytest.fixture
def feedback_service():
    """Create a FeedbackLoopService instance."""
    return FeedbackLoopService()


@pytest.mark.asyncio
async def test_record_feedback(feedback_service):
    """Test recording feedback."""
    result = await feedback_service.record_feedback(
        job_id="test-job-1",
        feedback_type="approval",
        rating=5,
        comments="Great post!",
    )

    assert result["feedback_id"] is not None
    assert result["job_id"] == "test-job-1"
    assert result["feedback_type"] == "approval"
    assert result["rating"] == 5
    assert result["comments"] == "Great post!"


@pytest.mark.asyncio
async def test_record_feedback_with_metadata(feedback_service):
    """Test recording feedback with metadata."""
    result = await feedback_service.record_feedback(
        job_id="test-job-1",
        feedback_type="rating",
        rating=4,
        metadata={"source": "dashboard", "user_id": "user-123"},
    )

    assert result["metadata"]["source"] == "dashboard"
    assert result["metadata"]["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_get_feedback_stats(feedback_service):
    """Test getting feedback statistics."""
    # Record some feedback
    await feedback_service.record_feedback("job-1", "approval", rating=5)
    await feedback_service.record_feedback("job-2", "rejection", rating=2)
    await feedback_service.record_feedback("job-3", "approval", rating=4)

    stats = await feedback_service.get_feedback_stats(days=30)

    assert "total_feedback" in stats
    assert "approval_rate" in stats
    assert stats["total_feedback"] >= 3


@pytest.mark.asyncio
async def test_get_feedback_stats_with_platform_filter(feedback_service):
    """Test getting feedback statistics with platform filter."""
    await feedback_service.record_feedback(
        "job-1", "approval", rating=5, metadata={"platform": "linkedin"}
    )
    await feedback_service.record_feedback(
        "job-2", "approval", rating=4, metadata={"platform": "hackernews"}
    )

    stats = await feedback_service.get_feedback_stats(days=30, platform="linkedin")

    assert "total_feedback" in stats
    assert "approval_rate" in stats


@pytest.mark.asyncio
async def test_get_feedback_stats_empty(feedback_service):
    """Test getting feedback statistics with no feedback."""
    stats = await feedback_service.get_feedback_stats(days=1)

    assert stats["total_feedback"] == 0
    assert stats["approval_rate"] == 0.0


@pytest.mark.asyncio
async def test_get_high_performing_patterns(feedback_service):
    """Test getting high performing patterns."""
    await feedback_service.record_feedback("job-1", "approval", rating=5)
    await feedback_service.record_feedback("job-2", "approval", rating=4)

    patterns = await feedback_service.get_high_performing_patterns(limit=5)

    assert isinstance(patterns, list)


@pytest.mark.asyncio
async def test_get_feedback_service_singleton():
    """Test that get_feedback_service returns a singleton."""
    service1 = get_feedback_service()
    service2 = get_feedback_service()

    assert service1 is service2
