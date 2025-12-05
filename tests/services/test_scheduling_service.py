"""
Tests for scheduling service.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from marketing_project.services.scheduling_service import (
    SchedulingService,
    get_scheduling_service,
)


@pytest.fixture
def scheduling_service():
    """Create a SchedulingService instance."""
    return SchedulingService()


@pytest.mark.asyncio
async def test_schedule_post(scheduling_service):
    """Test scheduling a post."""
    scheduled_time = datetime.now() + timedelta(days=1)

    result = await scheduling_service.schedule_post(
        job_id="test-job-1",
        content="Test post content",
        platform="linkedin",
        scheduled_time=scheduled_time,
        metadata={"hashtags": ["test"]},
    )

    assert result["schedule_id"] is not None
    assert result["status"] == "scheduled"
    assert "scheduled_time" in result


@pytest.mark.asyncio
async def test_list_scheduled_posts(scheduling_service):
    """Test listing scheduled posts."""
    # Schedule some posts
    scheduled_time = datetime.now() + timedelta(days=1)
    await scheduling_service.schedule_post(
        "job-1", "Content 1", "linkedin", scheduled_time
    )
    await scheduling_service.schedule_post(
        "job-2", "Content 2", "hackernews", scheduled_time
    )

    posts = await scheduling_service.list_scheduled_posts()

    assert len(posts) >= 2


@pytest.mark.asyncio
async def test_list_scheduled_posts_with_platform_filter(scheduling_service):
    """Test listing scheduled posts with platform filter."""
    scheduled_time = datetime.now() + timedelta(days=1)
    await scheduling_service.schedule_post(
        "job-1", "Content 1", "linkedin", scheduled_time
    )
    await scheduling_service.schedule_post(
        "job-2", "Content 2", "hackernews", scheduled_time
    )

    posts = await scheduling_service.list_scheduled_posts(platform="linkedin")

    assert all(p["platform"] == "linkedin" for p in posts)


@pytest.mark.asyncio
async def test_list_scheduled_posts_with_limit(scheduling_service):
    """Test listing scheduled posts with limit."""
    scheduled_time = datetime.now() + timedelta(days=1)
    for i in range(10):
        await scheduling_service.schedule_post(
            f"job-{i}", f"Content {i}", "linkedin", scheduled_time
        )

    posts = await scheduling_service.list_scheduled_posts(limit=5)

    assert len(posts) <= 5


@pytest.mark.asyncio
async def test_get_scheduled_post(scheduling_service):
    """Test getting a specific scheduled post."""
    scheduled_time = datetime.now() + timedelta(days=1)
    result = await scheduling_service.schedule_post(
        "job-1", "Content", "linkedin", scheduled_time
    )

    schedule_id = result["schedule_id"]
    post = await scheduling_service.get_scheduled_post(schedule_id)

    assert post is not None
    assert post["schedule_id"] == schedule_id
    assert post["job_id"] == "job-1"


@pytest.mark.asyncio
async def test_get_scheduled_post_not_found(scheduling_service):
    """Test getting a non-existent scheduled post."""
    post = await scheduling_service.get_scheduled_post("non-existent")

    assert post is None


@pytest.mark.asyncio
async def test_cancel_scheduled_post(scheduling_service):
    """Test canceling a scheduled post."""
    scheduled_time = datetime.now() + timedelta(days=1)
    result = await scheduling_service.schedule_post(
        "job-1", "Content", "linkedin", scheduled_time
    )

    schedule_id = result["schedule_id"]
    cancel_result = await scheduling_service.cancel_scheduled_post(schedule_id)

    assert "status" in cancel_result
    assert cancel_result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_scheduled_post_not_found(scheduling_service):
    """Test canceling a non-existent scheduled post."""
    with pytest.raises(ValueError, match="not found"):
        await scheduling_service.cancel_scheduled_post("non-existent")


@pytest.mark.asyncio
async def test_get_scheduling_service_singleton():
    """Test that get_scheduling_service returns a singleton."""
    service1 = get_scheduling_service()
    service2 = get_scheduling_service()

    assert service1 is service2
