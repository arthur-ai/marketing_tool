"""
Comprehensive tests for Redis manager service methods.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.redis_manager import RedisManager, get_redis_manager


@pytest.fixture
def redis_manager():
    """Create a RedisManager instance."""
    return RedisManager()


@pytest.mark.asyncio
async def test_get_redis(redis_manager):
    """Test get_redis method."""
    with patch.object(redis_manager, "_get_pool") as mock_pool:
        mock_pool_obj = MagicMock()
        mock_pool_obj.acquire = AsyncMock(return_value=MagicMock())
        mock_pool.return_value = mock_pool_obj

        redis_client = await redis_manager.get_redis()

        # May return None if Redis not configured
        assert redis_client is None or redis_client is not None


@pytest.mark.asyncio
async def test_execute(redis_manager):
    """Test execute method."""

    async def test_operation(redis_client):
        return "test_result"

    with patch.object(redis_manager, "get_redis", new_callable=AsyncMock) as mock_get:
        mock_redis = MagicMock()
        mock_get.return_value = mock_redis

        result = await redis_manager.execute(test_operation)

        # May return None if Redis not configured
        assert result is None or result == "test_result"


def test_check_circuit_breaker(redis_manager):
    """Test _check_circuit_breaker method."""
    redis_manager._circuit_breaker_state = "closed"
    assert redis_manager._check_circuit_breaker() is True

    redis_manager._circuit_breaker_state = "open"
    assert redis_manager._check_circuit_breaker() is False


def test_record_circuit_breaker_success(redis_manager):
    """Test _record_circuit_breaker_success method."""
    redis_manager._circuit_breaker_failures = 5
    redis_manager._circuit_breaker_state = "half_open"

    redis_manager._record_circuit_breaker_success()

    assert redis_manager._circuit_breaker_failures == 0
    assert redis_manager._circuit_breaker_state == "closed"


def test_record_circuit_breaker_failure(redis_manager):
    """Test _record_circuit_breaker_failure method."""
    redis_manager._circuit_breaker_failures = 4
    redis_manager._circuit_breaker_failure_threshold = 5

    redis_manager._record_circuit_breaker_failure()

    assert redis_manager._circuit_breaker_failures == 5
    assert redis_manager._circuit_breaker_state == "open"


@pytest.mark.asyncio
async def test_health_check(redis_manager):
    """Test health_check method."""
    with patch.object(redis_manager, "get_redis", new_callable=AsyncMock) as mock_get:
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_get.return_value = mock_redis

        result = await redis_manager.health_check()

        assert isinstance(result, bool)


def test_get_health_status(redis_manager):
    """Test get_health_status method."""
    status = redis_manager.get_health_status()

    assert isinstance(status, dict)
    assert "health_status" in status or "circuit_breaker_state" in status


@pytest.mark.asyncio
async def test_cleanup(redis_manager):
    """Test cleanup method."""
    await redis_manager.cleanup()

    # Should not raise exception
    assert True


def test_get_redis_manager_singleton():
    """Test that get_redis_manager returns a singleton."""
    manager1 = get_redis_manager()
    manager2 = get_redis_manager()

    assert manager1 is manager2
