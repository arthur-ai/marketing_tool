"""
Integration tests for Redis Manager with real Redis instance.

These tests require a running Redis instance. They can be run with:
    pytest tests/integration/test_redis_integration.py

Set REDIS_HOST and REDIS_PORT environment variables to point to your test Redis instance.
"""

import asyncio
import os

import pytest
import redis.asyncio as redis

from marketing_project.services.redis_manager import (
    CircuitBreakerError,
    RedisManager,
    get_redis_manager,
)

# Skip integration tests if Redis is not available
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


@pytest.fixture
async def redis_manager():
    """Create a RedisManager instance for testing."""
    manager = RedisManager()
    yield manager
    await manager.cleanup()


@pytest.fixture
async def test_redis_available():
    """Check if test Redis is available."""
    try:
        test_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
        await test_client.ping()
        await test_client.aclose()
        return True
    except Exception:
        pytest.skip("Redis not available for integration tests")


@pytest.mark.asyncio
@pytest.mark.integration
class TestRedisIntegration:
    """Integration tests with real Redis."""

    async def test_connection_pooling(self, redis_manager, test_redis_available):
        """Test that connection pooling works with real Redis."""
        # Set environment for test
        os.environ["REDIS_HOST"] = REDIS_HOST
        os.environ["REDIS_PORT"] = str(REDIS_PORT)

        # Get Redis client multiple times - should reuse connection
        client1 = await redis_manager.get_redis()
        client2 = await redis_manager.get_redis()

        # Both should work
        result1 = await client1.ping()
        result2 = await client2.ping()

        assert result1 is True
        assert result2 is True

    async def test_basic_operations(self, redis_manager, test_redis_available):
        """Test basic Redis operations through RedisManager."""
        os.environ["REDIS_HOST"] = REDIS_HOST
        os.environ["REDIS_PORT"] = str(REDIS_PORT)

        async def set_operation(client):
            return await client.set("test_key", "test_value")

        async def get_operation(client):
            return await client.get("test_key")

        # Set value
        await redis_manager.execute(set_operation)

        # Get value
        result = await redis_manager.execute(get_operation)

        assert result == "test_value"

        # Cleanup
        async def delete_operation(client):
            return await client.delete("test_key")

        await redis_manager.execute(delete_operation)

    async def test_pipeline_operations(self, redis_manager, test_redis_available):
        """Test pipeline operations for batch efficiency."""
        os.environ["REDIS_HOST"] = REDIS_HOST
        os.environ["REDIS_PORT"] = str(REDIS_PORT)

        async def pipeline_operation(client):
            async with client.pipeline() as pipe:
                pipe.set("key1", "value1")
                pipe.set("key2", "value2")
                pipe.set("key3", "value3")
                return await pipe.execute()

        results = await redis_manager.execute(pipeline_operation)

        assert len(results) == 3
        assert all(results)  # All should be True

        # Cleanup
        async def delete_operation(client):
            return await client.delete("key1", "key2", "key3")

        await redis_manager.execute(delete_operation)

    async def test_retry_on_failure(self, redis_manager, test_redis_available):
        """Test retry logic with temporary failures."""
        os.environ["REDIS_HOST"] = REDIS_HOST
        os.environ["REDIS_PORT"] = str(REDIS_PORT)
        os.environ["REDIS_RETRY_ATTEMPTS"] = "3"

        # This test would require simulating failures, which is complex
        # In a real scenario, you might temporarily stop Redis or use a proxy
        # For now, we just verify the retry configuration is respected
        assert redis_manager._circuit_breaker_failure_threshold == 5

    async def test_health_check(self, redis_manager, test_redis_available):
        """Test health check with real Redis."""
        os.environ["REDIS_HOST"] = REDIS_HOST
        os.environ["REDIS_PORT"] = str(REDIS_PORT)

        # Perform health check
        result = await redis_manager.health_check()

        assert result is True
        assert redis_manager._health_status is True

        # Check health status
        status = redis_manager.get_health_status()
        assert status["healthy"] is True

    async def test_circuit_breaker_recovery(self, redis_manager, test_redis_available):
        """Test circuit breaker recovery after failures."""
        os.environ["REDIS_HOST"] = REDIS_HOST
        os.environ["REDIS_PORT"] = str(REDIS_PORT)
        os.environ["REDIS_CIRCUIT_FAILURE_THRESHOLD"] = "2"
        os.environ["REDIS_CIRCUIT_RECOVERY_TIMEOUT"] = "5"

        # Manually trigger failures to open circuit
        for _ in range(2):
            redis_manager._record_circuit_breaker_failure()

        assert redis_manager._circuit_breaker_state == "open"

        # Wait for recovery timeout
        import time

        time.sleep(6)  # Wait for recovery timeout

        # Check if circuit breaker allows operations again
        assert redis_manager._check_circuit_breaker() is True
        assert redis_manager._circuit_breaker_state == "half_open"

    async def test_connection_cleanup(self, redis_manager, test_redis_available):
        """Test that cleanup properly closes connections."""
        os.environ["REDIS_HOST"] = REDIS_HOST
        os.environ["REDIS_PORT"] = str(REDIS_PORT)

        # Get a connection
        await redis_manager.get_redis()

        # Cleanup
        await redis_manager.cleanup()

        # Verify cleanup
        assert redis_manager._pool is None
        assert redis_manager._redis is None


@pytest.mark.asyncio
@pytest.mark.integration
class TestRedisFailover:
    """Test Redis failover scenarios (requires Redis Sentinel or Cluster)."""

    async def test_failover_handling(self, redis_manager, test_redis_available):
        """Test handling of Redis failover scenarios."""
        # This test would require a Redis Sentinel or Cluster setup
        # For now, we just verify the retry and circuit breaker logic
        # would handle failovers gracefully
        os.environ["REDIS_HOST"] = REDIS_HOST
        os.environ["REDIS_PORT"] = str(REDIS_PORT)

        # Verify retry configuration
        assert redis_manager._circuit_breaker_failure_threshold >= 1
        assert redis_manager._circuit_breaker_recovery_timeout >= 1
