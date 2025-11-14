"""
Redis Manager Service.

Centralized Redis connection management with connection pooling, retry logic,
circuit breaker pattern, health monitoring, and CloudWatch metrics integration.
"""

import asyncio
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, Optional

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


class RedisManager:
    """
    Centralized Redis manager with connection pooling, retry logic, circuit breaker,
    health monitoring, and CloudWatch metrics (production only).
    """

    def __init__(self):
        """Initialize the Redis manager."""
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[redis.Redis] = None
        self._pool_lock = asyncio.Lock()
        self._health_status = True
        self._last_health_check: Optional[datetime] = None
        self._health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None

        # Metrics tracking
        self._operation_times: deque = deque(
            maxlen=1000
        )  # Store last 1000 operation times
        self._error_count = 0
        self._success_count = 0
        self._circuit_breaker_state = "closed"  # closed, open, half_open
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure: Optional[datetime] = None
        self._circuit_breaker_failure_threshold = int(
            os.getenv("REDIS_CIRCUIT_FAILURE_THRESHOLD", "5")
        )
        self._circuit_breaker_recovery_timeout = int(
            os.getenv("REDIS_CIRCUIT_RECOVERY_TIMEOUT", "60")
        )
        self._last_metrics_publish: Optional[datetime] = None
        self._metrics_publish_interval = 60  # seconds

        # CloudWatch client (only in production)
        self._cloudwatch_enabled = (
            os.getenv("ENABLE_CLOUDWATCH_METRICS", "false").lower() == "true"
        )
        self._cloudwatch_client = None
        if self._cloudwatch_enabled and BOTO3_AVAILABLE:
            try:
                self._cloudwatch_client = boto3.client(
                    "cloudwatch", region_name=os.getenv("AWS_REGION")
                )
                logger.info("CloudWatch metrics enabled for Redis operations")
            except Exception as e:
                logger.warning(f"Failed to initialize CloudWatch client: {e}")
                self._cloudwatch_enabled = False

    async def _get_pool(self) -> ConnectionPool:
        """Get or create connection pool."""
        async with self._pool_lock:
            if self._pool is None:
                # Parse connection from REDIS_URL or individual env vars
                redis_url = os.getenv("REDIS_URL")
                use_ssl = False

                if redis_url:
                    # Parse URL to extract components (supports both redis:// and rediss://)
                    from urllib.parse import urlparse

                    parsed = urlparse(redis_url)
                    # Detect SSL from URL scheme (rediss://) or explicit SSL setting
                    use_ssl = parsed.scheme == "rediss"
                    redis_host = parsed.hostname or os.getenv("REDIS_HOST", "localhost")
                    redis_port = parsed.port or int(os.getenv("REDIS_PORT", "6379"))
                    redis_db = (
                        int(parsed.path.lstrip("/"))
                        if parsed.path
                        else int(os.getenv("REDIS_DATABASE", "0"))
                    )
                    redis_password = parsed.password or os.getenv("REDIS_PASSWORD")
                else:
                    redis_host = os.getenv("REDIS_HOST", "localhost")
                    redis_port = int(os.getenv("REDIS_PORT", "6379"))
                    redis_db = int(os.getenv("REDIS_DATABASE", "0"))
                    redis_password = os.getenv("REDIS_PASSWORD")

                # Check for explicit SSL setting (for ElastiCache with transit encryption)
                # This allows using redis:// URL but enabling SSL via env var
                if os.getenv("REDIS_SSL", "").lower() in ("true", "1", "yes"):
                    use_ssl = True

                max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
                socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
                connect_timeout = float(os.getenv("REDIS_CONNECT_TIMEOUT", "5.0"))

                # Build connection pool parameters
                pool_kwargs = {
                    "host": redis_host,
                    "port": redis_port,
                    "db": redis_db,
                    "max_connections": max_connections,
                    "socket_timeout": socket_timeout,
                    "socket_connect_timeout": connect_timeout,
                    "retry_on_timeout": True,
                    "decode_responses": True,
                    "health_check_interval": 30,
                }

                # Add SSL configuration if needed (for ElastiCache transit encryption)
                if use_ssl:
                    import ssl

                    # ElastiCache uses self-signed certificates, so we need to disable cert verification
                    pool_kwargs["ssl"] = True
                    pool_kwargs["ssl_cert_reqs"] = ssl.CERT_NONE
                    pool_kwargs["ssl_check_hostname"] = False
                    logger.info(
                        "SSL/TLS enabled for Redis connection (transit encryption)"
                    )

                # Only add password if it's set (Redis allows no password)
                if redis_password:
                    pool_kwargs["password"] = redis_password

                self._pool = ConnectionPool(**pool_kwargs)
                ssl_status = "with SSL" if use_ssl else "without SSL"
                logger.info(
                    f"Created Redis connection pool: {redis_host}:{redis_port} "
                    f"(db={redis_db}, max_connections={max_connections}, "
                    f"timeouts: socket={socket_timeout}s, connect={connect_timeout}s, {ssl_status})"
                )
            return self._pool

    async def get_redis(self) -> redis.Redis:
        """Get Redis client with connection pooling."""
        if self._redis is None:
            pool = await self._get_pool()
            self._redis = redis.Redis(connection_pool=pool)

            # Start health check task
            if self._health_check_task is None:
                self._health_check_task = asyncio.create_task(self._health_check_loop())

        return self._redis

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operation."""
        if self._circuit_breaker_state == "open":
            # Check if recovery timeout has passed
            if self._circuit_breaker_last_failure:
                elapsed = (
                    datetime.utcnow() - self._circuit_breaker_last_failure
                ).total_seconds()
                if elapsed >= self._circuit_breaker_recovery_timeout:
                    # Move to half-open state
                    self._circuit_breaker_state = "half_open"
                    self._circuit_breaker_failures = 0
                    logger.info("Circuit breaker moving to half-open state")
                    return True
            return False
        return True

    def _record_circuit_breaker_success(self):
        """Record successful operation for circuit breaker."""
        if self._circuit_breaker_state == "half_open":
            # Success in half-open state, close the circuit
            self._circuit_breaker_state = "closed"
            self._circuit_breaker_failures = 0
            logger.info("Circuit breaker closed after successful operation")
        elif self._circuit_breaker_state == "closed":
            # Reset failure count on success
            self._circuit_breaker_failures = 0

    def _record_circuit_breaker_failure(self):
        """Record failed operation for circuit breaker."""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = datetime.utcnow()

        if self._circuit_breaker_state == "half_open":
            # Failure in half-open state, open the circuit
            self._circuit_breaker_state = "open"
            logger.warning("Circuit breaker opened after failure in half-open state")
        elif self._circuit_breaker_failures >= self._circuit_breaker_failure_threshold:
            # Too many failures, open the circuit
            self._circuit_breaker_state = "open"
            logger.error(
                f"Circuit breaker opened after {self._circuit_breaker_failures} failures "
                f"(threshold: {self._circuit_breaker_failure_threshold})"
            )

    async def _execute_with_retry(
        self, operation: Callable[[redis.Redis], Awaitable[Any]], *args, **kwargs
    ) -> Any:
        """
        Execute Redis operation with retry logic and circuit breaker.

        Args:
            operation: Async function that takes redis.Redis as first arg
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the operation
        """
        retry_attempts = int(os.getenv("REDIS_RETRY_ATTEMPTS", "3"))
        backoff_min = int(os.getenv("REDIS_RETRY_BACKOFF_MIN", "2"))
        backoff_max = int(os.getenv("REDIS_RETRY_BACKOFF_MAX", "10"))

        # Check circuit breaker before attempting operation
        if not self._check_circuit_breaker():
            operation_name = getattr(operation, "__name__", "unknown")
            last_failure_time = (
                self._circuit_breaker_last_failure.isoformat()
                if self._circuit_breaker_last_failure
                else "unknown"
            )
            logger.warning(
                f"Redis operation '{operation_name}' blocked by circuit breaker (state: {self._circuit_breaker_state}, "
                f"failures: {self._circuit_breaker_failures}/{self._circuit_breaker_failure_threshold}, "
                f"last failure: {last_failure_time})",
                extra={
                    "operation": operation_name,
                    "circuit_breaker_state": self._circuit_breaker_state,
                    "circuit_breaker_failures": self._circuit_breaker_failures,
                    "circuit_breaker_threshold": self._circuit_breaker_failure_threshold,
                },
            )
            raise CircuitBreakerError(
                f"Circuit breaker is open (state: {self._circuit_breaker_state}, "
                f"failures: {self._circuit_breaker_failures}/{self._circuit_breaker_failure_threshold}). "
                f"Operation '{operation_name}' blocked. Recovery timeout: {self._circuit_breaker_recovery_timeout}s"
            )

        @retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=backoff_min, max=backoff_max),
            retry=retry_if_exception_type(
                (redis.ConnectionError, redis.TimeoutError, redis.RedisError)
            ),
            reraise=True,
        )
        async def _retry_operation():
            redis_client = await self.get_redis()
            start_time = time.time()
            try:
                result = await operation(redis_client, *args, **kwargs)
                duration = time.time() - start_time
                self._operation_times.append(duration)
                self._success_count += 1
                self._record_circuit_breaker_success()
                return result
            except (redis.ConnectionError, redis.TimeoutError, redis.RedisError) as e:
                duration = time.time() - start_time
                self._operation_times.append(duration)
                self._error_count += 1
                self._record_circuit_breaker_failure()
                # Enhanced error context
                operation_name = getattr(operation, "__name__", "unknown")
                logger.error(
                    f"Redis operation '{operation_name}' failed after {duration:.3f}s: {type(e).__name__}: {e}",
                    extra={
                        "redis_error_type": type(e).__name__,
                        "operation": operation_name,
                        "duration_seconds": duration,
                    },
                )
                raise
            except Exception as e:
                duration = time.time() - start_time
                self._operation_times.append(duration)
                self._error_count += 1
                self._record_circuit_breaker_failure()
                operation_name = getattr(operation, "__name__", "unknown")
                logger.error(
                    f"Unexpected error in Redis operation '{operation_name}' after {duration:.3f}s: {type(e).__name__}: {e}",
                    exc_info=True,
                    extra={
                        "error_type": type(e).__name__,
                        "operation": operation_name,
                        "duration_seconds": duration,
                    },
                )
                raise

        try:
            return await _retry_operation()
        except RetryError as e:
            operation_name = getattr(operation, "__name__", "unknown")
            logger.error(
                f"Redis operation '{operation_name}' failed after {retry_attempts} retry attempts: {e}",
                extra={
                    "operation": operation_name,
                    "retry_attempts": retry_attempts,
                    "circuit_breaker_state": self._circuit_breaker_state,
                },
            )
            raise CircuitBreakerError(
                f"Redis operation '{operation_name}' failed after {retry_attempts} attempts. "
                f"Circuit breaker state: {self._circuit_breaker_state}"
            ) from e

    async def execute(
        self, operation: Callable[[redis.Redis], Awaitable[Any]], *args, **kwargs
    ) -> Any:
        """
        Execute a Redis operation with retry logic, circuit breaker, and metrics.

        Args:
            operation: Async function that takes redis.Redis as first arg
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the operation
        """
        try:
            result = await self._execute_with_retry(operation, *args, **kwargs)
            await self._publish_metrics_if_needed()
            return result
        except CircuitBreakerError as e:
            await self._publish_metrics_if_needed()
            raise
        except Exception as e:
            await self._publish_metrics_if_needed()
            raise

    async def _reset_connection(self):
        """Reset Redis connection and pool to force reconnection."""
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception as close_error:
                logger.debug(f"Error closing Redis client during reset: {close_error}")
            self._redis = None

        # Reset pool to force recreation on next use
        if self._pool:
            try:
                await self._pool.aclose()
            except Exception as pool_error:
                logger.debug(
                    f"Error closing connection pool during reset: {pool_error}"
                )
            self._pool = None

    async def _health_check_loop(self):
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def health_check(self) -> bool:
        """Check Redis health and update status."""
        # Use a shorter timeout for health checks to fail fast
        health_check_timeout = float(os.getenv("REDIS_HEALTH_CHECK_TIMEOUT", "3.0"))

        try:
            redis_client = await self.get_redis()
            # Add timeout wrapper to prevent health check from hanging
            try:
                await asyncio.wait_for(
                    redis_client.ping(), timeout=health_check_timeout
                )
            except asyncio.TimeoutError:
                raise redis.TimeoutError(
                    f"Health check ping timed out after {health_check_timeout}s"
                )

            self._health_status = True
            self._last_health_check = datetime.utcnow()
            self._record_circuit_breaker_success()
            return True
        except (redis.TimeoutError, asyncio.TimeoutError) as e:
            logger.warning(
                f"Redis health check failed: TimeoutError: {e}",
                extra={
                    "health_check_error_type": "TimeoutError",
                    "circuit_breaker_state": self._circuit_breaker_state,
                    "circuit_breaker_failures": self._circuit_breaker_failures,
                    "timeout_seconds": health_check_timeout,
                },
            )
            self._health_status = False
            self._last_health_check = datetime.utcnow()
            self._record_circuit_breaker_failure()
            # Force reconnection on next use
            await self._reset_connection()
            return False
        except Exception as e:
            logger.warning(
                f"Redis health check failed: {type(e).__name__}: {e}",
                extra={
                    "health_check_error_type": type(e).__name__,
                    "circuit_breaker_state": self._circuit_breaker_state,
                    "circuit_breaker_failures": self._circuit_breaker_failures,
                },
            )
            self._health_status = False
            self._last_health_check = datetime.utcnow()
            self._record_circuit_breaker_failure()
            # Force reconnection on next use
            await self._reset_connection()
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        pool_info = {}
        if self._pool:
            try:
                # Get pool statistics
                pool_info = {
                    "max_connections": getattr(self._pool, "max_connections", None),
                    "created_connections": getattr(
                        self._pool, "created_connections", None
                    ),
                }
            except Exception:
                # Pool info not available
                pass

        error_rate = 0.0
        if (self._success_count + self._error_count) > 0:
            error_rate = self._error_count / (self._success_count + self._error_count)

        return {
            "healthy": self._health_status,
            "last_health_check": (
                self._last_health_check.isoformat() if self._last_health_check else None
            ),
            "circuit_breaker_state": self._circuit_breaker_state,
            "circuit_breaker_failures": self._circuit_breaker_failures,
            "pool_info": pool_info,
            "total_operations": self._success_count + self._error_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "error_rate": error_rate,
        }

    async def _publish_metrics_if_needed(self):
        """Publish metrics to CloudWatch if enabled and interval has passed."""
        if not self._cloudwatch_enabled or not self._cloudwatch_client:
            return

        now = datetime.utcnow()
        if (
            self._last_metrics_publish
            and (now - self._last_metrics_publish).total_seconds()
            < self._metrics_publish_interval
        ):
            return

        try:
            metrics = []

            # Calculate latency percentiles
            if self._operation_times:
                sorted_times = sorted(self._operation_times)
                p50 = sorted_times[int(len(sorted_times) * 0.5)] * 1000  # Convert to ms
                p95 = (
                    sorted_times[int(len(sorted_times) * 0.95)] * 1000
                    if len(sorted_times) > 1
                    else p50
                )
                p99 = (
                    sorted_times[int(len(sorted_times) * 0.99)] * 1000
                    if len(sorted_times) > 1
                    else p50
                )

                metrics.extend(
                    [
                        {
                            "MetricName": "RedisOperationLatencyP50",
                            "Value": p50,
                            "Unit": "Milliseconds",
                        },
                        {
                            "MetricName": "RedisOperationLatencyP95",
                            "Value": p95,
                            "Unit": "Milliseconds",
                        },
                        {
                            "MetricName": "RedisOperationLatencyP99",
                            "Value": p99,
                            "Unit": "Milliseconds",
                        },
                    ]
                )

            # Error rate
            total_ops = self._success_count + self._error_count
            if total_ops > 0:
                error_rate = (self._error_count / total_ops) * 100
                metrics.append(
                    {
                        "MetricName": "RedisErrorRate",
                        "Value": error_rate,
                        "Unit": "Percent",
                    }
                )

            # Connection pool usage
            if self._pool:
                try:
                    max_conn = getattr(self._pool, "max_connections", 0)
                    created_conn = getattr(self._pool, "created_connections", 0)
                    pool_usage = (created_conn / max_conn) * 100 if max_conn > 0 else 0
                    metrics.append(
                        {
                            "MetricName": "RedisConnectionPoolUsage",
                            "Value": pool_usage,
                            "Unit": "Percent",
                        }
                    )
                except Exception:
                    # Pool info not available, skip this metric
                    pass

            # Circuit breaker state (1 = open, 0 = closed)
            circuit_state = 1 if self._circuit_breaker_state == "open" else 0
            metrics.append(
                {
                    "MetricName": "RedisCircuitBreakerOpen",
                    "Value": circuit_state,
                    "Unit": "Count",
                }
            )

            if metrics:
                # Publish metrics in batches
                for i in range(
                    0, len(metrics), 20
                ):  # CloudWatch allows 20 metrics per request
                    batch = metrics[i : i + 20]
                    self._cloudwatch_client.put_metric_data(
                        Namespace="MarketingTool/Redis", MetricData=batch
                    )

                self._last_metrics_publish = now
                logger.debug(f"Published {len(metrics)} Redis metrics to CloudWatch")

        except Exception as e:
            logger.warning(f"Failed to publish CloudWatch metrics: {e}")

    async def cleanup(self):
        """Cleanup Redis connections."""
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        # Close Redis client
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
            self._redis = None

        # Close connection pool
        if self._pool:
            try:
                await self._pool.aclose()
            except Exception as e:
                logger.warning(f"Error closing connection pool: {e}")
            self._pool = None

        logger.info("Redis manager cleaned up")


# Global Redis manager instance
_redis_manager: Optional[RedisManager] = None


def get_redis_manager() -> RedisManager:
    """Get or create the global Redis manager instance."""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
    return _redis_manager
