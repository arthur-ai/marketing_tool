"""
Tests for retry strategy service.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.retry_strategy import (
    CircuitBreaker,
    CircuitState,
    ErrorType,
    RetryMetadata,
    RetryStrategy,
)


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_initialization_custom_params(self):
        """Test circuit breaker with custom parameters."""
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout=120)
        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 120

    def test_record_success_closed(self):
        """Test recording success in closed state."""
        cb = CircuitBreaker()
        cb.failure_count = 2
        cb.record_success()
        assert cb.failure_count == 0

    def test_record_success_half_open(self):
        """Test recording success in half-open state."""
        cb = CircuitBreaker(half_open_max_calls=2)
        cb.state = CircuitState.HALF_OPEN
        cb.success_count = 1
        cb.half_open_calls = 1

        cb.record_success()

        assert cb.success_count == 2
        assert cb.half_open_calls == 2
        # Should close after reaching max calls
        assert cb.state == CircuitState.CLOSED

    def test_record_failure_closed(self):
        """Test recording failure in closed state."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.failure_count = 2

        cb.record_failure()

        assert cb.failure_count == 3
        assert cb.state == CircuitState.OPEN
        assert cb.last_failure_time is not None

    def test_record_failure_half_open(self):
        """Test recording failure in half-open state."""
        cb = CircuitBreaker()
        cb.state = CircuitState.HALF_OPEN

        cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.half_open_calls == 0

    def test_can_attempt_closed(self):
        """Test can_attempt in closed state."""
        cb = CircuitBreaker()
        assert cb.can_attempt() is True

    def test_can_attempt_open(self):
        """Test can_attempt in open state."""
        cb = CircuitBreaker(recovery_timeout=60)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = datetime.utcnow() - timedelta(seconds=30)

        assert cb.can_attempt() is False

    def test_can_attempt_open_recovery(self):
        """Test can_attempt in open state after recovery timeout."""
        cb = CircuitBreaker(recovery_timeout=60)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = datetime.utcnow() - timedelta(seconds=61)

        assert cb.can_attempt() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_can_attempt_half_open(self):
        """Test can_attempt in half-open state."""
        cb = CircuitBreaker(half_open_max_calls=2)
        cb.state = CircuitState.HALF_OPEN
        cb.half_open_calls = 1

        assert cb.can_attempt() is True

        cb.half_open_calls = 2
        assert cb.can_attempt() is False

    def test_get_state(self):
        """Test get_state method."""
        cb = CircuitBreaker()
        assert cb.get_state() == CircuitState.CLOSED

        cb.state = CircuitState.OPEN
        assert cb.get_state() == CircuitState.OPEN


class TestRetryStrategy:
    """Test RetryStrategy class."""

    def test_initialization(self):
        """Test retry strategy initialization."""
        strategy = RetryStrategy()
        assert strategy.max_retries == 3
        assert strategy.circuit_breaker is not None
        assert len(strategy.retry_history) == 0

    def test_initialization_custom(self):
        """Test retry strategy with custom parameters."""
        cb = CircuitBreaker()
        strategy = RetryStrategy(circuit_breaker=cb, max_retries=5)
        assert strategy.max_retries == 5
        assert strategy.circuit_breaker is cb

    def test_classify_error_network(self):
        """Test error classification for network errors."""
        error = ConnectionError("Connection failed")
        error_type = RetryStrategy.classify_error(error)
        assert error_type == ErrorType.NETWORK

        error = TimeoutError("Request timeout")
        error_type = RetryStrategy.classify_error(error)
        assert error_type == ErrorType.NETWORK

    def test_classify_error_rate_limit(self):
        """Test error classification for rate limit errors."""
        error = Exception("Rate limit exceeded")
        error_type = RetryStrategy.classify_error(error)
        assert error_type == ErrorType.RATE_LIMIT

        error = Exception("429 Too Many Requests")
        error_type = RetryStrategy.classify_error(error)
        assert error_type == ErrorType.RATE_LIMIT

    def test_classify_error_validation(self):
        """Test error classification for validation errors."""
        error = ValueError("Invalid input")
        error_type = RetryStrategy.classify_error(error)
        assert error_type == ErrorType.VALIDATION

        error = Exception("400 Bad Request")
        error_type = RetryStrategy.classify_error(error)
        assert error_type == ErrorType.VALIDATION

    def test_classify_error_server_error(self):
        """Test error classification for server errors."""
        error = Exception("500 Internal Server Error")
        error_type = RetryStrategy.classify_error(error)
        assert error_type == ErrorType.SERVER_ERROR

    def test_classify_error_client_error(self):
        """Test error classification for client errors."""
        error = Exception("401 Unauthorized")
        error_type = RetryStrategy.classify_error(error)
        assert error_type == ErrorType.CLIENT_ERROR

    def test_classify_error_unknown(self):
        """Test error classification for unknown errors."""
        error = Exception("Some random error")
        error_type = RetryStrategy.classify_error(error)
        assert error_type == ErrorType.UNKNOWN

    def test_get_retry_delay_network(self):
        """Test retry delay calculation for network errors."""
        strategy = RetryStrategy()
        assert strategy.get_retry_delay(0, ErrorType.NETWORK) == 1.0
        assert strategy.get_retry_delay(1, ErrorType.NETWORK) == 2.0
        assert strategy.get_retry_delay(2, ErrorType.NETWORK) == 4.0
        assert strategy.get_retry_delay(3, ErrorType.NETWORK) == 4.0  # Capped

    def test_get_retry_delay_rate_limit(self):
        """Test retry delay calculation for rate limit errors."""
        strategy = RetryStrategy()
        assert strategy.get_retry_delay(0, ErrorType.RATE_LIMIT) == 60.0
        assert strategy.get_retry_delay(1, ErrorType.RATE_LIMIT) == 120.0
        assert strategy.get_retry_delay(2, ErrorType.RATE_LIMIT) == 240.0

    def test_get_retry_delay_validation(self):
        """Test retry delay calculation for validation errors."""
        strategy = RetryStrategy()
        assert strategy.get_retry_delay(0, ErrorType.VALIDATION) == 0.0

    def test_should_retry_validation(self):
        """Test should_retry for validation errors."""
        strategy = RetryStrategy()
        assert strategy.should_retry(0, ErrorType.VALIDATION) is False

    def test_should_retry_client_error(self):
        """Test should_retry for client errors."""
        strategy = RetryStrategy()
        assert strategy.should_retry(0, ErrorType.CLIENT_ERROR) is True
        assert strategy.should_retry(1, ErrorType.CLIENT_ERROR) is False

    def test_should_retry_max_retries(self):
        """Test should_retry with max retries."""
        strategy = RetryStrategy(max_retries=2)
        assert strategy.should_retry(0, ErrorType.NETWORK) is True
        assert strategy.should_retry(1, ErrorType.NETWORK) is True
        assert strategy.should_retry(2, ErrorType.NETWORK) is False

    def test_should_retry_circuit_breaker_open(self):
        """Test should_retry when circuit breaker is open."""
        strategy = RetryStrategy()
        strategy.circuit_breaker.state = CircuitState.OPEN
        strategy.circuit_breaker.last_failure_time = datetime.utcnow()

        assert strategy.should_retry(0, ErrorType.NETWORK) is False

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """Test execute_with_retry with successful call."""
        strategy = RetryStrategy()

        async def mock_func():
            return "success"

        result = await strategy.execute_with_retry(mock_func)

        assert result == "success"
        assert len(strategy.retry_history) == 0

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_retry(self):
        """Test execute_with_retry with success after retry."""
        strategy = RetryStrategy(max_retries=2)
        call_count = 0

        async def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network error")
            return "success"

        result = await strategy.execute_with_retry(mock_func)

        assert result == "success"
        assert call_count == 2
        assert len(strategy.retry_history) == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self):
        """Test execute_with_retry with all retries exhausted."""
        strategy = RetryStrategy(max_retries=2)

        async def mock_func():
            raise ConnectionError("Network error")

        with pytest.raises(ConnectionError):
            await strategy.execute_with_retry(mock_func)

        assert len(strategy.retry_history) == 3  # 3 attempts

    @pytest.mark.asyncio
    async def test_execute_with_retry_circuit_breaker_open(self):
        """Test execute_with_retry when circuit breaker is open."""
        strategy = RetryStrategy()
        strategy.circuit_breaker.state = CircuitState.OPEN
        strategy.circuit_breaker.last_failure_time = datetime.utcnow()

        async def mock_func():
            return "success"

        with pytest.raises(Exception, match="Circuit breaker is open"):
            await strategy.execute_with_retry(mock_func)

    @pytest.mark.asyncio
    async def test_execute_with_retry_error_handler(self):
        """Test execute_with_retry with error handler."""
        strategy = RetryStrategy(max_retries=1)
        errors_caught = []

        def error_handler(error):
            errors_caught.append(error)

        async def mock_func():
            raise ConnectionError("Network error")

        with pytest.raises(ConnectionError):
            await strategy.execute_with_retry(mock_func, error_handler=error_handler)

        assert len(errors_caught) > 0

    def test_get_retry_history(self):
        """Test get_retry_history method."""
        strategy = RetryStrategy()
        strategy.retry_history = [
            RetryMetadata(
                attempt=1,
                error_type=ErrorType.NETWORK,
                error_message="Connection failed",
                delay=1.0,
            )
        ]

        history = strategy.get_retry_history()
        assert len(history) == 1
        assert history[0].error_type == ErrorType.NETWORK

    def test_reset(self):
        """Test reset method."""
        strategy = RetryStrategy()
        strategy.retry_history = [RetryMetadata(1, ErrorType.NETWORK, "error", 1.0)]
        strategy.circuit_breaker.failure_count = 5

        strategy.reset()

        assert len(strategy.retry_history) == 0
        assert strategy.circuit_breaker.failure_count == 0
