"""
Adaptive Retry Strategy with Circuit Breaker Pattern.

This module provides intelligent retry logic that adapts to different error types
and implements a circuit breaker pattern to prevent cascading failures.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of error types for retry strategy."""

    NETWORK = "network"  # Network connectivity issues
    RATE_LIMIT = "rate_limit"  # API rate limiting
    VALIDATION = "validation"  # Validation errors (no retry)
    TIMEOUT = "timeout"  # Request timeout
    SERVER_ERROR = "server_error"  # 5xx server errors
    CLIENT_ERROR = "client_error"  # 4xx client errors (usually no retry)
    UNKNOWN = "unknown"  # Unknown error type


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryMetadata:
    """Metadata about retry attempts."""

    attempt: int
    error_type: ErrorType
    error_message: str
    delay: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CircuitBreaker:
    """Circuit breaker implementation for API calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery (half-open)
            half_open_max_calls: Max calls to allow in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

    def record_success(self):
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            self.half_open_calls += 1
            if self.success_count >= self.half_open_max_calls:
                logger.info(
                    "Circuit breaker: Closing circuit after successful recovery"
                )
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.half_open_calls = 0
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def record_failure(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.warning(
                    f"Circuit breaker: Opening circuit after {self.failure_count} failures"
                )
                self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning(
                "Circuit breaker: Re-opening circuit after failed recovery attempt"
            )
            self.state = CircuitState.OPEN
            self.half_open_calls = 0
            self.success_count = 0

    def can_attempt(self) -> bool:
        """Check if a call can be attempted."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                time_since_failure = (
                    datetime.utcnow() - self.last_failure_time
                ).total_seconds()
                if time_since_failure >= self.recovery_timeout:
                    logger.info(
                        "Circuit breaker: Moving to half-open state for recovery test"
                    )
                    self.state = CircuitState.HALF_OPEN
                    self.failure_count = 0
                    self.success_count = 0
                    self.half_open_calls = 0
                    return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state


class RetryStrategy:
    """
    Adaptive retry strategy with error-type-specific handling.

    Implements different retry strategies based on error classification:
    - Network errors: Fast retry (1s, 2s, 4s)
    - Rate limit errors: Exponential backoff with longer delays (60s, 120s, 240s)
    - Validation errors: No retry (immediate failure)
    - Timeout errors: Retry with timeout increase
    - Server errors: Exponential backoff
    """

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        max_retries: int = 3,
    ):
        """
        Initialize retry strategy.

        Args:
            circuit_breaker: Optional circuit breaker instance
            max_retries: Maximum number of retries (default: 3)
        """
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.max_retries = max_retries
        self.retry_history: List[RetryMetadata] = []

    @staticmethod
    def classify_error(
        error: Exception, platform: Optional[str] = None, content: Optional[str] = None
    ) -> ErrorType:
        """
        Classify an error to determine retry strategy.
        Includes platform-specific error detection.

        Args:
            error: The exception that occurred
            platform: Optional platform name for platform-specific error detection
            content: Optional content for platform-specific error detection

        Returns:
            ErrorType classification
        """
        # Check for platform-specific errors first
        if platform:
            try:
                from marketing_project.services.platform_error_handler import (
                    PlatformErrorHandler,
                )

                is_platform_error, error_type, error_details = (
                    PlatformErrorHandler.detect_platform_error(error, platform, content)
                )
                if is_platform_error:
                    # Map platform errors to ErrorType
                    if error_type == "character_limit_exceeded":
                        return ErrorType.VALIDATION  # Don't retry, but can auto-fix
                    elif error_type in [
                        "format_error",
                        "hashtag_error",
                        "subject_line_error",
                    ]:
                        return ErrorType.VALIDATION  # Can auto-fix and retry
            except Exception as e:
                logger.warning(f"Failed to check for platform errors: {e}")

        error_str = str(error).lower()
        error_type = type(error).__name__

        # Network errors
        if any(
            keyword in error_str
            for keyword in [
                "connection",
                "network",
                "timeout",
                "dns",
                "resolve",
                "unreachable",
            ]
        ) or error_type in ("ConnectionError", "TimeoutError", "NetworkError"):
            return ErrorType.NETWORK

        # Rate limit errors
        if any(
            keyword in error_str
            for keyword in [
                "rate limit",
                "rate_limit",
                "429",
                "too many requests",
                "quota",
                "throttle",
            ]
        ):
            return ErrorType.RATE_LIMIT

        # Validation errors
        if any(
            keyword in error_str
            for keyword in [
                "validation",
                "invalid",
                "400",
                "bad request",
                "malformed",
            ]
        ) or error_type in ("ValidationError", "ValueError"):
            return ErrorType.VALIDATION

        # Timeout errors
        if "timeout" in error_str or error_type == "TimeoutError":
            return ErrorType.TIMEOUT

        # Server errors (5xx)
        if any(
            keyword in error_str
            for keyword in ["500", "502", "503", "504", "server error"]
        ):
            return ErrorType.SERVER_ERROR

        # Client errors (4xx)
        if any(
            keyword in error_str for keyword in ["401", "403", "404", "client error"]
        ):
            return ErrorType.CLIENT_ERROR

        return ErrorType.UNKNOWN

    def get_retry_delay(self, attempt: int, error_type: ErrorType) -> float:
        """
        Calculate retry delay based on attempt number and error type.

        Args:
            attempt: Current attempt number (0-indexed)
            error_type: Classification of the error

        Returns:
            Delay in seconds before next retry
        """
        if error_type == ErrorType.NETWORK:
            # Fast retry: 1s, 2s, 4s
            return min(2**attempt, 4.0)

        if error_type == ErrorType.RATE_LIMIT:
            # Longer delays: 60s, 120s, 240s
            return min(60 * (2**attempt), 240.0)

        if error_type == ErrorType.TIMEOUT:
            # Exponential backoff with base 2
            return min(2**attempt, 16.0)

        if error_type == ErrorType.SERVER_ERROR:
            # Exponential backoff
            return min(2**attempt, 8.0)

        if error_type == ErrorType.VALIDATION:
            # No retry
            return 0.0

        if error_type == ErrorType.CLIENT_ERROR:
            # Usually no retry, but allow one attempt
            return 0.0 if attempt > 0 else 1.0

        # Unknown: Default exponential backoff
        return min(2**attempt, 8.0)

    def should_retry(self, attempt: int, error_type: ErrorType) -> bool:
        """
        Determine if we should retry based on attempt and error type.

        Args:
            attempt: Current attempt number (0-indexed)
            error_type: Classification of the error

        Returns:
            True if should retry, False otherwise
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            logger.warning("Circuit breaker is open, not retrying")
            return False

        # Don't retry validation errors
        if error_type == ErrorType.VALIDATION:
            return False

        # Don't retry client errors after first attempt
        if error_type == ErrorType.CLIENT_ERROR and attempt > 0:
            return False

        # Check max retries
        if attempt >= self.max_retries:
            return False

        return True

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        error_handler: Optional[Callable[[Exception], None]] = None,
        **kwargs,
    ) -> Any:
        """
        Execute a function with adaptive retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            error_handler: Optional error handler callback
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            Exception: If all retries exhausted or circuit breaker is open
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                # Check circuit breaker before attempting
                if not self.circuit_breaker.can_attempt():
                    raise Exception(
                        f"Circuit breaker is {self.circuit_breaker.get_state().value}, "
                        "not attempting call"
                    )

                # Execute the function
                result = await func(*args, **kwargs)

                # Record success
                self.circuit_breaker.record_success()

                # Clear retry history on success
                if attempt == 0:
                    self.retry_history = []

                return result

            except Exception as e:
                last_error = e
                error_type = self.classify_error(e)

                # Record failure
                self.circuit_breaker.record_failure()

                # Call error handler if provided
                if error_handler:
                    try:
                        error_handler(e)
                    except Exception as handler_error:
                        logger.warning(
                            f"Error handler raised exception: {handler_error}"
                        )

                # Check if we should retry
                if not self.should_retry(attempt, error_type):
                    logger.error(
                        f"Not retrying after {attempt + 1} attempts. "
                        f"Error type: {error_type.value}, Error: {str(e)}"
                    )
                    break

                # Calculate delay
                delay = self.get_retry_delay(attempt, error_type)

                # Record retry metadata
                retry_meta = RetryMetadata(
                    attempt=attempt + 1,
                    error_type=error_type,
                    error_message=str(e),
                    delay=delay,
                )
                self.retry_history.append(retry_meta)

                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed "
                    f"({error_type.value}): {str(e)}. Retrying in {delay}s..."
                )

                # Wait before retry
                if delay > 0:
                    await asyncio.sleep(delay)

        # All retries exhausted
        if last_error:
            raise last_error
        raise Exception("All retries exhausted without error information")

    def get_retry_history(self) -> List[RetryMetadata]:
        """Get history of retry attempts."""
        return self.retry_history.copy()

    def reset(self):
        """Reset retry strategy state."""
        self.retry_history = []
        self.circuit_breaker = CircuitBreaker()
