"""
Tests for logging middleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marketing_project.middleware.logging import LoggingMiddleware, RequestIDMiddleware


def test_logging_middleware_initialization():
    """Test logging middleware initialization."""
    app = FastAPI()
    middleware = LoggingMiddleware(app, log_requests=True, log_responses=True)

    assert middleware.app == app
    assert middleware.log_requests is True
    assert middleware.log_responses is True


def test_logging_middleware_with_options():
    """Test logging middleware with different options."""
    app = FastAPI()
    middleware = LoggingMiddleware(app, log_requests=False, log_responses=False)

    assert middleware.log_requests is False
    assert middleware.log_responses is False


def test_logging_middleware_request_logging():
    """Test that requests are logged."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    # Add middleware
    app.add_middleware(LoggingMiddleware, log_requests=True, log_responses=True)

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    # Check that request ID header is added
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers


def test_request_id_middleware_initialization():
    """Test request ID middleware initialization."""
    app = FastAPI()
    middleware = RequestIDMiddleware(app, header_name="X-Request-ID")

    assert middleware.app == app
    assert middleware.header_name == "X-Request-ID"


def test_request_id_middleware_custom_header():
    """Test request ID middleware with custom header."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    # Add middleware with custom header
    app.add_middleware(RequestIDMiddleware, header_name="X-Custom-Request-ID")

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    # Check that custom request ID header is added
    assert "X-Custom-Request-ID" in response.headers
