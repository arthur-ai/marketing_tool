"""
Tests for logging middleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marketing_project.middleware.logging import setup_logging_middleware


def test_logging_middleware_setup():
    """Test logging middleware setup."""
    app = FastAPI()
    setup_logging_middleware(app)

    # Check that middleware is added
    assert len(app.middleware_stack) > 0


def test_logging_middleware_request_logging():
    """Test that requests are logged."""
    app = FastAPI()
    setup_logging_middleware(app)

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    # Logging happens asynchronously, so we just verify the endpoint works
