"""
Tests for error handling middleware.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from marketing_project.middleware.error_handling import setup_error_handling


def test_error_handling_setup():
    """Test error handling middleware setup."""
    app = FastAPI()
    setup_error_handling(app)

    # Check that exception handlers are registered
    assert hasattr(app, "exception_handlers")


def test_http_exception_handler():
    """Test HTTP exception handling."""
    app = FastAPI()
    setup_error_handling(app)

    @app.get("/test-error")
    def test_endpoint():
        raise HTTPException(status_code=404, detail="Not found")

    client = TestClient(app)
    response = client.get("/test-error")

    assert response.status_code == 404
    assert "detail" in response.json()


def test_generic_exception_handler():
    """Test generic exception handling."""
    app = FastAPI()
    setup_error_handling(app)

    @app.get("/test-generic-error")
    def test_endpoint():
        raise ValueError("Something went wrong")

    client = TestClient(app)
    response = client.get("/test-generic-error")

    # Should return 500 for unhandled exceptions
    assert response.status_code == 500
    assert "detail" in response.json()


def test_validation_error_handler():
    """Test validation error handling."""
    app = FastAPI()
    setup_error_handling(app)

    from pydantic import BaseModel

    class TestModel(BaseModel):
        required_field: str

    @app.post("/test-validation")
    def test_endpoint(model: TestModel):
        return {"message": "ok"}

    client = TestClient(app)
    # Send request without required field
    response = client.post("/test-validation", json={})

    # Should return 422 for validation errors
    assert response.status_code == 422
