"""
Tests for error handling middleware.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from marketing_project.middleware.error_handling import ErrorHandlingMiddleware


def test_error_handling_middleware_initialization():
    """Test error handling middleware initialization."""
    app = FastAPI()
    middleware = ErrorHandlingMiddleware(app, debug=False)

    assert middleware.app == app
    assert middleware.debug is False


def test_error_handling_middleware_with_debug():
    """Test error handling middleware with debug enabled."""
    app = FastAPI()
    middleware = ErrorHandlingMiddleware(app, debug=True)

    assert middleware.debug is True


def test_http_exception_handling():
    """Test HTTP exception handling through middleware."""
    app = FastAPI()

    @app.get("/test-error")
    def test_endpoint():
        raise HTTPException(status_code=404, detail="Not found")

    # Add middleware - must be added before routes are registered
    # Note: FastAPI's default exception handlers may take precedence
    # The middleware catches unhandled exceptions, but HTTPException
    # is typically handled by FastAPI's default handler
    app.add_middleware(ErrorHandlingMiddleware, debug=False)

    client = TestClient(app)
    response = client.get("/test-error")

    assert response.status_code == 404
    # FastAPI's default handler returns {"detail": "..."}
    # The middleware would return {"success": False, ...} but only for unhandled exceptions
    data = response.json()
    assert "detail" in data


def test_generic_exception_handling():
    """Test generic exception handling through middleware."""
    app = FastAPI()

    @app.get("/test-generic-error")
    def test_endpoint():
        raise ValueError("Something went wrong")

    # Add middleware
    app.add_middleware(ErrorHandlingMiddleware, debug=False)

    client = TestClient(app)
    response = client.get("/test-generic-error")

    # Should return 500 for unhandled exceptions
    assert response.status_code == 500
    data = response.json()
    assert "success" in data
    assert data["success"] is False


def test_validation_error_handling():
    """Test validation error handling."""
    app = FastAPI()

    from pydantic import BaseModel

    class TestModel(BaseModel):
        required_field: str

    @app.post("/test-validation")
    def test_endpoint(model: TestModel):
        return {"message": "ok"}

    # Add middleware
    app.add_middleware(ErrorHandlingMiddleware, debug=False)

    client = TestClient(app)
    # Send request without required field
    response = client.post("/test-validation", json={})

    # Should return 422 for validation errors
    assert response.status_code == 422
    # FastAPI's default validation error handler returns {"detail": [...]}
    # The middleware would format it differently, but FastAPI's handler takes precedence
    data = response.json()
    assert "detail" in data
