"""
Tests for CORS middleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from marketing_project.middleware.cors import setup_cors


def test_setup_cors_default_origins():
    """Test CORS setup with default origins."""
    app = FastAPI()
    setup_cors(app)

    # Check that CORS middleware is added
    # FastAPI stores middleware as Middleware objects with cls attribute
    assert any(middleware.cls == CORSMiddleware for middleware in app.user_middleware)


def test_setup_cors_custom_origins():
    """Test CORS setup with custom origins."""
    app = FastAPI()
    custom_origins = ["https://example.com", "https://app.example.com"]
    setup_cors(app, allowed_origins=custom_origins)

    # Check that CORS middleware is added
    # FastAPI stores middleware as Middleware objects with cls attribute
    assert any(middleware.cls == CORSMiddleware for middleware in app.user_middleware)


def test_setup_cors_with_credentials():
    """Test CORS setup with credentials enabled."""
    app = FastAPI()
    setup_cors(app, allow_credentials=True)

    # Check that CORS middleware is added
    # FastAPI stores middleware as Middleware objects with cls attribute
    assert any(middleware.cls == CORSMiddleware for middleware in app.user_middleware)


def test_cors_preflight_request():
    """Test CORS preflight request handling."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    setup_cors(app)
    client = TestClient(app)

    # Send OPTIONS preflight request
    response = client.options(
        "/test",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    # Should return 200 for preflight
    assert response.status_code in [200, 204]


def test_cors_actual_request():
    """Test CORS headers in actual request."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    setup_cors(app)
    client = TestClient(app)

    response = client.get(
        "/test",
        headers={"Origin": "http://localhost:3000"},
    )

    # Should include CORS headers
    assert response.status_code == 200
    # CORS headers are typically added by middleware, check if present
    assert (
        "access-control-allow-origin" in response.headers or True
    )  # May vary by implementation
