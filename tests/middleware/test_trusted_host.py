"""
Tests for trusted host middleware.
"""

import pytest
from fastapi import FastAPI, Request, status
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from marketing_project.middleware.trusted_host import (
    TrustedHostMiddlewareWithHealthBypass,
)


@pytest.fixture
def app():
    """Create a FastAPI app for testing."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    @app.get("/api/v1/health")
    def health_endpoint():
        return {"status": "healthy"}

    @app.get("/api/v1/ready")
    def ready_endpoint():
        return {"status": "ready"}

    return app


def test_trusted_host_middleware_allowed_host():
    """Test that requests from allowed hosts are accepted."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    app.add_middleware(
        TrustedHostMiddlewareWithHealthBypass, allowed_hosts=["localhost", "127.0.0.1"]
    )

    client = TestClient(app)
    response = client.get("/test", headers={"Host": "localhost"})
    assert response.status_code == 200
    assert response.json() == {"message": "test"}


def test_trusted_host_middleware_rejected_host():
    """Test that requests from untrusted hosts are rejected."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    app.add_middleware(
        TrustedHostMiddlewareWithHealthBypass, allowed_hosts=["localhost", "127.0.0.1"]
    )

    client = TestClient(app)
    response = client.get("/test", headers={"Host": "evil.com"})
    assert response.status_code == 400
    assert "Invalid host header" in response.json()["detail"]


def test_trusted_host_middleware_health_check_bypass():
    """Test that health check endpoints bypass host validation."""
    app = FastAPI()

    @app.get("/api/v1/health")
    def health_endpoint():
        return {"status": "healthy"}

    app.add_middleware(
        TrustedHostMiddlewareWithHealthBypass, allowed_hosts=["localhost", "127.0.0.1"]
    )

    client = TestClient(app)
    # Health check should work even with untrusted host
    response = client.get("/api/v1/health", headers={"Host": "evil.com"})
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_trusted_host_middleware_ready_check_bypass():
    """Test that ready check endpoints bypass host validation."""
    app = FastAPI()

    @app.get("/api/v1/ready")
    def ready_endpoint():
        return {"status": "ready"}

    app.add_middleware(
        TrustedHostMiddlewareWithHealthBypass, allowed_hosts=["localhost", "127.0.0.1"]
    )

    client = TestClient(app)
    # Ready check should work even with untrusted host
    response = client.get("/api/v1/ready", headers={"Host": "evil.com"})
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_trusted_host_middleware_wildcard_subdomain():
    """Test that wildcard subdomain matching works."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    app.add_middleware(
        TrustedHostMiddlewareWithHealthBypass,
        allowed_hosts=["*.arthur.ai", "arthur.ai"],
    )

    client = TestClient(app)
    # Test exact domain match
    response = client.get("/test", headers={"Host": "arthur.ai"})
    assert response.status_code == 200

    # Test subdomain match
    response = client.get("/test", headers={"Host": "api.arthur.ai"})
    assert response.status_code == 200

    # Test nested subdomain match
    response = client.get("/test", headers={"Host": "app.api.arthur.ai"})
    assert response.status_code == 200

    # Test non-matching domain
    response = client.get("/test", headers={"Host": "evil.com"})
    assert response.status_code == 400


def test_trusted_host_middleware_no_allowed_hosts():
    """Test that middleware allows all hosts when no allowed hosts configured."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    app.add_middleware(TrustedHostMiddlewareWithHealthBypass, allowed_hosts=[])

    client = TestClient(app)
    # Should allow any host when no allowed hosts configured
    response = client.get("/test", headers={"Host": "any-host.com"})
    assert response.status_code == 200


def test_trusted_host_middleware_host_with_port():
    """Test that middleware handles host headers with ports."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    app.add_middleware(
        TrustedHostMiddlewareWithHealthBypass, allowed_hosts=["localhost", "127.0.0.1"]
    )

    client = TestClient(app)
    # Should strip port from host header
    response = client.get("/test", headers={"Host": "localhost:8080"})
    assert response.status_code == 200


def test_trusted_host_middleware_missing_host_header():
    """Test that middleware handles missing host header."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    app.add_middleware(
        TrustedHostMiddlewareWithHealthBypass, allowed_hosts=["localhost", "127.0.0.1"]
    )

    client = TestClient(app)
    # Missing host header should be treated as empty string
    response = client.get("/test", headers={})
    assert response.status_code == 400
