"""
Integration tests for Keycloak authentication.

These tests verify the complete authentication flow from token validation
to user context extraction and role-based access control.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from marketing_project.middleware.keycloak_auth import (
    get_current_user,
    get_keycloak_jwks,
    get_public_key_from_jwks,
)
from marketing_project.models.user_context import UserContext
from tests.utils.keycloak_test_helpers import (
    create_expired_jwt,
    create_test_jwt,
    create_user_context,
    get_test_private_key,
    mock_keycloak_public_key,
)


@pytest.fixture
def mock_keycloak_env(monkeypatch):
    """Mock Keycloak environment variables."""
    monkeypatch.setenv("KEYCLOAK_SERVER_URL", "https://test-keycloak.com")
    monkeypatch.setenv("KEYCLOAK_REALM", "test-realm")
    monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "test-client")
    monkeypatch.setenv("KEYCLOAK_PUBLIC_KEY", mock_keycloak_public_key())


@pytest.fixture
def app():
    """Create a test FastAPI app with authentication."""
    app = FastAPI()

    @app.get("/protected")
    async def protected_endpoint(user: UserContext = Depends(get_current_user)):
        return {"user_id": user.user_id, "roles": user.roles}

    @app.get("/public")
    async def public_endpoint():
        return {"message": "public"}

    return app


class TestKeycloakIntegration:
    """Integration tests for Keycloak authentication flow."""

    @pytest.mark.asyncio
    async def test_full_authentication_flow(self, mock_keycloak_env, app):
        """Test complete authentication flow from token to user context."""
        token = create_test_jwt(
            user_id="test-user-123",
            email="test@example.com",
            username="testuser",
            roles=["admin", "user"],
        )

        client = TestClient(app)
        response = client.get(
            "/protected", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test-user-123"
        assert "admin" in data["roles"]
        assert "user" in data["roles"]

    @pytest.mark.asyncio
    async def test_public_endpoint_accessible(self, app):
        """Test that public endpoints don't require authentication."""
        client = TestClient(app)
        response = client.get("/public")

        assert response.status_code == 200
        assert response.json()["message"] == "public"

    @pytest.mark.asyncio
    async def test_protected_endpoint_requires_auth(self, app):
        """Test that protected endpoints require authentication."""
        client = TestClient(app)
        response = client.get("/protected")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_rejected(self, mock_keycloak_env, app):
        """Test that invalid tokens are rejected."""
        client = TestClient(app)
        response = client.get(
            "/protected", headers={"Authorization": "Bearer invalid-token"}
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_token_rejected(self, mock_keycloak_env, app):
        """Test that expired tokens are rejected."""
        token = create_expired_jwt()

        client = TestClient(app)
        response = client.get(
            "/protected", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_jwks_fetching(self, mock_keycloak_env):
        """Test JWKS fetching from Keycloak."""
        mock_jwks = {
            "keys": [
                {
                    "kid": "test-kid",
                    "kty": "RSA",
                    "use": "sig",
                    "n": "test-n",
                    "e": "AQAB",
                }
            ]
        }

        with patch("httpx.Client.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_jwks
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            jwks = get_keycloak_jwks()
            assert jwks == mock_jwks

    @pytest.mark.asyncio
    async def test_public_key_from_jwks(self, mock_keycloak_env):
        """Test extracting public key from JWKS using token kid."""
        token = create_test_jwt()
        mock_jwks = {
            "keys": [
                {
                    "kid": "test-kid",
                    "kty": "RSA",
                    "use": "sig",
                    "n": "test-n",
                    "e": "AQAB",
                }
            ]
        }

        # Mock get_unverified_header to return our test kid
        with patch("jwt.get_unverified_header") as mock_header:
            mock_header.return_value = {"kid": "test-kid", "alg": "RS256"}

            public_key = get_public_key_from_jwks(token, mock_jwks)
            # Should return a key object (jwk.construct result)
            assert public_key is not None

    @pytest.mark.asyncio
    async def test_role_based_access(self, mock_keycloak_env):
        """Test role-based access control integration."""
        from marketing_project.middleware.rbac import require_roles

        admin_token = create_test_jwt(roles=["admin"])
        user_token = create_test_jwt(roles=["user"])

        app = FastAPI()

        @app.get("/admin-only")
        async def admin_endpoint(user: UserContext = Depends(require_roles(["admin"]))):
            return {"message": "admin access granted"}

        client = TestClient(app)

        # Admin should have access
        response = client.get(
            "/admin-only", headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200

        # Regular user should be denied
        response = client.get(
            "/admin-only", headers={"Authorization": f"Bearer {user_token}"}
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_token_with_multiple_role_sources(self, mock_keycloak_env, app):
        """Test token with roles from multiple sources (realm + client)."""
        import jwt

        from tests.utils.keycloak_test_helpers import get_test_private_key

        now = int(time.time())
        claims = {
            "sub": "test-user-456",
            "iss": "https://test-keycloak.com/realms/test-realm",
            "aud": "test-client",
            "exp": now + 3600,
            "iat": now,
            "realm_access": {"roles": ["admin"]},
            "resource_access": {"test-client": {"roles": ["editor"]}},
        }
        token = jwt.encode(claims, get_test_private_key(), algorithm="RS256")

        client = TestClient(app)
        response = client.get(
            "/protected", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "admin" in data["roles"]
        assert "editor" in data["roles"]

    @pytest.mark.asyncio
    async def test_concurrent_requests_same_token(self, mock_keycloak_env, app):
        """Test that the same token works for concurrent requests."""
        token = create_test_jwt(user_id="concurrent-user")

        client = TestClient(app)

        # Simulate concurrent requests
        responses = []
        for _ in range(5):
            response = client.get(
                "/protected", headers={"Authorization": f"Bearer {token}"}
            )
            responses.append(response)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json()["user_id"] == "concurrent-user" for r in responses)
