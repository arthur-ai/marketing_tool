"""
Tests for Keycloak authentication middleware.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from marketing_project.middleware.keycloak_auth import (
    extract_token_from_header,
    get_current_user,
    is_public_path,
    validate_jwt_token,
)
from tests.utils.keycloak_test_helpers import (
    create_expired_jwt,
    create_test_jwt,
    get_test_private_key,
    mock_keycloak_public_key,
)


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    app = FastAPI()
    return app


@pytest.fixture
def mock_keycloak_env(monkeypatch):
    """Mock Keycloak environment variables."""
    monkeypatch.setenv("KEYCLOAK_SERVER_URL", "https://test-keycloak.com")
    monkeypatch.setenv("KEYCLOAK_REALM", "test-realm")
    monkeypatch.setenv("KEYCLOAK_CLIENT_ID", "test-client")
    monkeypatch.setenv("KEYCLOAK_PUBLIC_KEY", mock_keycloak_public_key())


class TestExtractTokenFromHeader:
    """Tests for token extraction from headers."""

    def test_extract_valid_bearer_token(self):
        """Test extraction of valid Bearer token."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer test-token-123"}

        token = extract_token_from_header(request)
        assert token == "test-token-123"

    def test_extract_token_with_whitespace(self):
        """Test extraction of token with whitespace."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer  test-token-123  "}

        token = extract_token_from_header(request)
        assert token == "test-token-123"

    def test_missing_authorization_header(self):
        """Test handling of missing Authorization header."""
        request = MagicMock(spec=Request)
        request.headers = {}

        token = extract_token_from_header(request)
        assert token is None

    def test_invalid_header_format(self):
        """Test handling of invalid header format."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "InvalidFormat token"}

        token = extract_token_from_header(request)
        assert token is None

    def test_empty_token(self):
        """Test handling of empty token."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer "}

        token = extract_token_from_header(request)
        assert token == ""


class TestValidateJWTToken:
    """Tests for JWT token validation."""

    @pytest.mark.asyncio
    async def test_validate_valid_token(self, mock_keycloak_env):
        """Test validation of valid JWT token."""
        token = create_test_jwt(
            user_id="test-user-123",
            email="test@example.com",
            username="testuser",
            roles=["user", "editor"],
        )

        # Mock the public key retrieval
        with patch(
            "marketing_project.middleware.keycloak_auth.get_keycloak_public_key_from_env",
            return_value=mock_keycloak_public_key(),
        ):
            claims = validate_jwt_token(token)
            assert claims["sub"] == "test-user-123"
            assert claims["email"] == "test@example.com"
            assert claims["preferred_username"] == "testuser"

    @pytest.mark.asyncio
    async def test_validate_expired_token(self, mock_keycloak_env):
        """Test validation of expired token."""
        token = create_expired_jwt()

        with patch(
            "marketing_project.middleware.keycloak_auth.get_keycloak_public_key_from_env",
            return_value=mock_keycloak_public_key(),
        ):
            with pytest.raises(Exception):  # Should raise HTTPException
                validate_jwt_token(token)

    @pytest.mark.asyncio
    async def test_validate_malformed_token(self, mock_keycloak_env):
        """Test validation of malformed token."""
        with pytest.raises(Exception):
            validate_jwt_token("not-a-valid-token")

    @pytest.mark.asyncio
    async def test_validate_token_missing_sub(self, mock_keycloak_env):
        """Test validation of token missing sub claim."""
        # Create token without sub claim
        import jwt

        from tests.utils.keycloak_test_helpers import get_test_private_key

        claims = {
            "iss": "https://test-keycloak.com/realms/test-realm",
            "aud": "test-client",
            "exp": 9999999999,
            "iat": 1000000000,
        }
        token = jwt.encode(claims, get_test_private_key(), algorithm="RS256")

        with patch(
            "marketing_project.middleware.keycloak_auth.get_keycloak_public_key_from_env",
            return_value=mock_keycloak_public_key(),
        ):
            with pytest.raises(Exception):
                validate_jwt_token(token)


class TestIsPublicPath:
    """Tests for public path checking."""

    def test_health_endpoint_is_public(self):
        """Test that health endpoint is public."""
        assert is_public_path("/api/v1/health") is True

    def test_ready_endpoint_is_public(self):
        """Test that ready endpoint is public."""
        assert is_public_path("/api/v1/ready") is True

    def test_docs_endpoint_is_public(self):
        """Test that docs endpoint is public."""
        assert is_public_path("/docs") is True

    def test_protected_endpoint_is_not_public(self):
        """Test that protected endpoints are not public."""
        assert is_public_path("/api/v1/analyze") is False
        assert is_public_path("/api/v1/pipeline") is False
        assert is_public_path("/api/v1/jobs") is False


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_get_current_user_with_valid_token(self, mock_keycloak_env):
        """Test getting current user with valid token."""
        token = create_test_jwt(
            user_id="test-user-123",
            email="test@example.com",
            username="testuser",
            roles=["user"],
        )

        request = MagicMock(spec=Request)
        request.headers = {"Authorization": f"Bearer {token}"}

        with patch(
            "marketing_project.middleware.keycloak_auth.get_keycloak_public_key_from_env",
            return_value=mock_keycloak_public_key(),
        ):
            user_context = await get_current_user(request)
            assert user_context.user_id == "test-user-123"
            assert user_context.email == "test@example.com"
            assert user_context.username == "testuser"
            assert "user" in user_context.roles

    @pytest.mark.asyncio
    async def test_get_current_user_missing_token(self, mock_keycloak_env):
        """Test getting current user without token."""
        request = MagicMock(spec=Request)
        request.headers = {}

        with pytest.raises(Exception):  # Should raise HTTPException
            await get_current_user(request)

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, mock_keycloak_env):
        """Test getting current user with invalid token."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer invalid-token"}

        with pytest.raises(Exception):  # Should raise HTTPException
            await get_current_user(request)
