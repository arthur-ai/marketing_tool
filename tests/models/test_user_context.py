"""
Tests for user context model.
"""

import pytest

from marketing_project.models.user_context import (
    UserContext,
    create_user_context_from_claims,
    extract_roles_from_claims,
)


class TestUserContext:
    """Tests for UserContext model."""

    def test_create_user_context(self):
        """Test creating a user context."""
        user = UserContext(
            user_id="test-123",
            email="test@example.com",
            username="testuser",
            roles=["admin", "user"],
            realm_roles=["admin"],
            client_roles=["user"],
        )

        assert user.user_id == "test-123"
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert "admin" in user.roles
        assert "user" in user.roles

    def test_has_role(self):
        """Test has_role method."""
        user = UserContext(
            user_id="test-123",
            roles=["admin", "editor"],
        )

        assert user.has_role("admin") is True
        assert user.has_role("editor") is True
        assert user.has_role("user") is False

    def test_has_any_role(self):
        """Test has_any_role method."""
        user = UserContext(
            user_id="test-123",
            roles=["admin", "editor"],
        )

        assert user.has_any_role(["admin", "user"]) is True
        assert user.has_any_role(["user", "guest"]) is False

    def test_has_all_roles(self):
        """Test has_all_roles method."""
        user = UserContext(
            user_id="test-123",
            roles=["admin", "editor", "user"],
        )

        assert user.has_all_roles(["admin", "editor"]) is True
        assert user.has_all_roles(["admin", "guest"]) is False


class TestExtractRolesFromClaims:
    """Tests for extract_roles_from_claims function."""

    def test_extract_realm_roles(self):
        """Test extracting realm roles."""
        claims = {"realm_access": {"roles": ["admin", "user"]}}

        roles = extract_roles_from_claims(claims)
        assert "admin" in roles
        assert "user" in roles

    def test_extract_client_roles(self):
        """Test extracting client roles."""
        claims = {"resource_access": {"test-client": {"roles": ["editor", "viewer"]}}}

        roles = extract_roles_from_claims(claims)
        assert "editor" in roles
        assert "viewer" in roles

    def test_extract_direct_roles(self):
        """Test extracting direct roles claim."""
        claims = {"roles": ["admin", "user"]}

        roles = extract_roles_from_claims(claims)
        assert "admin" in roles
        assert "user" in roles

    def test_extract_all_role_types(self):
        """Test extracting all role types."""
        claims = {
            "realm_access": {"roles": ["admin"]},
            "resource_access": {"test-client": {"roles": ["editor"]}},
            "roles": ["user"],
        }

        roles = extract_roles_from_claims(claims)
        assert "admin" in roles
        assert "editor" in roles
        assert "user" in roles
        assert len(roles) == 3

    def test_extract_roles_empty_claims(self):
        """Test extracting roles from empty claims."""
        claims = {}
        roles = extract_roles_from_claims(claims)
        assert roles == []


class TestCreateUserContextFromClaims:
    """Tests for create_user_context_from_claims function."""

    def test_create_from_claims(self):
        """Test creating user context from claims."""
        claims = {
            "sub": "test-123",
            "email": "test@example.com",
            "preferred_username": "testuser",
            "realm_access": {"roles": ["admin"]},
            "resource_access": {"test-client": {"roles": ["user"]}},
        }

        user = create_user_context_from_claims(claims)
        assert user.user_id == "test-123"
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert "admin" in user.roles
        assert "user" in user.roles
        assert "admin" in user.realm_roles
        assert "user" in user.client_roles

    def test_create_from_minimal_claims(self):
        """Test creating user context from minimal claims."""
        claims = {"sub": "test-123"}

        user = create_user_context_from_claims(claims)
        assert user.user_id == "test-123"
        assert user.email is None
        assert user.username is None
        assert user.roles == []
