"""
Tests for RBAC middleware.
"""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from marketing_project.middleware.rbac import (
    require_all_roles,
    require_any_role,
    require_roles,
)
from marketing_project.models.user_context import UserContext
from tests.utils.keycloak_test_helpers import create_user_context


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    app = FastAPI()
    return app


class TestRequireRoles:
    """Tests for require_roles dependency."""

    @pytest.mark.asyncio
    async def test_user_with_required_role(self):
        """Test that user with required role can access."""
        user = create_user_context(roles=["admin", "editor", "user"])
        role_checker = require_roles(["admin"])

        # Mock the dependency chain
        result = await role_checker(user=user)
        assert result == user

    @pytest.mark.asyncio
    async def test_user_without_required_role(self):
        """Test that user without required role gets 403."""
        user = create_user_context(roles=["user"])
        role_checker = require_roles(["admin"])

        with pytest.raises(Exception):  # Should raise HTTPException with 403
            await role_checker(user=user)

    @pytest.mark.asyncio
    async def test_user_with_any_required_role(self):
        """Test that user with any required role can access."""
        user = create_user_context(roles=["editor"])
        role_checker = require_roles(["admin", "editor"], require_all=False)

        result = await role_checker(user=user)
        assert result == user

    @pytest.mark.asyncio
    async def test_user_with_all_required_roles(self):
        """Test that user with all required roles can access."""
        user = create_user_context(roles=["admin", "editor"])
        role_checker = require_roles(["admin", "editor"], require_all=True)

        result = await role_checker(user=user)
        assert result == user

    @pytest.mark.asyncio
    async def test_user_without_all_required_roles(self):
        """Test that user without all required roles gets 403."""
        user = create_user_context(roles=["admin"])
        role_checker = require_roles(["admin", "editor"], require_all=True)

        with pytest.raises(Exception):  # Should raise HTTPException with 403
            await role_checker(user=user)

    def test_empty_roles_list_raises_error(self):
        """Test that empty roles list raises ValueError."""
        with pytest.raises(ValueError):
            require_roles([])


class TestRequireAnyRole:
    """Tests for require_any_role convenience function."""

    @pytest.mark.asyncio
    async def test_user_with_any_role(self):
        """Test that user with any of the roles can access."""
        user = create_user_context(roles=["editor"])
        role_checker = require_any_role("admin", "editor")

        result = await role_checker(user=user)
        assert result == user

    @pytest.mark.asyncio
    async def test_user_without_any_role(self):
        """Test that user without any of the roles gets 403."""
        user = create_user_context(roles=["user"])
        role_checker = require_any_role("admin", "editor")

        with pytest.raises(Exception):  # Should raise HTTPException with 403
            await role_checker(user=user)


class TestRequireAllRoles:
    """Tests for require_all_roles convenience function."""

    @pytest.mark.asyncio
    async def test_user_with_all_roles(self):
        """Test that user with all roles can access."""
        user = create_user_context(roles=["admin", "editor"])
        role_checker = require_all_roles("admin", "editor")

        result = await role_checker(user=user)
        assert result == user

    @pytest.mark.asyncio
    async def test_user_without_all_roles(self):
        """Test that user without all roles gets 403."""
        user = create_user_context(roles=["admin"])
        role_checker = require_all_roles("admin", "editor")

        with pytest.raises(Exception):  # Should raise HTTPException with 403
            await role_checker(user=user)
