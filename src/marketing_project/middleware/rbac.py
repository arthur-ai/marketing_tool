"""
Role-Based Access Control (RBAC) middleware for FastAPI.

This module provides FastAPI dependencies for enforcing role-based access control.
"""

import logging
from typing import List, Optional

from fastapi import Depends, HTTPException, status

from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.models.user_context import UserContext

logger = logging.getLogger("marketing_project.middleware.rbac")


def require_roles(required_roles: List[str], require_all: bool = False):
    """
    Create a FastAPI dependency that requires specific roles.

    Args:
        required_roles: List of role names that are required
        require_all: If True, user must have all roles. If False, user needs any role.

    Returns:
        FastAPI dependency function

    Example:
        @router.get("/admin")
        async def admin_endpoint(user: UserContext = Depends(require_roles(["admin"]))):
            ...
    """
    if not required_roles:
        raise ValueError("required_roles cannot be empty")

    async def role_checker(
        user: UserContext = Depends(get_current_user),
    ) -> UserContext:
        """
        Check if user has required roles.

        Args:
            user: Current authenticated user

        Returns:
            UserContext if user has required roles

        Raises:
            HTTPException: If user doesn't have required roles
        """
        if require_all:
            has_access = user.has_all_roles(required_roles)
            error_detail = f"User must have all of the following roles: {', '.join(required_roles)}"
        else:
            has_access = user.has_any_role(required_roles)
            error_detail = f"User must have at least one of the following roles: {', '.join(required_roles)}"

        if not has_access:
            logger.warning(
                f"User {user.user_id} attempted to access endpoint requiring roles {required_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=error_detail,
            )

        return user

    return role_checker


def require_any_role(*roles: str):
    """
    Create a FastAPI dependency that requires any of the specified roles.

    This is a convenience function for require_roles with require_all=False.

    Args:
        *roles: Variable number of role names

    Returns:
        FastAPI dependency function

    Example:
        @router.get("/editor")
        async def editor_endpoint(user: UserContext = Depends(require_any_role("editor", "admin"))):
            ...
    """
    return require_roles(list(roles), require_all=False)


def require_all_roles(*roles: str):
    """
    Create a FastAPI dependency that requires all of the specified roles.

    This is a convenience function for require_roles with require_all=True.

    Args:
        *roles: Variable number of role names

    Returns:
        FastAPI dependency function

    Example:
        @router.get("/super-admin")
        async def super_admin_endpoint(
            user: UserContext = Depends(require_all_roles("admin", "superuser"))
        ):
            ...
    """
    return require_roles(list(roles), require_all=True)


def optional_user(
    user: Optional[UserContext] = Depends(get_current_user),
) -> Optional[UserContext]:
    """
    Optional authentication dependency.

    Returns user context if authenticated, None otherwise.
    Useful for endpoints that work for both authenticated and unauthenticated users.

    Args:
        user: Current user (may be None if not authenticated)

    Returns:
        UserContext if authenticated, None otherwise
    """
    return user
