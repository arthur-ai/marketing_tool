"""
Role-Based Access Control (RBAC) middleware for FastAPI.

This module provides FastAPI dependencies for enforcing role-based access control,
plus ownership-check helpers for jobs and approvals.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from fastapi import Depends, HTTPException, status

from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.models.user_context import UserContext

if TYPE_CHECKING:
    from marketing_project.models.approval_models import ApprovalRequest
    from marketing_project.services.job_manager import Job

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


async def verify_job_ownership(
    job_id: str,
    user: UserContext,
    job_manager,
) -> "Job":
    """
    Verify the current user owns the given job, then return it.

    Admins bypass the ownership check and can access any job.

    Args:
        job_id: Job ID to look up
        user: Authenticated user making the request
        job_manager: JobManager instance

    Returns:
        Job object if access is allowed

    Raises:
        HTTPException 404: If job not found
        HTTPException 403: If user doesn't own the job (non-admins only)
    """
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    if not user.has_role("admin") and job.user_id != user.user_id:
        logger.warning(
            f"User {user.user_id} attempted to access job {job_id} owned by {job.user_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not own this job",
        )
    return job


async def verify_approval_ownership(
    approval_id: str,
    user: UserContext,
    approval_manager,
    job_manager,
) -> "ApprovalRequest":
    """
    Verify the current user owns the job associated with the given approval.

    Admins bypass the ownership check.

    Args:
        approval_id: Approval ID to look up
        user: Authenticated user making the request
        approval_manager: ApprovalManager instance
        job_manager: JobManager instance

    Returns:
        ApprovalRequest if access is allowed

    Raises:
        HTTPException 404: If approval not found
        HTTPException 403: If user doesn't own the associated job (non-admins only)
    """
    approval = await approval_manager.get_approval(approval_id)
    if not approval:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Approval {approval_id} not found",
        )
    if not user.has_role("admin"):
        await verify_job_ownership(approval.job_id, user, job_manager)
    return approval
