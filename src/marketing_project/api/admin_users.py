"""
Admin user management endpoints.

Allows admins to list Keycloak users and manage their realm roles (e.g., promote
a user to the 'admin' role or revoke it) without accessing the Keycloak console
directly.

Uses the Keycloak Admin REST API via service-account client credentials.
The API client must have the 'realm-management' → 'manage-users' and
'manage-realm' service-account roles assigned in Keycloak.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..middleware.rbac import require_roles
from ..models.user_context import UserContext

logger = logging.getLogger("marketing_project.api.admin_users")

router = APIRouter()

# Keycloak configuration (same values used by keycloak_auth.py)
_KC_SERVER = os.getenv("KEYCLOAK_SERVER_URL", "").rstrip("/")
_KC_REALM = os.getenv("KEYCLOAK_REALM", "")
_KC_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "")
_KC_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "")
_KC_VERIFY_SSL = os.getenv("KEYCLOAK_VERIFY_SSL", "true").lower() == "true"


# ── Keycloak admin token helper ───────────────────────────────────────────────


async def _get_admin_token() -> str:
    """
    Obtain a short-lived access token using the service-account of the
    configured client.  The client must have the realm-management roles:
      - manage-users
      - view-users
      - manage-realm (for role assignment)
    """
    if not all([_KC_SERVER, _KC_REALM, _KC_CLIENT_ID, _KC_CLIENT_SECRET]):
        raise HTTPException(
            status_code=503,
            detail="Keycloak admin integration not configured (missing env vars)",
        )

    token_url = f"{_KC_SERVER}/realms/{_KC_REALM}/protocol/openid-connect/token"
    async with httpx.AsyncClient(verify=_KC_VERIFY_SSL, timeout=10) as client:
        resp = await client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": _KC_CLIENT_ID,
                "client_secret": _KC_CLIENT_SECRET,
            },
        )
    if resp.status_code != 200:
        logger.error(f"Failed to get Keycloak admin token: {resp.text}")
        raise HTTPException(
            status_code=503,
            detail="Failed to authenticate with Keycloak admin API",
        )
    return resp.json()["access_token"]


def _admin_base() -> str:
    return f"{_KC_SERVER}/admin/realms/{_KC_REALM}"


# ── Pydantic models ───────────────────────────────────────────────────────────


class UserSummary(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    enabled: bool
    roles: List[str] = []


class SetRolesRequest(BaseModel):
    roles: List[str]
    """
    Full list of realm roles to assign to the user.
    Any roles currently assigned but absent from this list will be revoked.
    Supported values: "admin", "user".
    """


# ── Helpers ───────────────────────────────────────────────────────────────────

MANAGED_ROLES = {"admin", "user"}


async def _get_realm_role(token: str, role_name: str) -> Dict[str, Any]:
    """Fetch role representation by name; raises 404 if not found."""
    url = f"{_admin_base()}/roles/{role_name}"
    async with httpx.AsyncClient(verify=_KC_VERIFY_SSL, timeout=10) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
    if resp.status_code == 404:
        raise HTTPException(
            status_code=400, detail=f"Role '{role_name}' not found in realm"
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Keycloak error: {resp.text}")
    return resp.json()


async def _get_user_realm_roles(token: str, user_id: str) -> List[Dict[str, Any]]:
    url = f"{_admin_base()}/users/{user_id}/role-mappings/realm"
    async with httpx.AsyncClient(verify=_KC_VERIFY_SSL, timeout=10) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Keycloak error: {resp.text}")
    return resp.json()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("", response_model=List[UserSummary])
async def list_users(
    search: Optional[str] = Query(None, description="Search by username/email"),
    first: int = Query(0, ge=0, description="Pagination offset"),
    max: int = Query(50, ge=1, le=200, description="Max results"),
    admin: UserContext = Depends(require_roles(["admin"])),
):
    """
    [Admin] List all users in the Keycloak realm.

    Includes each user's currently assigned realm roles.
    """
    try:
        token = await _get_admin_token()
        params: Dict[str, Any] = {"first": first, "max": max}
        if search:
            params["search"] = search

        async with httpx.AsyncClient(verify=_KC_VERIFY_SSL, timeout=15) as client:
            resp = await client.get(
                f"{_admin_base()}/users",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )

        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Keycloak error: {resp.text}")

        users = resp.json()
        result: List[UserSummary] = []

        for u in users:
            # Fetch realm roles for each user
            try:
                role_objs = await _get_user_realm_roles(token, u["id"])
                roles = [r["name"] for r in role_objs if r["name"] in MANAGED_ROLES]
            except Exception:
                roles = []

            result.append(
                UserSummary(
                    id=u["id"],
                    username=u.get("username", ""),
                    email=u.get("email"),
                    first_name=u.get("firstName"),
                    last_name=u.get("lastName"),
                    enabled=u.get("enabled", True),
                    roles=roles,
                )
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}", response_model=UserSummary)
async def get_user(
    user_id: str,
    admin: UserContext = Depends(require_roles(["admin"])),
):
    """[Admin] Get a single Keycloak user by ID, including their realm roles."""
    try:
        token = await _get_admin_token()

        async with httpx.AsyncClient(verify=_KC_VERIFY_SSL, timeout=10) as client:
            resp = await client.get(
                f"{_admin_base()}/users/{user_id}",
                headers={"Authorization": f"Bearer {token}"},
            )

        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Keycloak error: {resp.text}")

        u = resp.json()
        role_objs = await _get_user_realm_roles(token, user_id)
        roles = [r["name"] for r in role_objs if r["name"] in MANAGED_ROLES]

        return UserSummary(
            id=u["id"],
            username=u.get("username", ""),
            email=u.get("email"),
            first_name=u.get("firstName"),
            last_name=u.get("lastName"),
            enabled=u.get("enabled", True),
            roles=roles,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/roles")
async def get_user_roles(
    user_id: str,
    admin: UserContext = Depends(require_roles(["admin"])),
):
    """[Admin] Get the realm roles assigned to a Keycloak user."""
    try:
        token = await _get_admin_token()
        role_objs = await _get_user_realm_roles(token, user_id)
        roles = [r["name"] for r in role_objs if r["name"] in MANAGED_ROLES]
        return {"user_id": user_id, "roles": roles}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get roles for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{user_id}/roles", response_model=UserSummary)
async def set_user_roles(
    user_id: str,
    body: SetRolesRequest,
    admin: UserContext = Depends(require_roles(["admin"])),
):
    """
    [Admin] Replace the managed realm roles for a user.

    Roles not in `{"admin", "user"}` are ignored.
    The request body's `roles` list is treated as the desired final state:
    - Roles in the list that the user lacks → added.
    - Managed roles the user currently has but are absent from the list → removed.
    """
    try:
        # Validate requested roles
        desired = {r for r in body.roles if r in MANAGED_ROLES}
        unknown = set(body.roles) - MANAGED_ROLES
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown roles: {sorted(unknown)}. Allowed: {sorted(MANAGED_ROLES)}",
            )

        token = await _get_admin_token()

        # Current managed roles
        current_role_objs = await _get_user_realm_roles(token, user_id)
        current_managed = {
            r["name"]: r for r in current_role_objs if r["name"] in MANAGED_ROLES
        }

        to_add = desired - set(current_managed.keys())
        to_remove = set(current_managed.keys()) - desired

        # Add roles
        for role_name in to_add:
            role_rep = await _get_realm_role(token, role_name)
            async with httpx.AsyncClient(verify=_KC_VERIFY_SSL, timeout=10) as client:
                resp = await client.post(
                    f"{_admin_base()}/users/{user_id}/role-mappings/realm",
                    headers={"Authorization": f"Bearer {token}"},
                    json=[role_rep],
                )
            if resp.status_code not in (200, 204):
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to add role '{role_name}': {resp.text}",
                )

        # Remove roles
        for role_name in to_remove:
            role_rep = current_managed[role_name]
            async with httpx.AsyncClient(verify=_KC_VERIFY_SSL, timeout=10) as client:
                resp = await client.request(
                    "DELETE",
                    f"{_admin_base()}/users/{user_id}/role-mappings/realm",
                    headers={"Authorization": f"Bearer {token}"},
                    json=[role_rep],
                )
            if resp.status_code not in (200, 204):
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to remove role '{role_name}': {resp.text}",
                )

        logger.info(
            f"Admin {admin.user_id} updated roles for {user_id}: "
            f"added={to_add}, removed={to_remove}"
        )

        # Return updated user
        return await get_user(user_id, admin)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set roles for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
