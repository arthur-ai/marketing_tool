"""
User context model for Keycloak authentication.

This module defines the user context extracted from JWT tokens and provides
helper functions for role extraction and user information.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class UserContext(BaseModel):
    """User context extracted from JWT token."""

    user_id: str = Field(..., description="User ID (sub claim from JWT)")
    email: Optional[str] = Field(None, description="User email address")
    username: Optional[str] = Field(None, description="Username (preferred_username)")
    roles: List[str] = Field(default_factory=list, description="User roles")
    realm_roles: List[str] = Field(
        default_factory=list, description="Realm-level roles"
    )
    client_roles: List[str] = Field(
        default_factory=list, description="Client-level roles"
    )

    def has_role(self, role: str) -> bool:
        """
        Check if user has a specific role.

        Args:
            role: Role name to check

        Returns:
            True if user has the role, False otherwise
        """
        return role in self.roles

    def has_any_role(self, roles: List[str]) -> bool:
        """
        Check if user has any of the specified roles.

        Args:
            roles: List of role names to check

        Returns:
            True if user has any of the roles, False otherwise
        """
        return any(role in self.roles for role in roles)

    def has_all_roles(self, roles: List[str]) -> bool:
        """
        Check if user has all of the specified roles.

        Args:
            roles: List of role names to check

        Returns:
            True if user has all of the roles, False otherwise
        """
        return all(role in self.roles for role in roles)


def extract_roles_from_claims(claims: dict) -> List[str]:
    """
    Extract roles from JWT token claims.

    Keycloak can provide roles in different claim formats:
    - realm_access.roles: Realm-level roles
    - resource_access.{client_id}.roles: Client-level roles
    - roles: Direct roles claim (if configured)

    Args:
        claims: JWT token claims dictionary

    Returns:
        List of role names
    """
    roles: List[str] = []

    # Extract realm roles
    if "realm_access" in claims and isinstance(claims["realm_access"], dict):
        realm_roles = claims["realm_access"].get("roles", [])
        if isinstance(realm_roles, list):
            roles.extend(realm_roles)

    # Extract client roles (check all clients)
    if "resource_access" in claims and isinstance(claims["resource_access"], dict):
        for client_id, client_data in claims["resource_access"].items():
            if isinstance(client_data, dict):
                client_roles = client_data.get("roles", [])
                if isinstance(client_roles, list):
                    roles.extend(client_roles)

    # Extract direct roles claim (if present)
    if "roles" in claims and isinstance(claims["roles"], list):
        roles.extend(claims["roles"])

    # Remove duplicates and return
    return list(set(roles))


def create_user_context_from_claims(claims: dict) -> UserContext:
    """
    Create UserContext from JWT token claims.

    Args:
        claims: JWT token claims dictionary

    Returns:
        UserContext object with extracted user information
    """
    # Extract user ID (sub claim)
    user_id = claims.get("sub", "")

    # Extract email
    email = claims.get("email")

    # Extract username
    username = (
        claims.get("preferred_username") or claims.get("username") or claims.get("name")
    )

    # Extract roles
    roles = extract_roles_from_claims(claims)

    # Separate realm and client roles
    realm_roles: List[str] = []
    client_roles: List[str] = []

    if "realm_access" in claims and isinstance(claims["realm_access"], dict):
        realm_roles = claims["realm_access"].get("roles", [])
        if not isinstance(realm_roles, list):
            realm_roles = []

    if "resource_access" in claims and isinstance(claims["resource_access"], dict):
        for client_id, client_data in claims["resource_access"].items():
            if isinstance(client_data, dict):
                client_role_list = client_data.get("roles", [])
                if isinstance(client_role_list, list):
                    client_roles.extend(client_role_list)

    return UserContext(
        user_id=user_id,
        email=email,
        username=username,
        roles=roles,
        realm_roles=realm_roles,
        client_roles=client_roles,
    )
