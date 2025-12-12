"""
Test helpers for Keycloak authentication tests.

This module provides utilities for creating test JWT tokens and user contexts.
"""

import time
from typing import Dict, List, Optional

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from marketing_project.models.user_context import UserContext


# Generate a test RSA key pair for JWT signing
def generate_test_key_pair():
    """Generate a test RSA key pair for JWT signing."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    # Serialize keys
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return private_pem.decode(), public_pem.decode()


# Test key pair (generated once)
_TEST_PRIVATE_KEY, _TEST_PUBLIC_KEY = generate_test_key_pair()


def create_test_jwt(
    user_id: str = "test-user-123",
    email: Optional[str] = "test@example.com",
    username: Optional[str] = "testuser",
    roles: Optional[List[str]] = None,
    realm_roles: Optional[List[str]] = None,
    client_roles: Optional[List[str]] = None,
    issuer: str = "https://test-keycloak.com/realms/test-realm",
    audience: str = "test-client",
    expires_in: int = 3600,
    private_key: Optional[str] = None,
) -> str:
    """
    Create a test JWT token with the specified claims.

    Args:
        user_id: User ID (sub claim)
        email: User email
        username: Username (preferred_username)
        roles: List of all roles
        realm_roles: Realm-level roles
        client_roles: Client-level roles
        issuer: Token issuer
        audience: Token audience
        expires_in: Token expiration time in seconds
        private_key: Private key for signing (uses test key if not provided)

    Returns:
        JWT token string
    """
    now = int(time.time())
    roles = roles or []
    realm_roles = realm_roles or []
    client_roles = client_roles or []

    # Build claims
    claims = {
        "sub": user_id,
        "iss": issuer,
        "aud": audience,
        "exp": now + expires_in,
        "iat": now,
        "nbf": now,
    }

    if email:
        claims["email"] = email
        claims["email_verified"] = True

    if username:
        claims["preferred_username"] = username
        claims["name"] = username

    # Add realm access
    if realm_roles:
        claims["realm_access"] = {"roles": realm_roles}

    # Add resource access (client roles)
    if client_roles:
        claims["resource_access"] = {audience: {"roles": client_roles}}

    # Add direct roles claim if provided
    if roles:
        claims["roles"] = roles

    # Sign token
    key = private_key.encode() if private_key else _TEST_PRIVATE_KEY.encode()
    token = jwt.encode(claims, key, algorithm="RS256")

    return token


def create_expired_jwt(
    user_id: str = "test-user-123",
    issuer: str = "https://test-keycloak.com/realms/test-realm",
    audience: str = "test-client",
    private_key: Optional[str] = None,
) -> str:
    """
    Create an expired test JWT token.

    Args:
        user_id: User ID
        issuer: Token issuer
        audience: Token audience
        private_key: Private key for signing

    Returns:
        Expired JWT token string
    """
    now = int(time.time())
    claims = {
        "sub": user_id,
        "iss": issuer,
        "aud": audience,
        "exp": now - 3600,  # Expired 1 hour ago
        "iat": now - 7200,  # Issued 2 hours ago
        "nbf": now - 7200,
    }

    key = private_key.encode() if private_key else _TEST_PRIVATE_KEY.encode()
    token = jwt.encode(claims, key, algorithm="RS256")

    return token


def create_user_context(
    user_id: str = "test-user-123",
    email: Optional[str] = "test@example.com",
    username: Optional[str] = "testuser",
    roles: Optional[List[str]] = None,
    realm_roles: Optional[List[str]] = None,
    client_roles: Optional[List[str]] = None,
) -> UserContext:
    """
    Create a test UserContext object.

    Args:
        user_id: User ID
        email: User email
        username: Username
        roles: List of all roles
        realm_roles: Realm-level roles
        client_roles: Client-level roles

    Returns:
        UserContext object
    """
    return UserContext(
        user_id=user_id,
        email=email,
        username=username,
        roles=roles or [],
        realm_roles=realm_roles or [],
        client_roles=client_roles or [],
    )


def mock_keycloak_public_key() -> str:
    """
    Get the test public key for JWT verification.

    Returns:
        Public key as PEM string
    """
    return _TEST_PUBLIC_KEY


def get_test_private_key() -> str:
    """
    Get the test private key for JWT signing.

    Returns:
        Private key as PEM string
    """
    return _TEST_PRIVATE_KEY
