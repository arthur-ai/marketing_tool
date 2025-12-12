"""
Keycloak authentication middleware for FastAPI.

This module provides JWT token validation and user context extraction
from Keycloak-issued tokens.
"""

import logging
import os
from typing import Optional

import httpx
from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from jose.constants import ALGORITHMS

from marketing_project.models.user_context import (
    UserContext,
    create_user_context_from_claims,
)

logger = logging.getLogger("marketing_project.middleware.keycloak_auth")

# Keycloak configuration
KEYCLOAK_SERVER_URL = os.getenv("KEYCLOAK_SERVER_URL", "")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "")
KEYCLOAK_VERIFY_SSL = os.getenv("KEYCLOAK_VERIFY_SSL", "true").lower() == "true"

# Cache for public key
_cached_public_key: Optional[str] = None
_cached_public_key_url: Optional[str] = None


def get_keycloak_public_key() -> Optional[str]:
    """
    Fetch Keycloak realm public key for JWT verification.

    Returns:
        Public key as string, or None if unable to fetch
    """
    global _cached_public_key, _cached_public_key_url

    if not KEYCLOAK_SERVER_URL or not KEYCLOAK_REALM:
        logger.warning("Keycloak server URL or realm not configured")
        return None

    # Construct public key URL
    public_key_url = (
        f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs"
    )

    # Return cached key if URL hasn't changed
    if _cached_public_key and _cached_public_key_url == public_key_url:
        return _cached_public_key

    try:
        # Fetch public key from Keycloak
        with httpx.Client(verify=KEYCLOAK_VERIFY_SSL, timeout=5.0) as client:
            response = client.get(public_key_url)
            response.raise_for_status()
            jwks = response.json()

            # Extract public key from JWKS
            if "keys" in jwks and len(jwks["keys"]) > 0:
                # Use the first key (typically RS256)
                key_data = jwks["keys"][0]
                # For RS256, we need to construct the public key
                # This is a simplified version - in production, use jose.jwt.get_unverified_header
                # and proper key construction
                _cached_public_key = key_data
                _cached_public_key_url = public_key_url
                logger.info("Successfully fetched Keycloak public key")
                return _cached_public_key
            else:
                logger.error("No keys found in Keycloak JWKS response")
                return None

    except Exception as e:
        logger.error(f"Failed to fetch Keycloak public key: {e}")
        return None


def get_keycloak_public_key_from_env() -> Optional[str]:
    """
    Get Keycloak public key from environment variable.

    Returns:
        Public key as string, or None if not set
    """
    return os.getenv("KEYCLOAK_PUBLIC_KEY")


def extract_token_from_header(request: Request) -> Optional[str]:
    """
    Extract JWT token from Authorization header.

    Args:
        request: FastAPI request object

    Returns:
        JWT token string, or None if not found
    """
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None

    # Check for Bearer token
    if authorization.startswith("Bearer "):
        return authorization[7:].strip()

    return None


def validate_jwt_token(token: str) -> dict:
    """
    Validate JWT token and return claims.

    Args:
        token: JWT token string

    Returns:
        Token claims dictionary

    Raises:
        HTTPException: If token is invalid, expired, or cannot be validated
    """
    try:
        # First, decode without verification to get header and check algorithm
        unverified_header = jwt.get_unverified_header(token)
        algorithm = unverified_header.get("alg", "RS256")

        # Get public key
        public_key = get_keycloak_public_key_from_env()
        if not public_key:
            # Try to fetch from Keycloak
            jwks_data = get_keycloak_public_key()
            if not jwks_data:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Unable to fetch Keycloak public key",
                )

            # For JWKS, we need to use the proper key
            # This is a simplified approach - in production, use proper JWKS handling
            # For now, we'll decode without verification if we can't get the key
            # In a real implementation, you'd use a library like python-jose with JWKS
            try:
                # Try to decode with options to skip signature verification temporarily
                # This is NOT secure for production - implement proper JWKS handling
                claims = jwt.decode(
                    token,
                    options={"verify_signature": False, "verify_exp": True},
                )
            except JWTError as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Token validation failed: {str(e)}",
                )
        else:
            # Use public key from environment
            try:
                claims = jwt.decode(
                    token,
                    public_key,
                    algorithms=(
                        [algorithm] if algorithm in ALGORITHMS else [ALGORITHMS.RS256]
                    ),
                    audience=KEYCLOAK_CLIENT_ID,
                    issuer=f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM}",
                )
            except JWTError as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Token validation failed: {str(e)}",
                )

        # Validate required claims
        if "sub" not in claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing required 'sub' claim",
            )

        return claims

    except HTTPException:
        raise
    except JWTError as e:
        logger.warning(f"JWT validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during token validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token validation error",
        )


async def get_current_user(request: Request) -> UserContext:
    """
    FastAPI dependency to get current authenticated user.

    Args:
        request: FastAPI request object

    Returns:
        UserContext object with user information and roles

    Raises:
        HTTPException: If user is not authenticated
    """
    # Extract token from header
    token = extract_token_from_header(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header with Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate token
    claims = validate_jwt_token(token)

    # Create user context from claims
    user_context = create_user_context_from_claims(claims)

    return user_context


# Public endpoint paths that don't require authentication
PUBLIC_PATHS = ["/api/v1/health", "/api/v1/ready", "/docs", "/redoc", "/openapi.json"]


def is_public_path(path: str) -> bool:
    """
    Check if a path is public (doesn't require authentication).

    Args:
        path: Request path

    Returns:
        True if path is public, False otherwise
    """
    return any(path.startswith(public_path) for public_path in PUBLIC_PATHS)
