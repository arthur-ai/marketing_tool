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
from jose import JWTError, jwk, jwt
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

# Cache for JWKS
_cached_jwks: Optional[dict] = None
_cached_jwks_url: Optional[str] = None


def get_keycloak_jwks() -> Optional[dict]:
    """
    Fetch Keycloak realm JWKS (JSON Web Key Set) for JWT verification.

    Returns:
        JWKS dictionary, or None if unable to fetch
    """
    global _cached_jwks, _cached_jwks_url

    if not KEYCLOAK_SERVER_URL or not KEYCLOAK_REALM:
        logger.warning("Keycloak server URL or realm not configured")
        return None

    # Construct JWKS URL
    jwks_url = (
        f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs"
    )

    # Return cached JWKS if URL hasn't changed
    if _cached_jwks and _cached_jwks_url == jwks_url:
        return _cached_jwks

    try:
        # Fetch JWKS from Keycloak
        with httpx.Client(verify=KEYCLOAK_VERIFY_SSL, timeout=5.0) as client:
            response = client.get(jwks_url)
            response.raise_for_status()
            jwks = response.json()

            if "keys" in jwks and len(jwks["keys"]) > 0:
                _cached_jwks = jwks
                _cached_jwks_url = jwks_url
                logger.info("Successfully fetched Keycloak JWKS")
                return _cached_jwks
            else:
                logger.error("No keys found in Keycloak JWKS response")
                return None

    except Exception as e:
        logger.error(f"Failed to fetch Keycloak JWKS: {e}")
        return None


def get_public_key_from_jwks(token: str, jwks: dict):
    """
    Get the public key from JWKS that matches the token's key ID (kid).

    Args:
        token: JWT token string
        jwks: JWKS dictionary from Keycloak

    Returns:
        Public key object for JWT verification, or None if not found
    """
    try:
        # Get the key ID from token header
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        if not kid:
            logger.warning("Token missing 'kid' in header")
            # Fallback to first key if no kid
            if "keys" in jwks and len(jwks["keys"]) > 0:
                key_data = jwks["keys"][0]
                return jwk.construct(key_data)
            return None

        # Find matching key in JWKS
        for key_data in jwks.get("keys", []):
            if key_data.get("kid") == kid:
                return jwk.construct(key_data)

        logger.warning(f"Key with kid '{kid}' not found in JWKS")
        # Fallback to first key if kid doesn't match
        if "keys" in jwks and len(jwks["keys"]) > 0:
            key_data = jwks["keys"][0]
            return jwk.construct(key_data)

        return None

    except Exception as e:
        logger.error(f"Error constructing public key from JWKS: {e}")
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

        # Get public key from environment or JWKS
        public_key_pem = get_keycloak_public_key_from_env()
        public_key = None

        if public_key_pem:
            # Use public key from environment variable (PEM format)
            public_key = public_key_pem
        else:
            # Fetch JWKS and get the matching public key
            jwks = get_keycloak_jwks()
            if not jwks:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Unable to fetch Keycloak JWKS",
                )

            # Get the public key that matches the token's kid
            public_key_obj = get_public_key_from_jwks(token, jwks)
            if not public_key_obj:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Unable to construct public key from JWKS",
                )
            # Convert jwk object to format expected by jwt.decode
            public_key = public_key_obj

        # Validate token with public key
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
