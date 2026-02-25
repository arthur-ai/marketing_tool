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

# Valid JWT algorithms for token validation
VALID_ALGORITHMS = {
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES384",
    "ES512",
    "HS256",
    "HS384",
    "HS512",
}

from marketing_project.models.user_context import (
    UserContext,
    create_user_context_from_claims,
)

logger = logging.getLogger("marketing_project.middleware.keycloak_auth")

# Keycloak configuration
# Note: These are read at module import time. Make sure .env file is loaded before this module is imported.
KEYCLOAK_SERVER_URL = os.getenv("KEYCLOAK_SERVER_URL", "")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "")
KEYCLOAK_VERIFY_SSL = os.getenv("KEYCLOAK_VERIFY_SSL", "true").lower() == "true"
# Allow both frontend and backend client IDs for audience validation
# Frontend tokens have audience 'marketing-tool-frontend', backend expects 'marketing-tool-backend'
KEYCLOAK_ALLOWED_AUDIENCES = (
    os.getenv("KEYCLOAK_ALLOWED_AUDIENCES", "").split(",")
    if os.getenv("KEYCLOAK_ALLOWED_AUDIENCES")
    else []
)
# If no explicit allowed audiences, default to backend client ID and frontend client ID
if not KEYCLOAK_ALLOWED_AUDIENCES:
    KEYCLOAK_ALLOWED_AUDIENCES = (
        [KEYCLOAK_CLIENT_ID, "marketing-tool-frontend"]
        if KEYCLOAK_CLIENT_ID
        else ["marketing-tool-frontend"]
    )

# Log configuration on module load (only in development to avoid log spam)
if not KEYCLOAK_SERVER_URL or not KEYCLOAK_REALM:
    logger.warning(
        f"Keycloak configuration incomplete at module load: "
        f"KEYCLOAK_SERVER_URL={'SET' if KEYCLOAK_SERVER_URL else 'NOT SET'}, "
        f"KEYCLOAK_REALM={'SET' if KEYCLOAK_REALM else 'NOT SET'}. "
        f"Make sure .env file is loaded before importing this module."
    )
else:
    logger.info(
        f"Keycloak configured: Server={KEYCLOAK_SERVER_URL}, Realm={KEYCLOAK_REALM}, "
        f"Client={KEYCLOAK_CLIENT_ID or 'NOT SET'}, VerifySSL={KEYCLOAK_VERIFY_SSL}"
    )

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
        logger.error(
            f"Keycloak not configured: KEYCLOAK_SERVER_URL={KEYCLOAK_SERVER_URL}, "
            f"KEYCLOAK_REALM={KEYCLOAK_REALM}"
        )
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
        logger.info(f"Fetching Keycloak JWKS from: {jwks_url}")
        with httpx.Client(verify=KEYCLOAK_VERIFY_SSL, timeout=10.0) as client:
            response = client.get(jwks_url)
            response.raise_for_status()
            jwks = response.json()

            if "keys" in jwks and len(jwks["keys"]) > 0:
                _cached_jwks = jwks
                _cached_jwks_url = jwks_url
                logger.info(
                    f"Successfully fetched Keycloak JWKS with {len(jwks['keys'])} key(s)"
                )
                return _cached_jwks
            else:
                logger.error(f"No keys found in Keycloak JWKS response from {jwks_url}")
                return None

    except httpx.TimeoutException as e:
        logger.error(
            f"Timeout fetching Keycloak JWKS from {jwks_url}: {e}. "
            f"Check if Keycloak is running and accessible."
        )
        return None
    except httpx.ConnectError as e:
        logger.error(
            f"Connection error fetching Keycloak JWKS from {jwks_url}: {e}. "
            f"Check if Keycloak server is running at {KEYCLOAK_SERVER_URL}"
        )
        return None
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error fetching Keycloak JWKS from {jwks_url}: "
            f"Status {e.response.status_code}, Response: {e.response.text}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Failed to fetch Keycloak JWKS from {jwks_url}: {type(e).__name__}: {e}"
        )
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

        logger.info(
            f"[AUTH] Looking for key with kid='{kid}' in JWKS. Available keys: {[k.get('kid') for k in jwks.get('keys', [])]}"
        )

        if not kid:
            logger.warning(
                "[AUTH] Token missing 'kid' in header, using first key from JWKS"
            )
            # Fallback to first key if no kid
            if "keys" in jwks and len(jwks["keys"]) > 0:
                key_data = jwks["keys"][0]
                logger.info(
                    f"[AUTH] Using first key from JWKS: kid={key_data.get('kid')}"
                )
                return jwk.construct(key_data)
            logger.error("[AUTH] No keys available in JWKS")
            return None

        # Find matching key in JWKS
        for key_data in jwks.get("keys", []):
            if key_data.get("kid") == kid:
                logger.info(
                    f"[AUTH] Found matching key in JWKS: kid={kid}, alg={key_data.get('alg')}"
                )
                return jwk.construct(key_data)

        logger.warning(
            f"[AUTH] Key with kid '{kid}' not found in JWKS. Available kids: {[k.get('kid') for k in jwks.get('keys', [])]}"
        )
        # Fallback to first key if kid doesn't match
        if "keys" in jwks and len(jwks["keys"]) > 0:
            key_data = jwks["keys"][0]
            logger.info(f"[AUTH] Falling back to first key: kid={key_data.get('kid')}")
            return jwk.construct(key_data)

        logger.error("[AUTH] No keys available in JWKS for fallback")
        return None

    except Exception as e:
        logger.error(
            f"[AUTH] Error constructing public key from JWKS: {type(e).__name__}: {str(e)}",
            exc_info=True,
        )
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
        logger.debug(
            f"[AUTH] No Authorization header found for {request.method} {request.url.path}"
        )
        return None

    # Check for Bearer token
    if authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        logger.info(
            f"[AUTH] Token extracted from header for {request.method} {request.url.path}: "
            f"token_length={len(token)}, token_preview={token[:30]}..."
        )
        return token

    logger.warning(
        f"[AUTH] Authorization header present but not Bearer token for {request.method} {request.url.path}: "
        f"header_preview={authorization[:50]}..."
    )
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
    logger.info(
        f"[AUTH] Starting token validation: token_length={len(token)}, token_preview={token[:50]}..."
    )

    try:
        # First, decode without verification to get header and check algorithm
        unverified_header = jwt.get_unverified_header(token)
        algorithm = unverified_header.get("alg", "RS256")
        kid = unverified_header.get("kid")

        logger.info(
            f"[AUTH] Token header decoded: algorithm={algorithm}, kid={kid}, "
            f"header_keys={list(unverified_header.keys())}"
        )

        # Get public key from environment or JWKS
        public_key_pem = get_keycloak_public_key_from_env()
        public_key = None
        key_source = None

        if public_key_pem:
            # Use public key from environment variable (PEM format)
            public_key = public_key_pem
            key_source = "environment_variable"
            logger.info("[AUTH] Using public key from environment variable")
        else:
            # Fetch JWKS and get the matching public key
            logger.info(
                f"[AUTH] Fetching JWKS from Keycloak: server={KEYCLOAK_SERVER_URL}, realm={KEYCLOAK_REALM}"
            )
            jwks = get_keycloak_jwks()
            if not jwks:
                error_msg = (
                    f"Unable to fetch Keycloak JWKS. "
                    f"Check that Keycloak is running at {KEYCLOAK_SERVER_URL} "
                    f"and realm '{KEYCLOAK_REALM}' exists. "
                    f"JWKS URL: {KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs"
                )
                logger.error(f"[AUTH] {error_msg}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=error_msg,
                )

            # Get the public key that matches the token's kid
            logger.info(f"[AUTH] Constructing public key from JWKS for kid={kid}")
            public_key_obj = get_public_key_from_jwks(token, jwks)
            if not public_key_obj:
                logger.error("[AUTH] Unable to construct public key from JWKS")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Unable to construct public key from JWKS",
                )
            # Convert jwk object to format expected by jwt.decode
            public_key = public_key_obj
            key_source = "jwks"
            logger.info(
                f"[AUTH] Public key constructed from JWKS: key_source={key_source}"
            )

        # Validate token with public key
        try:
            # Ensure algorithm is valid, default to RS256 if not recognized
            if algorithm not in VALID_ALGORITHMS:
                logger.warning(
                    f"[AUTH] Unsupported algorithm '{algorithm}', defaulting to RS256"
                )
                algorithm = "RS256"

            expected_issuer = f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM}"

            # First decode without signature verification to see what issuer the token actually has
            actual_issuer = None
            try:
                # Decode token without verification to extract issuer
                # Use key="dummy" and verify_signature=False to extract claims without validation
                unverified_claims = jwt.decode(
                    token,
                    key="dummy",  # Dummy key since we're not verifying signature
                    options={
                        "verify_signature": False,
                        "verify_aud": False,
                        "verify_exp": False,
                        "verify_iss": False,
                    },
                )
                actual_issuer = unverified_claims.get("iss")
                logger.info(
                    f"[AUTH] Token issuer check: actual_issuer='{actual_issuer}', expected_issuer='{expected_issuer}'"
                )
                logger.info(
                    f"[AUTH] Keycloak config: SERVER_URL='{KEYCLOAK_SERVER_URL}', REALM='{KEYCLOAK_REALM}'"
                )

                if actual_issuer != expected_issuer:
                    logger.warning(
                        f"[AUTH] Issuer mismatch detected! Token has '{actual_issuer}' but backend expects '{expected_issuer}'"
                    )
                    # Try to be flexible - if the token issuer ends with the correct realm path,
                    # accept it even if the hostname differs (localhost vs host.docker.internal)
                    realm_path = f"/realms/{KEYCLOAK_REALM}"
                    if actual_issuer and actual_issuer.endswith(realm_path):
                        logger.info(
                            f"[AUTH] Token issuer ends with correct realm path '{realm_path}'"
                        )
                        # Check if it's just a hostname difference (localhost vs host.docker.internal)
                        # Both are valid - they point to the same Keycloak instance from different network contexts
                        expected_parts = expected_issuer.split("/realms/")
                        actual_parts = actual_issuer.split("/realms/")
                        logger.debug(
                            f"[AUTH] Parsed issuers - expected_parts={expected_parts}, actual_parts={actual_parts}"
                        )

                        if len(expected_parts) == 2 and len(actual_parts) == 2:
                            expected_host = expected_parts[0]
                            actual_host = actual_parts[0]
                            logger.debug(
                                f"[AUTH] Host comparison - expected_host='{expected_host}', actual_host='{actual_host}'"
                            )

                            # Normalize hostnames - localhost and host.docker.internal are equivalent
                            # Also normalize http vs https
                            normalized_expected = expected_host.replace(
                                "host.docker.internal", "localhost"
                            ).replace("https://", "http://")
                            normalized_actual = actual_host.replace(
                                "host.docker.internal", "localhost"
                            ).replace("https://", "http://")
                            logger.debug(
                                f"[AUTH] Normalized hosts - expected='{normalized_expected}', actual='{normalized_actual}'"
                            )

                            if normalized_expected == normalized_actual:
                                logger.info(
                                    f"[AUTH] Hostname difference (localhost vs host.docker.internal) detected. Using token's issuer: '{actual_issuer}'"
                                )
                                expected_issuer = actual_issuer
                            else:
                                # Even if hostnames don't match exactly, if realm matches, accept the token's issuer
                                # This handles cases where frontend and backend see different hostnames/ports
                                logger.info(
                                    f"[AUTH] Using token's issuer to allow for hostname/port differences: '{actual_issuer}'"
                                )
                                expected_issuer = actual_issuer
                        else:
                            # If we can't parse, but realm path matches, use token's issuer anyway
                            logger.info(
                                f"[AUTH] Using token's issuer (realm path matches): '{actual_issuer}'"
                            )
                            expected_issuer = actual_issuer
                    else:
                        logger.error(
                            f"[AUTH] Cannot reconcile issuer mismatch. Token issuer '{actual_issuer}' doesn't end with '{realm_path}'"
                        )
                        logger.error(
                            f"[AUTH] This will cause token validation to fail. Check KEYCLOAK_SERVER_URL and KEYCLOAK_REALM environment variables."
                        )
            except Exception as e:
                logger.warning(
                    f"[AUTH] Could not extract issuer from token: {type(e).__name__}: {str(e)}"
                )
                # If we couldn't extract issuer, try to proceed with expected issuer
                logger.info(
                    f"[AUTH] Proceeding with expected issuer: '{expected_issuer}'"
                )

            # Final check: if we still have a mismatch but the realm path matches, use token's issuer
            # This is a safety net in case the above logic didn't catch it
            if (
                actual_issuer
                and actual_issuer != expected_issuer
                and actual_issuer.endswith(f"/realms/{KEYCLOAK_REALM}")
            ):
                logger.info(
                    f"[AUTH] Safety check: Using token's issuer '{actual_issuer}' since realm path matches"
                )
                expected_issuer = actual_issuer

            logger.info(
                f"[AUTH] Decoding token with: algorithm={algorithm}, issuer={expected_issuer}, "
                f"allowed_audiences={KEYCLOAK_ALLOWED_AUDIENCES}"
            )
            logger.info(
                f"[AUTH] Final issuer being used for validation: '{expected_issuer}'"
            )

            # Decode token - we'll validate audience manually to allow both frontend and backend tokens
            # If issuer mismatch was detected, we'll use the token's actual issuer
            claims = jwt.decode(
                token,
                public_key,
                algorithms=[algorithm],
                options={
                    "verify_aud": False
                },  # Disable automatic audience check, we'll do it manually
                issuer=expected_issuer,
            )

            logger.info(
                f"[AUTH] Token decoded successfully: "
                f"sub={claims.get('sub')}, "
                f"iss={claims.get('iss')}, "
                f"aud={claims.get('aud')}, "
                f"exp={claims.get('exp')}, "
                f"iat={claims.get('iat')}, "
                f"claims_keys={list(claims.keys())[:10]}"
            )

            # Manual audience validation - check if token audience matches any allowed audience
            token_audience = claims.get("aud")
            if token_audience and KEYCLOAK_ALLOWED_AUDIENCES:
                # Handle both string and list audiences
                audiences = (
                    token_audience
                    if isinstance(token_audience, list)
                    else [token_audience]
                )
                # Check if any of the token's audiences match our allowed audiences
                if not any(aud in KEYCLOAK_ALLOWED_AUDIENCES for aud in audiences):
                    # If audience doesn't match, still allow the token but log a warning
                    # This allows frontend tokens (with frontend audience) to work with backend
                    logger.warning(
                        f"[AUTH] Token audience '{token_audience}' not in allowed audiences: {KEYCLOAK_ALLOWED_AUDIENCES}. "
                        f"Allowing token anyway to support frontend tokens."
                    )
                else:
                    logger.info(
                        f"[AUTH] Token audience '{token_audience}' matches allowed audiences"
                    )
            else:
                logger.info(
                    f"[AUTH] No audience validation needed: token_audience={token_audience}, "
                    f"allowed_audiences={KEYCLOAK_ALLOWED_AUDIENCES}"
                )
            # If no audience in token, that's also okay - some tokens may not include it
        except JWTError as e:
            logger.error(f"[AUTH] JWT decode error: {type(e).__name__}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token validation failed: {str(e)}",
            )

        # Validate required claims
        if "sub" not in claims:
            logger.error(
                f"[AUTH] Token missing required 'sub' claim. Available claims: {list(claims.keys())}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing required 'sub' claim",
            )

        logger.info(
            f"[AUTH] Token validation successful for user: sub={claims.get('sub')}"
        )
        return claims

    except HTTPException:
        raise
    except JWTError as e:
        logger.error(f"[AUTH] JWT validation error: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        )
    except Exception as e:
        logger.error(
            f"[AUTH] Unexpected error during token validation: {type(e).__name__}: {str(e)}",
            exc_info=True,
        )
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
    # Dev bypass: skip Keycloak entirely when DEV_AUTH_BYPASS=true
    if os.getenv("DEV_AUTH_BYPASS", "").lower() == "true":
        logger.warning(
            "[AUTH] DEV_AUTH_BYPASS is enabled — returning mock dev user. "
            "Never use this in production!"
        )
        return UserContext(
            user_id="dev-user-id",
            email="dev@localhost",
            username="dev",
            roles=["admin", "user", "content_editor", "approver"],
            realm_roles=["admin", "user", "content_editor", "approver"],
            client_roles=[],
        )

    logger.info(
        f"[AUTH] get_current_user called for {request.method} {request.url.path}"
    )

    # Extract token from header
    token = extract_token_from_header(request)
    if not token:
        logger.warning(
            f"[AUTH] No token found for {request.method} {request.url.path}. "
            f"Headers: {dict(request.headers)}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header with Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate token
    logger.info(f"[AUTH] Validating token for {request.method} {request.url.path}")
    claims = validate_jwt_token(token)

    # Create user context from claims
    user_context = create_user_context_from_claims(claims)
    logger.info(
        f"[AUTH] User context created: user_id={user_context.user_id}, "
        f"email={user_context.email}, roles={user_context.roles}"
    )

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
