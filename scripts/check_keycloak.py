#!/usr/bin/env python3
"""
Diagnostic script to check Keycloak connectivity and JWKS availability.

Usage:
    python scripts/check_keycloak.py
"""

import os
import sys

import httpx
from dotenv import find_dotenv, load_dotenv

# Add parent directory to path to import from marketing_project
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Load .env file from project root (same as backend does)
# First try to find .env in project root
env_file = os.path.join(project_root, ".env")
if os.path.exists(env_file):
    dotenv_path = env_file
else:
    # Fallback to find_dotenv which searches from current directory upward
    dotenv_path = find_dotenv()

if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)

KEYCLOAK_SERVER_URL = os.getenv("KEYCLOAK_SERVER_URL", "")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "")
KEYCLOAK_VERIFY_SSL = os.getenv("KEYCLOAK_VERIFY_SSL", "true").lower() == "true"


def check_keycloak():
    """Check Keycloak connectivity and JWKS availability."""
    print("=" * 60)
    print("Keycloak Configuration Check")
    print("=" * 60)

    if dotenv_path:
        print(f"\n📄 Loaded .env file: {dotenv_path}")
    else:
        print(f"\n⚠️  No .env file loaded - using system environment variables only")
        print(
            f"   Tip: Create a .env file in the project root or set environment variables"
        )

    # Check environment variables
    print(f"\n1. Environment Variables:")
    print(f"   KEYCLOAK_SERVER_URL: {KEYCLOAK_SERVER_URL or 'NOT SET'}")
    print(f"   KEYCLOAK_REALM: {KEYCLOAK_REALM or 'NOT SET'}")
    print(f"   KEYCLOAK_VERIFY_SSL: {KEYCLOAK_VERIFY_SSL}")

    if not KEYCLOAK_SERVER_URL or not KEYCLOAK_REALM:
        print("\n❌ ERROR: KEYCLOAK_SERVER_URL and KEYCLOAK_REALM must be set!")
        print("\n   Example:")
        print("   export KEYCLOAK_SERVER_URL=http://localhost:8080")
        print("   export KEYCLOAK_REALM=marketing-tool")
        return False

    # Construct URLs
    base_url = f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM}"
    jwks_url = f"{base_url}/protocol/openid-connect/certs"
    well_known_url = f"{base_url}/.well-known/openid-configuration"

    print(f"\n2. Constructed URLs:")
    print(f"   Realm Base: {base_url}")
    print(f"   JWKS URL: {jwks_url}")
    print(f"   Well-Known: {well_known_url}")

    # Test connectivity
    print(f"\n3. Connectivity Tests:")

    # Test 1: Check if Keycloak server is reachable
    try:
        print(f"   Testing connection to {KEYCLOAK_SERVER_URL}...")
        with httpx.Client(verify=KEYCLOAK_VERIFY_SSL, timeout=5.0) as client:
            response = client.get(KEYCLOAK_SERVER_URL, follow_redirects=True)
            print(
                f"   ✅ Keycloak server is reachable (Status: {response.status_code})"
            )
    except httpx.ConnectError as e:
        print(f"   ❌ Cannot connect to Keycloak server: {e}")
        print(f"      Make sure Keycloak is running at {KEYCLOAK_SERVER_URL}")
        return False
    except httpx.TimeoutException:
        print(f"   ❌ Timeout connecting to Keycloak server")
        print(f"      Check if Keycloak is running and accessible")
        return False
    except Exception as e:
        print(f"   ⚠️  Unexpected error: {e}")

    # Test 2: Check realm exists
    try:
        print(f"   Testing realm '{KEYCLOAK_REALM}'...")
        with httpx.Client(verify=KEYCLOAK_VERIFY_SSL, timeout=5.0) as client:
            response = client.get(well_known_url)
            response.raise_for_status()
            config = response.json()
            print(f"   ✅ Realm '{KEYCLOAK_REALM}' exists")
            print(f"      Issuer: {config.get('issuer', 'N/A')}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"   ❌ Realm '{KEYCLOAK_REALM}' not found (404)")
            print(f"      Check if the realm name is correct")
            return False
        else:
            print(f"   ❌ Error accessing realm: Status {e.response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error checking realm: {e}")
        return False

    # Test 3: Check JWKS endpoint
    try:
        print(f"   Testing JWKS endpoint...")
        with httpx.Client(verify=KEYCLOAK_VERIFY_SSL, timeout=10.0) as client:
            response = client.get(jwks_url)
            response.raise_for_status()
            jwks = response.json()

            if "keys" in jwks and len(jwks["keys"]) > 0:
                print(f"   ✅ JWKS endpoint is accessible")
                print(f"      Found {len(jwks['keys'])} key(s) in JWKS")
                for i, key in enumerate(jwks["keys"][:3], 1):  # Show first 3 keys
                    kid = key.get("kid", "N/A")
                    alg = key.get("alg", "N/A")
                    kty = key.get("kty", "N/A")
                    print(f"      Key {i}: kid={kid}, alg={alg}, kty={kty}")
                if len(jwks["keys"]) > 3:
                    print(f"      ... and {len(jwks['keys']) - 3} more key(s)")
            else:
                print(f"   ⚠️  JWKS endpoint accessible but no keys found")
                return False
    except httpx.HTTPStatusError as e:
        print(f"   ❌ JWKS endpoint error: Status {e.response.status_code}")
        print(f"      Response: {e.response.text[:200]}")
        return False
    except Exception as e:
        print(f"   ❌ Error accessing JWKS endpoint: {e}")
        return False

    print(f"\n{'=' * 60}")
    print("✅ All checks passed! Keycloak is properly configured.")
    print(f"{'=' * 60}\n")
    return True


if __name__ == "__main__":
    success = check_keycloak()
    sys.exit(0 if success else 1)
