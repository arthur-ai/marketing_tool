#!/usr/bin/env python3
"""
Verify that the backend can access Keycloak JWKS endpoint.
This simulates what the backend does when validating tokens.

Usage:
    python scripts/verify_backend_keycloak.py
"""

import os
import sys

import httpx
from dotenv import find_dotenv, load_dotenv

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Load .env file
env_file = os.path.join(project_root, ".env")
if os.path.exists(env_file):
    dotenv_path = env_file
else:
    dotenv_path = find_dotenv()

if dotenv_path:
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"✅ Loaded .env from: {dotenv_path}\n")
else:
    print("⚠️  No .env file found\n")

KEYCLOAK_SERVER_URL = os.getenv("KEYCLOAK_SERVER_URL", "")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "")
KEYCLOAK_VERIFY_SSL = os.getenv("KEYCLOAK_VERIFY_SSL", "true").lower() == "true"

print("=" * 60)
print("Backend Keycloak Configuration Verification")
print("=" * 60)
print(f"\nConfiguration:")
print(f"  KEYCLOAK_SERVER_URL: {KEYCLOAK_SERVER_URL or 'NOT SET'}")
print(f"  KEYCLOAK_REALM: {KEYCLOAK_REALM or 'NOT SET'}")
print(f"  KEYCLOAK_VERIFY_SSL: {KEYCLOAK_VERIFY_SSL}")

if not KEYCLOAK_SERVER_URL or not KEYCLOAK_REALM:
    print(
        "\n❌ ERROR: KEYCLOAK_SERVER_URL and KEYCLOAK_REALM must be set in .env file!"
    )
    sys.exit(1)

jwks_url = (
    f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs"
)

print(f"\nTesting JWKS fetch (same as backend does):")
print(f"  URL: {jwks_url}")

try:
    with httpx.Client(verify=KEYCLOAK_VERIFY_SSL, timeout=10.0) as client:
        response = client.get(jwks_url)
        response.raise_for_status()
        jwks = response.json()

        if "keys" in jwks and len(jwks["keys"]) > 0:
            print(f"  ✅ Success! Retrieved {len(jwks['keys'])} key(s)")
            print(f"\n✅ Backend should be able to fetch JWKS successfully!")
            print(f"\n💡 If backend still shows 503 errors, RESTART the backend server")
            print(f"   to pick up the .env configuration.")
        else:
            print(f"  ⚠️  Response received but no keys found")
            sys.exit(1)
except httpx.ConnectError as e:
    print(f"  ❌ Connection error: {e}")
    print(f"     Make sure Keycloak is running at {KEYCLOAK_SERVER_URL}")
    sys.exit(1)
except httpx.TimeoutException:
    print(f"  ❌ Timeout connecting to Keycloak")
    sys.exit(1)
except httpx.HTTPStatusError as e:
    print(f"  ❌ HTTP error: Status {e.response.status_code}")
    print(f"     Response: {e.response.text[:200]}")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ Error: {type(e).__name__}: {e}")
    sys.exit(1)
