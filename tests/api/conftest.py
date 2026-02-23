"""
Shared fixtures for API tests.

Provides an autouse fixture that overrides Keycloak authentication on the
main server app so tests that import `server.app` directly don't hit real
Keycloak and get 401 Unauthorized responses.
"""

import pytest

from marketing_project.middleware.keycloak_auth import get_current_user
from tests.utils.keycloak_test_helpers import create_user_context


@pytest.fixture(autouse=True)
def override_server_app_auth():
    """Override get_current_user on server.app for every API test.

    Tests that create their own bare FastAPI() app still need to set
    app.dependency_overrides themselves (see individual client fixtures).
    """
    from marketing_project.server import app

    mock_admin = create_user_context(roles=["admin"])
    app.dependency_overrides[get_current_user] = lambda: mock_admin
    yield
    app.dependency_overrides.pop(get_current_user, None)
