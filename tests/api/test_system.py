"""
Tests for system API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from marketing_project.api.system import router
from marketing_project.server import app

# Add router to app
app.include_router(router)

client = TestClient(app)


def test_get_system_info():
    """Test getting system information."""
    response = client.get("/system/info")

    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert "python_version" in data
    assert "platform" in data
    assert "environment" in data
    assert "configuration" in data
    assert data["service"] == "marketing-project"
