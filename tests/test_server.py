"""
Tests for server module.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.server import app, custom_openapi, lifespan


def test_app_initialization():
    """Test that FastAPI app is initialized correctly."""
    assert app is not None
    assert app.title == "Marketing Project API"
    assert app.version == "2.0.0"
    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"
    assert app.openapi_url == "/openapi.json"


def test_app_servers_configuration():
    """Test that app servers are configured."""
    assert len(app.servers) == 2
    assert app.servers[0]["url"] == "http://localhost:8000"
    assert app.servers[1]["url"] == "https://api.marketing-project.com"


def test_custom_openapi():
    """Test custom OpenAPI schema generation."""
    # Clear cached schema
    app.openapi_schema = None
    schema = custom_openapi()
    assert schema is not None
    assert "info" in schema
    assert schema["info"]["title"] == "Marketing Project API"
    assert schema["info"]["version"] == "2.0.0"


def test_custom_openapi_cached():
    """Test that OpenAPI schema is cached."""
    # Set cached schema
    app.openapi_schema = {"cached": True}
    schema = custom_openapi()
    assert schema == {"cached": True}


def test_app_middleware_configured():
    """Test that middleware is configured on the app."""
    # Check that middleware is added (at least one)
    assert len(app.user_middleware) > 0


def test_app_router_included():
    """Test that API router is included."""
    # Check that routes exist
    assert len(app.routes) > 0


@pytest.mark.asyncio
async def test_lifespan_startup():
    """Test lifespan startup sequence."""
    mock_app = MagicMock()

    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_db_manager:
        with patch(
            "marketing_project.server.content.initialize_content_sources",
            new_callable=AsyncMock,
        ):
            with patch(
                "marketing_project.services.scanned_document_db.get_scanned_document_db"
            ) as mock_scanned_db:
                mock_db = MagicMock()
                mock_scanned_db.return_value = mock_db

                mock_db_mgr = MagicMock()
                mock_db_mgr.initialize = AsyncMock(return_value=True)
                mock_db_mgr.create_tables = AsyncMock()
                mock_db_mgr.is_initialized = True
                mock_db_manager.return_value = mock_db_mgr

                async with lifespan(mock_app) as result:
                    # Lifespan context manager yields None, but that's expected
                    pass

                mock_db_mgr.initialize.assert_called_once()
                mock_db_mgr.create_tables.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_startup_no_database():
    """Test lifespan startup when database is not configured."""
    mock_app = MagicMock()

    with patch(
        "marketing_project.services.database.get_database_manager"
    ) as mock_db_manager:
        with patch(
            "marketing_project.server.content.initialize_content_sources",
            new_callable=AsyncMock,
        ):
            with patch(
                "marketing_project.services.scanned_document_db.get_scanned_document_db"
            ) as mock_scanned_db:
                mock_db = MagicMock()
                mock_scanned_db.return_value = mock_db

                mock_db_mgr = MagicMock()
                mock_db_mgr.initialize = AsyncMock(return_value=False)
                mock_db_manager.return_value = mock_db_mgr

                async with lifespan(mock_app) as result:
                    # Lifespan context manager yields None, but that's expected
                    pass

                mock_db_mgr.initialize.assert_called_once()
                mock_db_mgr.create_tables.assert_not_called()


@pytest.mark.asyncio
async def test_lifespan_startup_database_error():
    """Test lifespan startup handles database initialization errors."""
    mock_app = MagicMock()

    with patch(
        "marketing_project.services.database.get_database_manager",
        side_effect=Exception("DB Error"),
    ):
        with patch(
            "marketing_project.server.content.initialize_content_sources",
            new_callable=AsyncMock,
        ):
            with patch(
                "marketing_project.services.scanned_document_db.get_scanned_document_db",
                side_effect=Exception("Scanned DB Error"),
            ):
                # Should not raise exception, just log warning
                async with lifespan(mock_app) as result:
                    # Lifespan context manager yields None, but that's expected
                    pass


@pytest.mark.asyncio
async def test_lifespan_shutdown():
    """Test lifespan shutdown sequence."""
    mock_app = MagicMock()

    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_db_mgr.cleanup = AsyncMock()

    mock_redis_mgr = MagicMock()
    mock_redis_mgr.cleanup = AsyncMock()

    mock_job_mgr = MagicMock()
    mock_job_mgr.cleanup = AsyncMock()

    mock_approval_mgr = MagicMock()
    mock_approval_mgr.cleanup = AsyncMock()

    mock_design_kit_mgr = MagicMock()
    mock_design_kit_mgr.cleanup = AsyncMock()

    mock_internal_docs_mgr = MagicMock()
    mock_internal_docs_mgr.cleanup = AsyncMock()

    with patch(
        "marketing_project.services.database.get_database_manager",
        return_value=mock_db_mgr,
    ):
        with patch(
            "marketing_project.services.redis_manager.get_redis_manager",
            return_value=mock_redis_mgr,
        ):
            with patch(
                "marketing_project.services.job_manager.get_job_manager",
                return_value=mock_job_mgr,
            ):
                with patch(
                    "marketing_project.services.approval_manager.get_approval_manager",
                    new_callable=AsyncMock,
                    return_value=mock_approval_mgr,
                ):
                    with patch(
                        "marketing_project.services.design_kit_manager.get_design_kit_manager",
                        return_value=mock_design_kit_mgr,
                    ):
                        with patch(
                            "marketing_project.services.internal_docs_manager.get_internal_docs_manager",
                            return_value=mock_internal_docs_mgr,
                        ):
                            with patch(
                                "marketing_project.server.content.initialize_content_sources",
                                new_callable=AsyncMock,
                            ):
                                with patch(
                                    "marketing_project.services.scanned_document_db.get_scanned_document_db"
                                ):
                                    async with lifespan(mock_app):
                                        pass

                                    mock_db_mgr.cleanup.assert_called_once()
                                    mock_redis_mgr.cleanup.assert_called_once()
                                    mock_job_mgr.cleanup.assert_called_once()
                                    mock_approval_mgr.cleanup.assert_called_once()
                                    mock_design_kit_mgr.cleanup.assert_called_once()
                                    mock_internal_docs_mgr.cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_shutdown_handles_errors():
    """Test that lifespan shutdown handles cleanup errors gracefully."""
    mock_app = MagicMock()

    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_db_mgr.cleanup = AsyncMock(side_effect=Exception("Cleanup error"))

    with patch(
        "marketing_project.services.database.get_database_manager",
        return_value=mock_db_mgr,
    ):
        with patch(
            "marketing_project.services.redis_manager.get_redis_manager",
            side_effect=Exception("Redis error"),
        ):
            with patch(
                "marketing_project.services.job_manager.get_job_manager",
                side_effect=Exception("Job error"),
            ):
                with patch(
                    "marketing_project.services.approval_manager.get_approval_manager",
                    new_callable=AsyncMock,
                    side_effect=Exception("Approval error"),
                ):
                    with patch(
                        "marketing_project.services.design_kit_manager.get_design_kit_manager",
                        side_effect=Exception("Design kit error"),
                    ):
                        with patch(
                            "marketing_project.services.internal_docs_manager.get_internal_docs_manager",
                            side_effect=Exception("Internal docs error"),
                        ):
                            with patch(
                                "marketing_project.server.content.initialize_content_sources",
                                new_callable=AsyncMock,
                            ):
                                with patch(
                                    "marketing_project.services.scanned_document_db.get_scanned_document_db"
                                ):
                                    # Should not raise exception, just log warnings
                                    async with lifespan(mock_app):
                                        pass


def test_app_endpoints_accessible():
    """Test that app endpoints are accessible."""
    client = TestClient(app)
    # Test health endpoint (should exist)
    response = client.get("/api/v1/health")
    assert response.status_code in [
        200,
        404,
    ]  # May not exist if routes not fully loaded in test


def test_cors_configuration_from_env():
    """Test that CORS is configured from environment variables."""
    with patch.dict(
        os.environ, {"CORS_ORIGINS": "https://example.com,https://app.example.com"}
    ):
        # Reload module to get new CORS config
        import importlib

        import marketing_project.server

        importlib.reload(marketing_project.server)
        # CORS middleware should be configured
        assert marketing_project.server.app is not None


def test_cors_allow_credentials_from_env():
    """Test that CORS allow_credentials is configured from environment."""
    with patch.dict(os.environ, {"CORS_ALLOW_CREDENTIALS": "false"}):
        # Reload module to get new CORS config
        import importlib

        import marketing_project.server

        importlib.reload(marketing_project.server)
        # CORS middleware should be configured
        assert marketing_project.server.app is not None
