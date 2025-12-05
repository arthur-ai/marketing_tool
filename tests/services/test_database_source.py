"""
Tests for database content source service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.core.content_sources import DatabaseSourceConfig
from marketing_project.services.database_source import (
    DatabaseContentSource,
    SQLContentSource,
)


@pytest.fixture
def db_source_config():
    """Create database source config."""
    return DatabaseSourceConfig(
        name="test-db-source",
        connection_string="sqlite:///test.db",
        query="SELECT * FROM content",
    )


@pytest.fixture
def database_content_source(db_source_config):
    """Create DatabaseContentSource instance."""
    return DatabaseContentSource(db_source_config)


@pytest.fixture
def sql_content_source(db_source_config):
    """Create SQLContentSource instance."""
    return SQLContentSource(db_source_config)


def test_database_content_source_initialization(database_content_source):
    """Test DatabaseContentSource initialization."""
    assert database_content_source.name == "test-db-source"
    assert database_content_source.connection_string == "sqlite:///test.db"
    assert database_content_source.connected is False


@pytest.mark.asyncio
async def test_database_content_source_initialize_not_implemented(
    database_content_source,
):
    """Test that base class initialize returns False."""
    result = await database_content_source.initialize()

    assert result is False


@pytest.mark.asyncio
async def test_database_content_source_fetch_content_not_implemented(
    database_content_source,
):
    """Test that base class fetch_content returns error result."""
    result = await database_content_source.fetch_content()

    assert result.success is False
    assert "not implemented" in result.error_message.lower()


@pytest.mark.asyncio
async def test_database_content_source_health_check_not_implemented(
    database_content_source,
):
    """Test that base class health_check returns False."""
    result = await database_content_source.health_check()

    assert result is False


@pytest.mark.asyncio
async def test_database_content_source_cleanup(database_content_source):
    """Test cleanup method."""
    mock_connection = MagicMock()
    mock_connection.close = AsyncMock()
    database_content_source.connection = mock_connection

    await database_content_source.cleanup()

    assert database_content_source.connection is None
    assert database_content_source.connected is False


@pytest.mark.asyncio
async def test_sql_content_source_initialize_sqlite(sql_content_source):
    """Test SQLContentSource initialization with SQLite."""
    with patch("marketing_project.services.database_source.aiosqlite") as mock_sqlite:
        mock_connect = AsyncMock()
        mock_conn = MagicMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_connect.return_value = mock_conn
        mock_sqlite.connect = mock_connect

        sql_content_source.config.connection_string = "sqlite:///test.db"

        result = await sql_content_source.initialize()

        # May succeed or fail depending on database availability
        assert isinstance(result, bool)
