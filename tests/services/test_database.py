"""
Tests for database service.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from marketing_project.services.database import DatabaseManager, get_database_manager


@pytest.fixture
def db_manager():
    """Create a DatabaseManager instance for testing."""
    return DatabaseManager()


def test_database_manager_initialization(db_manager):
    """Test that DatabaseManager initializes correctly."""
    assert db_manager._engine is None
    assert db_manager._session_factory is None
    assert db_manager._initialized is False


def test_get_database_url_from_database_url_env(db_manager):
    """Test getting database URL from DATABASE_URL environment variable."""
    with patch.dict(
        os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost/db"}
    ):
        url = db_manager._get_database_url()
        assert url is not None
        assert url.startswith("postgresql+asyncpg://")


def test_get_database_url_from_postgres_url_env(db_manager):
    """Test getting database URL from POSTGRES_URL environment variable."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("DATABASE_URL", None)
        with patch.dict(
            os.environ, {"POSTGRES_URL": "postgresql://user:pass@localhost/db"}
        ):
            url = db_manager._get_database_url()
            assert url is not None
            assert url.startswith("postgresql+asyncpg://")


def test_get_database_url_postgres_to_postgresql_asyncpg(db_manager):
    """Test conversion of postgres:// to postgresql+asyncpg://."""
    with patch.dict(os.environ, {"DATABASE_URL": "postgres://user:pass@localhost/db"}):
        url = db_manager._get_database_url()
        assert url == "postgresql+asyncpg://user:pass@localhost/db"


def test_get_database_url_postgresql_to_asyncpg(db_manager):
    """Test conversion of postgresql:// to postgresql+asyncpg://."""
    with patch.dict(
        os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost/db"}
    ):
        url = db_manager._get_database_url()
        assert url == "postgresql+asyncpg://user:pass@localhost/db"


def test_get_database_url_no_scheme(db_manager):
    """Test adding asyncpg scheme when no scheme provided."""
    with patch.dict(os.environ, {"DATABASE_URL": "user:pass@localhost/db"}):
        url = db_manager._get_database_url()
        assert url == "postgresql+asyncpg://user:pass@localhost/db"


def test_get_database_url_no_env_vars(db_manager):
    """Test that None is returned when no database URL is configured."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("DATABASE_URL", None)
        os.environ.pop("POSTGRES_URL", None)
        url = db_manager._get_database_url()
        assert url is None


@pytest.mark.asyncio
async def test_initialize_no_database_url(db_manager):
    """Test initialization when no database URL is configured."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("DATABASE_URL", None)
        os.environ.pop("POSTGRES_URL", None)
        result = await db_manager.initialize()
        assert result is False
        assert db_manager._initialized is False


@pytest.mark.asyncio
async def test_initialize_already_initialized(db_manager):
    """Test that initialize returns True if already initialized."""
    db_manager._initialized = True
    result = await db_manager.initialize()
    assert result is True


@pytest.mark.asyncio
async def test_initialize_success(db_manager):
    """Test successful database initialization."""
    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.begin = AsyncMock()
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock()

    with patch.dict(
        os.environ, {"DATABASE_URL": "postgresql+asyncpg://user:pass@localhost/db"}
    ):
        with patch(
            "marketing_project.services.database.create_async_engine",
            return_value=mock_engine,
        ):
            with patch(
                "marketing_project.services.database.async_sessionmaker"
            ) as mock_session_factory:
                result = await db_manager.initialize()
                assert result is True
                assert db_manager._initialized is True
                assert db_manager._engine is not None


@pytest.mark.asyncio
async def test_initialize_failure(db_manager):
    """Test database initialization failure."""
    with patch.dict(
        os.environ, {"DATABASE_URL": "postgresql+asyncpg://user:pass@localhost/db"}
    ):
        with patch(
            "marketing_project.services.database.create_async_engine",
            side_effect=Exception("Connection failed"),
        ):
            result = await db_manager.initialize()
            assert result is False
            assert db_manager._initialized is False
            assert db_manager._engine is None


@pytest.mark.asyncio
async def test_create_tables_not_initialized(db_manager):
    """Test that create_tables does nothing if database is not initialized."""
    db_manager._initialized = False
    await db_manager.create_tables()
    # Should not raise an exception


@pytest.mark.asyncio
async def test_create_tables_success(db_manager):
    """Test successful table creation."""
    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.begin = AsyncMock()
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_conn.run_sync = AsyncMock()

    db_manager._initialized = True
    db_manager._engine = mock_engine

    await db_manager.create_tables()
    mock_conn.run_sync.assert_called_once()


@pytest.mark.asyncio
async def test_get_session_not_initialized(db_manager):
    """Test that get_session raises error if database is not initialized."""
    db_manager._initialized = False
    with pytest.raises(RuntimeError, match="Database not initialized"):
        async with db_manager.get_session():
            pass


@pytest.mark.asyncio
async def test_get_session_success(db_manager):
    """Test successful session retrieval."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()

    mock_session_factory = AsyncMock()
    mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)

    db_manager._initialized = True
    db_manager._session_factory = mock_session_factory

    async with db_manager.get_session() as session:
        assert session is not None

    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_get_session_rollback_on_error(db_manager):
    """Test that session rolls back on error."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()

    mock_session_factory = AsyncMock()
    mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)

    db_manager._initialized = True
    db_manager._session_factory = mock_session_factory

    with pytest.raises(ValueError):
        async with db_manager.get_session() as session:
            raise ValueError("Test error")

    mock_session.rollback.assert_called_once()


@pytest.mark.asyncio
async def test_health_check_not_initialized(db_manager):
    """Test health check when database is not initialized."""
    db_manager._initialized = False
    result = await db_manager.health_check()
    assert result is False


@pytest.mark.asyncio
async def test_health_check_success(db_manager):
    """Test successful health check."""
    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.begin = AsyncMock()
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock()

    db_manager._initialized = True
    db_manager._engine = mock_engine

    result = await db_manager.health_check()
    assert result is True


@pytest.mark.asyncio
async def test_health_check_failure(db_manager):
    """Test health check failure."""
    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.begin = AsyncMock(side_effect=Exception("Connection failed"))

    db_manager._initialized = True
    db_manager._engine = mock_engine

    result = await db_manager.health_check()
    assert result is False


@pytest.mark.asyncio
async def test_cleanup(db_manager):
    """Test database cleanup."""
    mock_engine = MagicMock(spec=AsyncEngine)
    mock_engine.dispose = AsyncMock()

    db_manager._engine = mock_engine
    db_manager._session_factory = MagicMock()
    db_manager._initialized = True

    await db_manager.cleanup()

    mock_engine.dispose.assert_called_once()
    assert db_manager._engine is None
    assert db_manager._session_factory is None
    assert db_manager._initialized is False


def test_is_initialized_property(db_manager):
    """Test is_initialized property."""
    assert db_manager.is_initialized is False
    db_manager._initialized = True
    assert db_manager.is_initialized is True


def test_engine_property(db_manager):
    """Test engine property."""
    assert db_manager.engine is None
    mock_engine = MagicMock()
    db_manager._engine = mock_engine
    assert db_manager.engine is mock_engine


def test_get_database_manager_singleton():
    """Test that get_database_manager returns a singleton."""
    manager1 = get_database_manager()
    manager2 = get_database_manager()
    assert manager1 is manager2
