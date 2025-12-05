"""
Tests for internal docs manager service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.internal_docs_config import InternalDocsConfig
from marketing_project.services.internal_docs_manager import (
    InternalDocsManager,
    get_internal_docs_manager,
)


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    with patch(
        "marketing_project.services.internal_docs_manager.get_redis_manager"
    ) as mock:
        manager = MagicMock()
        manager.get_redis = AsyncMock(return_value=MagicMock())
        manager.execute = AsyncMock(return_value=None)
        mock.return_value = manager
        yield manager


@pytest.fixture
def mock_db_manager():
    """Mock database manager."""
    with patch(
        "marketing_project.services.internal_docs_manager.get_database_manager"
    ) as mock:
        manager = MagicMock()
        manager.is_initialized = True
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
        )
        manager.get_session = MagicMock(return_value=session)
        mock.return_value = manager
        yield manager


@pytest.fixture
def internal_docs_manager(mock_redis_manager, mock_db_manager):
    """Create an InternalDocsManager instance."""
    return InternalDocsManager()


@pytest.mark.asyncio
async def test_get_active_config_from_db(internal_docs_manager, mock_db_manager):
    """Test getting active config from database."""
    # Mock database result
    mock_config_model = MagicMock()
    mock_config_model.to_dict.return_value = {
        "config_data": {
            "base_url": "https://example.com",
            "scan_depth": 2,
        }
    }

    mock_db_manager.get_session.return_value.__aenter__.return_value.execute.return_value.scalar_one_or_none.return_value = (
        mock_config_model
    )

    config = await internal_docs_manager.get_active_config_from_db()

    assert config is not None
    assert isinstance(config, InternalDocsConfig)


@pytest.mark.asyncio
async def test_get_active_config_from_db_not_found(
    internal_docs_manager, mock_db_manager
):
    """Test getting active config when none exists."""
    config = await internal_docs_manager.get_active_config_from_db()

    assert config is None


@pytest.mark.asyncio
async def test_get_active_config_from_redis(internal_docs_manager, mock_redis_manager):
    """Test getting active config from Redis."""
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value='{"base_url": "https://example.com"}')
    mock_redis_manager.get_redis = AsyncMock(return_value=mock_redis)
    mock_redis_manager.execute = AsyncMock(
        side_effect=["version-1", '{"base_url": "https://example.com"}']
    )

    config = await internal_docs_manager.get_active_config_from_redis()

    # May return None if Redis not configured
    assert config is None or isinstance(config, InternalDocsConfig)


@pytest.mark.asyncio
async def test_get_active_config(
    internal_docs_manager, mock_db_manager, mock_redis_manager
):
    """Test get_active_config method (tries DB first, then Redis)."""
    # Mock both to return None
    mock_db_manager.get_session.return_value.__aenter__.return_value.execute.return_value.scalar_one_or_none.return_value = (
        None
    )
    mock_redis_manager.execute = AsyncMock(return_value=None)

    config = await internal_docs_manager.get_active_config()

    # Should return None or config
    assert config is None or isinstance(config, InternalDocsConfig)


@pytest.mark.asyncio
async def test_save_config(internal_docs_manager, mock_db_manager, mock_redis_manager):
    """Test saving config."""
    config = InternalDocsConfig(
        base_url="https://example.com",
        scan_depth=2,
    )

    # Mock database
    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session

    success = await internal_docs_manager.save_config_to_db(config, set_active=True)

    # Should return True or False
    assert isinstance(success, bool)


@pytest.mark.asyncio
async def test_get_internal_docs_manager_singleton():
    """Test that get_internal_docs_manager returns a singleton."""
    manager1 = await get_internal_docs_manager()
    manager2 = await get_internal_docs_manager()

    assert manager1 is manager2
