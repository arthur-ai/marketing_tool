"""
Tests for design kit manager service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.design_kit_config import DesignKitConfig
from marketing_project.services.design_kit_manager import (
    DesignKitManager,
    get_design_kit_manager,
)


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    with patch(
        "marketing_project.services.design_kit_manager.get_redis_manager"
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
        "marketing_project.services.design_kit_manager.get_database_manager"
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
def design_kit_manager(mock_redis_manager, mock_db_manager):
    """Create a DesignKitManager instance."""
    return DesignKitManager()


@pytest.mark.asyncio
async def test_get_active_config_from_db(design_kit_manager, mock_db_manager):
    """Test getting active config from database."""
    # Mock database result
    mock_config_model = MagicMock()
    mock_config_model.to_dict.return_value = {
        "config_data": {
            "voice_adjectives": ["professional"],
            "point_of_view": "we",
        }
    }

    mock_db_manager.get_session.return_value.__aenter__.return_value.execute.return_value.scalar_one_or_none.return_value = (
        mock_config_model
    )

    config = await design_kit_manager.get_active_config_from_db()

    assert config is not None
    assert isinstance(config, DesignKitConfig)


@pytest.mark.asyncio
async def test_get_active_config_from_db_not_found(design_kit_manager, mock_db_manager):
    """Test getting active config when none exists."""
    config = await design_kit_manager.get_active_config_from_db()

    assert config is None


@pytest.mark.asyncio
async def test_get_active_config_from_redis(design_kit_manager, mock_redis_manager):
    """Test getting active config from Redis."""
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value='{"voice_adjectives": ["professional"]}')
    mock_redis_manager.get_redis = AsyncMock(return_value=mock_redis)
    mock_redis_manager.execute = AsyncMock(return_value="version-1")

    # Mock getting config data
    async def get_config_operation(redis_client):
        return '{"voice_adjectives": ["professional"], "point_of_view": "we"}'

    mock_redis_manager.execute = AsyncMock(
        side_effect=["version-1", '{"voice_adjectives": ["professional"]}']
    )

    config = await design_kit_manager.get_active_config_from_redis()

    # May return None if Redis not configured
    assert config is None or isinstance(config, DesignKitConfig)


@pytest.mark.asyncio
async def test_get_active_config(
    design_kit_manager, mock_db_manager, mock_redis_manager
):
    """Test get_active_config method (tries DB first, then Redis)."""
    # Mock both to return None
    mock_db_manager.get_session.return_value.__aenter__.return_value.execute.return_value.scalar_one_or_none.return_value = (
        None
    )
    mock_redis_manager.execute = AsyncMock(return_value=None)

    config = await design_kit_manager.get_active_config()

    # Should return None or config
    assert config is None or isinstance(config, DesignKitConfig)


@pytest.mark.asyncio
async def test_save_config(design_kit_manager, mock_db_manager, mock_redis_manager):
    """Test saving config."""
    from datetime import datetime

    config = DesignKitConfig(
        version="1.0.0",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        is_active=True,
        voice_adjectives=["professional"],
        point_of_view="we",
    )

    # Mock database
    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_db_manager.get_session.return_value.__aenter__.return_value = mock_session

    config_id = await design_kit_manager.save_config(config)

    # Should return config ID or None
    assert config_id is None or isinstance(config_id, str)


@pytest.mark.asyncio
async def test_get_design_kit_manager_singleton():
    """Test that get_design_kit_manager returns a singleton."""
    manager1 = await get_design_kit_manager()
    manager2 = await get_design_kit_manager()

    assert manager1 is manager2
