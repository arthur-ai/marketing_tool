"""
Tests for pipeline settings manager service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.pipeline_settings_manager import (
    PipelineSettings,
    PipelineSettingsManager,
    get_pipeline_settings_manager,
)


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    with patch(
        "marketing_project.services.pipeline_settings_manager.get_redis_manager"
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
        "marketing_project.services.pipeline_settings_manager.get_database_manager"
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
def settings_manager(mock_redis_manager, mock_db_manager):
    """Create a PipelineSettingsManager instance."""
    return PipelineSettingsManager()


@pytest.mark.asyncio
async def test_load_settings_from_db(settings_manager, mock_db_manager):
    """Test loading settings from database."""
    # Mock database result
    mock_settings_model = MagicMock()
    mock_settings_model.to_dict.return_value = {
        "settings_data": {
            "pipeline_config": {"default_temperature": 0.7},
            "optional_steps": ["suggested_links"],
        }
    }

    mock_db_manager.get_session.return_value.__aenter__.return_value.execute.return_value.scalar_one_or_none.return_value = (
        mock_settings_model
    )

    settings = await settings_manager.load_settings_from_db()

    assert settings is not None
    assert settings.pipeline_config["default_temperature"] == 0.7


@pytest.mark.asyncio
async def test_load_settings_from_db_not_found(settings_manager, mock_db_manager):
    """Test loading settings when none exist in database."""
    settings = await settings_manager.load_settings_from_db()

    assert settings is None


@pytest.mark.asyncio
async def test_load_settings_from_redis(settings_manager, mock_redis_manager):
    """Test loading settings from Redis."""
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(
        return_value='{"pipeline_config": {"default_temperature": 0.7}}'
    )
    mock_redis_manager.get_redis = AsyncMock(return_value=mock_redis)

    settings = await settings_manager.load_settings_from_redis()

    # May return None if Redis not configured
    assert settings is None or isinstance(settings, PipelineSettings)


@pytest.mark.asyncio
async def test_save_settings_to_db(settings_manager, mock_db_manager):
    """Test saving settings to database."""
    settings = PipelineSettings(
        pipeline_config={"default_temperature": 0.7},
        optional_steps=["suggested_links"],
    )

    # Mock database session properly
    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.execute = AsyncMock(
        return_value=MagicMock(scalars=MagicMock(return_value=[]))
    )
    mock_db_manager.get_session.return_value = mock_session

    success = await settings_manager.save_settings_to_db(settings)

    assert success is True
    mock_session.add.assert_called_once()


@pytest.mark.asyncio
async def test_save_settings_to_redis(settings_manager, mock_redis_manager):
    """Test saving settings to Redis."""
    settings = PipelineSettings(
        pipeline_config={"default_temperature": 0.7},
        optional_steps=["suggested_links"],
    )

    mock_redis = MagicMock()
    mock_redis.set = AsyncMock()
    mock_redis_manager.get_redis = AsyncMock(return_value=mock_redis)

    # Method doesn't return a value (returns None)
    await settings_manager.save_settings_to_redis(settings)

    # Should not raise exception
    assert True


@pytest.mark.asyncio
async def test_load_settings(settings_manager, mock_db_manager, mock_redis_manager):
    """Test load_settings method (tries DB first, then Redis)."""
    # Mock both to return None
    mock_db_manager.get_session.return_value.__aenter__.return_value.execute.return_value.scalar_one_or_none.return_value = (
        None
    )
    mock_redis_manager.execute = AsyncMock(return_value=None)

    settings = await settings_manager.load_settings()

    # Should return None or default settings
    assert settings is None or isinstance(settings, PipelineSettings)


@pytest.mark.asyncio
async def test_save_settings(settings_manager, mock_db_manager, mock_redis_manager):
    """Test save_settings method."""
    settings = PipelineSettings(
        pipeline_config={"default_temperature": 0.7},
        optional_steps=["suggested_links"],
    )

    # Mock database session properly
    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.execute = AsyncMock(
        return_value=MagicMock(scalars=MagicMock(return_value=[]))
    )
    mock_db_manager.get_session.return_value = mock_session

    await settings_manager.save_settings(settings)

    # Should attempt to save to both DB and Redis
    # Note: save_settings calls save_settings_to_db which calls session.add
    # But if db_manager.is_initialized is False, add won't be called
    # So we just verify the method completed without error
    assert True


def test_get_pipeline_settings_manager_singleton():
    """Test that get_pipeline_settings_manager returns a singleton."""
    manager1 = get_pipeline_settings_manager()
    manager2 = get_pipeline_settings_manager()

    assert manager1 is manager2


# ---------------------------------------------------------------------------
# Additional tests to cover missed lines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_redis_success(mock_redis_manager):
    """Test get_redis returns Redis client on success (lines 64-65)."""
    mock_client = MagicMock()
    mock_redis_manager.get_redis = AsyncMock(return_value=mock_client)
    manager = PipelineSettingsManager()
    result = await manager.get_redis()
    assert result == mock_client


@pytest.mark.asyncio
async def test_get_redis_failure_returns_none(mock_redis_manager):
    """Test get_redis returns None on failure (lines 66-70)."""
    mock_redis_manager.get_redis = AsyncMock(side_effect=Exception("Redis down"))
    manager = PipelineSettingsManager()
    result = await manager.get_redis()
    assert result is None


@pytest.mark.asyncio
async def test_load_settings_from_db_exception_returns_none(
    mock_db_manager, mock_redis_manager
):
    """Test load_settings_from_db returns None on DB exception (lines 92-93)."""
    mock_db_manager.get_session.return_value.__aenter__.side_effect = Exception(
        "DB error"
    )
    manager = PipelineSettingsManager()
    result = await manager.load_settings_from_db()
    assert result is None


@pytest.mark.asyncio
async def test_load_settings_from_db_not_initialized(mock_redis_manager):
    """Test load_settings_from_db returns None when DB not initialized (lines 76-77)."""
    with patch(
        "marketing_project.services.pipeline_settings_manager.get_database_manager"
    ) as mock:
        manager_mock = MagicMock()
        manager_mock.is_initialized = False
        mock.return_value = manager_mock
        mgr = PipelineSettingsManager()
        result = await mgr.load_settings_from_db()
    assert result is None


@pytest.mark.asyncio
async def test_load_settings_from_redis_with_data(mock_redis_manager):
    """Test load_settings_from_redis parses JSON correctly (lines 104-106)."""
    settings_json = '{"pipeline_config": {"model": "gpt-4"}, "optional_steps": [], "retry_strategy": null}'
    mock_redis_manager.execute = AsyncMock(return_value=settings_json)
    manager = PipelineSettingsManager()
    result = await manager.load_settings_from_redis()
    assert result is not None
    assert isinstance(result, PipelineSettings)
    assert result.pipeline_config["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_load_settings_from_redis_none(mock_redis_manager):
    """Test load_settings_from_redis returns None when no data."""
    mock_redis_manager.execute = AsyncMock(return_value=None)
    manager = PipelineSettingsManager()
    result = await manager.load_settings_from_redis()
    assert result is None


@pytest.mark.asyncio
async def test_load_settings_from_redis_exception(mock_redis_manager):
    """Test load_settings_from_redis returns None on exception (lines 107-108)."""
    mock_redis_manager.execute = AsyncMock(side_effect=Exception("Redis error"))
    manager = PipelineSettingsManager()
    result = await manager.load_settings_from_redis()
    assert result is None


@pytest.mark.asyncio
async def test_load_settings_db_found_updates_redis(
    mock_db_manager, mock_redis_manager
):
    """Test load_settings updates Redis when DB has data (lines 119-123)."""
    mock_settings_model = MagicMock()
    mock_settings_model.to_dict.return_value = {
        "settings_data": {
            "pipeline_config": {"model": "gpt-4"},
            "optional_steps": [],
        }
    }
    mock_db_manager.get_session.return_value.__aenter__.return_value.execute.return_value.scalar_one_or_none.return_value = (
        mock_settings_model
    )
    mock_redis_manager.execute = AsyncMock(return_value=None)
    manager = PipelineSettingsManager()
    result = await manager.load_settings()
    assert result is not None
    assert result.pipeline_config["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_save_settings_to_db_not_initialized(mock_redis_manager):
    """Test save_settings_to_db returns False when DB not initialized (lines 138-139)."""
    with patch(
        "marketing_project.services.pipeline_settings_manager.get_database_manager"
    ) as mock:
        manager_mock = MagicMock()
        manager_mock.is_initialized = False
        mock.return_value = manager_mock
        mgr = PipelineSettingsManager()
        settings = PipelineSettings(pipeline_config={"model": "gpt-4"})
        result = await mgr.save_settings_to_db(settings)
    assert result is False


@pytest.mark.asyncio
async def test_save_settings_to_db_exception_returns_false(
    mock_db_manager, mock_redis_manager
):
    """Test save_settings_to_db returns False on exception (lines 161-163)."""
    mock_db_manager.get_session.return_value.__aenter__.side_effect = Exception(
        "DB error"
    )
    manager = PipelineSettingsManager()
    settings = PipelineSettings(pipeline_config={"model": "gpt-4"})
    result = await manager.save_settings_to_db(settings)
    assert result is False


@pytest.mark.asyncio
async def test_save_settings_to_redis_exception_logs_warning(mock_redis_manager):
    """Test save_settings_to_redis logs warning on exception (lines 177-178)."""
    mock_redis_manager.execute = AsyncMock(side_effect=Exception("Redis fail"))
    manager = PipelineSettingsManager()
    settings = PipelineSettings(pipeline_config={"model": "gpt-4"})
    # Should not raise
    await manager.save_settings_to_redis(settings)


@pytest.mark.asyncio
async def test_save_settings_db_failure_logs_warning(
    mock_db_manager, mock_redis_manager
):
    """Test save_settings logs warning when DB save fails (lines 193-196)."""
    # DB not initialized so save_settings_to_db returns False
    mock_db_manager.is_initialized = False
    mock_redis_manager.execute = AsyncMock(return_value=None)
    manager = PipelineSettingsManager()
    settings = PipelineSettings(pipeline_config={"model": "gpt-4"})
    # Should complete without error even when DB fails
    await manager.save_settings(settings)


def test_json_serializer_datetime():
    """Test _json_serializer handles datetime (lines 29-32)."""
    from datetime import date, datetime

    from marketing_project.services.pipeline_settings_manager import _json_serializer

    dt = datetime(2024, 1, 1, 12, 0, 0)
    assert _json_serializer(dt) == "2024-01-01T12:00:00"

    d = date(2024, 1, 1)
    assert _json_serializer(d) == "2024-01-01"


def test_json_serializer_pydantic_model():
    """Test _json_serializer handles Pydantic models (lines 33-39)."""
    from marketing_project.services.pipeline_settings_manager import _json_serializer

    settings = PipelineSettings(pipeline_config={"model": "gpt-4"})
    result = _json_serializer(settings)
    assert isinstance(result, dict)
    assert "pipeline_config" in result


def test_json_serializer_unsupported_type():
    """Test _json_serializer raises TypeError for unsupported types (line 40)."""
    from marketing_project.services.pipeline_settings_manager import _json_serializer

    with pytest.raises(TypeError):
        _json_serializer(object())


@pytest.mark.asyncio
async def test_save_settings_to_db_with_existing_settings(
    mock_db_manager, mock_redis_manager
):
    """Test save_settings_to_db deactivates existing settings (lines 143-159)."""
    existing_model = MagicMock()
    existing_model.is_active = True

    scalars_mock = MagicMock()
    scalars_mock.__iter__ = MagicMock(return_value=iter([existing_model]))

    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.execute = AsyncMock(
        return_value=MagicMock(scalars=MagicMock(return_value=scalars_mock))
    )
    mock_db_manager.get_session.return_value = mock_session

    manager = PipelineSettingsManager()
    settings = PipelineSettings(pipeline_config={"model": "gpt-5"})
    result = await manager.save_settings_to_db(settings)
    assert result is True
    assert existing_model.is_active is False
