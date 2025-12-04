"""
Pipeline Settings Manager Service.

Manages pipeline configuration settings storage and retrieval.
Settings are persisted in PostgreSQL database (source of truth) with Redis as cache layer.
This allows settings to be shared between API and worker processes with persistence.
"""

import json
import logging
from typing import Any, Dict, Optional

import redis.asyncio as redis
from pydantic import BaseModel
from sqlalchemy import select

from marketing_project.models.db_models import PipelineSettingsModel
from marketing_project.services.database import get_database_manager
from marketing_project.services.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)

# Redis key for storing pipeline settings
PIPELINE_SETTINGS_KEY = "pipeline:settings"


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for datetime and other non-serializable objects."""
    from datetime import date, datetime

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # Handle Pydantic BaseModel instances
    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump(mode="json")
        except (TypeError, ValueError):
            # Fallback to regular model_dump if mode='json' fails
            return obj.model_dump()
    raise TypeError(f"Type {type(obj)} not serializable")


class PipelineSettings(BaseModel):
    """Pipeline settings model."""

    pipeline_config: Dict[str, Any] = {}
    optional_steps: list[str] = []
    retry_strategy: Optional[Dict[str, Any]] = None


class PipelineSettingsManager:
    """
    Manages pipeline settings storage and retrieval.

    Settings are persisted in PostgreSQL database (source of truth) with Redis as cache layer.
    This allows settings to be shared between API and worker processes with persistence.
    """

    def __init__(self):
        self._redis_manager = get_redis_manager()

    async def get_redis(self) -> Optional[redis.Redis]:
        """Get Redis client from RedisManager."""
        try:
            return await self._redis_manager.get_redis()
        except Exception as e:
            logger.warning(
                f"Failed to get Redis connection: {e}. Settings will only be stored in memory."
            )
            return None

    async def load_settings_from_db(self) -> Optional[PipelineSettings]:
        """Load settings from database (source of truth)."""
        try:
            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                return None

            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(PipelineSettingsModel)
                    .where(PipelineSettingsModel.is_active == True)
                    .order_by(PipelineSettingsModel.id.desc())
                    .limit(1)
                )
                settings_model = result.scalar_one_or_none()

                if settings_model:
                    settings_dict = settings_model.to_dict()
                    settings_data = settings_dict.get("settings_data", {})
                    return PipelineSettings(**settings_data)
        except Exception as e:
            logger.warning(f"Failed to load settings from database: {e}")
        return None

    async def load_settings_from_redis(self) -> Optional[PipelineSettings]:
        """Load settings from Redis (cache fallback)."""
        try:

            async def get_operation(redis_client: redis.Redis):
                return await redis_client.get(PIPELINE_SETTINGS_KEY)

            settings_json = await self._redis_manager.execute(get_operation)
            if settings_json:
                settings_dict = json.loads(settings_json)
                return PipelineSettings(**settings_dict)
        except Exception as e:
            logger.warning(f"Failed to load settings from Redis: {e}")
        return None

    async def load_settings(self) -> Optional[PipelineSettings]:
        """
        Load settings from database (source of truth), fallback to Redis cache.

        Returns:
            PipelineSettings if found, None otherwise
        """
        # Try database first (source of truth)
        settings = await self.load_settings_from_db()
        if settings:
            # Update Redis cache
            await self.save_settings_to_redis(settings)
            return settings

        # Fallback to Redis cache
        settings = await self.load_settings_from_redis()
        return settings

    async def save_settings_to_db(self, settings: PipelineSettings) -> bool:
        """
        Save settings to database (source of truth).

        Returns:
            True if successful, False otherwise
        """
        try:
            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                return False

            async with db_manager.get_session() as session:
                # Deactivate all existing active settings
                result = await session.execute(
                    select(PipelineSettingsModel).where(
                        PipelineSettingsModel.is_active == True
                    )
                )
                for existing in result.scalars():
                    existing.is_active = False

                # Create new active settings
                settings_data = settings.model_dump(mode="json")
                settings_model = PipelineSettingsModel(
                    settings_data=settings_data, is_active=True
                )
                session.add(settings_model)

                await session.commit()
                logger.info("Saved pipeline settings to database")
                return True
        except Exception as e:
            logger.error(f"Failed to save settings to database: {e}")
            return False

    async def save_settings_to_redis(self, settings: PipelineSettings):
        """Save settings to Redis (cache)."""
        try:
            settings_json = json.dumps(
                settings.model_dump(mode="json"), default=_json_serializer
            )

            async def set_operation(redis_client: redis.Redis):
                await redis_client.set(PIPELINE_SETTINGS_KEY, settings_json)

            await self._redis_manager.execute(set_operation)
            logger.debug("Saved pipeline settings to Redis cache")
        except Exception as e:
            logger.warning(f"Failed to save settings to Redis cache: {e}")

    async def save_settings(self, settings: PipelineSettings):
        """
        Save settings to database (source of truth) and update Redis cache.

        Args:
            settings: PipelineSettings to save
        """
        # Save to database first (source of truth)
        db_success = await self.save_settings_to_db(settings)

        # Update Redis cache regardless of DB success (for backward compatibility)
        await self.save_settings_to_redis(settings)

        if not db_success:
            logger.warning(
                "Settings saved to Redis cache only (database not available)"
            )


# Singleton instance
_pipeline_settings_manager: Optional[PipelineSettingsManager] = None


def get_pipeline_settings_manager() -> PipelineSettingsManager:
    """Get singleton instance of PipelineSettingsManager."""
    global _pipeline_settings_manager
    if _pipeline_settings_manager is None:
        _pipeline_settings_manager = PipelineSettingsManager()
    return _pipeline_settings_manager
