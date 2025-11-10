"""
Design Kit Manager Service.

Manages design kit configuration storage and retrieval.
Uses Redis for persistence and sharing between API and worker processes.
Design Kit uses Internal Docs configuration to generate interlinking information.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from sqlalchemy import select

from marketing_project.models.db_models import DesignKitConfigModel
from marketing_project.models.design_kit_config import DesignKitConfig
from marketing_project.services.database import get_database_manager
from marketing_project.services.internal_docs_manager import get_internal_docs_manager
from marketing_project.services.redis_manager import get_redis_manager
from marketing_project.services.scanned_document_db import get_scanned_document_db

logger = logging.getLogger("marketing_project.services.design_kit_manager")

# Redis key prefix
DESIGN_KIT_CONFIG_KEY = "design_kit:config"
DESIGN_KIT_VERSIONS_KEY = "design_kit:versions"
DESIGN_KIT_ACTIVE_KEY = "design_kit:active"


class DesignKitManager:
    """
    Manages design kit configuration storage and retrieval.

    Configuration is stored in PostgreSQL database (source of truth) with Redis as cache layer.
    This allows configuration to be shared between API and worker processes with persistence.
    Design Kit uses Internal Docs configuration for interlinking information.
    """

    def __init__(self):
        """Initialize the design kit manager."""
        self._redis_manager = get_redis_manager()

    async def get_redis(self) -> Optional[redis.Redis]:
        """Get Redis client from RedisManager."""
        try:
            return await self._redis_manager.get_redis()
        except Exception as e:
            logger.warning(
                f"Failed to get Redis connection: {e}. Configuration will only be stored in memory."
            )
            return None

    async def get_active_config_from_db(self) -> Optional[DesignKitConfig]:
        """Get active configuration from database (source of truth)."""
        try:
            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                return None

            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(DesignKitConfigModel)
                    .where(DesignKitConfigModel.is_active == True)
                    .order_by(DesignKitConfigModel.id.desc())
                    .limit(1)
                )
                config_model = result.scalar_one_or_none()

                if config_model:
                    config_dict = config_model.to_dict()["config_data"]
                    config = DesignKitConfig(**config_dict)
                    # Update Redis cache
                    await self._save_config_to_redis(config)
                    return config
        except Exception as e:
            logger.warning(f"Failed to load active config from database: {e}")
        return None

    async def get_active_config_from_redis(self) -> Optional[DesignKitConfig]:
        """Get active configuration from Redis (cache fallback)."""
        try:
            # Get active version ID
            async def get_active_operation(redis_client: redis.Redis):
                return await redis_client.get(DESIGN_KIT_ACTIVE_KEY)

            active_version = await self._redis_manager.execute(get_active_operation)
            if not active_version:
                return None

            # Get configuration for active version
            config_key = f"{DESIGN_KIT_CONFIG_KEY}:{active_version}"

            async def get_config_operation(redis_client: redis.Redis):
                return await redis_client.get(config_key)

            config_data = await self._redis_manager.execute(get_config_operation)

            if not config_data:
                return None

            config_dict = json.loads(config_data)
            return DesignKitConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Failed to load active config from Redis: {e}")
        return None

    async def get_active_config(self) -> Optional[DesignKitConfig]:
        """
        Get the currently active design kit configuration.
        Loads from database (source of truth), falls back to Redis cache.

        Returns:
            DesignKitConfig if found, None otherwise
        """
        # Try database first (source of truth)
        config = await self.get_active_config_from_db()
        if config:
            # Enrich with interlinking rules from internal docs if available
            await self._enrich_with_internal_docs(config)
            return config

        # Fallback to Redis cache
        config = await self.get_active_config_from_redis()
        if config:
            # Enrich with interlinking rules from internal docs if available
            await self._enrich_with_internal_docs(config)
            return config

        logger.warning("No active design kit configuration found")
        return None

    async def get_config_by_version_from_db(
        self, version: str
    ) -> Optional[DesignKitConfig]:
        """Get configuration by version from database."""
        try:
            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                return None

            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(DesignKitConfigModel).where(
                        DesignKitConfigModel.version == version
                    )
                )
                config_model = result.scalar_one_or_none()

                if config_model:
                    config_dict = config_model.to_dict()["config_data"]
                    return DesignKitConfig(**config_dict)
        except Exception as e:
            logger.warning(
                f"Failed to load config version {version} from database: {e}"
            )
        return None

    async def get_config_by_version(self, version: str) -> Optional[DesignKitConfig]:
        """
        Get design kit configuration by version.
        Loads from database (source of truth), falls back to Redis cache.

        Args:
            version: Version identifier

        Returns:
            DesignKitConfig if found, None otherwise
        """
        # Try database first
        config = await self.get_config_by_version_from_db(version)
        if config:
            await self._enrich_with_internal_docs(config)
            return config

        # Fallback to Redis
        try:
            config_key = f"{DESIGN_KIT_CONFIG_KEY}:{version}"

            async def get_config_operation(redis_client: redis.Redis):
                return await redis_client.get(config_key)

            config_data = await self._redis_manager.execute(get_config_operation)

            if config_data:
                config_dict = json.loads(config_data)
                config = DesignKitConfig(**config_dict)
                await self._enrich_with_internal_docs(config)
                return config
        except Exception as e:
            logger.error(f"Error loading design kit config version {version}: {e}")

        return None

    async def get_config_by_content_type(
        self, content_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get content-type-specific design kit configuration.

        Args:
            content_type: Content type (blog_post, press_release, case_study)

        Returns:
            Merged configuration dict for the content type, or None if no active config
        """
        try:
            config = await self.get_active_config()
            if not config:
                return None

            return config.get_content_type_config(content_type)
        except Exception as e:
            logger.error(
                f"Error getting design kit config for content type {content_type}: {e}"
            )
            return None

    async def _enrich_with_internal_docs(self, config: DesignKitConfig):
        """
        Enrich design kit config with interlinking rules from internal docs.
        Uses both Redis config and database for comprehensive information.

        Args:
            config: DesignKitConfig to enrich
        """
        try:
            # First, get from Redis config
            internal_docs_manager = await get_internal_docs_manager()
            internal_docs_config = await internal_docs_manager.get_active_config()

            if internal_docs_config:
                # Update interlinking rules from internal docs config
                config.commonly_referenced_pages = (
                    internal_docs_config.commonly_referenced_pages
                )
                config.commonly_referenced_categories = (
                    internal_docs_config.commonly_referenced_categories
                )
                config.anchor_phrasing_patterns = (
                    internal_docs_config.anchor_phrasing_patterns
                )
                logger.debug(
                    "Enriched design kit config with internal docs interlinking rules from config"
                )

            # Also enrich from database for more comprehensive data
            try:
                db = get_scanned_document_db()

                # Get commonly referenced pages from database (more accurate)
                db_pages = db.get_commonly_referenced_pages(min_links=2)
                if db_pages:
                    # Merge with config pages, avoiding duplicates
                    existing_pages = set(config.commonly_referenced_pages or [])
                    existing_pages.update(db_pages)
                    config.commonly_referenced_pages = list(existing_pages)

                # Get anchor text patterns from database
                db_patterns = db.get_anchor_text_patterns()
                if db_patterns:
                    # Merge with config patterns, avoiding duplicates
                    existing_patterns = set(config.anchor_phrasing_patterns or [])
                    existing_patterns.update(db_patterns[:20])  # Top 20 patterns
                    config.anchor_phrasing_patterns = list(existing_patterns)

                # Get categories from database documents
                all_docs = db.get_all_active_documents()
                db_categories = set()
                for doc in all_docs:
                    db_categories.update(doc.metadata.categories)

                if db_categories:
                    # Merge with config categories
                    existing_categories = set(
                        config.commonly_referenced_categories or []
                    )
                    existing_categories.update(db_categories)
                    config.commonly_referenced_categories = list(existing_categories)

                logger.debug("Enriched design kit config with database information")
            except Exception as e:
                logger.warning(f"Failed to enrich design kit with database info: {e}")

        except Exception as e:
            logger.warning(f"Failed to enrich design kit with internal docs: {e}")

    async def _save_config_to_redis(self, config: DesignKitConfig):
        """Save configuration to Redis cache."""
        try:
            config_key = f"{DESIGN_KIT_CONFIG_KEY}:{config.version}"
            config_json = config.model_dump_json()

            async def save_operation(redis_client: redis.Redis):
                async with redis_client.pipeline() as pipe:
                    pipe.set(config_key, config_json)
                    pipe.sadd(DESIGN_KIT_VERSIONS_KEY, config.version)
                    if config.is_active:
                        pipe.set(DESIGN_KIT_ACTIVE_KEY, config.version)
                    await pipe.execute()

            await self._redis_manager.execute(save_operation)
        except Exception as e:
            logger.warning(f"Failed to save config to Redis cache: {e}")

    async def save_config_to_db(
        self, config: DesignKitConfig, set_active: bool = True
    ) -> bool:
        """
        Save configuration to database (source of truth).

        Returns:
            True if successful, False otherwise
        """
        try:
            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                return False

            async with db_manager.get_session() as session:
                # Get existing config by version
                result = await session.execute(
                    select(DesignKitConfigModel).where(
                        DesignKitConfigModel.version == config.version
                    )
                )
                config_model = result.scalar_one_or_none()

                if set_active:
                    # Deactivate other active configs
                    other_active_result = await session.execute(
                        select(DesignKitConfigModel).where(
                            DesignKitConfigModel.is_active == True
                        )
                    )
                    for other in other_active_result.scalars():
                        if other.version != config.version:
                            other.is_active = False

                config.is_active = set_active
                config_data = config.model_dump(mode="json")

                if config_model:
                    # Update existing
                    config_model.config_data = config_data
                    config_model.is_active = config.is_active
                    config_model.updated_at = datetime.utcnow()
                else:
                    # Create new
                    config_model = DesignKitConfigModel(
                        version=config.version,
                        config_data=config_data,
                        is_active=config.is_active,
                    )
                    session.add(config_model)

                await session.commit()
                logger.info(
                    f"Saved design kit config version {config.version} to database"
                )
                return True
        except Exception as e:
            logger.error(f"Failed to save config to database: {e}")
            return False

    async def save_config(
        self, config: DesignKitConfig, set_active: bool = True
    ) -> bool:
        """
        Save design kit configuration.
        Saves to database (source of truth) and updates Redis cache.

        Args:
            config: DesignKitConfig to save
            set_active: Whether to set this version as active

        Returns:
            True if successful, False otherwise
        """
        try:
            # Update timestamp
            config.updated_at = datetime.utcnow()

            # Enrich with internal docs before saving
            await self._enrich_with_internal_docs(config)

            # Save to database first (source of truth)
            db_success = await self.save_config_to_db(config, set_active)

            # Update Redis cache
            await self._save_config_to_redis(config)

            if db_success:
                logger.info(f"Saved design kit config version {config.version}")
                return True
            else:
                logger.warning(
                    f"Config saved to Redis cache only (database not available)"
                )
                return True  # Still return True for backward compatibility

        except Exception as e:
            logger.error(
                f"Error saving design kit config version {config.version}: {type(e).__name__}: {e}",
                extra={
                    "operation": "save_config",
                    "config_version": config.version,
                    "set_active": set_active,
                    "error_type": type(e).__name__,
                },
            )
            return False

    async def list_versions(self) -> List[str]:
        """
        List all available configuration versions.

        Returns:
            List of version identifiers
        """
        try:

            async def smembers_operation(redis_client: redis.Redis):
                return await redis_client.smembers(DESIGN_KIT_VERSIONS_KEY)

            versions = await self._redis_manager.execute(smembers_operation)
            return list(versions)
        except Exception as e:
            logger.error(f"Error listing design kit versions: {e}")
            return []

    async def activate_version(self, version: str) -> bool:
        """
        Activate a specific configuration version.

        Args:
            version: Version identifier to activate

        Returns:
            True if successful, False otherwise
        """
        try:
            config = await self.get_config_by_version(version)
            if not config:
                logger.error(f"Version {version} not found")
                return False

            # Activate version using pipeline for batch operations
            config_key = f"{DESIGN_KIT_CONFIG_KEY}:{version}"

            async def activate_operation(redis_client: redis.Redis):
                # Get old active version
                old_active = await redis_client.get(DESIGN_KIT_ACTIVE_KEY)

                async with redis_client.pipeline() as pipe:
                    # Deactivate previous active version if different
                    if old_active and old_active != version:
                        old_config = await self.get_config_by_version(old_active)
                        if old_config:
                            old_config.is_active = False
                            old_config_key = f"{DESIGN_KIT_CONFIG_KEY}:{old_active}"
                            pipe.set(old_config_key, old_config.model_dump_json())

                    # Activate new version
                    config.is_active = True
                    pipe.set(DESIGN_KIT_ACTIVE_KEY, version)
                    pipe.set(config_key, config.model_dump_json())
                    await pipe.execute()

            await self._redis_manager.execute(activate_operation)

            logger.info(f"Activated design kit config version {version}")
            return True

        except Exception as e:
            logger.error(f"Error activating design kit config version {version}: {e}")
            return False

    async def generate_config_with_ai(
        self, use_internal_docs: bool = True, job_id: Optional[str] = None
    ) -> DesignKitConfig:
        """
        Generate a comprehensive design kit configuration using AI/LLM.

        Uses the DesignKitPlugin to generate sensible defaults for all design kit
        fields based on best practices and common patterns.

        Args:
            use_internal_docs: Whether to enrich with internal docs configuration
            job_id: Optional job ID for progress tracking

        Returns:
            Generated DesignKitConfig with AI-populated fields
        """
        try:
            logger.info("Generating design kit configuration using AI...")

            # Update progress if job_id provided
            if job_id:
                from marketing_project.services.job_manager import get_job_manager

                job_manager = get_job_manager()
                await job_manager.update_job_progress(
                    job_id, 35, "Initializing AI generation"
                )

            # Use DesignKitPlugin for generation
            from marketing_project.plugins.design_kit.tasks import DesignKitPlugin

            plugin = DesignKitPlugin()
            generated_config = await plugin.generate_config(
                use_internal_docs=use_internal_docs, job_id=job_id
            )

            # Set metadata
            generated_config.version = "1.0.0"
            generated_config.created_at = datetime.utcnow()
            generated_config.updated_at = datetime.utcnow()
            generated_config.is_active = True

            # Update progress
            if job_id:
                from marketing_project.services.job_manager import get_job_manager

                job_manager = get_job_manager()
                await job_manager.update_job_progress(
                    job_id, 60, "Enriching with internal docs"
                )

            # Enrich with internal docs if requested
            if use_internal_docs:
                await self._enrich_with_internal_docs(generated_config)

            logger.info("Successfully generated design kit configuration using AI")
            return generated_config

        except Exception as e:
            logger.error(
                f"Error generating design kit config with AI: {e}", exc_info=True
            )
            # Fallback to basic default config
            logger.info("Falling back to basic default configuration")
            return DesignKitConfig(
                version="1.0.0",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_active=True,
            )

    async def cleanup(self):
        """Cleanup Redis connections."""
        # RedisManager cleanup is handled globally
        # This method is here for consistency with other managers
        pass


# Singleton instance
_design_kit_manager: Optional[DesignKitManager] = None


async def get_design_kit_manager() -> DesignKitManager:
    """Get or create the design kit manager singleton."""
    global _design_kit_manager
    if _design_kit_manager is None:
        _design_kit_manager = DesignKitManager()
    return _design_kit_manager
