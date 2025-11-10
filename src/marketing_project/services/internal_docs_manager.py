"""
Internal Docs Manager Service.

Manages internal documentation configuration storage and retrieval.
Uses Redis for persistence and sharing between API and worker processes.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from sqlalchemy import select

from marketing_project.models.db_models import InternalDocsConfigModel
from marketing_project.models.internal_docs_config import (
    InternalDocsConfig,
    ScannedDocument,
)
from marketing_project.services.database import get_database_manager
from marketing_project.services.redis_manager import get_redis_manager

logger = logging.getLogger("marketing_project.services.internal_docs_manager")

# Redis key prefix
INTERNAL_DOCS_CONFIG_KEY = "internal_docs:config"
INTERNAL_DOCS_VERSIONS_KEY = "internal_docs:versions"
INTERNAL_DOCS_ACTIVE_KEY = "internal_docs:active"


class InternalDocsManager:
    """
    Manages internal docs configuration storage and retrieval.

    Configuration is stored in PostgreSQL database (source of truth) with Redis as cache layer.
    This allows configuration to be shared between API and worker processes with persistence.
    """

    def __init__(self):
        """Initialize the internal docs manager."""
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

    async def get_active_config_from_db(self) -> Optional[InternalDocsConfig]:
        """Get active configuration from database (source of truth)."""
        try:
            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                return None

            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(InternalDocsConfigModel)
                    .where(InternalDocsConfigModel.is_active == True)
                    .order_by(InternalDocsConfigModel.id.desc())
                    .limit(1)
                )
                config_model = result.scalar_one_or_none()

                if config_model:
                    config_dict = config_model.to_dict()["config_data"]
                    config = InternalDocsConfig(**config_dict)
                    # Update Redis cache
                    await self._save_config_to_redis(config)
                    return config
        except Exception as e:
            logger.warning(f"Failed to load active config from database: {e}")
        return None

    async def get_active_config_from_redis(self) -> Optional[InternalDocsConfig]:
        """Get active configuration from Redis (cache fallback)."""
        try:

            async def get_active_operation(redis_client: redis.Redis):
                return await redis_client.get(INTERNAL_DOCS_ACTIVE_KEY)

            active_version = await self._redis_manager.execute(get_active_operation)
            if not active_version:
                return None

            config_key = f"{INTERNAL_DOCS_CONFIG_KEY}:{active_version}"

            async def get_config_operation(redis_client: redis.Redis):
                return await redis_client.get(config_key)

            config_data = await self._redis_manager.execute(get_config_operation)

            if not config_data:
                return None

            config_dict = json.loads(config_data)
            return InternalDocsConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Failed to load active config from Redis: {e}")
        return None

    async def get_active_config(self) -> Optional[InternalDocsConfig]:
        """
        Get the currently active internal docs configuration.
        Loads from database (source of truth), falls back to Redis cache.

        Returns:
            InternalDocsConfig if found, None otherwise
        """
        # Try database first (source of truth)
        config = await self.get_active_config_from_db()
        if config:
            return config

        # Fallback to Redis cache
        config = await self.get_active_config_from_redis()
        if config:
            return config

        logger.warning("No active internal docs configuration found")
        return None

    async def get_config_by_version_from_db(
        self, version: str
    ) -> Optional[InternalDocsConfig]:
        """Get configuration by version from database."""
        try:
            db_manager = get_database_manager()
            if not db_manager.is_initialized:
                return None

            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(InternalDocsConfigModel).where(
                        InternalDocsConfigModel.version == version
                    )
                )
                config_model = result.scalar_one_or_none()

                if config_model:
                    config_dict = config_model.to_dict()["config_data"]
                    return InternalDocsConfig(**config_dict)
        except Exception as e:
            logger.warning(
                f"Failed to load config version {version} from database: {e}"
            )
        return None

    async def get_config_by_version(self, version: str) -> Optional[InternalDocsConfig]:
        """
        Get internal docs configuration by version.
        Loads from database (source of truth), falls back to Redis cache.

        Args:
            version: Version identifier

        Returns:
            InternalDocsConfig if found, None otherwise
        """
        # Try database first
        config = await self.get_config_by_version_from_db(version)
        if config:
            return config

        # Fallback to Redis
        try:
            config_key = f"{INTERNAL_DOCS_CONFIG_KEY}:{version}"

            async def get_config_operation(redis_client: redis.Redis):
                return await redis_client.get(config_key)

            config_data = await self._redis_manager.execute(get_config_operation)

            if config_data:
                config_dict = json.loads(config_data)
                return InternalDocsConfig(**config_dict)
        except Exception as e:
            logger.error(f"Error loading internal docs config version {version}: {e}")

        return None

    async def _save_config_to_redis(self, config: InternalDocsConfig):
        """Save configuration to Redis cache."""
        try:
            config_key = f"{INTERNAL_DOCS_CONFIG_KEY}:{config.version}"
            config_json = config.model_dump_json()

            async def save_operation(redis_client: redis.Redis):
                async with redis_client.pipeline() as pipe:
                    pipe.set(config_key, config_json)
                    pipe.sadd(INTERNAL_DOCS_VERSIONS_KEY, config.version)
                    if config.is_active:
                        pipe.set(INTERNAL_DOCS_ACTIVE_KEY, config.version)
                    await pipe.execute()

            await self._redis_manager.execute(save_operation)
        except Exception as e:
            logger.warning(f"Failed to save config to Redis cache: {e}")

    async def save_config_to_db(
        self, config: InternalDocsConfig, set_active: bool = True
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
                    select(InternalDocsConfigModel).where(
                        InternalDocsConfigModel.version == config.version
                    )
                )
                config_model = result.scalar_one_or_none()

                if set_active:
                    # Deactivate other active configs
                    other_active_result = await session.execute(
                        select(InternalDocsConfigModel).where(
                            InternalDocsConfigModel.is_active == True
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
                else:
                    # Create new
                    config_model = InternalDocsConfigModel(
                        version=config.version,
                        config_data=config_data,
                        is_active=config.is_active,
                    )
                    session.add(config_model)

                await session.commit()
                logger.info(
                    f"Saved internal docs config version {config.version} to database"
                )
                return True
        except Exception as e:
            logger.error(f"Failed to save config to database: {e}")
            return False

    async def save_config(
        self, config: InternalDocsConfig, set_active: bool = True
    ) -> bool:
        """
        Save internal docs configuration.
        Saves to database (source of truth) and updates Redis cache.

        Always updates the existing active config if one exists, or creates a new one.
        Only one active config is allowed at a time.

        Args:
            config: InternalDocsConfig to save
            set_active: Whether to set this version as active (always True for single config)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if there's an existing active config
            existing_active = await self.get_active_config()

            if existing_active:
                # Update existing config instead of creating a new one
                # Preserve the original version and created_at
                config.version = existing_active.version
                config.created_at = existing_active.created_at
                logger.info(
                    f"Updating existing internal docs config (version {config.version})"
                )
            else:
                # No existing config, use default version
                if not config.version or config.version == "1.0.0":
                    config.version = "1.0.0"
                if not config.created_at:
                    config.created_at = datetime.utcnow()
                logger.info(
                    f"Creating new internal docs config (version {config.version})"
                )

            # Update timestamp
            config.updated_at = datetime.utcnow()
            config.is_active = True  # Always active since there's only one config

            # Save to database first (source of truth)
            db_success = await self.save_config_to_db(config, set_active)

            # Update Redis cache
            await self._save_config_to_redis(config)

            if db_success:
                logger.info(f"Saved internal docs config version {config.version}")
                return True
            else:
                logger.warning(
                    f"Config saved to Redis cache only (database not available)"
                )
                return True  # Still return True for backward compatibility

        except Exception as e:
            logger.error(
                f"Error saving internal docs config version {config.version}: {type(e).__name__}: {e}",
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
                return await redis_client.smembers(INTERNAL_DOCS_VERSIONS_KEY)

            versions = await self._redis_manager.execute(smembers_operation)
            return list(versions)
        except Exception as e:
            logger.error(f"Error listing internal docs versions: {e}")
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
            config_key = f"{INTERNAL_DOCS_CONFIG_KEY}:{version}"

            async def activate_operation(redis_client: redis.Redis):
                # Get old active version
                old_active = await redis_client.get(INTERNAL_DOCS_ACTIVE_KEY)

                async with redis_client.pipeline() as pipe:
                    # Deactivate previous active version if different
                    if old_active and old_active != version:
                        old_config = await self.get_config_by_version(old_active)
                        if old_config:
                            old_config.is_active = False
                            old_config_key = f"{INTERNAL_DOCS_CONFIG_KEY}:{old_active}"
                            pipe.set(old_config_key, old_config.model_dump_json())

                    # Activate new version
                    config.is_active = True
                    pipe.set(INTERNAL_DOCS_ACTIVE_KEY, version)
                    pipe.set(config_key, config.model_dump_json())
                    await pipe.execute()

            await self._redis_manager.execute(activate_operation)

            logger.info(f"Activated internal docs config version {version}")
            return True

        except Exception as e:
            logger.error(
                f"Error activating internal docs config version {version}: {e}"
            )
            return False

    async def merge_scan_results(
        self, config: InternalDocsConfig, scanned_docs: List[ScannedDocument]
    ) -> InternalDocsConfig:
        """
        Merge scanned documents with existing configuration.

        Args:
            config: Existing InternalDocsConfig
            scanned_docs: List of newly scanned documents

        Returns:
            Updated InternalDocsConfig with merged documents
        """
        # Create a set of existing URLs for quick lookup
        existing_urls = {doc.url for doc in config.scanned_documents}

        # Add new documents that don't already exist
        new_docs = [doc for doc in scanned_docs if doc.url not in existing_urls]
        config.scanned_documents.extend(new_docs)

        # Update timestamp
        config.updated_at = datetime.utcnow()

        logger.info(
            f"Merged {len(new_docs)} new documents into config (total: {len(config.scanned_documents)})"
        )
        return config

    async def add_document(
        self, config: InternalDocsConfig, doc: ScannedDocument
    ) -> InternalDocsConfig:
        """
        Add a single document to the configuration.

        Args:
            config: InternalDocsConfig to update
            doc: ScannedDocument to add

        Returns:
            Updated InternalDocsConfig
        """
        # Check if document already exists
        existing_urls = {d.url for d in config.scanned_documents}
        if doc.url in existing_urls:
            # Update existing document
            for i, existing_doc in enumerate(config.scanned_documents):
                if existing_doc.url == doc.url:
                    config.scanned_documents[i] = doc
                    break
        else:
            # Add new document
            config.scanned_documents.append(doc)

        config.updated_at = datetime.utcnow()
        logger.info(f"Added/updated document: {doc.title} ({doc.url})")
        return config

    async def remove_document(
        self, config: InternalDocsConfig, doc_url: str
    ) -> InternalDocsConfig:
        """
        Remove a document from the configuration by URL.

        Args:
            config: InternalDocsConfig to update
            doc_url: URL of document to remove

        Returns:
            Updated InternalDocsConfig
        """
        original_count = len(config.scanned_documents)
        config.scanned_documents = [
            doc for doc in config.scanned_documents if doc.url != doc_url
        ]

        removed_count = original_count - len(config.scanned_documents)
        if removed_count > 0:
            config.updated_at = datetime.utcnow()
            logger.info(f"Removed document: {doc_url}")
        else:
            logger.warning(f"Document not found for removal: {doc_url}")

        return config

    async def cleanup(self):
        """Cleanup Redis connections."""
        # RedisManager cleanup is handled globally
        # This method is here for consistency with other managers
        pass


# Singleton instance
_internal_docs_manager: Optional[InternalDocsManager] = None


async def get_internal_docs_manager() -> InternalDocsManager:
    """Get or create the internal docs manager singleton."""
    global _internal_docs_manager
    if _internal_docs_manager is None:
        _internal_docs_manager = InternalDocsManager()
    return _internal_docs_manager
