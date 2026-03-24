"""
Profound Settings Manager.

Loads and stores Profound API settings from the database (singleton row).
get_profound_api_key() is the primary entry point used by ProfoundClient — it
returns the API key from the DB first, falling back to the PROFOUND_API_KEY
environment variable if no DB record exists.
"""

import logging
import os
from typing import Optional, Tuple

from sqlalchemy import select

from marketing_project.models.db_models import ProfoundSettingsModel
from marketing_project.models.profound_models import (
    ProfoundSettingsRequest,
    ProfoundSettingsResponse,
)
from marketing_project.services.database import get_database_manager

logger = logging.getLogger(__name__)

_SINGLETON_ID = 1  # We keep a single row; id=1 is canonical.


class ProfoundSettingsManager:
    """DB-backed manager for Profound integration settings."""

    async def get(self) -> Optional[ProfoundSettingsModel]:
        """Fetch the settings row from the DB (may return None if never configured)."""
        db = get_database_manager()
        try:
            async with db.get_session() as session:
                result = await session.execute(select(ProfoundSettingsModel))
                return result.scalars().first()
        except Exception as exc:
            exc_name = type(exc).__name__
            if "InvalidToken" in exc_name:
                logger.error(
                    "Decryption failed for Profound settings — ENCRYPTION_KEY may have "
                    "been rotated. Re-save settings via the admin UI. Error: %s",
                    exc,
                )
            else:
                logger.warning("Failed to fetch Profound settings: %s", exc)
            return None

    async def upsert(self, req: ProfoundSettingsRequest) -> None:
        """Create or update Profound settings."""
        db = get_database_manager()
        async with db.get_session() as session:
            result = await session.execute(select(ProfoundSettingsModel))
            record = result.scalars().first()

            if record is None:
                record = ProfoundSettingsModel()
                session.add(record)

            record.is_enabled = req.is_enabled
            if req.api_key is not None:
                record.api_key = req.api_key
            if req.default_category_id is not None:
                record.default_category_id = req.default_category_id

    async def delete(self) -> bool:
        """Delete Profound settings row. Returns True if a row existed."""
        db = get_database_manager()
        async with db.get_session() as session:
            result = await session.execute(select(ProfoundSettingsModel))
            record = result.scalars().first()
            if record is None:
                return False
            await session.delete(record)
            return True

    async def get_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (api_key, default_category_id) for use by ProfoundClient.

        Resolution order:
          1. DB record (if is_enabled=True and api_key is set)
          2. Environment variables (PROFOUND_API_KEY, PROFOUND_CATEGORY_ID)
          3. (None, None) — persona injection disabled; pipeline uses default keywords
        """
        record = await self.get()
        if record is not None and record.is_enabled and record.api_key:
            return record.api_key, record.default_category_id

        # Fall back to environment variables
        env_key = os.getenv("PROFOUND_API_KEY", "") or None
        env_cat = os.getenv("PROFOUND_CATEGORY_ID", "") or None
        return env_key, env_cat

    async def to_response(self) -> ProfoundSettingsResponse:
        """Return a safe response object (never exposes the raw api_key)."""
        record = await self.get()
        if record is None:
            return ProfoundSettingsResponse(
                is_enabled=False,
                has_api_key=False,
                default_category_id=None,
                created_at=None,
                updated_at=None,
            )
        return ProfoundSettingsResponse(
            is_enabled=record.is_enabled,
            has_api_key=bool(record.api_key),
            default_category_id=record.default_category_id,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


_manager: Optional[ProfoundSettingsManager] = None


def get_profound_settings_manager() -> ProfoundSettingsManager:
    global _manager
    if _manager is None:
        _manager = ProfoundSettingsManager()
    return _manager
