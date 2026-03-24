"""
Pydantic request/response schemas for Profound settings.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ProfoundSettingsRequest(BaseModel):
    """Request body for updating Profound settings."""

    is_enabled: bool = True
    api_key: Optional[str] = None  # None means "keep existing"
    default_category_id: Optional[str] = None


class ProfoundSettingsResponse(BaseModel):
    """Response body for Profound settings (never exposes raw api_key)."""

    is_enabled: bool
    has_api_key: bool
    default_category_id: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
