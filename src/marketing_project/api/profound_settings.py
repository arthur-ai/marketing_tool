"""
API endpoints for Profound integration settings.

All endpoints require the 'admin' role.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from marketing_project.middleware.rbac import require_roles
from marketing_project.models.profound_models import (
    ProfoundSettingsRequest,
    ProfoundSettingsResponse,
)
from marketing_project.models.user_context import UserContext
from marketing_project.services.profound_settings_manager import (
    get_profound_settings_manager,
)

logger = logging.getLogger("marketing_project.api.profound_settings")

router = APIRouter(prefix="/settings/profound", tags=["Profound Settings"])


class ProfoundTestResponse(BaseModel):
    success: bool
    message: str
    personas_count: int = 0


@router.get("", response_model=ProfoundSettingsResponse)
async def get_profound_settings(
    user: UserContext = Depends(require_roles(["admin"])),
):
    """Get Profound integration settings (never returns raw API key)."""
    mgr = get_profound_settings_manager()
    return await mgr.to_response()


@router.put("", response_model=ProfoundSettingsResponse)
async def update_profound_settings(
    req: ProfoundSettingsRequest,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """Create or update Profound integration settings."""
    mgr = get_profound_settings_manager()
    await mgr.upsert(req)
    return await mgr.to_response()


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profound_settings(
    user: UserContext = Depends(require_roles(["admin"])),
):
    """Remove Profound settings. Pipeline will fall back to default keyword extraction."""
    mgr = get_profound_settings_manager()
    deleted = await mgr.delete()
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Profound settings configured.",
        )


@router.post("/test", response_model=ProfoundTestResponse)
async def test_profound_connection(
    user: UserContext = Depends(require_roles(["admin"])),
):
    """Test the Profound API connection using the stored (or env-var) credentials."""
    mgr = get_profound_settings_manager()
    api_key, default_category_id = await mgr.get_credentials()

    if not api_key:
        return ProfoundTestResponse(
            success=False,
            message="No Profound API key configured. Add one via PUT /settings/profound.",
        )

    if not default_category_id:
        return ProfoundTestResponse(
            success=False,
            message=(
                "API key is set but no default_category_id configured. "
                "Add a category ID to test persona fetching."
            ),
        )

    try:
        from marketing_project.services.profound_client import ProfoundClient

        client = ProfoundClient(api_key=api_key)
        personas = await client.get_category_personas(default_category_id)
        return ProfoundTestResponse(
            success=True,
            message=f"Connection successful. Fetched {len(personas)} personas for category {default_category_id}.",
            personas_count=len(personas),
        )
    except Exception as exc:
        logger.warning("Profound test connection failed: %s", exc)
        return ProfoundTestResponse(
            success=False,
            message=f"Connection failed: {exc}",
        )
