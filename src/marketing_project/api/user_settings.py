"""
Per-user settings API endpoints.

Users manage their own pipeline/approval overrides via /users/me/settings.
Admins can read and update any user's settings via /users/{user_id}/settings.

Resolved settings (global defaults merged with user overrides) are computed
at job creation time and stored in job metadata for stable execution.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select

from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.middleware.rbac import require_roles
from marketing_project.models.db_models import (
    ApprovalSettingsModel,
    PipelineSettingsModel,
    UserSettingsModel,
)
from marketing_project.models.user_context import UserContext
from marketing_project.models.user_settings_models import (
    ResolvedUserSettings,
    UserSettings,
    UserSettingsResponse,
)
from marketing_project.services.database import get_database_manager

logger = logging.getLogger("marketing_project.api.user_settings")

router = APIRouter()

# Global setting defaults (used when neither global nor user settings exist)
_DEFAULT_MODEL = "gpt-5.1"
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_OPTIONAL_STEPS = ["suggested_links", "design_kit"]
_DEFAULT_APPROVAL_AGENTS = [
    "content_pipeline",
    "article_generation",
    "marketing_brief",
    "seo_keywords",
    "seo_optimization",
    "content_formatting",
    "transcript_preprocessing_approval",
    "blog_post_preprocessing_approval",
    "suggested_links",
    "social_media_marketing_brief",
    "social_media_angle_hook",
    "social_media_post_generation",
]


async def resolve_user_settings(user_id: str) -> ResolvedUserSettings:
    """
    Merge global pipeline/approval settings with per-user overrides.

    Resolution order (higher = wins):
      1. Hard-coded fallback defaults
      2. Active global PipelineSettingsModel / ApprovalSettingsModel rows
      3. UserSettingsModel row for this user_id (if any)

    The result is a fully-specified ResolvedUserSettings suitable for
    storing in job.metadata["user_settings"] at job creation time.
    """
    db_manager = get_database_manager()

    # ── Global defaults ──────────────────────────────────────────────────────
    global_model = _DEFAULT_MODEL
    global_temperature = _DEFAULT_TEMPERATURE
    global_disabled_steps: list = []
    global_require_approval = False
    global_approval_agents = list(_DEFAULT_APPROVAL_AGENTS)
    global_auto_approve_threshold = None
    global_timeout_seconds = None

    if db_manager.is_initialized:
        try:
            async with db_manager.get_session() as session:
                # Global pipeline settings
                pipeline_row = (
                    await session.execute(
                        select(PipelineSettingsModel)
                        .where(PipelineSettingsModel.is_active == True)  # noqa: E712
                        .order_by(PipelineSettingsModel.id.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()

                if pipeline_row:
                    data = pipeline_row.to_dict().get("settings_data", {})
                    cfg = data.get("pipeline_config", {})
                    global_model = cfg.get("default_model", global_model)
                    global_temperature = cfg.get(
                        "default_temperature", global_temperature
                    )
                    # optional_steps → treat non-optional steps as disabled by default
                    # (optional_steps are the ones that CAN be skipped by user)
                    # We keep global_disabled_steps = [] here since disabling is per-user

                # Global approval settings
                approval_row = (
                    await session.execute(
                        select(ApprovalSettingsModel)
                        .order_by(ApprovalSettingsModel.id.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()

                if approval_row:
                    ad = approval_row.to_dict()
                    global_require_approval = ad.get(
                        "require_approval", global_require_approval
                    )
                    global_approval_agents = (
                        ad.get("approval_agents", global_approval_agents)
                        or global_approval_agents
                    )
                    global_auto_approve_threshold = ad.get("auto_approve_threshold")
                    global_timeout_seconds = ad.get("timeout_seconds")

                # ── User overrides ────────────────────────────────────────────────
                user_row = (
                    await session.execute(
                        select(UserSettingsModel).where(
                            UserSettingsModel.user_id == user_id
                        )
                    )
                ).scalar_one_or_none()

        except Exception as e:
            logger.warning(
                f"Failed to load settings from DB for user {user_id}: {e}. "
                "Using defaults."
            )
            user_row = None
    else:
        user_row = None

    # Apply user overrides where fields are not None
    if user_row:
        ud = user_row.to_dict()
        disabled_steps = (
            ud["disabled_steps"]
            if ud["disabled_steps"] is not None
            else global_disabled_steps
        )
        require_approval = (
            ud["require_approval"]
            if ud["require_approval"] is not None
            else global_require_approval
        )
        approval_agents = (
            ud["approval_agents"]
            if ud["approval_agents"] is not None
            else global_approval_agents
        )
        auto_approve_threshold = (
            ud["auto_approve_threshold"]
            if ud["auto_approve_threshold"] is not None
            else global_auto_approve_threshold
        )
        approval_timeout_seconds = (
            ud["approval_timeout_seconds"]
            if ud["approval_timeout_seconds"] is not None
            else global_timeout_seconds
        )
        preferred_model = (
            ud["preferred_model"] if ud["preferred_model"] is not None else global_model
        )
        preferred_temperature = (
            ud["preferred_temperature"]
            if ud["preferred_temperature"] is not None
            else global_temperature
        )
    else:
        disabled_steps = global_disabled_steps
        require_approval = global_require_approval
        approval_agents = global_approval_agents
        auto_approve_threshold = global_auto_approve_threshold
        approval_timeout_seconds = global_timeout_seconds
        preferred_model = global_model
        preferred_temperature = global_temperature

    return ResolvedUserSettings(
        disabled_steps=disabled_steps,
        require_approval=require_approval,
        approval_agents=approval_agents,
        auto_approve_threshold=auto_approve_threshold,
        approval_timeout_seconds=approval_timeout_seconds,
        preferred_model=preferred_model,
        preferred_temperature=float(preferred_temperature),
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/me/settings", response_model=UserSettingsResponse)
async def get_my_settings(user: UserContext = Depends(get_current_user)):
    """
    Get the current user's personal settings.

    Returns the stored per-user overrides. Fields that are None inherit
    from the global defaults at job creation time.
    """
    db_manager = get_database_manager()
    if not db_manager.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_manager.get_session() as session:
            row = (
                await session.execute(
                    select(UserSettingsModel).where(
                        UserSettingsModel.user_id == user.user_id
                    )
                )
            ).scalar_one_or_none()

        if not row:
            raise HTTPException(
                status_code=404,
                detail="No personal settings found. All fields inherit from global defaults.",
            )

        data = row.to_dict()
        return UserSettingsResponse(**data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get settings for user {user.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve settings")


@router.put("/me/settings", response_model=UserSettingsResponse)
async def upsert_my_settings(
    settings: UserSettings,
    user: UserContext = Depends(get_current_user),
):
    """
    Create or update the current user's personal settings.

    Only fields explicitly provided override the global default.
    Send null/omit a field to revert it to the global default.
    """
    db_manager = get_database_manager()
    if not db_manager.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_manager.get_session() as session:
            row = (
                await session.execute(
                    select(UserSettingsModel).where(
                        UserSettingsModel.user_id == user.user_id
                    )
                )
            ).scalar_one_or_none()

            if row:
                row.disabled_steps = settings.disabled_steps
                row.require_approval = settings.require_approval
                row.approval_agents = settings.approval_agents
                row.auto_approve_threshold = settings.auto_approve_threshold
                row.approval_timeout_seconds = settings.approval_timeout_seconds
                row.preferred_model = settings.preferred_model
                row.preferred_temperature = settings.preferred_temperature
            else:
                row = UserSettingsModel(
                    user_id=user.user_id,
                    disabled_steps=settings.disabled_steps,
                    require_approval=settings.require_approval,
                    approval_agents=settings.approval_agents,
                    auto_approve_threshold=settings.auto_approve_threshold,
                    approval_timeout_seconds=settings.approval_timeout_seconds,
                    preferred_model=settings.preferred_model,
                    preferred_temperature=settings.preferred_temperature,
                )
                session.add(row)

            await session.commit()
            await session.refresh(row)

        data = row.to_dict()
        logger.info(f"Settings upserted for user {user.user_id}")
        return UserSettingsResponse(**data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upsert settings for user {user.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save settings")


@router.delete("/me/settings", status_code=204)
async def delete_my_settings(user: UserContext = Depends(get_current_user)):
    """
    Delete the current user's personal settings.

    After deletion all fields revert to global defaults.
    """
    db_manager = get_database_manager()
    if not db_manager.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_manager.get_session() as session:
            row = (
                await session.execute(
                    select(UserSettingsModel).where(
                        UserSettingsModel.user_id == user.user_id
                    )
                )
            ).scalar_one_or_none()

            if row:
                await session.delete(row)
                await session.commit()
                logger.info(f"Settings deleted for user {user.user_id}")

        return Response(status_code=204)

    except Exception as e:
        logger.error(f"Failed to delete settings for user {user.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete settings")


# ── Admin endpoints ───────────────────────────────────────────────────────────


@router.get("/{user_id}/settings", response_model=UserSettingsResponse)
async def admin_get_user_settings(
    user_id: str,
    admin: UserContext = Depends(require_roles(["admin"])),
):
    """[Admin] Get settings for any user by their Keycloak user_id."""
    db_manager = get_database_manager()
    if not db_manager.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_manager.get_session() as session:
            row = (
                await session.execute(
                    select(UserSettingsModel).where(
                        UserSettingsModel.user_id == user_id
                    )
                )
            ).scalar_one_or_none()

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"No personal settings found for user {user_id}.",
            )

        return UserSettingsResponse(**row.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin failed to get settings for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve settings")


@router.put("/{user_id}/settings", response_model=UserSettingsResponse)
async def admin_update_user_settings(
    user_id: str,
    settings: UserSettings,
    admin: UserContext = Depends(require_roles(["admin"])),
):
    """[Admin] Create or update settings for any user by their Keycloak user_id."""
    db_manager = get_database_manager()
    if not db_manager.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_manager.get_session() as session:
            row = (
                await session.execute(
                    select(UserSettingsModel).where(
                        UserSettingsModel.user_id == user_id
                    )
                )
            ).scalar_one_or_none()

            if row:
                row.disabled_steps = settings.disabled_steps
                row.require_approval = settings.require_approval
                row.approval_agents = settings.approval_agents
                row.auto_approve_threshold = settings.auto_approve_threshold
                row.approval_timeout_seconds = settings.approval_timeout_seconds
                row.preferred_model = settings.preferred_model
                row.preferred_temperature = settings.preferred_temperature
            else:
                row = UserSettingsModel(
                    user_id=user_id,
                    disabled_steps=settings.disabled_steps,
                    require_approval=settings.require_approval,
                    approval_agents=settings.approval_agents,
                    auto_approve_threshold=settings.auto_approve_threshold,
                    approval_timeout_seconds=settings.approval_timeout_seconds,
                    preferred_model=settings.preferred_model,
                    preferred_temperature=settings.preferred_temperature,
                )
                session.add(row)

            await session.commit()
            await session.refresh(row)

        logger.info(f"Admin {admin.user_id} updated settings for user {user_id}")
        return UserSettingsResponse(**row.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Admin {admin.user_id} failed to update settings for user {user_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to save settings")
