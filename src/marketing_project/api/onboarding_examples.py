"""
API endpoints for onboarding examples (quick-start job templates).

GET  /api/v1/onboarding-examples          — any authenticated user, active only
GET  /api/v1/onboarding-examples/admin    — admin: all examples including inactive
POST /api/v1/onboarding-examples/admin    — admin: create
GET  /api/v1/onboarding-examples/admin/{id} — admin: get one
PATCH /api/v1/onboarding-examples/admin/{id} — admin: partial update
DELETE /api/v1/onboarding-examples/admin/{id} — admin: delete
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.middleware.rbac import require_roles
from marketing_project.models.onboarding_models import (
    OnboardingExampleCreateRequest,
    OnboardingExampleResponse,
    OnboardingExamplesListResponse,
    OnboardingExampleUpdateRequest,
)
from marketing_project.models.user_context import UserContext
from marketing_project.services.onboarding_examples_manager import (
    get_onboarding_examples_manager,
)

logger = logging.getLogger("marketing_project.api.onboarding_examples")

router = APIRouter(prefix="/onboarding-examples", tags=["Onboarding Examples"])


@router.get("", response_model=OnboardingExamplesListResponse)
async def list_active_examples(
    limit: int = Query(100, ge=1, le=500, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    user: UserContext = Depends(get_current_user),
):
    """Return active onboarding examples ordered by display_order."""
    try:
        mgr = get_onboarding_examples_manager()
        examples = await mgr.list_active(limit=limit, offset=offset)
        total = await mgr.count_active()
        return OnboardingExamplesListResponse(examples=examples, total=total)
    except Exception as e:
        logger.error(f"Failed to list active onboarding examples: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve onboarding examples"
        )


@router.get("/admin", response_model=OnboardingExamplesListResponse)
async def list_all_examples(
    limit: int = Query(1000, ge=1, le=1000, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    user: UserContext = Depends(require_roles(["admin"])),
):
    """[Admin] Return all examples including inactive."""
    try:
        mgr = get_onboarding_examples_manager()
        examples = await mgr.list_all(limit=limit, offset=offset)
        total = await mgr.count_all()
        return OnboardingExamplesListResponse(examples=examples, total=total)
    except Exception as e:
        logger.error(f"Failed to list all onboarding examples: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve onboarding examples"
        )


@router.post(
    "/admin",
    response_model=OnboardingExampleResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_example(
    req: OnboardingExampleCreateRequest,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """[Admin] Create a new onboarding example."""
    try:
        mgr = get_onboarding_examples_manager()
        return await mgr.create(req)
    except Exception as e:
        logger.error(f"Failed to create onboarding example: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to create onboarding example"
        )


@router.get("/admin/{example_id}", response_model=OnboardingExampleResponse)
async def get_example(
    example_id: int,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """[Admin] Get a single onboarding example by ID."""
    try:
        mgr = get_onboarding_examples_manager()
        example = await mgr.get(example_id)
        if example is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Onboarding example {example_id} not found",
            )
        return example
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get onboarding example {example_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve onboarding example"
        )


@router.patch("/admin/{example_id}", response_model=OnboardingExampleResponse)
async def update_example(
    example_id: int,
    req: OnboardingExampleUpdateRequest,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """[Admin] Partially update an onboarding example."""
    try:
        mgr = get_onboarding_examples_manager()
        example = await mgr.update(example_id, req)
        if example is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Onboarding example {example_id} not found",
            )
        return example
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update onboarding example {example_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to update onboarding example"
        )


@router.delete("/admin/{example_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_example(
    example_id: int,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """[Admin] Delete an onboarding example."""
    try:
        mgr = get_onboarding_examples_manager()
        deleted = await mgr.delete(example_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Onboarding example {example_id} not found",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete onboarding example {example_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to delete onboarding example"
        )
