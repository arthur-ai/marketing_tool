"""
Service layer for onboarding examples (quick-start job templates).
"""

import logging
from typing import List, Optional

from sqlalchemy import func, select, update

from marketing_project.models.db_models import OnboardingExampleModel
from marketing_project.models.onboarding_models import (
    OnboardingExampleCreateRequest,
    OnboardingExampleResponse,
    OnboardingExampleUpdateRequest,
)
from marketing_project.services.database import get_database_manager as get_db_manager

logger = logging.getLogger("marketing_project.services.onboarding_examples_manager")


class OnboardingExamplesManager:
    """Manages CRUD operations for onboarding example templates."""

    async def list_active(
        self, limit: int = 100, offset: int = 0
    ) -> List[OnboardingExampleResponse]:
        """Return active examples ordered by display_order (paginated)."""
        db = get_db_manager()
        async with db.get_session() as session:
            result = await session.execute(
                select(OnboardingExampleModel)
                .where(OnboardingExampleModel.is_active.is_(True))
                .order_by(OnboardingExampleModel.display_order)
                .offset(offset)
                .limit(limit)
            )
            rows = result.scalars().all()
            return [OnboardingExampleResponse(**row.to_dict()) for row in rows]

    async def count_active(self) -> int:
        """Return the total number of active examples (unaffected by pagination)."""
        db = get_db_manager()
        async with db.get_session() as session:
            result = await session.execute(
                select(func.count())
                .select_from(OnboardingExampleModel)
                .where(OnboardingExampleModel.is_active.is_(True))
            )
            return result.scalar_one()

    async def count_all(self) -> int:
        """Return the total number of examples including inactive."""
        db = get_db_manager()
        async with db.get_session() as session:
            result = await session.execute(
                select(func.count()).select_from(OnboardingExampleModel)
            )
            return result.scalar_one()

    async def list_all(
        self, limit: int = 1000, offset: int = 0
    ) -> List[OnboardingExampleResponse]:
        """Return all examples (including inactive) ordered by display_order."""
        db = get_db_manager()
        async with db.get_session() as session:
            result = await session.execute(
                select(OnboardingExampleModel)
                .order_by(OnboardingExampleModel.display_order)
                .offset(offset)
                .limit(limit)
            )
            rows = result.scalars().all()
            return [OnboardingExampleResponse(**row.to_dict()) for row in rows]

    async def get(self, example_id: int) -> Optional[OnboardingExampleResponse]:
        """Return a single example by ID, or None if not found."""
        db = get_db_manager()
        async with db.get_session() as session:
            result = await session.execute(
                select(OnboardingExampleModel).where(
                    OnboardingExampleModel.id == example_id
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return OnboardingExampleResponse(**row.to_dict())

    async def create(
        self, req: OnboardingExampleCreateRequest
    ) -> OnboardingExampleResponse:
        """Create a new onboarding example."""
        db = get_db_manager()
        async with db.get_session() as session:
            row = OnboardingExampleModel(
                title=req.title,
                description=req.description,
                job_type=req.job_type,
                input_data=req.input_data,
                display_order=req.display_order,
                is_active=req.is_active,
            )
            session.add(row)
            await session.flush()
            await session.refresh(row)
            return OnboardingExampleResponse(**row.to_dict())

    async def update(
        self, example_id: int, req: OnboardingExampleUpdateRequest
    ) -> Optional[OnboardingExampleResponse]:
        """Partial-update an existing example. Returns None if not found."""
        db = get_db_manager()
        async with db.get_session() as session:
            result = await session.execute(
                select(OnboardingExampleModel).where(
                    OnboardingExampleModel.id == example_id
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None

            patch = req.model_dump(exclude_unset=True)
            for field, value in patch.items():
                setattr(row, field, value)

            await session.flush()
            await session.refresh(row)
            return OnboardingExampleResponse(**row.to_dict())

    async def delete(self, example_id: int) -> bool:
        """Delete an example. Returns True if deleted, False if not found."""
        db = get_db_manager()
        async with db.get_session() as session:
            result = await session.execute(
                select(OnboardingExampleModel).where(
                    OnboardingExampleModel.id == example_id
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return False
            await session.delete(row)
            return True


_manager: Optional[OnboardingExamplesManager] = None


def get_onboarding_examples_manager() -> OnboardingExamplesManager:
    global _manager
    if _manager is None:
        _manager = OnboardingExamplesManager()
    return _manager
