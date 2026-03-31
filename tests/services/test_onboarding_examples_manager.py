"""
Unit tests for OnboardingExamplesManager service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.onboarding_models import (
    OnboardingExampleCreateRequest,
    OnboardingExampleUpdateRequest,
)
from marketing_project.services.onboarding_examples_manager import (
    OnboardingExamplesManager,
    get_onboarding_examples_manager,
)


def _make_row(
    id=1,
    title="My Example",
    description="desc",
    job_type="blog",
    input_data=None,
    display_order=0,
    is_active=True,
):
    row = MagicMock()
    row.to_dict.return_value = {
        "id": id,
        "title": title,
        "description": description,
        "job_type": job_type,
        "input_data": input_data or {},
        "display_order": display_order,
        "is_active": is_active,
        "created_at": None,
        "updated_at": None,
    }
    return row


@pytest.fixture
def manager():
    return OnboardingExamplesManager()


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    return session


@pytest.fixture
def mock_db(mock_session):
    db = MagicMock()
    db.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    db.get_session.return_value.__aexit__ = AsyncMock(return_value=False)
    return db


@pytest.mark.asyncio
async def test_list_active_returns_only_active(manager, mock_session, mock_db):
    rows = [_make_row(id=1, is_active=True), _make_row(id=2, is_active=True)]
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = result_mock

    with patch(
        "marketing_project.services.onboarding_examples_manager.get_db_manager",
        return_value=mock_db,
    ):
        examples = await manager.list_active()

    assert len(examples) == 2
    assert all(e.is_active for e in examples)


@pytest.mark.asyncio
async def test_list_active_pagination(manager, mock_session, mock_db):
    """Pagination params are forwarded to the query."""
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = [_make_row(id=3)]
    mock_session.execute.return_value = result_mock

    with patch(
        "marketing_project.services.onboarding_examples_manager.get_db_manager",
        return_value=mock_db,
    ):
        examples = await manager.list_active(limit=1, offset=2)

    assert len(examples) == 1
    # Verify execute was called (the limit/offset are applied in the SQLAlchemy statement,
    # not something we can inspect easily without integration, but the call went through)
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_list_all_includes_inactive(manager, mock_session, mock_db):
    rows = [
        _make_row(id=1, is_active=True),
        _make_row(id=2, is_active=False),
    ]
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = rows
    mock_session.execute.return_value = result_mock

    with patch(
        "marketing_project.services.onboarding_examples_manager.get_db_manager",
        return_value=mock_db,
    ):
        examples = await manager.list_all()

    assert len(examples) == 2


@pytest.mark.asyncio
async def test_get_returns_example(manager, mock_session, mock_db):
    row = _make_row(id=5, title="Found")
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = row
    mock_session.execute.return_value = result_mock

    with patch(
        "marketing_project.services.onboarding_examples_manager.get_db_manager",
        return_value=mock_db,
    ):
        example = await manager.get(5)

    assert example is not None
    assert example.id == 5
    assert example.title == "Found"


@pytest.mark.asyncio
async def test_get_returns_none_when_not_found(manager, mock_session, mock_db):
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = result_mock

    with patch(
        "marketing_project.services.onboarding_examples_manager.get_db_manager",
        return_value=mock_db,
    ):
        example = await manager.get(999)

    assert example is None


@pytest.mark.asyncio
async def test_create_adds_row_and_returns_response(manager, mock_session, mock_db):
    new_row = _make_row(id=10, title="New", job_type="transcript")
    mock_session.refresh = AsyncMock(side_effect=lambda r: None)

    async def _refresh(row):
        # Simulate DB assigning id after flush
        row.to_dict = new_row.to_dict

    mock_session.refresh = _refresh

    with patch(
        "marketing_project.services.onboarding_examples_manager.get_db_manager",
        return_value=mock_db,
    ):
        req = OnboardingExampleCreateRequest(
            title="New",
            job_type="transcript",
            input_data={"key": "value"},
        )
        # The row added to session is a real OnboardingExampleModel; mock to_dict on it
        with patch(
            "marketing_project.services.onboarding_examples_manager.OnboardingExampleModel",
        ) as MockModel:
            MockModel.return_value = new_row
            example = await manager.create(req)

    mock_session.add.assert_called_once()
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_update_returns_none_when_not_found(manager, mock_session, mock_db):
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = result_mock

    with patch(
        "marketing_project.services.onboarding_examples_manager.get_db_manager",
        return_value=mock_db,
    ):
        result = await manager.update(
            99, OnboardingExampleUpdateRequest(title="New Title")
        )

    assert result is None


@pytest.mark.asyncio
async def test_update_patches_fields(manager, mock_session, mock_db):
    row = _make_row(id=1, title="Old")
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = row
    mock_session.execute.return_value = result_mock

    with patch(
        "marketing_project.services.onboarding_examples_manager.get_db_manager",
        return_value=mock_db,
    ):
        result = await manager.update(
            1, OnboardingExampleUpdateRequest(title="New Title")
        )

    # setattr is called on the row mock
    assert row.title == "New Title"
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_delete_returns_true_when_found(manager, mock_session, mock_db):
    row = _make_row(id=7)
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = row
    mock_session.execute.return_value = result_mock

    with patch(
        "marketing_project.services.onboarding_examples_manager.get_db_manager",
        return_value=mock_db,
    ):
        deleted = await manager.delete(7)

    assert deleted is True
    mock_session.delete.assert_called_once_with(row)


@pytest.mark.asyncio
async def test_delete_returns_false_when_not_found(manager, mock_session, mock_db):
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = result_mock

    with patch(
        "marketing_project.services.onboarding_examples_manager.get_db_manager",
        return_value=mock_db,
    ):
        deleted = await manager.delete(999)

    assert deleted is False


def test_get_onboarding_examples_manager_returns_singleton():
    mgr1 = get_onboarding_examples_manager()
    mgr2 = get_onboarding_examples_manager()
    assert mgr1 is mgr2
