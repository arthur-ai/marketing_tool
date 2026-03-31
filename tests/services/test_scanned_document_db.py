"""
Tests for scanned document database service.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.scanned_document_db import (
    ScannedDocumentDB,
    ScannedDocumentMetadata,
)
from marketing_project.services.scanned_document_db import (
    ScannedDocumentDatabase,
    get_scanned_document_db,
)


def _make_db_mock(scalar_result=None, scalars_result=None, rowcount=0):
    """Build a mock db_manager with a fully async session."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = scalar_result
    if scalars_result is not None:
        mock_result.scalars.return_value.all.return_value = scalars_result
    mock_result.rowcount = rowcount

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()

    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_db_mgr, mock_session, mock_result


def _sample_doc():
    return ScannedDocumentDB(
        title="Test Document",
        url="https://example.com/test",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(content_text="Test content"),
    )


@pytest.fixture
def scanned_doc_db():
    """Create a ScannedDocumentDatabase instance (no args required)."""
    return ScannedDocumentDatabase()


def test_initialization(scanned_doc_db):
    """ScannedDocumentDatabase instantiates without arguments."""
    assert scanned_doc_db is not None
    assert isinstance(scanned_doc_db, ScannedDocumentDatabase)


@pytest.mark.asyncio
async def test_save_document_new(scanned_doc_db):
    """Test saving a new document inserts and returns an ID."""
    mock_model = MagicMock()
    mock_model.id = 42
    mock_db_mgr, mock_session, mock_result = _make_db_mock(scalar_result=None)
    # scalar_one_or_none returns None → new insert path
    mock_result.scalar_one_or_none.return_value = None

    async def fake_flush():
        mock_model.id = 42

    mock_session.flush = fake_flush
    mock_session.add = MagicMock()

    # Patch add so we can set id on the added object
    added_objects = []

    def capture_add(obj):
        obj.id = 42
        added_objects.append(obj)

    mock_session.add = capture_add

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        doc = _sample_doc()
        doc_id = await scanned_doc_db.save_document(doc)

    assert doc_id == 42


@pytest.mark.asyncio
async def test_save_document_update(scanned_doc_db):
    """Test saving an existing document updates it and returns the existing ID."""
    existing = MagicMock()
    existing.id = 7
    existing.scan_count = 1

    mock_db_mgr, _, mock_result = _make_db_mock(scalar_result=existing)
    mock_result.scalar_one_or_none.return_value = existing

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        doc = _sample_doc()
        doc_id = await scanned_doc_db.save_document(doc)

    assert doc_id == 7


@pytest.mark.asyncio
async def test_get_document_by_url_not_found(scanned_doc_db):
    """Returns None when the document is not in the DB."""
    mock_db_mgr, _, mock_result = _make_db_mock(scalar_result=None)
    mock_result.scalar_one_or_none.return_value = None

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_document_by_url("https://example.com/missing")

    assert result is None


@pytest.mark.asyncio
async def test_get_document_by_url_found(scanned_doc_db):
    """Returns a ScannedDocumentDB when the document exists."""
    mock_row = MagicMock()
    mock_row.id = 1
    mock_row.title = "Test Document"
    mock_row.url = "https://example.com/test"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = datetime.now()
    mock_row.metadata_json = {"content_text": "Test content"}
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = []

    mock_db_mgr, _, mock_result = _make_db_mock(scalar_result=mock_row)
    mock_result.scalar_one_or_none.return_value = mock_row

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_document_by_url("https://example.com/test")

    assert result is not None
    assert result.title == "Test Document"


@pytest.mark.asyncio
async def test_get_all_active_documents(scanned_doc_db):
    """Returns a list of active documents."""
    mock_row = MagicMock()
    mock_row.id = 1
    mock_row.title = "Doc"
    mock_row.url = "https://example.com/1"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = datetime.now()
    mock_row.metadata_json = {}
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = []

    mock_db_mgr, _, mock_result = _make_db_mock(scalars_result=[mock_row])

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        docs = await scanned_doc_db.get_all_active_documents()

    assert len(docs) >= 1


@pytest.mark.asyncio
async def test_delete_document(scanned_doc_db):
    """Delete returns True when a row is deleted."""
    mock_db_mgr, _, mock_result = _make_db_mock(rowcount=1)
    mock_result.rowcount = 1

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        success = await scanned_doc_db.delete_document("https://example.com/test")

    assert success is True


@pytest.mark.asyncio
async def test_delete_document_not_found(scanned_doc_db):
    """Delete returns False when no row is deleted."""
    mock_db_mgr, _, mock_result = _make_db_mock(rowcount=0)
    mock_result.rowcount = 0

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        success = await scanned_doc_db.delete_document("https://example.com/missing")

    assert success is False


@pytest.mark.asyncio
async def test_save_document_db_not_initialized(scanned_doc_db):
    """Raises RuntimeError when DB is not initialized."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = False

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        with pytest.raises(RuntimeError, match="Database not initialized"):
            await scanned_doc_db.save_document(_sample_doc())


def test_get_scanned_document_db_singleton():
    """get_scanned_document_db returns the same instance each time."""
    db1 = get_scanned_document_db()
    db2 = get_scanned_document_db()
    assert db1 is db2
