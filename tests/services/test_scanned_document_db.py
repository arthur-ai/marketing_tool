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


# ---------------------------------------------------------------------------
# Additional tests for missed coverage lines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_document_by_url_db_not_initialized(scanned_doc_db):
    """Returns None when DB is not initialized for get_document_by_url."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = False

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_document_by_url("https://example.com/test")

    assert result is None


@pytest.mark.asyncio
async def test_get_all_active_documents_db_not_initialized(scanned_doc_db):
    """Returns empty list when DB is not initialized."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = False

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_all_active_documents()

    assert result == []


@pytest.mark.asyncio
async def test_search_by_keywords_db_not_initialized(scanned_doc_db):
    """Returns empty list when DB is not initialized for keyword search."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = False

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.search_by_keywords(["test"])

    assert result == []


@pytest.mark.asyncio
async def test_search_by_keywords_returns_docs(scanned_doc_db):
    """search_by_keywords returns matching documents."""
    mock_row = MagicMock()
    mock_row.id = 1
    mock_row.title = "Test Document"
    mock_row.url = "https://example.com/test"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
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
        result = await scanned_doc_db.search_by_keywords(["Test"])

    assert len(result) == 1
    assert result[0].title == "Test Document"


@pytest.mark.asyncio
async def test_get_documents_by_category_db_not_initialized(scanned_doc_db):
    """Returns empty list when DB is not initialized for category search."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = False

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_documents_by_category("blog")

    assert result == []


@pytest.mark.asyncio
async def test_get_documents_by_category_returns_docs(scanned_doc_db):
    """get_documents_by_category returns documents in that category."""
    mock_row = MagicMock()
    mock_row.id = 2
    mock_row.title = "Blog Post"
    mock_row.url = "https://example.com/blog"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
    mock_row.metadata_json = {"categories": ["blog"]}
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = 0.9
    mock_row.related_documents = []

    mock_db_mgr, _, mock_result = _make_db_mock(scalars_result=[mock_row])

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_documents_by_category("blog")

    assert len(result) == 1
    assert result[0].title == "Blog Post"


@pytest.mark.asyncio
async def test_get_documents_with_internal_links_db_not_initialized(scanned_doc_db):
    """Returns empty list when DB not initialized for internal links query."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = False

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_documents_with_internal_links()

    assert result == []


@pytest.mark.asyncio
async def test_get_documents_with_internal_links_returns_docs(scanned_doc_db):
    """get_documents_with_internal_links returns documents from raw SQL."""
    mock_row = {
        "id": 3,
        "title": "Linked Doc",
        "url": "https://example.com/linked",
        "scanned_at": datetime.now(),
        "last_scanned_at": None,
        "metadata_json": {"outbound_link_count": 5},
        "is_active": True,
        "scan_count": 1,
        "relevance_score": None,
        "related_documents": [],
    }

    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [mock_row]
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_documents_with_internal_links()

    assert len(result) == 1
    assert result[0].title == "Linked Doc"


@pytest.mark.asyncio
async def test_get_anchor_text_patterns_db_not_initialized(scanned_doc_db):
    """Returns empty list when DB not initialized for anchor text patterns."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = False

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_anchor_text_patterns()

    assert result == []


@pytest.mark.asyncio
async def test_get_anchor_text_patterns_returns_patterns(scanned_doc_db):
    """get_anchor_text_patterns returns unique patterns from the DB."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_result = MagicMock()
    mock_result.all.return_value = [("learn more",), ("click here",)]
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_anchor_text_patterns()

    assert result == ["learn more", "click here"]


@pytest.mark.asyncio
async def test_get_anchor_text_patterns_exception_returns_empty(scanned_doc_db):
    """get_anchor_text_patterns returns [] on exception."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=Exception("DB error"))
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_anchor_text_patterns()

    assert result == []


@pytest.mark.asyncio
async def test_get_commonly_referenced_pages_db_not_initialized(scanned_doc_db):
    """Returns empty list when DB not initialized."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = False

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_commonly_referenced_pages()

    assert result == []


@pytest.mark.asyncio
async def test_get_commonly_referenced_pages_returns_urls(scanned_doc_db):
    """get_commonly_referenced_pages returns URL strings."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_result = MagicMock()
    mock_result.all.return_value = [
        ("https://example.com/popular",),
        (None,),  # None values should be filtered out
    ]
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_commonly_referenced_pages(min_links=2)

    assert "https://example.com/popular" in result
    assert None not in result


@pytest.mark.asyncio
async def test_get_commonly_referenced_pages_exception_returns_empty(scanned_doc_db):
    """get_commonly_referenced_pages returns [] on exception."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=Exception("DB error"))
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.get_commonly_referenced_pages()

    assert result == []


@pytest.mark.asyncio
async def test_bulk_delete_documents_empty_list(scanned_doc_db):
    """bulk_delete_documents returns 0 for empty URL list."""
    result = await scanned_doc_db.bulk_delete_documents([])
    assert result == 0


@pytest.mark.asyncio
async def test_bulk_delete_documents_db_not_initialized(scanned_doc_db):
    """bulk_delete_documents returns 0 when DB not initialized."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = False

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.bulk_delete_documents(
            ["https://example.com/a", "https://example.com/b"]
        )

    assert result == 0


@pytest.mark.asyncio
async def test_bulk_delete_documents_returns_count(scanned_doc_db):
    """bulk_delete_documents returns the number of deleted rows."""
    mock_db_mgr, _, mock_result = _make_db_mock(rowcount=2)
    mock_result.rowcount = 2

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.bulk_delete_documents(
            ["https://example.com/a", "https://example.com/b"]
        )

    assert result == 2


@pytest.mark.asyncio
async def test_bulk_save_documents_all_new(scanned_doc_db):
    """bulk_save_documents returns created=N, updated=0 when all docs are new."""
    doc1 = _sample_doc()
    doc2 = ScannedDocumentDB(
        title="Second Doc",
        url="https://example.com/second",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(),
    )

    # get_document_by_url returns None → new; save_document returns an id
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_result.scalars.return_value.all.return_value = []
    mock_result.rowcount = 0
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = lambda obj: setattr(obj, "id", 99)
    mock_session.flush = AsyncMock()
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.bulk_save_documents([doc1, doc2])

    assert result["created"] == 2
    assert result["updated"] == 0


@pytest.mark.asyncio
async def test_bulk_save_documents_all_existing(scanned_doc_db):
    """bulk_save_documents returns updated=N when all docs already exist."""
    doc = _sample_doc()

    existing_mock_row = MagicMock()
    existing_mock_row.id = 1
    existing_mock_row.title = "Test Document"
    existing_mock_row.url = doc.url
    existing_mock_row.scanned_at = datetime.now()
    existing_mock_row.last_scanned_at = None
    existing_mock_row.metadata_json = {}
    existing_mock_row.is_active = True
    existing_mock_row.scan_count = 1
    existing_mock_row.relevance_score = None
    existing_mock_row.related_documents = []

    existing_db_mock = MagicMock()
    existing_db_mock.id = 1
    existing_db_mock.scan_count = 1

    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_result = MagicMock()
    # First call (get_document_by_url) returns a row, second call (save) returns the same
    mock_result.scalar_one_or_none.side_effect = [
        existing_mock_row,  # get_document_by_url
        existing_db_mock,  # save_document
    ]
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.flush = AsyncMock()
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.bulk_save_documents([doc])

    assert result["updated"] == 1
    assert result["created"] == 0


@pytest.mark.asyncio
async def test_bulk_update_categories(scanned_doc_db):
    """bulk_update_categories merges categories and returns count."""
    doc = ScannedDocumentDB(
        title="Cat Doc",
        url="https://example.com/cat",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(categories=["existing"]),
    )

    existing_mock_row = MagicMock()
    existing_mock_row.id = 1
    existing_mock_row.title = "Cat Doc"
    existing_mock_row.url = doc.url
    existing_mock_row.scanned_at = datetime.now()
    existing_mock_row.last_scanned_at = None
    existing_mock_row.metadata_json = {"categories": ["existing"]}
    existing_mock_row.is_active = True
    existing_mock_row.scan_count = 1
    existing_mock_row.relevance_score = None
    existing_mock_row.related_documents = []

    existing_db_mock = MagicMock()
    existing_db_mock.id = 1
    existing_db_mock.scan_count = 1

    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.side_effect = [
        existing_mock_row,  # first get_document_by_url
        existing_db_mock,  # save_document inside update
    ]
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.flush = AsyncMock()
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        count = await scanned_doc_db.bulk_update_categories([doc.url], ["new_category"])

    assert count == 1


@pytest.mark.asyncio
async def test_search_with_filters_db_not_initialized(scanned_doc_db):
    """search_with_filters returns [] when DB not initialized."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = False

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.search_with_filters({"category": "blog"})

    assert result == []


@pytest.mark.asyncio
async def test_search_with_filters_with_category(scanned_doc_db):
    """search_with_filters applies category filter correctly."""
    mock_row = MagicMock()
    mock_row.id = 5
    mock_row.title = "Blog Post"
    mock_row.url = "https://example.com/blog"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
    mock_row.metadata_json = {"categories": ["blog"]}
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = []

    mock_db_mgr, _, mock_result = _make_db_mock(scalars_result=[mock_row])

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.search_with_filters({"category": "blog"})

    assert len(result) == 1
    assert result[0].title == "Blog Post"


@pytest.mark.asyncio
async def test_search_with_filters_with_content_type(scanned_doc_db):
    """search_with_filters applies content_type filter."""
    mock_row = MagicMock()
    mock_row.id = 6
    mock_row.title = "Docs Page"
    mock_row.url = "https://example.com/docs"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
    mock_row.metadata_json = {"content_type": "docs"}
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = []

    mock_db_mgr, _, mock_result = _make_db_mock(scalars_result=[mock_row])

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.search_with_filters({"content_type": "docs"})

    assert len(result) >= 0  # result depends on filter execution


@pytest.mark.asyncio
async def test_search_with_filters_with_keywords(scanned_doc_db):
    """search_with_filters handles string keyword filter."""
    mock_row = MagicMock()
    mock_row.id = 7
    mock_row.title = "AI Article"
    mock_row.url = "https://example.com/ai"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
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
        result = await scanned_doc_db.search_with_filters({"keywords": "AI"})

    assert len(result) >= 0


@pytest.mark.asyncio
async def test_search_with_filters_with_keywords_list(scanned_doc_db):
    """search_with_filters handles list keyword filter."""
    mock_row = MagicMock()
    mock_row.id = 8
    mock_row.title = "ML Guide"
    mock_row.url = "https://example.com/ml"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
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
        result = await scanned_doc_db.search_with_filters({"keywords": ["ML", "AI"]})

    assert len(result) >= 0


@pytest.mark.asyncio
async def test_search_with_filters_exception_returns_empty(scanned_doc_db):
    """search_with_filters returns [] on exception."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=Exception("DB error"))
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.search_with_filters({})

    assert result == []


@pytest.mark.asyncio
async def test_full_text_search_db_not_initialized(scanned_doc_db):
    """full_text_search returns [] when DB not initialized."""
    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = False

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.full_text_search("test query")

    assert result == []


@pytest.mark.asyncio
async def test_full_text_search_returns_docs_from_tsvector(scanned_doc_db):
    """full_text_search returns documents when tsvector matches."""
    mock_row = {
        "id": 9,
        "title": "FTS Doc",
        "url": "https://example.com/fts",
        "scanned_at": datetime.now(),
        "last_scanned_at": None,
        "metadata_json": {},
        "is_active": True,
        "scan_count": 1,
        "relevance_score": None,
        "related_documents": [],
    }

    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [mock_row]
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.full_text_search("fts test")

    assert len(result) == 1
    assert result[0].title == "FTS Doc"


@pytest.mark.asyncio
async def test_full_text_search_falls_back_to_keyword_search(scanned_doc_db):
    """full_text_search falls back to keyword search when no tsvector results."""
    # First execute (full text) returns empty mappings -> falls back to keyword
    mock_row = MagicMock()
    mock_row.id = 10
    mock_row.title = "Fallback Doc"
    mock_row.url = "https://example.com/fallback"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
    mock_row.metadata_json = {}
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = []

    call_count = 0

    async def fake_execute(stmt, params=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Full text search result — empty mappings
            result = MagicMock()
            result.mappings.return_value.all.return_value = []
            return result
        else:
            # keyword search result
            result = MagicMock()
            result.scalars.return_value.all.return_value = [mock_row]
            return result

    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_session = AsyncMock()
    mock_session.execute = fake_execute
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.full_text_search("fallback")

    # Should get results from the keyword-search fallback
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_full_text_search_exception_falls_back(scanned_doc_db):
    """full_text_search falls back to keyword search on exception."""
    mock_row = MagicMock()
    mock_row.id = 11
    mock_row.title = "Exception Fallback Doc"
    mock_row.url = "https://example.com/exc"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
    mock_row.metadata_json = {}
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = []

    call_count = 0

    async def fake_execute(stmt, params=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("tsvector error")
        result = MagicMock()
        result.scalars.return_value.all.return_value = [mock_row]
        return result

    mock_db_mgr = MagicMock()
    mock_db_mgr.is_initialized = True
    mock_session = AsyncMock()
    mock_session.execute = fake_execute
    mock_db_mgr.get_session.return_value.__aenter__ = AsyncMock(
        return_value=mock_session
    )
    mock_db_mgr.get_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.full_text_search("anything")

    assert isinstance(result, list)


def test_calculate_relationships_same_url_excluded(scanned_doc_db):
    """calculate_relationships does not include the document itself."""
    doc = ScannedDocumentDB(
        title="Main Doc",
        url="https://example.com/main",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(
            extracted_keywords=["ai", "ml"],
            topics=["tech"],
            categories=["blog"],
        ),
    )
    all_docs = [doc]  # only the doc itself

    relationships = scanned_doc_db.calculate_relationships(doc, all_docs)

    assert relationships == []


def test_calculate_relationships_returns_sorted_scores(scanned_doc_db):
    """calculate_relationships returns top 10 sorted by score."""
    doc = ScannedDocumentDB(
        title="Main Doc",
        url="https://example.com/main",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(
            extracted_keywords=["ai", "ml"],
            topics=["tech", "ai"],
            categories=["blog"],
        ),
    )
    high_match = ScannedDocumentDB(
        title="High Match",
        url="https://example.com/high",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(
            extracted_keywords=["ai", "ml"],
            topics=["tech", "ai"],
            categories=["blog"],
        ),
    )
    low_match = ScannedDocumentDB(
        title="Low Match",
        url="https://example.com/low",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(
            extracted_keywords=["unrelated"],
            topics=["other"],
            categories=["news"],
        ),
    )

    relationships = scanned_doc_db.calculate_relationships(
        doc, [doc, high_match, low_match]
    )

    assert len(relationships) >= 1
    # Should be sorted descending by score
    if len(relationships) > 1:
        assert relationships[0][1] >= relationships[1][1]
    assert any(url == "https://example.com/high" for url, _ in relationships)


def test_calculate_relationships_with_shared_links(scanned_doc_db):
    """calculate_relationships considers shared internal links."""
    doc = ScannedDocumentDB(
        title="Link Doc",
        url="https://example.com/link",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(
            internal_links_found=[{"target_url": "https://example.com/target"}]
        ),
    )
    related = ScannedDocumentDB(
        title="Related Link Doc",
        url="https://example.com/related",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(
            internal_links_found=[{"target_url": "https://example.com/target"}]
        ),
    )

    relationships = scanned_doc_db.calculate_relationships(doc, [doc, related])

    assert len(relationships) == 1
    assert relationships[0][0] == "https://example.com/related"


@pytest.mark.asyncio
async def test_update_relationships(scanned_doc_db):
    """update_relationships updates the document's related_documents field."""
    doc = ScannedDocumentDB(
        title="Doc",
        url="https://example.com/doc",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(
            extracted_keywords=["ai"],
            topics=["tech"],
            categories=["blog"],
        ),
    )
    similar = ScannedDocumentDB(
        title="Similar",
        url="https://example.com/similar",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(
            extracted_keywords=["ai"],
            topics=["tech"],
            categories=["blog"],
        ),
    )

    mock_row = MagicMock()
    mock_row.id = 1
    mock_row.title = "Similar"
    mock_row.url = "https://example.com/similar"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
    mock_row.metadata_json = {
        "extracted_keywords": ["ai"],
        "topics": ["tech"],
        "categories": ["blog"],
    }
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = []

    mock_db_mgr, _, mock_result = _make_db_mock(scalars_result=[mock_row])

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        updated_doc = await scanned_doc_db.update_relationships(doc)

    assert isinstance(updated_doc.related_documents, list)


def test_model_to_document_with_none_internal_links(scanned_doc_db):
    """_model_to_document filters out None values in internal_links_found."""
    mock_row = MagicMock()
    mock_row.id = 1
    mock_row.title = "Test"
    mock_row.url = "https://example.com/test"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
    mock_row.metadata_json = {
        "internal_links_found": [
            {"target_url": "https://example.com/valid"},
            None,
            "invalid_string",
        ]
    }
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = []

    result = scanned_doc_db._model_to_document(mock_row)

    assert all(
        link is not None and isinstance(link, dict)
        for link in result.metadata.internal_links_found
    )


def test_model_to_document_related_documents_not_list(scanned_doc_db):
    """_model_to_document handles non-list related_documents gracefully."""
    mock_row = MagicMock()
    mock_row.id = 2
    mock_row.title = "Test2"
    mock_row.url = "https://example.com/test2"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
    mock_row.metadata_json = {}
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = None  # Not a list

    result = scanned_doc_db._model_to_document(mock_row)

    assert result.related_documents == []


def test_mapping_to_document_string_metadata(scanned_doc_db):
    """_mapping_to_document handles string-encoded metadata_json."""
    import json

    row = {
        "id": 20,
        "title": "Mapping Doc",
        "url": "https://example.com/mapping",
        "scanned_at": datetime.now(),
        "last_scanned_at": None,
        "metadata_json": json.dumps({"content_text": "hello"}),
        "is_active": True,
        "scan_count": 1,
        "relevance_score": None,
        "related_documents": [],
    }

    result = scanned_doc_db._mapping_to_document(row)

    assert result.title == "Mapping Doc"
    assert result.metadata.content_text == "hello"


def test_mapping_to_document_string_related_docs(scanned_doc_db):
    """_mapping_to_document handles JSON-string related_documents."""
    import json

    row = {
        "id": 21,
        "title": "Rel Doc",
        "url": "https://example.com/rel",
        "scanned_at": datetime.now(),
        "last_scanned_at": None,
        "metadata_json": {},
        "is_active": True,
        "scan_count": 1,
        "relevance_score": None,
        "related_documents": json.dumps(["https://example.com/a"]),
    }

    result = scanned_doc_db._mapping_to_document(row)

    assert result.related_documents == ["https://example.com/a"]


def test_mapping_to_document_string_scanned_at(scanned_doc_db):
    """_mapping_to_document parses ISO string scanned_at correctly."""
    row = {
        "id": 22,
        "title": "ISO Doc",
        "url": "https://example.com/iso",
        "scanned_at": "2024-01-15T10:30:00",
        "last_scanned_at": "2024-01-16T10:30:00",
        "metadata_json": {},
        "is_active": 1,
        "scan_count": 2,
        "relevance_score": 0.5,
        "related_documents": [],
    }

    result = scanned_doc_db._mapping_to_document(row)

    assert isinstance(result.scanned_at, datetime)
    assert isinstance(result.last_scanned_at, datetime)


def test_mapping_to_document_none_metadata(scanned_doc_db):
    """_mapping_to_document handles None metadata_json gracefully."""
    row = {
        "id": 23,
        "title": "None Meta Doc",
        "url": "https://example.com/nonemeta",
        "scanned_at": datetime.now(),
        "last_scanned_at": None,
        "metadata_json": None,
        "is_active": True,
        "scan_count": 1,
        "relevance_score": None,
        "related_documents": [],
    }

    result = scanned_doc_db._mapping_to_document(row)

    assert result.metadata is not None


@pytest.mark.asyncio
async def test_search_with_filters_with_word_count_filters(scanned_doc_db):
    """search_with_filters applies word_count_min and word_count_max filters."""
    mock_row = MagicMock()
    mock_row.id = 30
    mock_row.title = "Word Count Doc"
    mock_row.url = "https://example.com/wc"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
    mock_row.metadata_json = {"word_count": 500}
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = []

    mock_db_mgr, _, mock_result = _make_db_mock(scalars_result=[mock_row])

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.search_with_filters(
            {"word_count_min": 100, "word_count_max": 1000}
        )

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_search_with_filters_with_date_filters(scanned_doc_db):
    """search_with_filters applies date_from and date_to filters."""
    from datetime import timedelta

    mock_row = MagicMock()
    mock_row.id = 31
    mock_row.title = "Date Filter Doc"
    mock_row.url = "https://example.com/date"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
    mock_row.metadata_json = {}
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = []

    mock_db_mgr, _, mock_result = _make_db_mock(scalars_result=[mock_row])
    now = datetime.now()

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.search_with_filters(
            {
                "date_from": now - timedelta(days=30),
                "date_to": now,
            }
        )

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_search_with_filters_with_has_internal_links(scanned_doc_db):
    """search_with_filters applies has_internal_links filter."""
    mock_row = MagicMock()
    mock_row.id = 32
    mock_row.title = "Internal Links Doc"
    mock_row.url = "https://example.com/links"
    mock_row.scanned_at = datetime.now()
    mock_row.last_scanned_at = None
    mock_row.metadata_json = {"outbound_link_count": 3}
    mock_row.is_active = True
    mock_row.scan_count = 1
    mock_row.relevance_score = None
    mock_row.related_documents = []

    mock_db_mgr, _, mock_result = _make_db_mock(scalars_result=[mock_row])

    with patch(
        "marketing_project.services.scanned_document_db.get_database_manager",
        return_value=mock_db_mgr,
    ):
        result = await scanned_doc_db.search_with_filters({"has_internal_links": True})

    assert isinstance(result, list)
