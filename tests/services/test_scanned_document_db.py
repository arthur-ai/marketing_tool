"""
Tests for scanned document database service.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from marketing_project.models.scanned_document_db import (
    ScannedDocumentDB,
    ScannedDocumentMetadata,
)
from marketing_project.services.scanned_document_db import (
    ScannedDocumentDatabase,
    get_scanned_document_db,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def scanned_doc_db(temp_db_path):
    """Create a ScannedDocumentDatabase instance."""
    return ScannedDocumentDatabase(db_path=temp_db_path)


def test_initialization(scanned_doc_db, temp_db_path):
    """Test database initialization."""
    assert scanned_doc_db.db_path == temp_db_path
    assert Path(temp_db_path).exists()


def test_ensure_db_directory(scanned_doc_db):
    """Test _ensure_db_directory method."""
    scanned_doc_db._ensure_db_directory()

    # Directory should exist
    assert Path(scanned_doc_db.db_path).parent.exists()


def test_init_database(scanned_doc_db):
    """Test _init_database method."""
    scanned_doc_db._init_database()

    # Database file should exist
    assert Path(scanned_doc_db.db_path).exists()


def test_save_document(scanned_doc_db):
    """Test saving a document."""
    from marketing_project.models.scanned_document_db import ScannedDocumentDB

    document = ScannedDocumentDB(
        title="Test Document",
        url="https://example.com/test",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(
            content_text="Test content",
            content_type="blog_post",
        ),
    )

    doc_id = scanned_doc_db.save_document(document)

    assert doc_id is not None
    assert isinstance(doc_id, int)


def test_get_document_by_url(scanned_doc_db):
    """Test getting document by URL."""
    from datetime import datetime

    from marketing_project.models.scanned_document_db import ScannedDocumentDB

    document = ScannedDocumentDB(
        title="Test Document",
        url="https://example.com/test",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(
            content_text="Test content",
        ),
    )

    doc_id = scanned_doc_db.save_document(document)
    doc = scanned_doc_db.get_document_by_url("https://example.com/test")

    assert doc is not None
    assert doc.id == doc_id
    assert doc.title == "Test Document"


def test_get_all_active_documents(scanned_doc_db):
    """Test getting all active documents."""
    from datetime import datetime

    from marketing_project.models.scanned_document_db import ScannedDocumentDB

    doc1 = ScannedDocumentDB(
        title="Doc 1",
        url="https://example.com/1",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(content_text="Content 1"),
    )
    doc2 = ScannedDocumentDB(
        title="Doc 2",
        url="https://example.com/2",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(content_text="Content 2"),
    )

    scanned_doc_db.save_document(doc1)
    scanned_doc_db.save_document(doc2)

    docs = scanned_doc_db.get_all_active_documents()

    assert len(docs) >= 2


def test_update_document(scanned_doc_db):
    """Test updating a document."""
    from datetime import datetime

    from marketing_project.models.scanned_document_db import ScannedDocumentDB

    document = ScannedDocumentDB(
        title="Test Document",
        url="https://example.com/test",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(content_text="Test content"),
    )

    doc_id = scanned_doc_db.save_document(document)

    updated_doc = ScannedDocumentDB(
        title="Updated Document",
        url="https://example.com/test",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(content_text="Updated content"),
    )

    updated_id = scanned_doc_db.save_document(updated_doc)

    assert updated_id == doc_id  # Should update existing

    doc = scanned_doc_db.get_document_by_url("https://example.com/test")
    assert doc.title == "Updated Document"


def test_delete_document(scanned_doc_db):
    """Test deleting a document."""
    from datetime import datetime

    from marketing_project.models.scanned_document_db import ScannedDocumentDB

    document = ScannedDocumentDB(
        title="Test Document",
        url="https://example.com/test",
        scanned_at=datetime.now(),
        metadata=ScannedDocumentMetadata(content_text="Test content"),
    )

    scanned_doc_db.save_document(document)
    success = scanned_doc_db.delete_document("https://example.com/test")

    assert success is True

    doc = scanned_doc_db.get_document_by_url("https://example.com/test")
    assert doc is None


def test_get_scanned_document_db_singleton():
    """Test that get_scanned_document_db returns a singleton."""
    db1 = get_scanned_document_db()
    db2 = get_scanned_document_db()

    assert db1 is db2
