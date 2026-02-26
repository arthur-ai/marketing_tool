"""
Database Service for Scanned Internal Documents.

Manages storage and retrieval of scanned internal documents with rich metadata.
Uses SQLite by default, but can be configured for other databases.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from marketing_project.models.scanned_document_db import (
    ScannedDocumentDB,
    ScannedDocumentMetadata,
)

logger = logging.getLogger("marketing_project.services.scanned_document_db")

# Database file path
DB_PATH = os.getenv("SCANNED_DOCS_DB_PATH", "data/scanned_documents.db")


class ScannedDocumentDatabase:
    """Database service for scanned internal documents."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection."""
        self.db_path = db_path or DB_PATH
        self._ensure_db_directory()
        self._init_database()

    def _ensure_db_directory(self):
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize database schema with versioning."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create schema version table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Get current schema version
        cursor.execute("SELECT MAX(version) FROM schema_version")
        result = cursor.fetchone()
        current_version = result[0] if result[0] else 0

        # Schema version 1: Initial schema
        if current_version < 1:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scanned_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    scanned_at TEXT NOT NULL,
                    last_scanned_at TEXT,
                    metadata_json TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    scan_count INTEGER DEFAULT 1,
                    relevance_score REAL,
                    related_documents_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indexes for common queries
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_url ON scanned_documents(url)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_is_active ON scanned_documents(is_active)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_scanned_at ON scanned_documents(scanned_at)"
            )

            cursor.execute("INSERT INTO schema_version (version) VALUES (1)")
            logger.info("Applied schema version 1")

        # Schema version 2: Add FTS5 table for full-text search
        if current_version < 2:
            try:
                cursor.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS scanned_documents_fts USING fts5(
                        url UNINDEXED,
                        title,
                        content_text,
                        keywords
                    )
                """
                )

                # Populate FTS5 table with existing data
                # Note: For keywords, we need to convert JSON array to space-separated string
                # SQLite doesn't have great JSON array handling, so we'll do it in Python
                cursor.execute(
                    "SELECT url, title, metadata_json FROM scanned_documents WHERE is_active = 1"
                )
                rows = cursor.fetchall()
                for row in rows:
                    try:
                        metadata = json.loads(row[2])
                        keywords = metadata.get("extracted_keywords", [])
                        keywords_str = (
                            " ".join(keywords) if isinstance(keywords, list) else ""
                        )
                        content_text = metadata.get("content_text", "") or ""

                        cursor.execute(
                            """
                            INSERT INTO scanned_documents_fts(url, title, content_text, keywords)
                            VALUES (?, ?, ?, ?)
                        """,
                            (row[0], row[1], content_text, keywords_str),
                        )
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # Skip rows with invalid JSON
                        continue

                cursor.execute("INSERT INTO schema_version (version) VALUES (2)")
                logger.info("Applied schema version 2 (FTS5)")
            except sqlite3.OperationalError as e:
                logger.warning(
                    f"FTS5 not available: {e}. Full-text search will use fallback."
                )

        # Future migrations would go here:
        # if current_version < 3:
        #     cursor.execute("ALTER TABLE scanned_documents ADD COLUMN new_field TEXT")
        #     cursor.execute("INSERT INTO schema_version (version) VALUES (3)")

        conn.commit()
        conn.close()
        logger.info(
            f"Initialized scanned documents database at {self.db_path} (schema version: {current_version or 1})"
        )

    def save_document(self, document: ScannedDocumentDB) -> int:
        """
        Save or update a scanned document in the database.

        Args:
            document: ScannedDocumentDB to save

        Returns:
            int: Database ID of the saved document
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Check if document exists
            cursor.execute(
                "SELECT id, scan_count FROM scanned_documents WHERE url = ?",
                (document.url,),
            )
            existing = cursor.fetchone()

            metadata_json = json.dumps(
                document.metadata.model_dump(mode="json"), default=str
            )
            related_docs_json = json.dumps(document.related_documents, default=str)

            now = datetime.now(timezone.utc).isoformat()

            if existing:
                # Update existing document
                doc_id, scan_count = existing
                cursor.execute(
                    """
                    UPDATE scanned_documents
                    SET title = ?,
                        scanned_at = ?,
                        last_scanned_at = ?,
                        metadata_json = ?,
                        is_active = ?,
                        scan_count = ?,
                        relevance_score = ?,
                        related_documents_json = ?,
                        updated_at = ?
                    WHERE id = ?
                """,
                    (
                        document.title,
                        document.scanned_at.isoformat(),
                        now,
                        metadata_json,
                        1 if document.is_active else 0,
                        scan_count + 1,
                        document.relevance_score,
                        related_docs_json,
                        now,
                        doc_id,
                    ),
                )

                # Update FTS5 index
                try:
                    # Convert keywords list to space-separated string for FTS5
                    keywords_str = " ".join(document.metadata.extracted_keywords or [])
                    # FTS5 doesn't support ON CONFLICT, so delete first then insert
                    cursor.execute(
                        "DELETE FROM scanned_documents_fts WHERE url = ?",
                        (document.url,),
                    )
                    cursor.execute(
                        """
                        INSERT INTO scanned_documents_fts(url, title, content_text, keywords)
                        VALUES (?, ?, ?, ?)
                    """,
                        (
                            document.url,
                            document.title,
                            document.metadata.content_text or "",
                            keywords_str,
                        ),
                    )
                except sqlite3.OperationalError:
                    pass  # FTS5 table might not exist

                logger.info(f"Updated scanned document: {document.url} (ID: {doc_id})")
                return doc_id
            else:
                # Insert new document
                cursor.execute(
                    """
                    INSERT INTO scanned_documents
                    (title, url, scanned_at, last_scanned_at, metadata_json, is_active,
                     scan_count, relevance_score, related_documents_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        document.title,
                        document.url,
                        document.scanned_at.isoformat(),
                        now,
                        metadata_json,
                        1 if document.is_active else 0,
                        document.scan_count,
                        document.relevance_score,
                        related_docs_json,
                        now,
                        now,
                    ),
                )
                doc_id = cursor.lastrowid

                # Insert into FTS5 index
                try:
                    # Convert keywords list to space-separated string for FTS5
                    keywords_str = " ".join(document.metadata.extracted_keywords or [])
                    cursor.execute(
                        """
                        INSERT INTO scanned_documents_fts(url, title, content_text, keywords)
                        VALUES (?, ?, ?, ?)
                    """,
                        (
                            document.url,
                            document.title,
                            document.metadata.content_text or "",
                            keywords_str,
                        ),
                    )
                except sqlite3.OperationalError:
                    pass  # FTS5 table might not exist

                logger.info(
                    f"Saved new scanned document: {document.url} (ID: {doc_id})"
                )
                return doc_id
        finally:
            conn.commit()
            conn.close()

    def get_document_by_url(self, url: str) -> Optional[ScannedDocumentDB]:
        """Get a document by URL."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM scanned_documents WHERE url = ?", (url,))
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_document(row)
        finally:
            conn.close()

    def get_all_active_documents(self) -> List[ScannedDocumentDB]:
        """Get all active documents."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT * FROM scanned_documents WHERE is_active = 1 ORDER BY scanned_at DESC"
            )
            rows = cursor.fetchall()
            return [self._row_to_document(row) for row in rows]
        finally:
            conn.close()

    def search_by_keywords(
        self, keywords: List[str], limit: int = 50
    ) -> List[ScannedDocumentDB]:
        """
        Search documents by keywords (searches in title, content, and extracted keywords).

        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # Build search query
            keyword_patterns = " OR ".join(["metadata_json LIKE ?" for _ in keywords])
            params = [f"%{kw}%" for kw in keywords]

            query = f"""
                SELECT * FROM scanned_documents
                WHERE is_active = 1
                AND (title LIKE ? OR {keyword_patterns})
                ORDER BY scanned_at DESC
                LIMIT ?
            """
            params = [f"%{keywords[0]}%"] + params + [limit]

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_document(row) for row in rows]
        finally:
            conn.close()

    def get_documents_by_category(self, category: str) -> List[ScannedDocumentDB]:
        """Get documents by category."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT * FROM scanned_documents
                WHERE is_active = 1
                AND metadata_json LIKE ?
                ORDER BY scanned_at DESC
            """,
                (f'%"categories":%"{category}"%',),
            )
            rows = cursor.fetchall()
            return [self._row_to_document(row) for row in rows]
        finally:
            conn.close()

    def get_documents_with_internal_links(self) -> List[ScannedDocumentDB]:
        """Get documents that contain internal links."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT * FROM scanned_documents
                WHERE is_active = 1
                AND metadata_json LIKE '%"internal_links_found":%'
                AND json_extract(metadata_json, '$.outbound_link_count') > 0
                ORDER BY json_extract(metadata_json, '$.outbound_link_count') DESC
            """
            )
            rows = cursor.fetchall()
            return [self._row_to_document(row) for row in rows]
        finally:
            conn.close()

    def get_anchor_text_patterns(self) -> List[str]:
        """Get all unique anchor text patterns from all documents."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT DISTINCT json_each.value
                FROM scanned_documents,
                json_each(json_extract(metadata_json, '$.anchor_text_patterns'))
                WHERE is_active = 1
            """
            )
            patterns = [row[0] for row in cursor.fetchall()]
            return patterns
        finally:
            conn.close()

    def get_commonly_referenced_pages(self, min_links: int = 2) -> List[str]:
        """
        Get pages that are commonly referenced (linked to from multiple documents).

        Args:
            min_links: Minimum number of documents that must link to a page

        Returns:
            List of commonly referenced URLs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Try modern SQLite JSON syntax first
            try:
                cursor.execute(
                    """
                    SELECT json_each.value->>'$.target_url' as target_url, COUNT(*) as link_count
                    FROM scanned_documents,
                    json_each(json_extract(metadata_json, '$.internal_links_found'))
                    WHERE is_active = 1
                    GROUP BY target_url
                    HAVING link_count >= ?
                    ORDER BY link_count DESC
                """,
                    (min_links,),
                )
                results = [row[0] for row in cursor.fetchall()]
                return results
            except sqlite3.OperationalError:
                # Fallback for older SQLite versions - parse JSON in Python
                logger.debug(
                    "Using fallback method for commonly referenced pages (older SQLite)"
                )
                cursor.execute(
                    """
                    SELECT metadata_json FROM scanned_documents WHERE is_active = 1
                """
                )
                url_counts = {}
                for row in cursor.fetchall():
                    try:
                        metadata = json.loads(row[0])
                        links = metadata.get("internal_links_found", [])
                        for link in links:
                            target_url = link.get("target_url")
                            if target_url:
                                url_counts[target_url] = (
                                    url_counts.get(target_url, 0) + 1
                                )
                    except (json.JSONDecodeError, KeyError):
                        continue

                # Filter by min_links and sort
                results = [
                    url for url, count in url_counts.items() if count >= min_links
                ]
                results.sort(key=lambda x: url_counts[x], reverse=True)
                return results
        except Exception as e:
            logger.warning(f"Error getting commonly referenced pages: {e}")
            return []
        finally:
            conn.close()

    def delete_document(self, url: str) -> bool:
        """Delete a document by URL (hard delete)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM scanned_documents WHERE url = ?", (url,))
            # Also remove from FTS5 index if it exists
            try:
                cursor.execute(
                    "DELETE FROM scanned_documents_fts WHERE url = ?", (url,)
                )
            except sqlite3.OperationalError:
                pass  # FTS5 table might not exist yet
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted document: {url}")
            return deleted
        finally:
            conn.close()

    def bulk_delete_documents(self, urls: List[str]) -> int:
        """Bulk delete documents by URLs (hard delete)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            placeholders = ",".join(["?" for _ in urls])
            cursor.execute(
                f"DELETE FROM scanned_documents WHERE url IN ({placeholders})", urls
            )
            # Also remove from FTS5 index
            try:
                cursor.execute(
                    f"DELETE FROM scanned_documents_fts WHERE url IN ({placeholders})",
                    urls,
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()
            deleted = cursor.rowcount
            logger.info(f"Bulk deleted {deleted} documents")
            return deleted
        finally:
            conn.close()

    def bulk_save_documents(self, documents: List[ScannedDocumentDB]) -> Dict[str, int]:
        """Bulk save documents."""
        created = 0
        updated = 0

        for doc in documents:
            doc_id = self.save_document(doc)
            # Check if it was new or updated by checking scan_count
            existing = self.get_document_by_url(doc.url)
            if existing and existing.scan_count > doc.scan_count:
                updated += 1
            else:
                created += 1

        return {"created": created, "updated": updated}

    def bulk_update_categories(self, urls: List[str], categories: List[str]) -> int:
        """Add categories to existing documents (merge, not replace)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        updated = 0
        try:
            for url in urls:
                doc = self.get_document_by_url(url)
                if doc:
                    # Merge categories
                    existing_cats = set(doc.metadata.categories or [])
                    new_cats = set(categories)
                    merged_cats = list(existing_cats | new_cats)

                    # Update metadata
                    doc.metadata.categories = merged_cats
                    metadata_json = json.dumps(doc.metadata.model_dump(), default=str)

                    cursor.execute(
                        """
                        UPDATE scanned_documents
                        SET metadata_json = ?, updated_at = ?
                        WHERE url = ?
                    """,
                        (metadata_json, datetime.now(timezone.utc).isoformat(), url),
                    )
                    updated += 1

            conn.commit()
            logger.info(f"Bulk updated categories for {updated} documents")
        finally:
            conn.close()

        return updated

    def search_with_filters(
        self, filters: Dict[str, Any], limit: int = 50
    ) -> List[ScannedDocumentDB]:
        """
        Search documents with multiple filters.

        Args:
            filters: Dictionary with filter criteria:
                - keywords: List[str] or str
                - category: str
                - content_type: str
                - date_from: str (ISO format)
                - date_to: str (ISO format)
                - word_count_min: int
                - word_count_max: int
                - has_internal_links: bool
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Start with base query
            query = "SELECT * FROM scanned_documents WHERE is_active = 1"
            params = []

            # Apply filters
            if filters.get("category"):
                query += " AND metadata_json LIKE ?"
                params.append(f'%"categories":%"{filters["category"]}"%')

            if filters.get("content_type"):
                query += " AND metadata_json LIKE ?"
                params.append(f'%"content_type":"{filters["content_type"]}"%')

            # Try JSON extract first, fallback to LIKE if it fails
            if filters.get("word_count_min"):
                try:
                    query += " AND json_extract(metadata_json, '$.word_count') >= ?"
                    params.append(filters["word_count_min"])
                except Exception:
                    # Fallback for older SQLite versions
                    query += " AND metadata_json LIKE ?"
                    params.append(f'%"word_count":{filters["word_count_min"]}%')

            if filters.get("word_count_max"):
                try:
                    query += " AND json_extract(metadata_json, '$.word_count') <= ?"
                    params.append(filters["word_count_max"])
                except Exception:
                    # Fallback for older SQLite versions
                    query += " AND metadata_json LIKE ?"
                    params.append(f'%"word_count":{filters["word_count_max"]}%')

            if filters.get("has_internal_links"):
                try:
                    query += (
                        " AND json_extract(metadata_json, '$.outbound_link_count') > 0"
                    )
                except Exception:
                    # Fallback
                    query += " AND metadata_json LIKE ?"
                    params.append('%"outbound_link_count":%')

            if filters.get("date_from"):
                query += " AND scanned_at >= ?"
                params.append(filters["date_from"])

            if filters.get("date_to"):
                query += " AND scanned_at <= ?"
                params.append(filters["date_to"])

            # Keyword search (if provided)
            if filters.get("keywords"):
                keywords = filters["keywords"]
                if isinstance(keywords, str):
                    keywords = [keywords]
                keyword_patterns = " OR ".join(
                    ["metadata_json LIKE ?" for _ in keywords]
                )
                query += f" AND (title LIKE ? OR {keyword_patterns})"
                params.append(f"%{keywords[0]}%")
                params.extend([f"%{kw}%" for kw in keywords])

            query += " ORDER BY scanned_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_document(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"SQLite error in search_with_filters: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in search_with_filters: {e}", exc_info=True)
            return []
        finally:
            if conn:
                conn.close()

    def full_text_search(self, query: str, limit: int = 50) -> List[ScannedDocumentDB]:
        """
        Full-text search using FTS5.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Try FTS5 search
            cursor.execute(
                """
                SELECT s.* FROM scanned_documents s
                JOIN scanned_documents_fts fts ON s.url = fts.url
                WHERE fts MATCH ?
                AND s.is_active = 1
                ORDER BY scanned_at DESC
                LIMIT ?
            """,
                (query, limit),
            )
            rows = cursor.fetchall()
            return [self._row_to_document(row) for row in rows]
        except sqlite3.OperationalError as e:
            # FTS5 table might not exist, fallback to keyword search
            logger.warning(
                f"FTS5 table not found or error: {e}, falling back to keyword search"
            )
            try:
                return self.search_by_keywords([query], limit=limit)
            except Exception as fallback_error:
                logger.error(f"Fallback keyword search also failed: {fallback_error}")
                return []
        except sqlite3.Error as e:
            logger.error(f"SQLite error in full_text_search: {e}")
            # Try fallback
            try:
                return self.search_by_keywords([query], limit=limit)
            except Exception:
                return []
        except Exception as e:
            logger.error(f"Error in full_text_search: {e}", exc_info=True)
            return []
        finally:
            if conn:
                conn.close()

    def calculate_relationships(
        self, document: ScannedDocumentDB
    ) -> List[tuple[str, float]]:
        """
        Calculate relationships for a document.

        Args:
            document: Document to calculate relationships for

        Returns:
            List of (url, score) tuples for top 10 related documents
        """
        all_docs = self.get_all_active_documents()
        relationships = []

        doc_keywords = set(document.metadata.extracted_keywords or [])
        doc_topics = set(document.metadata.topics or [])
        doc_categories = set(document.metadata.categories or [])
        doc_link_targets = {
            link.get("target_url")
            for link in (document.metadata.internal_links_found or [])
            if link and link.get("target_url")
        }

        for other_doc in all_docs:
            if other_doc.url == document.url:
                continue

            other_keywords = set(other_doc.metadata.extracted_keywords or [])
            other_topics = set(other_doc.metadata.topics or [])
            other_categories = set(other_doc.metadata.categories or [])
            other_link_targets = {
                link.get("target_url")
                for link in (other_doc.metadata.internal_links_found or [])
                if link and link.get("target_url")
            }

            # Calculate Jaccard similarities
            def jaccard(set1: set, set2: set) -> float:
                if not set1 and not set2:
                    return 0.0
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                return intersection / union if union > 0 else 0.0

            keyword_score = jaccard(doc_keywords, other_keywords) * 0.3
            topic_score = jaccard(doc_topics, other_topics) * 0.3
            category_score = jaccard(doc_categories, other_categories) * 0.2

            # Shared links score
            shared_links = len(doc_link_targets & other_link_targets)
            max_links = max(len(doc_link_targets), len(other_link_targets), 1)
            shared_links_score = (shared_links / max_links) * 0.2

            # Combined score
            total_score = (
                keyword_score + topic_score + category_score + shared_links_score
            )

            if total_score > 0:
                relationships.append((other_doc.url, total_score))

        # Sort by score and return top 10
        relationships.sort(key=lambda x: x[1], reverse=True)
        return relationships[:10]

    def update_relationships(self, document: ScannedDocumentDB) -> ScannedDocumentDB:
        """Calculate and update relationships for a document."""
        relationships = self.calculate_relationships(document)
        document.related_documents = [url for url, _ in relationships]
        return document

    def _row_to_document(self, row: sqlite3.Row) -> ScannedDocumentDB:
        """Convert database row to ScannedDocumentDB model."""
        metadata_dict = json.loads(row["metadata_json"])
        related_docs = (
            json.loads(row["related_documents_json"])
            if row["related_documents_json"]
            else []
        )

        # Sanitize internal_links_found to remove None values
        if (
            "internal_links_found" in metadata_dict
            and metadata_dict["internal_links_found"]
        ):
            metadata_dict["internal_links_found"] = [
                link
                for link in metadata_dict["internal_links_found"]
                if link is not None and isinstance(link, dict)
            ]

        return ScannedDocumentDB(
            id=row["id"],
            title=row["title"],
            url=row["url"],
            scanned_at=datetime.fromisoformat(row["scanned_at"]),
            last_scanned_at=(
                datetime.fromisoformat(row["last_scanned_at"])
                if row["last_scanned_at"]
                else None
            ),
            metadata=ScannedDocumentMetadata(**metadata_dict),
            is_active=bool(row["is_active"]),
            scan_count=row["scan_count"],
            relevance_score=row["relevance_score"],
            related_documents=related_docs,
        )


# Singleton instance
_db_instance: Optional[ScannedDocumentDatabase] = None


def get_scanned_document_db(db_path: Optional[str] = None) -> ScannedDocumentDatabase:
    """Get or create the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ScannedDocumentDatabase(db_path)
    return _db_instance
