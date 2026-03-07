"""
Database Service for Scanned Internal Documents.

Manages storage and retrieval of scanned internal documents with rich metadata.
Uses PostgreSQL via SQLAlchemy async (same connection as the rest of the app).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Text, and_, cast, or_, select, text

from marketing_project.models.db_models import ScannedDocumentModel
from marketing_project.models.scanned_document_db import (
    ScannedDocumentDB,
    ScannedDocumentMetadata,
)
from marketing_project.services.database import get_database_manager

logger = logging.getLogger("marketing_project.services.scanned_document_db")


class ScannedDocumentDatabase:
    """Database service for scanned internal documents using PostgreSQL."""

    async def save_document(self, document: ScannedDocumentDB) -> int:
        """
        Save or update a scanned document in the database.

        Returns:
            int: Database ID of the saved document
        """
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            raise RuntimeError("Database not initialized")

        metadata_dict = document.metadata.model_dump(mode="json")
        related_docs = list(document.related_documents or [])
        now = datetime.now(timezone.utc)

        async with db_manager.get_session() as session:
            stmt = select(ScannedDocumentModel).where(
                ScannedDocumentModel.url == document.url
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.title = document.title
                existing.scanned_at = document.scanned_at
                existing.last_scanned_at = now
                existing.metadata_json = metadata_dict
                existing.is_active = document.is_active
                existing.scan_count = existing.scan_count + 1
                existing.relevance_score = document.relevance_score
                existing.related_documents = related_docs
                existing.updated_at = now
                doc_id = existing.id
                logger.info(f"Updated scanned document: {document.url} (ID: {doc_id})")
            else:
                new_doc = ScannedDocumentModel(
                    title=document.title,
                    url=document.url,
                    scanned_at=document.scanned_at,
                    last_scanned_at=now,
                    metadata_json=metadata_dict,
                    is_active=document.is_active,
                    scan_count=document.scan_count,
                    relevance_score=document.relevance_score,
                    related_documents=related_docs,
                )
                session.add(new_doc)
                await session.flush()
                doc_id = new_doc.id
                logger.info(
                    f"Saved new scanned document: {document.url} (ID: {doc_id})"
                )

            return doc_id

    async def get_document_by_url(self, url: str) -> Optional[ScannedDocumentDB]:
        """Get a document by URL."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return None

        async with db_manager.get_session() as session:
            stmt = select(ScannedDocumentModel).where(ScannedDocumentModel.url == url)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return self._model_to_document(row)

    async def get_all_active_documents(self) -> List[ScannedDocumentDB]:
        """Get all active documents."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return []

        async with db_manager.get_session() as session:
            stmt = (
                select(ScannedDocumentModel)
                .where(ScannedDocumentModel.is_active == True)
                .order_by(ScannedDocumentModel.scanned_at.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._model_to_document(row) for row in rows]

    async def search_by_keywords(
        self, keywords: List[str], limit: int = 50
    ) -> List[ScannedDocumentDB]:
        """Search documents by keywords (title and metadata text)."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return []

        async with db_manager.get_session() as session:
            conditions = []
            for kw in keywords:
                pattern = f"%{kw}%"
                conditions.append(ScannedDocumentModel.title.ilike(pattern))
                conditions.append(
                    cast(ScannedDocumentModel.metadata_json, Text).ilike(pattern)
                )
            stmt = (
                select(ScannedDocumentModel)
                .where(
                    and_(
                        ScannedDocumentModel.is_active == True,
                        or_(*conditions),
                    )
                )
                .order_by(ScannedDocumentModel.scanned_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._model_to_document(row) for row in rows]

    async def get_documents_by_category(self, category: str) -> List[ScannedDocumentDB]:
        """Get documents by category."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return []

        async with db_manager.get_session() as session:
            # Use JSONB array contains operator via raw text fragment
            stmt = (
                select(ScannedDocumentModel)
                .where(
                    and_(
                        ScannedDocumentModel.is_active == True,
                        ScannedDocumentModel.metadata_json["categories"].contains(
                            [category]
                        ),
                    )
                )
                .order_by(ScannedDocumentModel.scanned_at.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._model_to_document(row) for row in rows]

    async def get_documents_with_internal_links(self) -> List[ScannedDocumentDB]:
        """Get documents that contain internal links."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return []

        async with db_manager.get_session() as session:
            result = await session.execute(
                text(
                    """
                    SELECT * FROM scanned_documents
                    WHERE is_active = true
                    AND COALESCE((metadata_json->>'outbound_link_count')::int, 0) > 0
                    ORDER BY (metadata_json->>'outbound_link_count')::int DESC
                    """
                )
            )
            rows = result.mappings().all()
            return [self._mapping_to_document(row) for row in rows]

    async def get_anchor_text_patterns(self) -> List[str]:
        """Get all unique anchor text patterns from all documents."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return []

        try:
            async with db_manager.get_session() as session:
                result = await session.execute(
                    text(
                        """
                        SELECT DISTINCT value
                        FROM scanned_documents,
                        LATERAL jsonb_array_elements_text(
                            CASE WHEN jsonb_typeof(metadata_json->'anchor_text_patterns') = 'array'
                                 THEN metadata_json->'anchor_text_patterns'
                                 ELSE '[]'::jsonb
                            END
                        ) AS value
                        WHERE is_active = true AND value IS NOT NULL
                        """
                    )
                )
                return [row[0] for row in result.all()]
        except Exception as e:
            logger.warning(f"Error getting anchor text patterns: {e}")
            return []

    async def get_commonly_referenced_pages(self, min_links: int = 2) -> List[str]:
        """Get pages commonly referenced (linked to from multiple documents)."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return []

        try:
            async with db_manager.get_session() as session:
                result = await session.execute(
                    text(
                        """
                        SELECT elem->>'target_url' AS target_url, COUNT(*) AS cnt
                        FROM scanned_documents,
                        LATERAL jsonb_array_elements(
                            CASE WHEN jsonb_typeof(metadata_json->'internal_links_found') = 'array'
                                 THEN metadata_json->'internal_links_found'
                                 ELSE '[]'::jsonb
                            END
                        ) AS elem
                        WHERE is_active = true
                        GROUP BY target_url
                        HAVING elem->>'target_url' IS NOT NULL AND COUNT(*) >= :min_links
                        ORDER BY cnt DESC
                        """
                    ),
                    {"min_links": min_links},
                )
                return [row[0] for row in result.all() if row[0]]
        except Exception as e:
            logger.warning(f"Error getting commonly referenced pages: {e}")
            return []

    async def delete_document(self, url: str) -> bool:
        """Delete a document by URL (hard delete)."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return False

        async with db_manager.get_session() as session:
            result = await session.execute(
                text("DELETE FROM scanned_documents WHERE url = :url RETURNING id"),
                {"url": url},
            )
            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Deleted document: {url}")
            return deleted

    async def bulk_delete_documents(self, urls: List[str]) -> int:
        """Bulk delete documents by URLs."""
        if not urls:
            return 0

        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return 0

        async with db_manager.get_session() as session:
            result = await session.execute(
                text(
                    "DELETE FROM scanned_documents WHERE url = ANY(:urls) RETURNING id"
                ),
                {"urls": list(urls)},
            )
            deleted = result.rowcount
            logger.info(f"Bulk deleted {deleted} documents")
            return deleted

    async def bulk_save_documents(
        self, documents: List[ScannedDocumentDB]
    ) -> Dict[str, int]:
        """Bulk save documents."""
        created = 0
        updated = 0
        for doc in documents:
            existing = await self.get_document_by_url(doc.url)
            await self.save_document(doc)
            if existing:
                updated += 1
            else:
                created += 1
        return {"created": created, "updated": updated}

    async def bulk_update_categories(
        self, urls: List[str], categories: List[str]
    ) -> int:
        """Add categories to existing documents (merge, not replace)."""
        updated = 0
        for url in urls:
            doc = await self.get_document_by_url(url)
            if doc:
                existing_cats = set(doc.metadata.categories or [])
                new_cats = set(categories)
                doc.metadata.categories = list(existing_cats | new_cats)
                await self.save_document(doc)
                updated += 1
        logger.info(f"Bulk updated categories for {updated} documents")
        return updated

    async def search_with_filters(
        self, filters: Dict[str, Any], limit: int = 50
    ) -> List[ScannedDocumentDB]:
        """Search documents with multiple filters."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return []

        try:
            async with db_manager.get_session() as session:
                conditions = [ScannedDocumentModel.is_active == True]

                if filters.get("category"):
                    conditions.append(
                        ScannedDocumentModel.metadata_json["categories"].contains(
                            [filters["category"]]
                        )
                    )

                if filters.get("content_type"):
                    conditions.append(
                        ScannedDocumentModel.metadata_json["content_type"].as_string()
                        == filters["content_type"]
                    )

                if filters.get("word_count_min"):
                    conditions.append(
                        cast(
                            ScannedDocumentModel.metadata_json[
                                "word_count"
                            ].as_string(),
                            Text,
                        ).cast(Text)
                        != None
                    )
                    # Use raw text for numeric comparison
                    conditions.append(
                        text(
                            f"(metadata_json->>'word_count')::int >= {int(filters['word_count_min'])}"
                        )
                    )

                if filters.get("word_count_max"):
                    conditions.append(
                        text(
                            f"(metadata_json->>'word_count')::int <= {int(filters['word_count_max'])}"
                        )
                    )

                if filters.get("has_internal_links"):
                    conditions.append(
                        text(
                            "COALESCE((metadata_json->>'outbound_link_count')::int, 0) > 0"
                        )
                    )

                if filters.get("date_from"):
                    conditions.append(
                        ScannedDocumentModel.scanned_at >= filters["date_from"]
                    )

                if filters.get("date_to"):
                    conditions.append(
                        ScannedDocumentModel.scanned_at <= filters["date_to"]
                    )

                if filters.get("keywords"):
                    keywords = filters["keywords"]
                    if isinstance(keywords, str):
                        keywords = [keywords]
                    kw_conditions = []
                    for kw in keywords:
                        pattern = f"%{kw}%"
                        kw_conditions.append(ScannedDocumentModel.title.ilike(pattern))
                        kw_conditions.append(
                            cast(ScannedDocumentModel.metadata_json, Text).ilike(
                                pattern
                            )
                        )
                    conditions.append(or_(*kw_conditions))

                stmt = (
                    select(ScannedDocumentModel)
                    .where(and_(*conditions))
                    .order_by(ScannedDocumentModel.scanned_at.desc())
                    .limit(limit)
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [self._model_to_document(row) for row in rows]
        except Exception as e:
            logger.error(f"Error in search_with_filters: {e}", exc_info=True)
            return []

    async def full_text_search(
        self, query: str, limit: int = 50
    ) -> List[ScannedDocumentDB]:
        """Full-text search using PostgreSQL tsvector."""
        db_manager = get_database_manager()
        if not db_manager.is_initialized:
            return []

        try:
            async with db_manager.get_session() as session:
                result = await session.execute(
                    text(
                        """
                        SELECT * FROM scanned_documents
                        WHERE is_active = true
                        AND to_tsvector('english',
                            title || ' ' || COALESCE(metadata_json->>'content_text', '')
                        ) @@ plainto_tsquery('english', :query)
                        ORDER BY scanned_at DESC
                        LIMIT :limit
                        """
                    ),
                    {"query": query, "limit": limit},
                )
                rows = result.mappings().all()
                if rows:
                    return [self._mapping_to_document(row) for row in rows]
                # Fallback to keyword ILIKE search
                return await self.search_by_keywords([query], limit=limit)
        except Exception as e:
            logger.warning(
                f"Full-text search failed ({e}), falling back to keyword search"
            )
            try:
                return await self.search_by_keywords([query], limit=limit)
            except Exception:
                return []

    def calculate_relationships(
        self, document: ScannedDocumentDB, all_docs: List[ScannedDocumentDB]
    ) -> List[tuple]:
        """
        Calculate relationships for a document given all active documents.

        Returns:
            List of (url, score) tuples for top 10 related documents
        """
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

            def jaccard(set1: set, set2: set) -> float:
                if not set1 and not set2:
                    return 0.0
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                return intersection / union if union > 0 else 0.0

            keyword_score = jaccard(doc_keywords, other_keywords) * 0.3
            topic_score = jaccard(doc_topics, other_topics) * 0.3
            category_score = jaccard(doc_categories, other_categories) * 0.2

            shared_links = len(doc_link_targets & other_link_targets)
            max_links = max(len(doc_link_targets), len(other_link_targets), 1)
            shared_links_score = (shared_links / max_links) * 0.2

            total_score = (
                keyword_score + topic_score + category_score + shared_links_score
            )
            if total_score > 0:
                relationships.append((other_doc.url, total_score))

        relationships.sort(key=lambda x: x[1], reverse=True)
        return relationships[:10]

    async def update_relationships(
        self, document: ScannedDocumentDB
    ) -> ScannedDocumentDB:
        """Calculate and update relationships for a document."""
        all_docs = await self.get_all_active_documents()
        relationships = self.calculate_relationships(document, all_docs)
        document.related_documents = [url for url, _ in relationships]
        return document

    def _model_to_document(self, row: ScannedDocumentModel) -> ScannedDocumentDB:
        """Convert SQLAlchemy model instance to ScannedDocumentDB."""
        metadata_dict = row.metadata_json if isinstance(row.metadata_json, dict) else {}
        related_docs = (
            row.related_documents if isinstance(row.related_documents, list) else []
        )

        # Sanitize internal_links_found
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
            id=row.id,
            title=row.title,
            url=row.url,
            scanned_at=row.scanned_at,
            last_scanned_at=row.last_scanned_at,
            metadata=ScannedDocumentMetadata(**metadata_dict),
            is_active=row.is_active,
            scan_count=row.scan_count,
            relevance_score=row.relevance_score,
            related_documents=related_docs,
        )

    def _mapping_to_document(self, row: Any) -> ScannedDocumentDB:
        """Convert a raw SQL result mapping to ScannedDocumentDB."""
        import json as _json

        metadata_raw = row["metadata_json"]
        if isinstance(metadata_raw, str):
            metadata_dict = _json.loads(metadata_raw)
        elif isinstance(metadata_raw, dict):
            metadata_dict = metadata_raw
        else:
            metadata_dict = {}

        related_raw = row["related_documents"]
        if isinstance(related_raw, str):
            related_docs = _json.loads(related_raw)
        elif isinstance(related_raw, list):
            related_docs = related_raw
        else:
            related_docs = []

        if (
            "internal_links_found" in metadata_dict
            and metadata_dict["internal_links_found"]
        ):
            metadata_dict["internal_links_found"] = [
                link
                for link in metadata_dict["internal_links_found"]
                if link is not None and isinstance(link, dict)
            ]

        scanned_at = row["scanned_at"]
        if isinstance(scanned_at, str):
            scanned_at = datetime.fromisoformat(scanned_at)

        last_scanned_at = row.get("last_scanned_at")
        if isinstance(last_scanned_at, str):
            last_scanned_at = datetime.fromisoformat(last_scanned_at)

        return ScannedDocumentDB(
            id=row["id"],
            title=row["title"],
            url=row["url"],
            scanned_at=scanned_at,
            last_scanned_at=last_scanned_at,
            metadata=ScannedDocumentMetadata(**metadata_dict),
            is_active=bool(row["is_active"]),
            scan_count=row["scan_count"],
            relevance_score=row.get("relevance_score"),
            related_documents=related_docs,
        )


# Singleton instance
_db_instance: Optional[ScannedDocumentDatabase] = None


def get_scanned_document_db() -> ScannedDocumentDatabase:
    """Get or create the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = ScannedDocumentDatabase()
    return _db_instance
