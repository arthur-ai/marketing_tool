"""
Database content source implementation for Marketing Project.

This module provides database content sources that can fetch content from various
database systems including SQL and NoSQL databases.

Classes:
    DatabaseContentSource: Base class for database content sources
    SQLContentSource: Fetches content from SQL databases
    MongoDBContentSource: Fetches content from MongoDB
    RedisContentSource: Fetches content from Redis
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiosqlite
import aiomysql
import asyncpg
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient

from marketing_project.core.content_sources import (
    ContentSource,
    ContentSourceResult,
    ContentSourceStatus,
    DatabaseSourceConfig,
)

logger = logging.getLogger("marketing_project.services.database_source")


class DatabaseContentSource(ContentSource):
    """Base class for database content sources."""

    def __init__(self, config: DatabaseSourceConfig):
        super().__init__(config)
        self.config: DatabaseSourceConfig = config
        self.connection = None
        self.connected = False

    async def initialize(self) -> bool:
        """Initialize database connection."""
        # This method should be implemented by subclasses
        return False

    async def fetch_content(self, limit: Optional[int] = None) -> ContentSourceResult:
        """Fetch content from database."""
        # This method should be implemented by subclasses
        return ContentSourceResult(
            source_name=self.config.name,
            content_items=[],
            total_count=0,
            success=False,
            error_message="Method not implemented",
        )

    async def health_check(self) -> bool:
        """Check database connection health."""
        # This method should be implemented by subclasses
        return False

    async def cleanup(self) -> None:
        """Close database connection."""
        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
            finally:
                self.connection = None
                self.connected = False


class SQLContentSource(DatabaseContentSource):
    """Content source for SQL databases."""

    def __init__(self, config: DatabaseSourceConfig):
        super().__init__(config)
        self.connection = None

    async def initialize(self) -> bool:
        """Initialize SQL database connection."""
        try:
            # Import appropriate database driver based on connection string
            connection_string = self.config.connection_string

            if connection_string.startswith("sqlite"):
                self.connection = await aiosqlite.connect(
                    connection_string.replace("sqlite://", "")
                )
            elif connection_string.startswith("postgresql"):
                self.connection = await asyncpg.connect(connection_string)
            elif connection_string.startswith("mysql"):
                # Parse MySQL connection string
                # Format: mysql://user:password@host:port/database
                parts = connection_string.replace("mysql://", "").split("/")
                db_name = parts[1] if len(parts) > 1 else "test"
                auth_host = parts[0].split("@")
                if len(auth_host) == 2:
                    auth, host_port = auth_host
                    user, password = auth.split(":") if ":" in auth else (auth, "")
                    host, port = (
                        host_port.split(":") if ":" in host_port else (host_port, 3306)
                    )
                    port = int(port)
                else:
                    user, password, host, port = "root", "", "localhost", 3306

                self.connection = await aiomysql.connect(
                    host=host, port=port, user=user, password=password, db=db_name
                )
            else:
                raise ValueError(
                    f"Unsupported database type in connection string: {connection_string}"
                )

            self.connected = True
            self.status = ContentSourceStatus.ACTIVE
            logger.info(f"SQL database source {self.config.name} initialized")
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize SQL database source {self.config.name}: {e}"
            )
            self.status = ContentSourceStatus.ERROR
            return False

    async def fetch_content(self, limit: Optional[int] = None) -> ContentSourceResult:
        """Fetch content from SQL database."""
        if not self.connected or not self.connection:
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message="Database not connected",
            )

        try:
            # Build query
            query = self.config.query
            if self.config.where_clause:
                query += f" WHERE {self.config.where_clause}"
            if self.config.order_by:
                query += f" ORDER BY {self.config.order_by}"

            # Apply limit
            db_limit = limit or self.config.limit or self.config.batch_size
            if db_limit:
                query += f" LIMIT {db_limit}"

            # Execute query
            if hasattr(self.connection, "execute"):
                # SQLite
                cursor = await self.connection.execute(query)
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            else:
                # PostgreSQL/MySQL
                rows = await self.connection.fetch(query)
                columns = list(rows[0].keys()) if rows else []

            # Convert rows to content items
            content_items = []
            for row in rows:
                if isinstance(row, (list, tuple)):
                    # Convert tuple/list to dict
                    row_dict = dict(zip(columns, row))
                else:
                    # Already a dict (asyncpg)
                    row_dict = dict(row)

                content_item = self._convert_row_to_content_item(row_dict)
                if content_item:
                    content_items.append(content_item)

            return ContentSourceResult(
                source_name=self.config.name,
                content_items=content_items,
                total_count=len(content_items),
                success=True,
                metadata={"query": query, "rows_returned": len(content_items)},
            )

        except Exception as e:
            logger.error(
                f"Failed to fetch content from SQL database {self.config.name}: {e}"
            )
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message=str(e),
            )

    def _convert_row_to_content_item(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert database row to content item."""
        try:
            # Map common database fields to content item fields
            content_item = {
                "id": str(row.get("id", row.get("_id", hash(str(row))))),
                "title": row.get(
                    "title", row.get("name", row.get("subject", "Untitled"))
                ),
                "content": row.get(
                    "content",
                    row.get("body", row.get("text", row.get("description", ""))),
                ),
                "snippet": row.get(
                    "snippet", row.get("excerpt", row.get("summary", ""))
                ),
                "created_at": row.get(
                    "created_at",
                    row.get("date", row.get("timestamp", datetime.now().isoformat())),
                ),
                "source_url": row.get("url", row.get("link", "")),
                "metadata": {
                    "database_source": self.config.name,
                    "table": self.config.table_name,
                    "raw_row": row,
                },
            }

            # Add type-specific fields based on available columns
            if "speakers" in row or "participants" in row:
                content_item.update(
                    {
                        "speakers": row.get("speakers", row.get("participants", [])),
                        "duration": row.get("duration"),
                        "transcript_type": row.get("type", "podcast"),
                    }
                )
            elif "version" in row or "release" in row:
                content_item.update(
                    {
                        "version": row.get("version", row.get("release", "1.0.0")),
                        "changes": row.get("changes", row.get("updates", [])),
                        "features": row.get("features", row.get("new_features", [])),
                        "bug_fixes": row.get("bug_fixes", row.get("fixes", [])),
                    }
                )
            elif "author" in row or "writer" in row:
                content_item.update(
                    {
                        "author": row.get("author", row.get("writer")),
                        "tags": row.get("tags", row.get("categories", [])),
                        "category": row.get("category", row.get("section")),
                        "word_count": row.get(
                            "word_count", len(content_item["content"].split())
                        ),
                    }
                )

            return content_item

        except Exception as e:
            logger.warning(f"Failed to convert database row to content item: {e}")
            return None

    async def health_check(self) -> bool:
        """Check SQL database connection health."""
        if not self.connected or not self.connection:
            return False

        try:
            # Execute a simple query to test connection
            if hasattr(self.connection, "execute"):
                await self.connection.execute("SELECT 1")
            else:
                await self.connection.fetch("SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False


class MongoDBContentSource(DatabaseContentSource):
    """Content source for MongoDB databases."""

    def __init__(self, config: DatabaseSourceConfig):
        super().__init__(config)
        self.client = None
        self.database = None
        self.collection = None

    async def initialize(self) -> bool:
        """Initialize MongoDB connection."""
        try:
            # Parse connection string
            connection_string = self.config.connection_string
            if not connection_string.startswith("mongodb"):
                connection_string = f"mongodb://{connection_string}"

            # Connect to MongoDB
            self.client = AsyncIOMotorClient(connection_string)

            # Get database and collection
            db_name = self.config.metadata.get("database", "marketing_project")
            collection_name = self.config.table_name or "content"

            self.database = self.client[db_name]
            self.collection = self.database[collection_name]

            # Test connection
            await self.client.admin.command("ping")

            self.connected = True
            self.status = ContentSourceStatus.ACTIVE
            logger.info(f"MongoDB source {self.config.name} initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MongoDB source {self.config.name}: {e}")
            self.status = ContentSourceStatus.ERROR
            return False

    async def fetch_content(self, limit: Optional[int] = None) -> ContentSourceResult:
        """Fetch content from MongoDB."""
        if not self.connected or not self.collection:
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message="MongoDB not connected",
            )

        try:
            # Build query
            query = {}
            if self.config.where_clause:
                # Parse where clause (simple JSON format)
                try:
                    query = json.loads(self.config.where_clause)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid where clause JSON: {self.config.where_clause}"
                    )

            # Build projection (select specific fields)
            projection = None
            if self.config.columns:
                projection = {col: 1 for col in self.config.columns}

            # Build sort
            sort = None
            if self.config.order_by:
                sort = [(self.config.order_by, 1)]  # 1 for ascending, -1 for descending

            # Execute query
            cursor = self.collection.find(query, projection)

            if sort:
                cursor = cursor.sort(sort)

            db_limit = limit or self.config.limit or self.config.batch_size
            if db_limit:
                cursor = cursor.limit(db_limit)

            # Convert to list
            rows = await cursor.to_list(length=db_limit)

            # Convert to content items
            content_items = []
            for row in rows:
                content_item = self._convert_document_to_content_item(row)
                if content_item:
                    content_items.append(content_item)

            return ContentSourceResult(
                source_name=self.config.name,
                content_items=content_items,
                total_count=len(content_items),
                success=True,
                metadata={"query": query, "documents_returned": len(content_items)},
            )

        except Exception as e:
            logger.error(
                f"Failed to fetch content from MongoDB {self.config.name}: {e}"
            )
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message=str(e),
            )

    def _convert_document_to_content_item(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MongoDB document to content item."""
        try:
            # Handle ObjectId
            if "_id" in doc:
                doc["id"] = str(doc["_id"])

            content_item = {
                "id": str(doc.get("id", doc.get("_id", hash(str(doc))))),
                "title": doc.get(
                    "title", doc.get("name", doc.get("subject", "Untitled"))
                ),
                "content": doc.get(
                    "content",
                    doc.get("body", doc.get("text", doc.get("description", ""))),
                ),
                "snippet": doc.get(
                    "snippet", doc.get("excerpt", doc.get("summary", ""))
                ),
                "created_at": doc.get(
                    "created_at",
                    doc.get("date", doc.get("timestamp", datetime.now().isoformat())),
                ),
                "source_url": doc.get("url", doc.get("link", "")),
                "metadata": {
                    "database_source": self.config.name,
                    "collection": self.config.table_name,
                    "raw_document": doc,
                },
            }

            # Add type-specific fields
            if "speakers" in doc or "participants" in doc:
                content_item.update(
                    {
                        "speakers": doc.get("speakers", doc.get("participants", [])),
                        "duration": doc.get("duration"),
                        "transcript_type": doc.get("type", "podcast"),
                    }
                )
            elif "version" in doc or "release" in doc:
                content_item.update(
                    {
                        "version": doc.get("version", doc.get("release", "1.0.0")),
                        "changes": doc.get("changes", doc.get("updates", [])),
                        "features": doc.get("features", doc.get("new_features", [])),
                        "bug_fixes": doc.get("bug_fixes", doc.get("fixes", [])),
                    }
                )
            elif "author" in doc or "writer" in doc:
                content_item.update(
                    {
                        "author": doc.get("author", doc.get("writer")),
                        "tags": doc.get("tags", doc.get("categories", [])),
                        "category": doc.get("category", doc.get("section")),
                        "word_count": doc.get(
                            "word_count", len(content_item["content"].split())
                        ),
                    }
                )

            return content_item

        except Exception as e:
            logger.warning(f"Failed to convert MongoDB document to content item: {e}")
            return None

    async def health_check(self) -> bool:
        """Check MongoDB connection health."""
        if not self.connected or not self.client:
            return False

        try:
            await self.client.admin.command("ping")
            return True
        except Exception as e:
            logger.warning(f"MongoDB health check failed: {e}")
            return False

    async def cleanup(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
        await super().cleanup()


class RedisContentSource(DatabaseContentSource):
    """Content source for Redis databases."""

    def __init__(self, config: DatabaseSourceConfig):
        super().__init__(config)
        self.redis = None

    async def initialize(self) -> bool:
        """Initialize Redis connection."""
        try:
            # Parse connection string
            connection_string = self.config.connection_string
            if connection_string.startswith("redis://"):
                self.redis = redis.from_url(connection_string)
            else:
                # Parse host:port format
                host, port = (
                    connection_string.split(":")
                    if ":" in connection_string
                    else (connection_string, 6379)
                )
                self.redis = redis.Redis(host=host, port=int(port))

            # Test connection
            await self.redis.ping()

            self.connected = True
            self.status = ContentSourceStatus.ACTIVE
            logger.info(f"Redis source {self.config.name} initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Redis source {self.config.name}: {e}")
            self.status = ContentSourceStatus.ERROR
            return False

    async def fetch_content(self, limit: Optional[int] = None) -> ContentSourceResult:
        """Fetch content from Redis."""
        if not self.connected or not self.redis:
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message="Redis not connected",
            )

        try:
            content_items = []

            # Get keys matching pattern
            pattern = self.config.metadata.get("key_pattern", "*")
            keys = await self.redis.keys(pattern)

            # Apply limit
            if limit:
                keys = keys[:limit]

            # Fetch values
            for key in keys:
                try:
                    value = await self.redis.get(key)
                    if value:
                        # Try to parse as JSON
                        try:
                            data = json.loads(value)
                        except json.JSONDecodeError:
                            # Treat as plain text
                            data = {"content": value.decode("utf-8")}

                        content_item = self._convert_redis_value_to_content_item(
                            data, key
                        )
                        if content_item:
                            content_items.append(content_item)

                except Exception as e:
                    logger.warning(f"Failed to process Redis key {key}: {e}")
                    continue

            return ContentSourceResult(
                source_name=self.config.name,
                content_items=content_items,
                total_count=len(content_items),
                success=True,
                metadata={"keys_processed": len(keys)},
            )

        except Exception as e:
            logger.error(f"Failed to fetch content from Redis {self.config.name}: {e}")
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message=str(e),
            )

    def _convert_redis_value_to_content_item(
        self, data: Dict[str, Any], key: str
    ) -> Dict[str, Any]:
        """Convert Redis value to content item."""
        try:
            content_item = {
                "id": str(data.get("id", key)),
                "title": data.get("title", data.get("name", key)),
                "content": data.get(
                    "content", data.get("body", data.get("text", str(data)))
                ),
                "snippet": data.get("snippet", data.get("excerpt", "")),
                "created_at": data.get("created_at", datetime.now().isoformat()),
                "source_url": data.get("url", ""),
                "metadata": {
                    "redis_key": key,
                    "database_source": self.config.name,
                    "raw_data": data,
                },
            }

            return content_item

        except Exception as e:
            logger.warning(f"Failed to convert Redis value to content item: {e}")
            return None

    async def health_check(self) -> bool:
        """Check Redis connection health."""
        if not self.connected or not self.redis:
            return False

        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    async def cleanup(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
        await super().cleanup()
