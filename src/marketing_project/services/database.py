"""
Database Manager Service.

Centralized PostgreSQL database connection management using SQLAlchemy async.
Handles connection pooling, lifecycle, and health checks.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

# SQLAlchemy declarative base for models
Base = declarative_base()


class DatabaseManager:
    """
    Centralized database manager with SQLAlchemy async engine.
    Manages connection pooling and lifecycle.
    """

    def __init__(self):
        """Initialize the database manager."""
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._initialized = False

    def _get_database_url(self) -> Optional[str]:
        """
        Get database URL from environment variables.

        Checks DATABASE_URL first, then POSTGRES_URL as fallback.

        Returns:
            Database URL string or None if not configured
        """
        # Try DATABASE_URL first
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            # Ensure it's a postgresql:// URL (convert postgres:// if needed)
            if database_url.startswith("postgres://"):
                database_url = database_url.replace(
                    "postgres://", "postgresql+asyncpg://", 1
                )
            elif database_url.startswith("postgresql://"):
                database_url = database_url.replace(
                    "postgresql://", "postgresql+asyncpg://", 1
                )
            elif not database_url.startswith("postgresql+asyncpg://"):
                # If it doesn't have a scheme, assume it's a connection string and add asyncpg
                database_url = f"postgresql+asyncpg://{database_url}"
            return database_url

        # Fallback to POSTGRES_URL
        postgres_url = os.getenv("POSTGRES_URL")
        if postgres_url:
            if postgres_url.startswith("postgres://"):
                postgres_url = postgres_url.replace(
                    "postgres://", "postgresql+asyncpg://", 1
                )
            elif postgres_url.startswith("postgresql://"):
                postgres_url = postgres_url.replace(
                    "postgresql://", "postgresql+asyncpg://", 1
                )
            elif not postgres_url.startswith("postgresql+asyncpg://"):
                postgres_url = f"postgresql+asyncpg://{postgres_url}"
            return postgres_url

        return None

    async def initialize(self) -> bool:
        """
        Initialize database connection.

        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            logger.debug("Database already initialized")
            return True

        database_url = self._get_database_url()
        if not database_url:
            logger.warning(
                "No database URL configured (DATABASE_URL or POSTGRES_URL). Database features will be disabled."
            )
            return False

        try:
            # Create async engine with connection pooling
            self._engine = create_async_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before using
                echo=False,  # Set to True for SQL query logging
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            # Test connection
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            self._initialized = True
            logger.info("Database connection initialized successfully")
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize database connection: {e}", exc_info=True
            )
            self._engine = None
            self._session_factory = None
            return False

    async def create_tables(self):
        """
        Create all database tables defined in models.

        This should be called after importing all model classes.
        """
        if not self._initialized or not self._engine:
            logger.warning("Database not initialized. Cannot create tables.")
            return

        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}", exc_info=True)
            raise

        await self._run_migrations()

    async def _run_migrations(self):
        """
        Apply incremental schema migrations for columns added after initial table creation.
        Uses IF NOT EXISTS / DO NOTHING patterns so re-runs are safe.
        """
        migrations = [
            # Added user_id to jobs table
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS user_id VARCHAR",
            "CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs (user_id)",
            # Approvals table indexes (table created via create_all)
            "CREATE INDEX IF NOT EXISTS idx_approvals_job_id_status ON approvals (job_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_approvals_status_created ON approvals (status, created_at)",
            # Migrate String float columns to FLOAT (safe if already correct type)
            "ALTER TABLE approval_settings ALTER COLUMN auto_approve_threshold TYPE FLOAT USING auto_approve_threshold::FLOAT",
            "ALTER TABLE user_settings ALTER COLUMN auto_approve_threshold TYPE FLOAT USING auto_approve_threshold::FLOAT",
            "ALTER TABLE user_settings ALTER COLUMN preferred_temperature TYPE FLOAT USING preferred_temperature::FLOAT",
            "ALTER TABLE approvals ALTER COLUMN confidence_score TYPE FLOAT USING confidence_score::FLOAT",
            "ALTER TABLE step_results ALTER COLUMN execution_time TYPE FLOAT USING execution_time::FLOAT",
        ]
        try:
            async with self._engine.begin() as conn:
                for statement in migrations:
                    await conn.execute(text(statement))
            logger.info("Database migrations applied successfully")
        except Exception as e:
            logger.error(f"Failed to apply database migrations: {e}", exc_info=True)
            raise

    @asynccontextmanager
    async def get_session(self):
        """
        Get an async database session.

        Usage:
            async with db_manager.get_session() as session:
                # Use session here
                result = await session.execute(...)
        """
        if not self._initialized or not self._session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def health_check(self) -> bool:
        """
        Check database connection health.

        Returns:
            True if healthy, False otherwise
        """
        if not self._initialized or not self._engine:
            return False

        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._initialized = False
            logger.info("Database connections cleaned up")

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._initialized

    @property
    def engine(self) -> Optional[AsyncEngine]:
        """Get the SQLAlchemy engine."""
        return self._engine


# Singleton instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Get or create the database manager singleton.

    Returns:
        DatabaseManager instance
    """
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager
