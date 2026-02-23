#!/usr/bin/env python3
"""
Database migration script to add user_id column to jobs table.

This script:
1. Adds user_id column to jobs table (nullable)
2. Backfills user_id from job_metadata->>'triggered_by_user_id' if available
3. Adds index on user_id for querying jobs by user

Usage:
    python scripts/migrate_add_user_id_to_jobs.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import text

from marketing_project.services.database import get_database_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate():
    """Run the migration."""
    db_manager = get_database_manager()

    if not db_manager.is_initialized:
        logger.error("Database not initialized. Please initialize the database first.")
        return False

    try:
        async with db_manager.get_session() as session:
            # Check if user_id column already exists
            check_column_query = text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'jobs' AND column_name = 'user_id'
            """
            )
            result = await session.execute(check_column_query)
            column_exists = result.scalar_one_or_none() is not None

            if column_exists:
                logger.info("user_id column already exists. Skipping column creation.")
            else:
                # Step 1: Add user_id column
                logger.info("Adding user_id column to jobs table...")
                add_column_query = text(
                    """
                    ALTER TABLE jobs
                    ADD COLUMN user_id VARCHAR NULL
                """
                )
                await session.execute(add_column_query)
                await session.commit()
                logger.info("✓ Added user_id column")

            # Step 2: Backfill user_id from metadata
            logger.info("Backfilling user_id from job_metadata...")
            backfill_query = text(
                """
                UPDATE jobs
                SET user_id = job_metadata->>'triggered_by_user_id'
                WHERE user_id IS NULL
                  AND job_metadata IS NOT NULL
                  AND job_metadata->>'triggered_by_user_id' IS NOT NULL
                  AND job_metadata->>'triggered_by_user_id' != ''
            """
            )
            result = await session.execute(backfill_query)
            updated_count = result.rowcount
            await session.commit()
            logger.info(f"✓ Backfilled user_id for {updated_count} jobs")

            # Step 3: Check if index already exists
            check_index_query = text(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'jobs' AND indexname = 'idx_jobs_user_id'
            """
            )
            result = await session.execute(check_index_query)
            index_exists = result.scalar_one_or_none() is not None

            if index_exists:
                logger.info(
                    "idx_jobs_user_id index already exists. Skipping index creation."
                )
            else:
                # Step 3: Add index on user_id
                logger.info("Adding index on user_id column...")
                add_index_query = text(
                    """
                    CREATE INDEX idx_jobs_user_id ON jobs(user_id)
                """
                )
                await session.execute(add_index_query)
                await session.commit()
                logger.info("✓ Added index on user_id")

            logger.info("Migration completed successfully!")
            return True

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(migrate())
    sys.exit(0 if success else 1)
