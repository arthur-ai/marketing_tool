"""
S3-based content source implementation for Marketing Project.

This module provides a content source that reads content from AWS S3.
"""

import fnmatch
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from marketing_project.core.content_sources import (
    ContentSource,
    ContentSourceResult,
    ContentSourceStatus,
    S3SourceConfig,
)
from marketing_project.core.models import (
    BlogPostContext,
    ContentContext,
    ReleaseNotesContext,
    TranscriptContext,
)
from marketing_project.core.utils import convert_dict_to_content_context
from marketing_project.services.s3_storage import S3Storage

logger = logging.getLogger("marketing_project.services.s3_source")


class S3ContentSource(ContentSource):
    """Content source that reads from AWS S3."""

    def __init__(self, config: S3SourceConfig):
        super().__init__(config)
        self.config: S3SourceConfig = config
        self.s3_storage: Optional[S3Storage] = None
        self.file_cache: Dict[str, datetime] = {}

    async def initialize(self) -> bool:
        """Initialize the S3 content source."""
        try:
            logger.info(f"Initializing S3 content source '{self.config.name}'")

            # Initialize S3 storage
            self.s3_storage = S3Storage(
                bucket_name=self.config.bucket_name,
                region=self.config.region,
                prefix=self.config.prefix,
            )

            if not self.s3_storage.is_available():
                logger.error(
                    f"S3 is not available for source '{self.config.name}'. "
                    "Check AWS credentials and bucket configuration."
                )
                return False

            # List files to verify access
            files = await self.s3_storage.list_files()
            logger.info(
                f"S3 source '{self.config.name}' initialized. Found {len(files)} files in bucket."
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize S3 source '{self.config.name}': {e}")
            return False

    def _matches_pattern_with_prefix(
        self, s3_key: str, patterns: List[str], prefix: str
    ) -> bool:
        """Check if S3 key matches any of the patterns with a given prefix."""
        if not patterns:
            return True

        # Remove prefix from key for pattern matching
        key_without_prefix = s3_key
        if prefix and s3_key.startswith(prefix):
            key_without_prefix = s3_key[len(prefix) :]

        for pattern in patterns:
            # Handle ** patterns - they should match any number of directories
            if "**" in pattern:
                # For patterns like **/*.json, extract the file part (e.g., *.json)
                if "/" in pattern:
                    # Extract the file pattern part after **/
                    file_pattern = (
                        pattern.split("/", 1)[-1] if "/" in pattern else pattern
                    )
                    # Match if the key (or any part of it) matches the file pattern
                    # This allows **/*.json to match blog_posts/filename.json
                    if fnmatch.fnmatch(key_without_prefix, file_pattern):
                        return True
                    # Also check if any suffix of the path matches
                    parts = key_without_prefix.split("/")
                    for i in range(len(parts)):
                        suffix = "/".join(parts[i:])
                        if fnmatch.fnmatch(suffix, file_pattern):
                            return True
                else:
                    # Pattern is just ** - matches everything
                    return True
            else:
                # Simple pattern matching (no **)
                if fnmatch.fnmatch(key_without_prefix, pattern) or fnmatch.fnmatch(
                    s3_key, pattern
                ):
                    return True
        return False

    def _matches_pattern(self, s3_key: str, patterns: List[str]) -> bool:
        """Check if S3 key matches any of the patterns using the configured prefix."""
        return self._matches_pattern_with_prefix(s3_key, patterns, self.config.prefix)

    def _is_supported_format(self, s3_key: str) -> bool:
        """Check if file format is supported."""
        file_ext = Path(s3_key).suffix.lower()
        return file_ext in self.config.supported_formats

    async def fetch_content(
        self, limit: Optional[int] = None, include_cached: bool = True
    ) -> ContentSourceResult:
        """Fetch content from S3."""
        try:
            if not self.s3_storage or not self.s3_storage.is_available():
                return ContentSourceResult(
                    source_name=self.config.name,
                    content_items=[],
                    total_count=0,
                    success=False,
                    error_message="S3 storage is not available",
                )

            # List all files in S3 with the configured prefix
            all_files = await self.s3_storage.list_files()
            logger.info(
                f"Found {len(all_files)} files in S3 bucket with prefix '{self.config.prefix}' for source '{self.config.name}'"
            )

            if all_files:
                logger.info(f"Sample S3 files found: {all_files[:5]}")

            # Filter files by pattern and format
            filtered_files = []
            for s3_key in all_files:
                is_supported = self._is_supported_format(s3_key)
                matches_pattern = self._matches_pattern(
                    s3_key, self.config.file_patterns
                )

                if not is_supported:
                    logger.debug(f"Skipping {s3_key}: unsupported format")
                elif not matches_pattern:
                    logger.debug(
                        f"Skipping {s3_key}: doesn't match patterns {self.config.file_patterns}"
                    )

                if is_supported and matches_pattern:
                    filtered_files.append(s3_key)

            logger.info(
                f"Filtered to {len(filtered_files)} matching files for source '{self.config.name}' (from {len(all_files)} total files)"
            )

            if filtered_files:
                logger.info(f"Sample filtered files: {filtered_files[:5]}")

            # Apply limit
            if limit:
                filtered_files = filtered_files[:limit]

            # Fetch and parse content
            content_items = []
            for s3_key in filtered_files:
                try:
                    logger.debug(f"Reading S3 file: {s3_key}")
                    content_item = await self._read_s3_file(s3_key)
                    if content_item:
                        content_items.append(content_item)
                        self.file_cache[s3_key] = datetime.now()
                        logger.debug(f"Successfully parsed content from {s3_key}")
                    else:
                        logger.warning(
                            f"Failed to parse content from {s3_key}: returned None"
                        )
                except Exception as e:
                    logger.warning(f"Failed to read file {s3_key}: {e}", exc_info=True)
                    continue

            logger.info(
                f"Successfully fetched {len(content_items)} content items from S3 source '{self.config.name}' (attempted {len(filtered_files)} files)"
            )

            return ContentSourceResult(
                source_name=self.config.name,
                content_items=content_items,
                total_count=len(content_items),
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to fetch content from S3 source: {e}")
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message=str(e),
            )

    async def _read_s3_file(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """Read and parse a single file from S3."""
        try:
            # Get file content from S3
            logger.debug(f"Fetching content from S3 for key: {s3_key}")
            content_bytes = await self.s3_storage.get_file_content(s3_key)
            if not content_bytes:
                logger.warning(f"No content bytes returned for S3 key: {s3_key}")
                return None

            # Decode content
            content = content_bytes.decode(self.config.encoding)
            logger.debug(f"Decoded {len(content)} characters from {s3_key}")

            # Parse based on file extension
            file_ext = Path(s3_key).suffix.lower()

            if file_ext in [".json"]:
                data = json.loads(content)
                return self._convert_to_content_item(data, s3_key)
            elif file_ext in [".yaml", ".yml"]:
                data = yaml.safe_load(content)
                return self._convert_to_content_item(data, s3_key)
            elif file_ext in [".md", ".txt"]:
                return self._convert_text_to_content_item(content, s3_key)
            else:
                # Try to parse as JSON first, then fall back to text
                try:
                    data = json.loads(content)
                    return self._convert_to_content_item(data, s3_key)
                except json.JSONDecodeError:
                    return self._convert_text_to_content_item(content, s3_key)

        except Exception as e:
            logger.warning(f"Failed to read S3 file {s3_key}: {e}")
            return None

    def _convert_to_content_item(
        self, data: Dict[str, Any], s3_key: str
    ) -> Dict[str, Any]:
        """Convert parsed data to content item format."""
        # Extract content type from data or infer from structure
        content_type = data.get("type", "blog_post")

        # Ensure required fields
        if "id" not in data:
            data["id"] = Path(s3_key).stem
        if "title" not in data:
            data["title"] = Path(s3_key).stem
        if "content" not in data:
            data["content"] = str(data)

        # Add metadata
        data["source"] = self.config.name
        data["source_type"] = "s3"
        data["s3_key"] = s3_key
        if "created_at" not in data:
            data["created_at"] = datetime.utcnow().isoformat() + "Z"

        return data

    def _convert_text_to_content_item(
        self, content: str, s3_key: str
    ) -> Dict[str, Any]:
        """Convert text content to content item format."""
        # Infer content type from S3 key path
        content_type = "blog_post"
        if "transcript" in s3_key.lower():
            content_type = "transcript"
        elif "release" in s3_key.lower() or "release_notes" in s3_key.lower():
            content_type = "release_notes"

        return {
            "id": Path(s3_key).stem,
            "title": Path(s3_key).stem,
            "content": content,
            "type": content_type,
            "source": self.config.name,
            "source_type": "s3",
            "s3_key": s3_key,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

    async def health_check(self) -> bool:
        """Check if the S3 source is healthy."""
        try:
            if not self.s3_storage or not self.s3_storage.is_available():
                return False

            # Try to list files as a health check
            await self.s3_storage.list_files(max_keys=1)
            return True

        except Exception as e:
            logger.error(f"Health check failed for S3 source '{self.config.name}': {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup S3 source resources."""
        self.file_cache.clear()
        # S3 client doesn't need explicit cleanup
