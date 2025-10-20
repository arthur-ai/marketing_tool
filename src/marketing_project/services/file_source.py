"""
File-based content source implementation for Marketing Project.

This module provides file-based content sources that can read content from local files,
uploaded files, and watch directories for new content.

Classes:
    FileContentSource: Reads content from local files
    DirectoryWatcherSource: Watches directories for new files
    UploadedFileSource: Handles uploaded file content
"""

import asyncio
import glob
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from marketing_project.core.content_sources import (
    ContentSource,
    ContentSourceResult,
    ContentSourceStatus,
    FileSourceConfig,
)
from marketing_project.core.models import (
    BlogPostContext,
    ContentContext,
    ReleaseNotesContext,
    TranscriptContext,
)
from marketing_project.core.utils import convert_dict_to_content_context

logger = logging.getLogger("marketing_project.services.file_source")


class FileContentSource(ContentSource):
    """Content source that reads from local files."""

    def __init__(self, config: FileSourceConfig):
        super().__init__(config)
        self.config: FileSourceConfig = config
        self.file_cache: Dict[str, datetime] = {}

    async def initialize(self) -> bool:
        """Initialize the file source."""
        try:
            # Validate file paths and patterns
            valid_paths = []

            # Check individual file paths
            for file_path in self.config.file_paths:
                if os.path.exists(file_path):
                    valid_paths.append(file_path)
                else:
                    logger.warning(f"File not found: {file_path}")

            # Check file patterns
            for pattern in self.config.file_patterns:
                matches = glob.glob(pattern, recursive=True)
                valid_paths.extend(matches)

            if not valid_paths:
                logger.error("No valid files found for file source")
                return False

            # Don't cache files during initialization - let fetch handle caching
            # This ensures files are processed on first fetch

            self.status = ContentSourceStatus.ACTIVE
            logger.info(f"File source initialized with {len(valid_paths)} files")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize file source: {e}")
            self.status = ContentSourceStatus.ERROR
            return False

    async def fetch_content(
        self, limit: Optional[int] = None, include_cached: bool = True
    ) -> ContentSourceResult:
        """Fetch content from files.

        Args:
            limit: Maximum number of files to fetch
            include_cached: If True, returns all files. If False, only returns new/modified files.
        """
        try:
            content_items = []
            processed_count = 0
            new_files_count = 0
            cached_files_count = 0

            # Get all file paths
            all_paths = []
            for file_path in self.config.file_paths:
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    all_paths.append(file_path)

            for pattern in self.config.file_patterns:
                matches = glob.glob(pattern, recursive=True)
                # Filter out directories
                matches = [m for m in matches if os.path.isfile(m)]
                all_paths.extend(matches)

            # Remove duplicates and sort
            all_paths = sorted(list(set(all_paths)))

            # Apply limit
            if limit:
                all_paths = all_paths[:limit]

            for file_path in all_paths:
                try:
                    # Check if file has been modified
                    current_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    cached_mtime = self.file_cache.get(file_path)
                    is_new_or_modified = (
                        not cached_mtime or current_mtime > cached_mtime
                    )

                    # Skip cached files if include_cached is False
                    if not include_cached and not is_new_or_modified:
                        cached_files_count += 1
                        continue

                    # Read file content
                    content_item = await self._read_file(file_path)
                    if content_item:
                        content_items.append(content_item)
                        processed_count += 1
                        if is_new_or_modified:
                            new_files_count += 1
                        else:
                            cached_files_count += 1

                    # Update cache
                    self.file_cache[file_path] = current_mtime

                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
                    continue

            return ContentSourceResult(
                source_name=self.config.name,
                content_items=content_items,
                total_count=processed_count,
                success=True,
                metadata={
                    "files_processed": processed_count,
                    "new_files": new_files_count,
                    "cached_files": cached_files_count,
                },
            )

        except Exception as e:
            logger.error(f"Failed to fetch content from file source: {e}")
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message=str(e),
            )

    async def _read_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Read and parse a single file."""
        try:
            file_ext = Path(file_path).suffix.lower()

            # Read file content
            async with aiofiles.open(
                file_path, "r", encoding=self.config.encoding
            ) as f:
                content = await f.read()

            # Parse based on file extension
            if file_ext in [".json"]:
                data = json.loads(content)
                return self._convert_to_content_item(data, file_path)
            elif file_ext in [".yaml", ".yml"]:
                data = yaml.safe_load(content)
                return self._convert_to_content_item(data, file_path)
            elif file_ext in [".md", ".txt"]:
                return self._convert_text_to_content_item(content, file_path)
            else:
                # Try to parse as JSON first, then fall back to text
                try:
                    data = json.loads(content)
                    return self._convert_to_content_item(data, file_path)
                except json.JSONDecodeError:
                    return self._convert_text_to_content_item(content, file_path)

        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return None

    def _convert_to_content_item(
        self, data: Dict[str, Any], file_path: str
    ) -> Dict[str, Any]:
        """Convert parsed data to content item."""
        # Ensure required fields exist
        content_item = {
            "id": data.get(
                "id", f"file_{Path(file_path).stem}_{datetime.now().timestamp()}"
            ),
            "title": data.get("title", Path(file_path).stem),
            "content": data.get("content", ""),
            "snippet": data.get("snippet", data.get("content", "")[:200] + "..."),
            "created_at": data.get("created_at", datetime.now().isoformat()),
            "source_url": f"file://{file_path}",
            "metadata": {
                **{k: str(v) for k, v in data.get("metadata", {}).items()},
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_size": str(os.path.getsize(file_path)),
            },
        }

        # Add type-specific fields and determine content type
        if "speakers" in data or "transcript_type" in data:
            content_item.update(
                {
                    "speakers": data.get("speakers", []),
                    "duration": data.get("duration"),
                    "transcript_type": data.get("transcript_type", "podcast"),
                    "content_type": "transcript",
                }
            )
        elif "version" in data or "changes" in data:
            content_item.update(
                {
                    "version": data.get("version", "1.0.0"),
                    "changes": data.get("changes", []),
                    "features": data.get("features", []),
                    "bug_fixes": data.get("bug_fixes", []),
                    "content_type": "release_notes",
                }
            )
        elif "author" in data or "tags" in data:
            content_item.update(
                {
                    "author": data.get("author"),
                    "tags": data.get("tags", []),
                    "category": data.get("category"),
                    "word_count": data.get(
                        "word_count", len(content_item["content"].split())
                    ),
                    "content_type": "blog_post",
                }
            )
        else:
            # Default to blog_post if no specific type indicators
            content_item["content_type"] = "blog_post"

        return content_item

    def _convert_text_to_content_item(
        self, content: str, file_path: str
    ) -> Dict[str, Any]:
        """Convert plain text to content item."""
        lines = content.split("\n")
        title = lines[0] if lines else Path(file_path).stem

        # Determine content type based on file name and content
        content_type = "blog_post"  # default
        if "transcript" in file_path.lower() or "speakers:" in content.lower():
            content_type = "transcript"
        elif "release" in file_path.lower() or "version" in content.lower():
            content_type = "release_notes"

        # Parse transcript-specific fields if it's a transcript
        if content_type == "transcript":
            speakers = []
            duration = None
            transcript_type = "podcast"

            # Extract speakers from content
            for line in lines:
                if line.startswith("**Speakers:**"):
                    speakers_text = line.replace("**Speakers:**", "").strip()
                    speakers = [s.strip() for s in speakers_text.split(",")]
                    break

            # Extract duration
            for line in lines:
                if line.startswith("**Duration:**"):
                    duration = line.replace("**Duration:**", "").strip()
                    break

            # Extract type
            for line in lines:
                if line.startswith("**Type:**"):
                    transcript_type = line.replace("**Type:**", "").strip()
                    break

            return {
                "id": f"file_{Path(file_path).stem}_{datetime.now().timestamp()}",
                "title": title,
                "content_type": content_type,
                "content": content,
                "snippet": content[:200] + "..." if len(content) > 200 else content,
                "created_at": datetime.now().isoformat(),
                "source_url": f"file://{file_path}",
                "speakers": speakers,
                "duration": duration,
                "transcript_type": transcript_type,
                "metadata": {
                    "file_path": file_path,
                    "file_name": Path(file_path).name,
                    "file_size": str(len(content.encode("utf-8"))),
                    "word_count": str(len(content.split())),
                },
            }

        return {
            "id": f"file_{Path(file_path).stem}_{datetime.now().timestamp()}",
            "title": title,
            "content_type": content_type,
            "content": content,
            "snippet": content[:200] + "..." if len(content) > 200 else content,
            "created_at": datetime.now().isoformat(),
            "source_url": f"file://{file_path}",
            "metadata": {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_size": str(len(content.encode("utf-8"))),
                "word_count": str(len(content.split())),
            },
        }

    async def health_check(self) -> bool:
        """Check if file source is healthy."""
        try:
            # Check if at least one file exists and is readable
            for file_path in self.config.file_paths:
                if os.path.exists(file_path) and os.access(file_path, os.R_OK):
                    return True

            # Check patterns
            for pattern in self.config.file_patterns:
                if glob.glob(pattern):
                    return True

            return False

        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup file source resources."""
        self.file_cache.clear()


class DirectoryWatcherSource(FileContentSource):
    """File source that watches directories for changes."""

    def __init__(self, config: FileSourceConfig):
        super().__init__(config)
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[FileSystemEventHandler] = None
        self.pending_files: List[str] = []
        self.lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize directory watcher."""
        if not await super().initialize():
            return False

        if not self.config.watch_directory:
            return True

        try:
            # Set up file system event handler
            self.event_handler = FileEventHandler(self)

            # Set up observer
            self.observer = Observer()

            # Watch directories from file paths
            for file_path in self.config.file_paths:
                dir_path = os.path.dirname(file_path)
                if os.path.isdir(dir_path):
                    self.observer.schedule(self.event_handler, dir_path, recursive=True)

            # Watch directories from patterns
            for pattern in self.config.file_patterns:
                # Extract directory from pattern
                if "*" in pattern:
                    dir_pattern = pattern.split("*")[0]
                    if os.path.isdir(dir_pattern):
                        self.observer.schedule(
                            self.event_handler, dir_pattern, recursive=True
                        )

            self.observer.start()
            logger.info("Directory watcher started")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize directory watcher: {e}")
            return False

    async def fetch_content(self, limit: Optional[int] = None) -> ContentSourceResult:
        """Fetch content including pending files from directory watcher."""
        async with self.lock:
            # Process pending files first
            if self.pending_files:
                pending_files = self.pending_files.copy()
                self.pending_files.clear()

                for file_path in pending_files:
                    if file_path.endswith(tuple(self.config.supported_formats)):
                        try:
                            content_item = await self._read_file(file_path)
                            if content_item:
                                # Add to results
                                pass  # This will be handled by the parent class
                        except Exception as e:
                            logger.warning(
                                f"Failed to process pending file {file_path}: {e}"
                            )

            # Call parent fetch_content
            return await super().fetch_content(limit)

    def add_pending_file(self, file_path: str) -> None:
        """Add a file to pending processing queue."""
        if file_path not in self.pending_files:
            self.pending_files.append(file_path)

    async def cleanup(self) -> None:
        """Cleanup directory watcher resources."""
        if self.observer:
            self.observer.stop()
            self.observer.join()

        await super().cleanup()


class FileEventHandler(FileSystemEventHandler):
    """File system event handler for directory watcher."""

    def __init__(self, source: DirectoryWatcherSource):
        self.source = source

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self.source.add_pending_file(event.src_path)

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self.source.add_pending_file(event.src_path)


class UploadedFileSource(FileContentSource):
    """Content source for handling uploaded files."""

    def __init__(self, config: FileSourceConfig, upload_directory: str = "uploads"):
        super().__init__(config)
        self.upload_directory = upload_directory
        self.uploaded_files: List[str] = []

    async def initialize(self) -> bool:
        """Initialize uploaded file source."""
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_directory, exist_ok=True)

        # Set file paths to upload directory
        self.config.file_paths = [self.upload_directory]
        self.config.file_patterns = [os.path.join(self.upload_directory, "**/*")]

        return await super().initialize()

    async def add_uploaded_file(
        self, file_path: str, content_data: Dict[str, Any]
    ) -> bool:
        """Add an uploaded file to the source."""
        try:
            # Save content to file
            file_ext = Path(file_path).suffix.lower()
            full_path = os.path.join(self.upload_directory, file_path)

            # Ensure directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            if file_ext in [".json"]:
                async with aiofiles.open(
                    full_path, "w", encoding=self.config.encoding
                ) as f:
                    await f.write(json.dumps(content_data, indent=2))
            elif file_ext in [".yaml", ".yml"]:
                async with aiofiles.open(
                    full_path, "w", encoding=self.config.encoding
                ) as f:
                    await f.write(yaml.dump(content_data, default_flow_style=False))
            else:
                # Save as text
                content = content_data.get("content", str(content_data))
                async with aiofiles.open(
                    full_path, "w", encoding=self.config.encoding
                ) as f:
                    await f.write(content)

            self.uploaded_files.append(full_path)
            return True

        except Exception as e:
            logger.error(f"Failed to add uploaded file {file_path}: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup uploaded files."""
        # Optionally clean up uploaded files
        for file_path in self.uploaded_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup uploaded file {file_path}: {e}")

        await super().cleanup()
