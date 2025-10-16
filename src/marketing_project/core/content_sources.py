"""
Content source models and configuration for Marketing Project.

This module defines data models and configuration structures for various content sources
that can be integrated into the marketing pipeline.

Classes:
    ContentSourceConfig: Base configuration for content sources
    FileSourceConfig: Configuration for file-based content sources
    APISourceConfig: Configuration for API-based content sources
    DatabaseSourceConfig: Configuration for database content sources
    WebScrapingSourceConfig: Configuration for web scraping content sources
    ContentSourceManager: Manages multiple content sources
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union

from pydantic import BaseModel, Field


class ContentSourceType(str, Enum):
    """Types of content sources supported."""

    FILE = "file"
    API = "api"
    DATABASE = "database"
    WEB_SCRAPING = "web_scraping"
    WEBHOOK = "webhook"
    RSS = "rss"
    SOCIAL_MEDIA = "social_media"


class ContentSourceStatus(str, Enum):
    """Status of content source."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    CONFIGURING = "configuring"


class ContentSourceConfig(BaseModel):
    """Base configuration for content sources."""

    name: str
    source_type: ContentSourceType
    enabled: bool = True
    priority: int = 1  # Higher number = higher priority
    batch_size: int = 10
    polling_interval: int = 300  # seconds
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    filters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FileSourceConfig(ContentSourceConfig):
    """Configuration for file-based content sources."""

    source_type: ContentSourceType = ContentSourceType.FILE
    file_paths: List[str] = Field(default_factory=list)
    file_patterns: List[str] = Field(default_factory=list)  # glob patterns
    watch_directory: bool = False
    supported_formats: List[str] = Field(
        default_factory=lambda: [".txt", ".md", ".json", ".yaml", ".yml"]
    )
    encoding: str = "utf-8"


class APISourceConfig(ContentSourceConfig):
    """Configuration for API-based content sources."""

    source_type: ContentSourceType = ContentSourceType.API
    base_url: str
    endpoints: List[str] = Field(default_factory=list)
    headers: Dict[str, str] = Field(default_factory=dict)
    auth_type: str = "none"  # none, bearer, basic, api_key
    auth_config: Dict[str, str] = Field(default_factory=dict)
    rate_limit: int = 100  # requests per minute
    pagination: Dict[str, Any] = Field(default_factory=dict)


class DatabaseSourceConfig(ContentSourceConfig):
    """Configuration for database content sources."""

    source_type: ContentSourceType = ContentSourceType.DATABASE
    connection_string: str
    query: str
    table_name: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    where_clause: Optional[str] = None
    order_by: Optional[str] = None
    limit: Optional[int] = None


class WebScrapingSourceConfig(ContentSourceConfig):
    """Configuration for web scraping content sources."""

    source_type: ContentSourceType = ContentSourceType.WEB_SCRAPING
    urls: List[str] = Field(default_factory=list)
    selectors: Dict[str, str] = Field(
        default_factory=dict
    )  # CSS selectors for content extraction
    user_agent: str = "Marketing Project Bot 1.0"
    respect_robots_txt: bool = True
    delay_between_requests: float = 1.0  # seconds
    max_pages: int = 10


class WebhookSourceConfig(ContentSourceConfig):
    """Configuration for webhook content sources."""

    source_type: ContentSourceType = ContentSourceType.WEBHOOK
    webhook_url: str
    secret: Optional[str] = None
    events: List[str] = Field(default_factory=list)  # events to listen for
    verify_signature: bool = True


class RSSSourceConfig(ContentSourceConfig):
    """Configuration for RSS feed content sources."""

    source_type: ContentSourceType = ContentSourceType.RSS
    feed_urls: List[str] = Field(default_factory=list)
    max_items: int = 50
    include_content: bool = True


class SocialMediaSourceConfig(ContentSourceConfig):
    """Configuration for social media content sources."""

    source_type: ContentSourceType = ContentSourceType.SOCIAL_MEDIA
    platform: str  # twitter, linkedin, facebook, etc.
    api_credentials: Dict[str, str] = Field(default_factory=dict)
    hashtags: List[str] = Field(default_factory=list)
    user_handles: List[str] = Field(default_factory=list)
    max_posts: int = 100


# Union type for all source configurations
SourceConfig = Union[
    FileSourceConfig,
    APISourceConfig,
    DatabaseSourceConfig,
    WebScrapingSourceConfig,
    WebhookSourceConfig,
    RSSSourceConfig,
    SocialMediaSourceConfig,
]


class ContentSourceResult(BaseModel):
    """Result from a content source operation."""

    source_name: str
    content_items: List[Dict[str, Any]]
    total_count: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class ContentSource(ABC):
    """Abstract base class for content sources."""

    def __init__(self, config: ContentSourceConfig):
        self.config = config
        self.status = ContentSourceStatus.CONFIGURING
        self.last_run: Optional[datetime] = None
        self.error_count = 0

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the content source."""
        pass

    @abstractmethod
    async def fetch_content(self, limit: Optional[int] = None) -> ContentSourceResult:
        """Fetch content from the source."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the source is healthy."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the source."""
        return {
            "name": self.config.name,
            "type": self.config.source_type,
            "status": self.status,
            "enabled": self.config.enabled,
            "last_run": self.last_run,
            "error_count": self.error_count,
            "priority": self.config.priority,
        }


class ContentSourceManager:
    """Manages multiple content sources."""

    def __init__(self):
        self.sources: Dict[str, ContentSource] = {}
        self.running = False
        self.tasks: List[asyncio.Task] = []

    async def add_source(self, source: ContentSource) -> bool:
        """Add a content source to the manager."""
        try:
            if await source.initialize():
                self.sources[source.config.name] = source
                source.status = ContentSourceStatus.ACTIVE
                return True
            else:
                source.status = ContentSourceStatus.ERROR
                return False
        except Exception as e:
            source.status = ContentSourceStatus.ERROR
            source.error_count += 1
            return False

    async def remove_source(self, name: str) -> bool:
        """Remove a content source from the manager."""
        if name in self.sources:
            source = self.sources[name]
            await source.cleanup()
            del self.sources[name]
            return True
        return False

    async def fetch_all_content(
        self, limit_per_source: Optional[int] = None
    ) -> List[ContentSourceResult]:
        """Fetch content from all active sources."""
        results = []
        active_sources = [
            s for s in self.sources.values() if s.status == ContentSourceStatus.ACTIVE
        ]

        # Sort by priority (higher priority first)
        active_sources.sort(key=lambda x: x.config.priority, reverse=True)

        for source in active_sources:
            try:
                result = await source.fetch_content(limit_per_source)
                results.append(result)
                source.last_run = datetime.now()
            except Exception as e:
                source.error_count += 1
                if source.error_count >= source.config.retry_attempts:
                    source.status = ContentSourceStatus.ERROR

                results.append(
                    ContentSourceResult(
                        source_name=source.config.name,
                        content_items=[],
                        total_count=0,
                        success=False,
                        error_message=str(e),
                    )
                )

        return results

    async def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sources."""
        return {name: source.get_status() for name, source in self.sources.items()}

    async def start_polling(self) -> None:
        """Start polling all sources at their configured intervals."""
        self.running = True

        for source in self.sources.values():
            if (
                source.status == ContentSourceStatus.ACTIVE
                and source.config.polling_interval > 0
            ):
                task = asyncio.create_task(self._poll_source(source))
                self.tasks.append(task)

    async def stop_polling(self) -> None:
        """Stop polling all sources."""
        self.running = False

        for task in self.tasks:
            task.cancel()

        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    async def _poll_source(self, source: ContentSource) -> None:
        """Poll a single source at its configured interval."""
        while self.running and source.status == ContentSourceStatus.ACTIVE:
            try:
                await source.fetch_content()
                await asyncio.sleep(source.config.polling_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                source.error_count += 1
                if source.error_count >= source.config.retry_attempts:
                    source.status = ContentSourceStatus.ERROR
                    break
                await asyncio.sleep(source.config.polling_interval)

    def get_all_sources(self) -> List[ContentSource]:
        """Get all content sources."""
        return list(self.sources.values())

    def get_source(self, name: str) -> Optional[ContentSource]:
        """Get a specific content source by name."""
        return self.sources.get(name)

    async def cleanup(self) -> None:
        """Cleanup all sources."""
        await self.stop_polling()

        for source in self.sources.values():
            await source.cleanup()

        self.sources.clear()
