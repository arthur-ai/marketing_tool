"""
API-based content source implementation for Marketing Project.

This module provides API-based content sources that can fetch content from REST APIs,
webhooks, and other HTTP-based services.

Classes:
    APIContentSource: Fetches content from REST APIs
    WebhookContentSource: Handles webhook-based content
    RSSContentSource: Fetches content from RSS feeds
    SocialMediaSource: Fetches content from social media APIs
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import aiohttp
import feedparser

from marketing_project.core.content_sources import (
    APISourceConfig,
    ContentSource,
    ContentSourceResult,
    ContentSourceStatus,
    RSSSourceConfig,
    SocialMediaSourceConfig,
    WebhookSourceConfig,
)

logger = logging.getLogger("marketing_project.services.api_source")


class APIContentSource(ContentSource):
    """Content source that fetches content from REST APIs."""

    def __init__(self, config: APISourceConfig):
        super().__init__(config)
        self.config: APISourceConfig = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = asyncio.Semaphore(config.rate_limit)
        self.last_request_time = 0

    async def initialize(self) -> bool:
        """Initialize the API source."""
        try:
            # Create HTTP session with timeout
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Test connection
            if await self.health_check():
                self.status = ContentSourceStatus.ACTIVE
                logger.info(f"API source {self.config.name} initialized successfully")
                return True
            else:
                self.status = ContentSourceStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"Failed to initialize API source {self.config.name}: {e}")
            self.status = ContentSourceStatus.ERROR
            return False

    async def fetch_content(self, limit: Optional[int] = None) -> ContentSourceResult:
        """Fetch content from API endpoints."""
        if not self.session:
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message="API source not initialized",
            )

        try:
            content_items = []
            total_count = 0

            # Process each endpoint
            endpoints = self.config.endpoints or [""]

            for endpoint in endpoints:
                try:
                    # Apply rate limiting
                    async with self.rate_limiter:
                        await self._rate_limit_delay()

                        # Fetch from endpoint
                        endpoint_items = await self._fetch_from_endpoint(
                            endpoint, limit
                        )
                        content_items.extend(endpoint_items)
                        total_count += len(endpoint_items)

                except Exception as e:
                    logger.warning(f"Failed to fetch from endpoint {endpoint}: {e}")
                    continue

            return ContentSourceResult(
                source_name=self.config.name,
                content_items=content_items,
                total_count=total_count,
                success=True,
                metadata={"endpoints_processed": len(endpoints)},
            )

        except Exception as e:
            logger.error(
                f"Failed to fetch content from API source {self.config.name}: {e}"
            )
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message=str(e),
            )

    async def _fetch_from_endpoint(
        self, endpoint: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch content from a specific endpoint."""
        url = urljoin(self.config.base_url, endpoint)
        headers = self.config.headers.copy()

        # Add authentication
        await self._add_authentication(headers)

        # Add pagination parameters
        params = {}
        if limit and self.config.pagination:
            params.update(self.config.pagination.get("params", {}))
            if "limit_param" in self.config.pagination:
                params[self.config.pagination["limit_param"]] = limit

        async with self.session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            data = await response.json()

            # Extract content items from response
            items = self._extract_content_items(data)

            # Apply limit
            if limit:
                items = items[:limit]

            return items

    async def _add_authentication(self, headers: Dict[str, str]) -> None:
        """Add authentication to headers."""
        if self.config.auth_type == "bearer":
            token = self.config.auth_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        elif self.config.auth_type == "api_key":
            key_name = self.config.auth_config.get("key_name", "X-API-Key")
            key_value = self.config.auth_config.get("key_value")
            if key_value:
                headers[key_name] = key_value
        elif self.config.auth_type == "basic":
            username = self.config.auth_config.get("username")
            password = self.config.auth_config.get("password")
            if username and password:
                credentials = base64.b64encode(
                    f"{username}:{password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"

    def _extract_content_items(self, data: Any) -> List[Dict[str, Any]]:
        """Extract content items from API response."""
        items = []

        # Handle different response formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Common patterns for paginated responses
            if "data" in data:
                items = data["data"]
            elif "items" in data:
                items = data["items"]
            elif "results" in data:
                items = data["results"]
            elif "content" in data:
                items = data["content"]
            else:
                # Single item
                items = [data]

        # Convert to standard format
        content_items = []
        for item in items:
            if isinstance(item, dict):
                content_item = self._convert_to_content_item(item)
                content_items.append(content_item)

        return content_items

    def _convert_to_content_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert API item to standard content item format."""
        # Extract common fields with fallbacks
        content_item = {
            "id": item.get("id", item.get("_id", str(hash(str(item))))),
            "title": item.get(
                "title", item.get("name", item.get("subject", "Untitled"))
            ),
            "content": item.get(
                "content",
                item.get("body", item.get("text", item.get("description", ""))),
            ),
            "snippet": item.get(
                "snippet", item.get("excerpt", item.get("summary", ""))
            ),
            "created_at": item.get(
                "created_at",
                item.get("date", item.get("timestamp", datetime.now().isoformat())),
            ),
            "source_url": item.get("url", item.get("link", item.get("permalink", ""))),
            "metadata": {
                **item.get("metadata", {}),
                "api_source": self.config.name,
                "raw_data": item,
            },
        }

        # Add type-specific fields based on content structure
        if "speakers" in item or "participants" in item:
            content_item.update(
                {
                    "speakers": item.get("speakers", item.get("participants", [])),
                    "duration": item.get("duration"),
                    "transcript_type": item.get("type", "podcast"),
                }
            )
        elif "version" in item or "release" in item:
            content_item.update(
                {
                    "version": item.get("version", item.get("release", "1.0.0")),
                    "changes": item.get("changes", item.get("updates", [])),
                    "features": item.get("features", item.get("new_features", [])),
                    "bug_fixes": item.get("bug_fixes", item.get("fixes", [])),
                }
            )
        elif "author" in item or "writer" in item:
            content_item.update(
                {
                    "author": item.get("author", item.get("writer")),
                    "tags": item.get("tags", item.get("categories", [])),
                    "category": item.get("category", item.get("section")),
                    "word_count": item.get(
                        "word_count", len(content_item["content"].split())
                    ),
                }
            )

        return content_item

    async def _rate_limit_delay(self) -> None:
        """Apply rate limiting delay."""
        now = datetime.now().timestamp()
        time_since_last = now - self.last_request_time
        min_interval = 60.0 / self.config.rate_limit  # Convert to seconds per request

        if time_since_last < min_interval:
            delay = min_interval - time_since_last
            await asyncio.sleep(delay)

        self.last_request_time = datetime.now().timestamp()

    async def health_check(self) -> bool:
        """Check if API source is healthy."""
        if not self.session:
            return False

        try:
            # Try to make a simple request to check connectivity
            test_url = urljoin(self.config.base_url, "health")
            headers = self.config.headers.copy()
            await self._add_authentication(headers)

            async with self.session.get(test_url, headers=headers) as response:
                return response.status < 500

        except Exception:
            # If health endpoint doesn't exist, try the base URL
            try:
                async with self.session.get(self.config.base_url) as response:
                    return response.status < 500
            except Exception:
                return False

    async def cleanup(self) -> None:
        """Cleanup API source resources."""
        if self.session:
            await self.session.close()


class WebhookContentSource(ContentSource):
    """Content source that handles webhook-based content."""

    def __init__(self, config: WebhookSourceConfig):
        super().__init__(config)
        self.config: WebhookSourceConfig = config
        self.received_webhooks: List[Dict[str, Any]] = []
        self.lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize webhook source."""
        try:
            self.status = ContentSourceStatus.ACTIVE
            logger.info(f"Webhook source {self.config.name} initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize webhook source {self.config.name}: {e}")
            self.status = ContentSourceStatus.ERROR
            return False

    async def fetch_content(self, limit: Optional[int] = None) -> ContentSourceResult:
        """Fetch content from received webhooks."""
        async with self.lock:
            try:
                # Process received webhooks
                content_items = []
                processed_count = 0

                webhooks_to_process = self.received_webhooks.copy()
                if limit:
                    webhooks_to_process = webhooks_to_process[:limit]

                for webhook_data in webhooks_to_process:
                    content_item = self._convert_webhook_to_content_item(webhook_data)
                    if content_item:
                        content_items.append(content_item)
                        processed_count += 1

                # Clear processed webhooks
                self.received_webhooks = self.received_webhooks[processed_count:]

                return ContentSourceResult(
                    source_name=self.config.name,
                    content_items=content_items,
                    total_count=processed_count,
                    success=True,
                    metadata={"webhooks_processed": processed_count},
                )

            except Exception as e:
                logger.error(
                    f"Failed to fetch content from webhook source {self.config.name}: {e}"
                )
                return ContentSourceResult(
                    source_name=self.config.name,
                    content_items=[],
                    total_count=0,
                    success=False,
                    error_message=str(e),
                )

    async def receive_webhook(
        self, webhook_data: Dict[str, Any], signature: Optional[str] = None
    ) -> bool:
        """Receive and validate a webhook."""
        try:
            # Verify signature if configured
            if self.config.verify_signature and self.config.secret:
                if not self._verify_webhook_signature(webhook_data, signature):
                    logger.warning(f"Invalid webhook signature for {self.config.name}")
                    return False

            # Check if event type is allowed
            event_type = webhook_data.get("event", webhook_data.get("type", "unknown"))
            if self.config.events and event_type not in self.config.events:
                logger.debug(
                    f"Event type {event_type} not in allowed events for {self.config.name}"
                )
                return False

            # Add to received webhooks
            async with self.lock:
                webhook_data["received_at"] = datetime.now().isoformat()
                self.received_webhooks.append(webhook_data)

            logger.info(f"Received webhook for {self.config.name}: {event_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to receive webhook for {self.config.name}: {e}")
            return False

    def _verify_webhook_signature(
        self, webhook_data: Dict[str, Any], signature: Optional[str]
    ) -> bool:
        """Verify webhook signature."""
        if not signature or not self.config.secret:
            return False

        try:
            # Create payload string
            payload = json.dumps(webhook_data, sort_keys=True)

            # Calculate expected signature
            expected_signature = hmac.new(
                self.config.secret.encode(), payload.encode(), hashlib.sha256
            ).hexdigest()

            # Compare signatures
            return hmac.compare_digest(signature, expected_signature)

        except Exception as e:
            logger.error(f"Failed to verify webhook signature: {e}")
            return False

    def _convert_webhook_to_content_item(
        self, webhook_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Convert webhook data to content item."""
        try:
            # Extract content from webhook payload
            content_data = webhook_data.get(
                "data", webhook_data.get("payload", webhook_data)
            )

            if not isinstance(content_data, dict):
                return None

            return {
                "id": webhook_data.get("id", str(hash(str(webhook_data)))),
                "title": content_data.get(
                    "title", webhook_data.get("event", "Webhook Event")
                ),
                "content": content_data.get(
                    "content", content_data.get("body", str(content_data))
                ),
                "snippet": content_data.get("snippet", content_data.get("summary", "")),
                "created_at": webhook_data.get(
                    "received_at", datetime.now().isoformat()
                ),
                "source_url": content_data.get("url", ""),
                "metadata": {
                    "webhook_source": self.config.name,
                    "event_type": webhook_data.get("event", "unknown"),
                    "raw_webhook": webhook_data,
                },
            }

        except Exception as e:
            logger.warning(f"Failed to convert webhook to content item: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if webhook source is healthy."""
        return self.status == ContentSourceStatus.ACTIVE

    async def cleanup(self) -> None:
        """Cleanup webhook source resources."""
        self.received_webhooks.clear()


class RSSContentSource(ContentSource):
    """Content source that fetches content from RSS feeds."""

    def __init__(self, config: RSSSourceConfig):
        super().__init__(config)
        self.config: RSSSourceConfig = config
        self.feed_cache: Dict[str, datetime] = {}

    async def initialize(self) -> bool:
        """Initialize RSS source."""
        try:
            self.status = ContentSourceStatus.ACTIVE
            logger.info(
                f"RSS source {self.config.name} initialized with {len(self.config.feed_urls)} feeds"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RSS source {self.config.name}: {e}")
            self.status = ContentSourceStatus.ERROR
            return False

    async def fetch_content(self, limit: Optional[int] = None) -> ContentSourceResult:
        """Fetch content from RSS feeds."""
        try:
            content_items = []
            total_count = 0

            for feed_url in self.config.feed_urls:
                try:
                    feed_items = await self._fetch_feed(feed_url, limit)
                    content_items.extend(feed_items)
                    total_count += len(feed_items)

                except Exception as e:
                    logger.warning(f"Failed to fetch RSS feed {feed_url}: {e}")
                    continue

            return ContentSourceResult(
                source_name=self.config.name,
                content_items=content_items,
                total_count=total_count,
                success=True,
                metadata={"feeds_processed": len(self.config.feed_urls)},
            )

        except Exception as e:
            logger.error(
                f"Failed to fetch content from RSS source {self.config.name}: {e}"
            )
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message=str(e),
            )

    async def _fetch_feed(
        self, feed_url: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch content from a single RSS feed."""
        try:
            # Parse RSS feed
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                logger.warning(f"RSS feed {feed_url} has parsing issues")

            items = []
            entries = (
                feed.entries[:limit] if limit else feed.entries[: self.config.max_items]
            )

            for entry in entries:
                content_item = self._convert_rss_entry_to_content_item(entry, feed_url)
                if content_item:
                    items.append(content_item)

            return items

        except Exception as e:
            logger.error(f"Failed to parse RSS feed {feed_url}: {e}")
            return []

    def _convert_rss_entry_to_content_item(
        self, entry: Any, feed_url: str
    ) -> Dict[str, Any]:
        """Convert RSS entry to content item."""
        try:
            # Extract content
            content = ""
            if hasattr(entry, "content") and entry.content:
                content = (
                    entry.content[0].value
                    if isinstance(entry.content, list)
                    else str(entry.content)
                )
            elif hasattr(entry, "summary"):
                content = entry.summary
            elif hasattr(entry, "description"):
                content = entry.description

            # Clean HTML tags if needed
            if content and "<" in content:
                content = re.sub(r"<[^>]+>", "", content)

            return {
                "id": entry.get("id", entry.get("guid", str(hash(str(entry))))),
                "title": entry.get("title", "Untitled"),
                "content": content,
                "snippet": entry.get(
                    "summary", content[:200] + "..." if len(content) > 200 else content
                ),
                "created_at": entry.get(
                    "published", entry.get("updated", datetime.now().isoformat())
                ),
                "source_url": entry.get("link", ""),
                "metadata": {
                    "feed_url": feed_url,
                    "author": entry.get("author"),
                    "tags": [tag.term for tag in entry.get("tags", [])],
                    "raw_entry": dict(entry),
                },
            }

        except Exception as e:
            logger.warning(f"Failed to convert RSS entry to content item: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if RSS source is healthy."""
        try:
            # Test one feed
            if self.config.feed_urls:
                feed = feedparser.parse(self.config.feed_urls[0])
                return not feed.bozo
            return False
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup RSS source resources."""
        self.feed_cache.clear()
