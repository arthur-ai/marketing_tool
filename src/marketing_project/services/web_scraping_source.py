"""
Web scraping content source implementation for Marketing Project.

This module provides web scraping content sources that can extract content from
websites using various scraping techniques and libraries.

Classes:
    WebScrapingContentSource: Scrapes content from websites
    SeleniumScrapingSource: Uses Selenium for JavaScript-heavy sites
    BeautifulSoupScrapingSource: Uses BeautifulSoup for static content
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from marketing_project.core.content_sources import (
    ContentSource,
    ContentSourceResult,
    ContentSourceStatus,
    WebScrapingSourceConfig,
)

logger = logging.getLogger("marketing_project.services.web_scraping_source")


class WebScrapingContentSource(ContentSource):
    """Content source that scrapes content from websites."""

    def __init__(self, config: WebScrapingSourceConfig):
        super().__init__(config)
        self.config: WebScrapingSourceConfig = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.visited_urls: set = set()
        self.robots_cache: Dict[str, bool] = {}

    async def initialize(self) -> bool:
        """Initialize web scraping source."""
        try:
            # Create HTTP session with headers
            headers = {
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10),
            )

            # Check robots.txt if configured
            if self.config.respect_robots_txt:
                await self._check_robots_txt()

            self.status = ContentSourceStatus.ACTIVE
            logger.info(f"Web scraping source {self.config.name} initialized")
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize web scraping source {self.config.name}: {e}"
            )
            self.status = ContentSourceStatus.ERROR
            return False

    async def fetch_content(self, limit: Optional[int] = None) -> ContentSourceResult:
        """Fetch content from configured URLs."""
        if not self.session:
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message="Web scraping source not initialized",
            )

        try:
            content_items = []
            urls_to_process = (
                self.config.urls[:limit]
                if limit
                else self.config.urls[: self.config.max_pages]
            )

            for url in urls_to_process:
                try:
                    # Check if we can scrape this URL
                    if not await self._can_scrape_url(url):
                        logger.debug(f"Skipping URL due to robots.txt: {url}")
                        continue

                    # Scrape the URL
                    url_items = await self._scrape_url(url)
                    content_items.extend(url_items)

                    # Respect delay between requests
                    if self.config.delay_between_requests > 0:
                        await asyncio.sleep(self.config.delay_between_requests)

                except Exception as e:
                    logger.warning(f"Failed to scrape URL {url}: {e}")
                    continue

            return ContentSourceResult(
                source_name=self.config.name,
                content_items=content_items,
                total_count=len(content_items),
                success=True,
                metadata={"urls_processed": len(urls_to_process)},
            )

        except Exception as e:
            logger.error(
                f"Failed to fetch content from web scraping source {self.config.name}: {e}"
            )
            return ContentSourceResult(
                source_name=self.config.name,
                content_items=[],
                total_count=0,
                success=False,
                error_message=str(e),
            )

    async def _scrape_url(self, url: str) -> List[Dict[str, Any]]:
        """Scrape content from a single URL."""
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                html_content = await response.text()

                # Parse HTML content
                content_items = await self._parse_html(html_content, url)

                # Mark URL as visited
                self.visited_urls.add(url)

                return content_items

        except Exception as e:
            logger.warning(f"Failed to scrape URL {url}: {e}")
            return []

    async def _parse_html(self, html_content: str, url: str) -> List[Dict[str, Any]]:
        """Parse HTML content and extract content items."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            content_items = []

            # Extract content using configured selectors
            selectors = self.config.selectors

            # Extract title
            title = ""
            if "title" in selectors:
                title_elem = soup.select_one(selectors["title"])
                if title_elem:
                    title = title_elem.get_text().strip()
            else:
                # Fallback to page title
                title_elem = soup.find("title")
                if title_elem:
                    title = title_elem.get_text().strip()

            # Extract main content
            content = ""
            if "content" in selectors:
                content_elem = soup.select_one(selectors["content"])
                if content_elem:
                    content = content_elem.get_text().strip()
            else:
                # Fallback to article or main content
                content_elem = (
                    soup.find("article")
                    or soup.find("main")
                    or soup.find("div", class_="content")
                )
                if content_elem:
                    content = content_elem.get_text().strip()

            # Extract snippet
            snippet = ""
            if "snippet" in selectors:
                snippet_elem = soup.select_one(selectors["snippet"])
                if snippet_elem:
                    snippet = snippet_elem.get_text().strip()
            else:
                # Generate snippet from content
                snippet = content[:200] + "..." if len(content) > 200 else content

            # Extract metadata
            metadata = {
                "url": url,
                "domain": urlparse(url).netloc,
                "scraped_at": datetime.now().isoformat(),
                "content_length": len(content),
                "word_count": len(content.split()),
            }

            # Extract additional fields using selectors
            for field, selector in selectors.items():
                if field not in ["title", "content", "snippet"]:
                    elem = soup.select_one(selector)
                    if elem:
                        metadata[field] = elem.get_text().strip()

            # Extract links for potential further scraping
            links = []
            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)
                if self._is_valid_url(full_url):
                    links.append(full_url)
            metadata["links"] = links

            # Create content item
            if content:  # Only create item if we have content
                content_item = {
                    "id": f"scraped_{hash(url)}_{datetime.now().timestamp()}",
                    "title": title or "Scraped Content",
                    "content": content,
                    "snippet": snippet,
                    "created_at": datetime.now().isoformat(),
                    "source_url": url,
                    "metadata": metadata,
                }

                content_items.append(content_item)

            return content_items

        except Exception as e:
            logger.warning(f"Failed to parse HTML from {url}: {e}")
            return []

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for scraping."""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and parsed.scheme in ["http", "https"]
        except Exception:
            return False

    async def _check_robots_txt(self) -> None:
        """Check robots.txt for all configured domains."""
        domains = set()
        for url in self.config.urls:
            domain = urlparse(url).netloc
            domains.add(domain)

        for domain in domains:
            try:
                robots_url = f"https://{domain}/robots.txt"
                async with self.session.get(robots_url) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        self.robots_cache[domain] = self._parse_robots_txt(
                            robots_content
                        )
                    else:
                        self.robots_cache[domain] = True  # Allow if no robots.txt
            except Exception:
                self.robots_cache[domain] = True  # Allow if can't fetch robots.txt

    def _parse_robots_txt(self, robots_content: str) -> bool:
        """Parse robots.txt to check if scraping is allowed."""
        # Simple robots.txt parser
        user_agent = self.config.user_agent.lower()
        current_ua = None
        disallowed = []

        for line in robots_content.split("\n"):
            line = line.strip()
            if line.startswith("User-agent:"):
                current_ua = line.split(":", 1)[1].strip().lower()
            elif line.startswith("Disallow:") and current_ua in ["*", user_agent]:
                disallowed.append(line.split(":", 1)[1].strip())

        # For now, just return True (allow scraping)
        # In a production system, you'd implement proper robots.txt parsing
        return True

    async def _can_scrape_url(self, url: str) -> bool:
        """Check if we can scrape a URL based on robots.txt."""
        if not self.config.respect_robots_txt:
            return True

        domain = urlparse(url).netloc
        return self.robots_cache.get(domain, True)

    async def health_check(self) -> bool:
        """Check if web scraping source is healthy."""
        if not self.session:
            return False

        try:
            # Test with a simple request
            test_url = "https://httpbin.org/get"
            async with self.session.get(test_url) as response:
                return response.status < 500
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup web scraping source resources."""
        if self.session:
            await self.session.close()
        self.visited_urls.clear()
        self.robots_cache.clear()


class SeleniumScrapingSource(WebScrapingContentSource):
    """Web scraping source that uses Selenium for JavaScript-heavy sites."""

    def __init__(self, config: WebScrapingSourceConfig):
        super().__init__(config)
        self.driver = None

    async def initialize(self) -> bool:
        """Initialize Selenium scraping source."""
        try:
            # Configure Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Initialize Chrome driver
            self.driver = webdriver.Chrome(options=chrome_options)
            
            self.connected = True
            self.status = ContentSourceStatus.ACTIVE
            logger.info(f"Selenium scraping source {self.config.name} initialized")
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize Selenium scraping source {self.config.name}: {e}"
            )
            self.status = ContentSourceStatus.ERROR
            return False

    async def _scrape_url(self, url: str) -> List[Dict[str, Any]]:
        """Scrape URL using Selenium."""
        try:
            self.driver.get(url)

            # Wait for content to load
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except ImportError:
                logger.warning("Selenium not available, skipping wait")
                await asyncio.sleep(2)  # Simple delay instead

            # Get page source
            html_content = self.driver.page_source

            # Parse using parent class method
            return await self._parse_html(html_content, url)

        except Exception as e:
            logger.warning(f"Failed to scrape URL with Selenium {url}: {e}")
            return []

    async def health_check(self) -> bool:
        """Check if Selenium source is healthy."""
        if not self.driver:
            return False

        try:
            self.driver.get("https://httpbin.org/get")
            return True
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup Selenium resources."""
        if self.driver:
            self.driver.quit()
        await super().cleanup()


class BeautifulSoupScrapingSource(WebScrapingContentSource):
    """Web scraping source that uses BeautifulSoup for static content."""

    def __init__(self, config: WebScrapingSourceConfig):
        super().__init__(config)
        # This is the same as the base WebScrapingContentSource
        # but we can add BeautifulSoup-specific optimizations here
        pass

    async def _parse_html(self, html_content: str, url: str) -> List[Dict[str, Any]]:
        """Parse HTML with BeautifulSoup optimizations."""
        try:
            # Use lxml parser for better performance if available
            try:
                soup = BeautifulSoup(html_content, "lxml")
            except ImportError:
                soup = BeautifulSoup(html_content, "html.parser")

            content_items = []

            # Enhanced content extraction
            selectors = self.config.selectors

            # Extract title with multiple fallbacks
            title = self._extract_title(soup, selectors)

            # Extract main content with multiple strategies
            content = self._extract_content(soup, selectors)

            # Extract snippet
            snippet = self._extract_snippet(soup, selectors, content)

            # Extract metadata
            metadata = self._extract_metadata(soup, url, selectors)

            # Create content item
            if content:
                content_item = {
                    "id": f"scraped_{hash(url)}_{datetime.now().timestamp()}",
                    "title": title or "Scraped Content",
                    "content": content,
                    "snippet": snippet,
                    "created_at": datetime.now().isoformat(),
                    "source_url": url,
                    "metadata": metadata,
                }

                content_items.append(content_item)

            return content_items

        except Exception as e:
            logger.warning(f"Failed to parse HTML with BeautifulSoup from {url}: {e}")
            return []

    def _extract_title(self, soup, selectors: Dict[str, str]) -> str:
        """Extract title with multiple fallbacks."""
        if "title" in selectors:
            title_elem = soup.select_one(selectors["title"])
            if title_elem:
                return title_elem.get_text().strip()

        # Fallback strategies
        for selector in ["h1", "title", ".title", ".headline", "[class*='title']"]:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text().strip()

        return ""

    def _extract_content(self, soup, selectors: Dict[str, str]) -> str:
        """Extract main content with multiple strategies."""
        if "content" in selectors:
            content_elem = soup.select_one(selectors["content"])
            if content_elem:
                return content_elem.get_text().strip()

        # Fallback strategies
        for selector in [
            "article",
            "main",
            ".content",
            ".post-content",
            ".entry-content",
            "[class*='content']",
            "[class*='post']",
            "[class*='article']",
        ]:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text().strip()

        # Last resort: get all text from body
        body = soup.find("body")
        if body:
            return body.get_text().strip()

        return ""

    def _extract_snippet(self, soup, selectors: Dict[str, str], content: str) -> str:
        """Extract snippet with multiple strategies."""
        if "snippet" in selectors:
            snippet_elem = soup.select_one(selectors["snippet"])
            if snippet_elem:
                return snippet_elem.get_text().strip()

        # Fallback strategies
        for selector in [".excerpt", ".summary", ".description", "[class*='excerpt']"]:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text().strip()

        # Generate from content
        return content[:200] + "..." if len(content) > 200 else content

    def _extract_metadata(
        self, soup, url: str, selectors: Dict[str, str]
    ) -> Dict[str, Any]:
        """Extract metadata from the page."""
        metadata = {
            "url": url,
            "domain": urlparse(url).netloc,
            "scraped_at": datetime.now().isoformat(),
            "content_length": len(soup.get_text()),
            "word_count": len(soup.get_text().split()),
        }

        # Extract meta tags
        meta_tags = soup.find_all("meta")
        for meta in meta_tags:
            name = meta.get("name") or meta.get("property")
            content = meta.get("content")
            if name and content:
                metadata[f"meta_{name}"] = content

        # Extract additional fields using selectors
        for field, selector in selectors.items():
            if field not in ["title", "content", "snippet"]:
                elem = soup.select_one(selector)
                if elem:
                    metadata[field] = elem.get_text().strip()

        # Extract links
        links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(url, href)
            if self._is_valid_url(full_url):
                links.append(full_url)
        metadata["links"] = links

        return metadata
