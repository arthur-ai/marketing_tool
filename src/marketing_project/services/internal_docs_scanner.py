"""
Internal Docs Scanner Service.

This service scans and retrieves internal documentation from URLs,
extracting document information and analyzing content patterns.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp
from bs4 import BeautifulSoup

from marketing_project.models.internal_docs_config import (
    InternalDocsConfig,
    ScannedDocument,
)
from marketing_project.models.scanned_document_db import (
    ScannedDocumentDB,
    ScannedDocumentMetadata,
)
from marketing_project.services.scanned_document_db import get_scanned_document_db

logger = logging.getLogger("marketing_project.services.internal_docs_scanner")


class InternalDocsScanner:
    """Service for scanning internal documentation from URLs."""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.visited_urls: Set[str] = set()
        self.base_domain: Optional[str] = None

    async def initialize(self):
        """Initialize the scanner with HTTP session."""
        if self.session is None:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; InternalDocsScanner/1.0)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10),
            )
            logger.info("Internal docs scanner initialized")

    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None

    async def scan_from_base_url(
        self,
        base_url: str,
        max_depth: int = 3,
        follow_external: bool = False,
        max_pages: int = 100,
    ) -> List[ScannedDocument]:
        """
        Scan internal documentation by crawling from a base URL.

        Args:
            base_url: Base URL to start crawling from
            max_depth: Maximum depth to crawl (default: 3)
            follow_external: Whether to follow external links (default: False)
            max_pages: Maximum number of pages to scan (default: 100)

        Returns:
            List of ScannedDocument objects
        """
        await self.initialize()

        try:
            parsed_base = urlparse(base_url)
            self.base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
            self.visited_urls.clear()

            scanned_docs: List[ScannedDocument] = []
            urls_to_visit: List[tuple[str, int]] = [(base_url, 0)]  # (url, depth)

            while urls_to_visit and len(scanned_docs) < max_pages:
                current_url, depth = urls_to_visit.pop(0)

                if current_url in self.visited_urls or depth > max_depth:
                    continue

                if not follow_external:
                    parsed_current = urlparse(current_url)
                    if parsed_current.netloc != parsed_base.netloc:
                        continue

                try:
                    doc = await self._scan_single_url(current_url)
                    if doc:
                        scanned_docs.append(doc)
                        self.visited_urls.add(current_url)

                        # Extract links for further crawling
                        if depth < max_depth:
                            links = await self._extract_links(current_url)
                            for link in links:
                                if link not in self.visited_urls:
                                    urls_to_visit.append((link, depth + 1))

                    # Respect rate limiting
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.warning(f"Failed to scan URL {current_url}: {e}")
                    continue

            logger.info(f"Scanned {len(scanned_docs)} documents from {base_url}")
            return scanned_docs

        except Exception as e:
            logger.error(f"Error scanning from base URL {base_url}: {e}")
            return []
        finally:
            await self.cleanup()

    async def scan_from_url_list(
        self, urls: List[str], max_concurrent: int = 5
    ) -> List[ScannedDocument]:
        """
        Scan internal documentation from a list of specific URLs.

        Args:
            urls: List of URLs to scan
            max_concurrent: Maximum number of concurrent requests (default: 5)

        Returns:
            List of ScannedDocument objects
        """
        await self.initialize()

        try:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def scan_with_semaphore(url: str) -> Optional[ScannedDocument]:
                async with semaphore:
                    return await self._scan_single_url(url)

            tasks = [scan_with_semaphore(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            scanned_docs: List[ScannedDocument] = []
            for result in results:
                if isinstance(result, ScannedDocument):
                    scanned_docs.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Error scanning URL: {result}")

            logger.info(f"Scanned {len(scanned_docs)} documents from {len(urls)} URLs")
            return scanned_docs

        except Exception as e:
            logger.error(f"Error scanning from URL list: {e}")
            return []
        finally:
            await self.cleanup()

    async def _scan_single_url(
        self, url: str, save_to_db: bool = True
    ) -> Optional[ScannedDocument]:
        """
        Scan a single URL and extract document information with rich metadata.

        Args:
            url: URL to scan
            save_to_db: Whether to save to database (default: True)

        Returns:
            ScannedDocument (for config) and saves ScannedDocumentDB to database
        """
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.debug(f"Skipping {url}: status {response.status}")
                    return None

                html_content = await response.text()
                soup = BeautifulSoup(html_content, "html.parser")

                # Extract basic information
                title = self._extract_title(soup, url)
                content = self._extract_content(soup)

                # Extract rich metadata
                metadata = self._extract_rich_metadata(soup, url, content)

                # Create scanned document for config
                doc = ScannedDocument(
                    title=title, url=url, scanned_at=datetime.now(timezone.utc)
                )

                # Save to database with rich metadata
                if save_to_db:
                    try:
                        db = get_scanned_document_db()
                        db_doc = ScannedDocumentDB(
                            title=title,
                            url=url,
                            scanned_at=datetime.now(timezone.utc),
                            metadata=metadata,
                        )
                        # Calculate relationships before saving
                        db_doc = db.update_relationships(db_doc)
                        db.save_document(db_doc)
                        logger.info(f"Saved scanned document to database: {url}")
                    except Exception as e:
                        logger.warning(f"Failed to save to database: {e}")

                return doc

        except Exception as e:
            logger.warning(f"Failed to scan URL {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract document title from HTML."""
        # Try multiple strategies
        title_elem = (
            soup.find("h1")
            or soup.find("title")
            or soup.find("meta", property="og:title")
            or soup.find("meta", attrs={"name": "title"})
        )

        if title_elem:
            if title_elem.name == "meta":
                return title_elem.get("content", "").strip()
            return title_elem.get_text().strip()

        # Fallback to URL
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if path:
            return path.split("/")[-1].replace("-", " ").replace("_", " ").title()

        return "Untitled Document"

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML."""
        # Try common content selectors
        content_elem = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"content|post|article", re.I))
            or soup.find("div", id=re.compile(r"content|post|article", re.I))
        )

        if content_elem:
            # Remove script and style tags
            for script in content_elem(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            return content_elem.get_text(separator=" ", strip=True)

        # Fallback to body
        body = soup.find("body")
        if body:
            for script in body(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            return body.get_text(separator=" ", strip=True)

        return ""

    async def _extract_links(self, url: str) -> List[str]:
        """Extract links from a page for crawling."""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return []

                html_content = await response.text()
                soup = BeautifulSoup(html_content, "html.parser")

                links: List[str] = []
                parsed_base = urlparse(url)

                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    absolute_url = urljoin(url, href)
                    parsed_link = urlparse(absolute_url)

                    # Only include same-domain links
                    if parsed_link.netloc == parsed_base.netloc:
                        # Remove fragments
                        clean_url = urlunparse(
                            (
                                parsed_link.scheme,
                                parsed_link.netloc,
                                parsed_link.path,
                                parsed_link.params,
                                parsed_link.query,
                                "",  # Remove fragment
                            )
                        )
                        if clean_url not in links:
                            links.append(clean_url)

                return links

        except Exception as e:
            logger.warning(f"Failed to extract links from {url}: {e}")
            return []

    def _is_valid_document_url(self, url: str) -> bool:
        """
        Check if URL is a valid document (not image/js/css).

        Args:
            url: URL to check

        Returns:
            True if URL is a valid document, False otherwise
        """
        # Parse URL to get path
        parsed = urlparse(url)
        path = parsed.path.lower()

        # Invalid file extensions (images, scripts, styles)
        invalid_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".svg",
            ".webp",
            ".ico",
            ".bmp",
            ".js",
            ".css",
            ".json",
            ".xml",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".mp4",
            ".mp3",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".webm",
            ".zip",
            ".tar",
            ".gz",
            ".rar",
            ".7z",
        ]

        # Check if path ends with invalid extension
        for ext in invalid_extensions:
            if path.endswith(ext):
                return False

        # Valid document extensions
        valid_extensions = [
            ".html",
            ".htm",
            ".pdf",
            ".doc",
            ".docx",
            ".txt",
            ".md",
            ".rtf",
            ".odt",
            ".pages",
        ]

        # Check if path ends with valid extension
        for ext in valid_extensions:
            if path.endswith(ext):
                return True

        # If no extension or extension not in lists, check if it looks like a document page
        # (has path segments, not just domain)
        if (
            path
            and path != "/"
            and not any(path.endswith(ext) for ext in invalid_extensions)
        ):
            # Assume it's a document page if it has a path
            return True

        return False

    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing fragments and cleaning up.

        Args:
            url: URL to normalize

        Returns:
            Normalized URL without fragment
        """
        parsed = urlparse(url)
        # Remove fragment, keep everything else
        normalized = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                "",  # Remove fragment
            )
        )
        return normalized

    def _extract_rich_metadata(
        self, soup: BeautifulSoup, url: str, content: str
    ) -> ScannedDocumentMetadata:
        """Extract rich metadata from HTML and content."""
        metadata = ScannedDocumentMetadata()

        # Content information
        metadata.content_text = content
        metadata.content_summary = self._generate_summary(content)
        metadata.word_count = len(content.split()) if content else 0

        # Structure information
        metadata.headings = self._extract_headings(soup)
        metadata.sections = self._extract_sections(soup)
        metadata.content_type = self._infer_content_type(url, soup, content)

        # Keywords and topics
        metadata.extracted_keywords = self._extract_keywords(content)
        metadata.topics = self._extract_topics(content, metadata.headings)
        metadata.categories = self._extract_categories(url, content)

        # Internal linking information
        internal_links = self._extract_internal_links(soup, url)
        metadata.internal_links_found = internal_links
        metadata.anchor_text_patterns = [
            link.get("anchor_text", "")
            for link in internal_links
            if link and link.get("anchor_text")
        ]
        metadata.outbound_link_count = len(internal_links)

        # SEO metadata
        metadata.meta_description = self._extract_meta_description(soup)
        metadata.meta_keywords = self._extract_meta_keywords(soup)
        metadata.canonical_url = self._extract_canonical_url(soup, url)

        # Additional metadata
        metadata.author = self._extract_author(soup)
        metadata.last_modified = self._extract_last_modified(soup)
        metadata.language = self._extract_language(soup)
        metadata.reading_time_minutes = self._calculate_reading_time(
            metadata.word_count
        )

        # Quality metrics
        metadata.readability_score = self._calculate_readability(content)
        metadata.completeness_score = self._assess_completeness(
            content, metadata.headings
        )

        return metadata

    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate a brief summary of the content."""
        if not content:
            return ""
        sentences = content.split(". ")
        summary = ". ".join(sentences[:3])
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary

    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract all headings from the document."""
        headings = []
        for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            for heading in soup.find_all(level):
                text = heading.get_text().strip()
                if text:
                    headings.append(text)
        return headings

    def _extract_sections(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract document sections with structure."""
        sections = []
        current_section = None

        for elem in soup.find_all(["h1", "h2", "h3", "p"]):
            if elem.name in ["h1", "h2", "h3"]:
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "heading": elem.get_text().strip(),
                    "level": int(elem.name[1]),
                    "content": "",
                }
            elif current_section and elem.name == "p":
                current_section["content"] += elem.get_text().strip() + " "

        if current_section:
            sections.append(current_section)

        return sections

    def _infer_content_type(self, url: str, soup: BeautifulSoup, content: str) -> str:
        """Infer the type of content."""
        url_lower = url.lower()
        if "blog" in url_lower or "post" in url_lower:
            return "blog"
        elif "docs" in url_lower or "documentation" in url_lower:
            return "docs"
        elif "guide" in url_lower or "tutorial" in url_lower:
            return "guide"
        elif "api" in url_lower:
            return "api"
        else:
            return "page"

    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from content (simple frequency-based)."""
        if not content:
            return []

        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r"\b[a-z]{4,}\b", content.lower())
        word_freq = {}
        for word in words:
            if word not in [
                "this",
                "that",
                "with",
                "from",
                "have",
                "been",
                "will",
                "would",
            ]:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]

    def _extract_topics(self, content: str, headings: List[str]) -> List[str]:
        """Extract main topics from headings and content."""
        topics = []
        # Use H1 and H2 headings as topics
        for heading in headings[:5]:
            if len(heading.split()) <= 5:  # Short headings are likely topics
                topics.append(heading)
        return topics

    def _extract_categories(self, url: str, content: str) -> List[str]:
        """Extract categories from URL path and content."""
        categories = []
        parsed = urlparse(url)
        path_parts = [
            p
            for p in parsed.path.strip("/").split("/")
            if p and p not in ["docs", "blog", "api"]
        ]
        if len(path_parts) > 1:
            categories.extend(path_parts[:-1])  # All but last are categories
        return categories

    def _extract_internal_links(
        self, soup: BeautifulSoup, base_url: str
    ) -> List[Dict[str, str]]:
        """Extract internal links with anchor text."""
        links = []
        parsed_base = urlparse(base_url)

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            anchor_text = a_tag.get_text().strip()

            if not anchor_text:
                continue

            absolute_url = urljoin(base_url, href)
            parsed_link = urlparse(absolute_url)

            # Only include same-domain links
            if parsed_link.netloc == parsed_base.netloc:
                links.append({"anchor_text": anchor_text, "target_url": absolute_url})

        return links

    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description."""
        meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", property="og:description"
        )
        if meta_desc:
            return meta_desc.get("content", "").strip()
        return None

    def _extract_meta_keywords(self, soup: BeautifulSoup) -> Optional[List[str]]:
        """Extract meta keywords."""
        meta_keywords = soup.find("meta", attrs={"name": "keywords"})
        if meta_keywords:
            keywords_str = meta_keywords.get("content", "")
            return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        return None

    def _extract_canonical_url(
        self, soup: BeautifulSoup, base_url: str
    ) -> Optional[str]:
        """Extract canonical URL."""
        canonical = soup.find("link", rel="canonical")
        if canonical:
            return urljoin(base_url, canonical.get("href", ""))
        return None

    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author information."""
        author = (
            soup.find("meta", property="article:author")
            or soup.find("meta", attrs={"name": "author"})
            or soup.find("span", class_=re.compile(r"author", re.I))
        )
        if author:
            if author.name == "meta":
                return author.get("content", "").strip()
            return author.get_text().strip()
        return None

    def _extract_last_modified(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract last modified date."""
        last_mod = soup.find("meta", attrs={"http-equiv": "last-modified"})
        if last_mod:
            try:
                return datetime.fromisoformat(last_mod.get("content", ""))
            except Exception:
                pass
        return None

    def _extract_language(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract content language."""
        lang = soup.find("html", lang=True)
        if lang:
            return lang.get("lang")
        meta_lang = soup.find("meta", attrs={"http-equiv": "content-language"})
        if meta_lang:
            return meta_lang.get("content")
        return None

    def _calculate_reading_time(self, word_count: int) -> float:
        """Calculate estimated reading time in minutes (assuming 200 words/min)."""
        if word_count == 0:
            return 0.0
        return round(word_count / 200.0, 1)

    def _calculate_readability(self, content: str) -> float:
        """Calculate basic readability score (simplified Flesch-like)."""
        if not content:
            return 0.0

        sentences = content.split(".")
        words = content.split()

        if len(sentences) == 0 or len(words) == 0:
            return 0.0

        avg_sentence_length = len(words) / len(sentences)
        # Simplified readability (higher is better, 0-100 scale)
        score = max(0, min(100, 100 - (avg_sentence_length * 2)))
        return round(score, 1)

    def _assess_completeness(self, content: str, headings: List[str]) -> float:
        """Assess content completeness (0.0-1.0)."""
        score = 0.0

        if content and len(content) > 100:
            score += 0.3
        if len(headings) > 0:
            score += 0.3
        if len(headings) >= 3:
            score += 0.2
        if len(content.split()) > 300:
            score += 0.2

        return round(score, 2)

    async def analyze_documents(
        self, documents: List[ScannedDocument]
    ) -> Dict[str, Any]:
        """
        Analyze scanned documents to extract patterns from database.

        Args:
            documents: List of scanned documents

        Returns:
            Dictionary with commonly_referenced_pages, commonly_referenced_categories,
            and anchor_phrasing_patterns
        """
        try:
            db = get_scanned_document_db()

            # Get commonly referenced pages from database
            commonly_referenced_pages = db.get_commonly_referenced_pages(min_links=2)

            # Get anchor text patterns from database
            anchor_patterns = db.get_anchor_text_patterns()

            # Extract categories from database documents
            all_docs = db.get_all_active_documents()
            categories_set = set()
            for doc in all_docs:
                categories_set.update(doc.metadata.categories)

            return {
                "commonly_referenced_pages": commonly_referenced_pages,
                "commonly_referenced_categories": sorted(list(categories_set)),
                "anchor_phrasing_patterns": anchor_patterns[:20],  # Limit to top 20
            }
        except Exception as e:
            logger.warning(f"Error analyzing documents from database: {e}")
            # Fallback to URL-based extraction
            pages: Set[str] = set()
            categories: Set[str] = set()

            for doc in documents:
                parsed = urlparse(doc.url)
                path_parts = [p for p in parsed.path.strip("/").split("/") if p]

                if path_parts:
                    pages.add(path_parts[-1])
                    if len(path_parts) > 1:
                        categories.update(path_parts[:-1])

            return {
                "commonly_referenced_pages": sorted(list(pages)),
                "commonly_referenced_categories": sorted(list(categories)),
                "anchor_phrasing_patterns": [],
            }


# Singleton instance
_scanner: Optional[InternalDocsScanner] = None


async def get_internal_docs_scanner() -> InternalDocsScanner:
    """Get or create the singleton scanner instance."""
    global _scanner
    if _scanner is None:
        _scanner = InternalDocsScanner()
    return _scanner
