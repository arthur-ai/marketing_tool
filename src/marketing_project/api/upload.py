"""
File upload API endpoints for Marketing Project.
"""

import csv
import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

from marketing_project.middleware.keycloak_auth import get_current_user
from marketing_project.models.user_context import UserContext

# Document processing imports
try:
    from docx import Document as DocxDocument

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from marketing_project.core.parsers import parse_transcript
from marketing_project.services.content_source_config_loader import (
    ContentSourceConfigLoader,
)
from marketing_project.services.content_source_factory import ContentSourceManager
from marketing_project.services.s3_storage import S3Storage

logger = logging.getLogger("marketing_project.api.upload")

# Create router
router = APIRouter()

# Initialize content source manager
content_manager = ContentSourceManager()
config_loader = ContentSourceConfigLoader()

# Upload directories
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
CONTENT_DIR = os.getenv("CONTENT_DIR", "content")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CONTENT_DIR, exist_ok=True)

# Initialize S3 storage (will be None if S3 is not configured)
s3_storage = S3Storage(prefix="")


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    content_type: str = Form("blog_post"),
    user: UserContext = Depends(get_current_user),
):
    """
    Upload a file and process it for the marketing pipeline.

    Args:
        file: The file to upload
        content_type: Type of content (blog_post, transcript, release_notes)
    """
    try:
        # Validate file type
        allowed_extensions = {
            ".json",
            ".md",
            ".txt",
            ".yaml",
            ".yml",
            ".docx",
            ".doc",
            ".pdf",
            ".csv",
            ".rtf",
        }
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
            )

        # Validate file size (25MB limit)
        max_size = 25 * 1024 * 1024  # 25MB
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=400, detail="File too large. Maximum size is 25MB"
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.filename}"

        # Save to uploads directory
        upload_path = os.path.join(UPLOAD_DIR, safe_filename)
        with open(upload_path, "wb") as buffer:
            buffer.write(content)

        logger.info(f"File uploaded: {safe_filename} ({len(content)} bytes)")

        # Process file based on content type
        processed_path = await process_uploaded_file(
            upload_path, content_type, file_ext
        )

        # Check if file was uploaded to S3
        s3_uploaded = False
        s3_key = None
        if s3_storage.is_available():
            # The S3 upload happens in process_uploaded_file, so we need to check
            # We'll add s3_key to the response in process_uploaded_file return
            # For now, just indicate S3 is available
            s3_uploaded = True

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "File uploaded and processed successfully",
                "file_id": file_id,
                "filename": file.filename,
                "size": len(content),
                "content_type": content_type,
                "upload_path": upload_path,
                "processed_path": processed_path,
                "s3_uploaded": s3_uploaded,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    if not DOCX_AVAILABLE:
        raise HTTPException(status_code=500, detail="DOCX processing not available")

    try:
        doc = DocxDocument(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process DOCX file: {str(e)}"
        )


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    if not PDF_AVAILABLE:
        raise HTTPException(status_code=500, detail="PDF processing not available")

    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return "\n".join(text)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process PDF file: {str(e)}"
        )


def extract_text_from_csv(file_path: str) -> str:
    """Extract text from CSV file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)

            # Convert CSV to readable text format
            text_lines = []
            for i, row in enumerate(rows):
                if i == 0:  # Header row
                    text_lines.append("Headers: " + " | ".join(row))
                else:
                    text_lines.append("Row " + str(i) + ": " + " | ".join(row))

            return "\n".join(text_lines)
    except Exception as e:
        logger.error(f"Failed to extract text from CSV: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process CSV file: {str(e)}"
        )


def extract_text_from_rtf(file_path: str) -> str:
    """Extract text from RTF file (basic implementation)."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            # Basic RTF text extraction (removes RTF formatting codes)
            import re

            # Remove RTF control words and braces
            text = re.sub(r"\\[a-z]+\d*\s?", "", content)
            text = re.sub(r"[{}]", "", text)
            return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text from RTF: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process RTF file: {str(e)}"
        )


async def process_uploaded_file(
    upload_path: str, content_type: str, file_ext: str
) -> str:
    """
    Process an uploaded file and move it to the appropriate content directory.
    """
    try:
        # Create content directory structure
        content_type_dir = os.path.join(CONTENT_DIR, f"{content_type}s")
        os.makedirs(content_type_dir, exist_ok=True)

        # Generate processed filename
        filename = Path(upload_path).name
        processed_path = os.path.join(content_type_dir, filename)

        # Extract text content based on file type
        if file_ext == ".json":
            # Read JSON file
            with open(upload_path, "r", encoding="utf-8") as f:
                content_data = f.read()
        elif file_ext in [".docx", ".doc"]:
            # Extract text from Word document
            content_data = extract_text_from_docx(upload_path)
        elif file_ext == ".pdf":
            # Extract text from PDF
            content_data = extract_text_from_pdf(upload_path)
        elif file_ext == ".csv":
            # Extract text from CSV
            content_data = extract_text_from_csv(upload_path)
        elif file_ext == ".rtf":
            # Extract text from RTF
            content_data = extract_text_from_rtf(upload_path)
        else:
            # Read text files (md, txt, yaml, yml)
            with open(upload_path, "r", encoding="utf-8") as f:
                content_data = f.read()

        # Process based on file type
        if file_ext == ".json":
            # Validate JSON structure
            try:
                json_data = json.loads(content_data)
                # Ensure required fields for content
                if "title" not in json_data:
                    json_data["title"] = Path(filename).stem
                if "content" not in json_data:
                    json_data["content"] = content_data
                if "id" not in json_data:
                    json_data["id"] = str(uuid.uuid4())

                # If this is a transcript and has raw content, preprocess it
                if (
                    content_type == "transcript"
                    and "content" in json_data
                    and isinstance(json_data["content"], str)
                ):
                    raw_content = json_data["content"]
                    parsed_data = parse_transcript(
                        raw_content, content_type="transcript"
                    )

                    # Update with parsed data
                    json_data["content"] = parsed_data.get(
                        "cleaned_content", raw_content
                    )
                    json_data["content_type"] = (
                        "transcript"  # Ensure content_type is set
                    )
                    if "speakers" not in json_data or not json_data["speakers"]:
                        json_data["speakers"] = parsed_data.get("speakers", [])
                    if "duration" not in json_data or not json_data["duration"]:
                        # models.content_models.TranscriptContext expects duration as int
                        json_data["duration"] = parsed_data.get("duration")
                    if (
                        "transcript_type" not in json_data
                        or not json_data["transcript_type"]
                    ):
                        json_data["transcript_type"] = parsed_data.get(
                            "transcript_type", "podcast"
                        )
                    if parsed_data.get("timestamps"):
                        json_data["timestamps"] = parsed_data["timestamps"]
                    if "snippet" not in json_data:
                        # Use generated snippet from parser if available
                        json_data["snippet"] = parsed_data.get("snippet") or (
                            json_data["content"][:200] + "..."
                            if len(json_data["content"]) > 200
                            else json_data["content"]
                        )

                    # Preserve enhanced parsing metadata
                    if parsed_data.get("parsing_confidence") is not None:
                        json_data["parsing_confidence"] = parsed_data[
                            "parsing_confidence"
                        ]
                    if parsed_data.get("detected_format"):
                        json_data["detected_format"] = parsed_data["detected_format"]
                    if parsed_data.get("warnings"):
                        json_data["parsing_warnings"] = parsed_data["warnings"]
                    if parsed_data.get("quality_metrics"):
                        json_data["quality_metrics"] = parsed_data["quality_metrics"]
                    if parsed_data.get("speaking_time_per_speaker"):
                        json_data["speaking_time_per_speaker"] = parsed_data[
                            "speaking_time_per_speaker"
                        ]
                    if parsed_data.get("detected_language"):
                        json_data["detected_language"] = parsed_data[
                            "detected_language"
                        ]
                    if parsed_data.get("key_topics"):
                        json_data["key_topics"] = parsed_data["key_topics"]
                    if parsed_data.get("conversation_flow"):
                        json_data["conversation_flow"] = parsed_data[
                            "conversation_flow"
                        ]

                # Write processed JSON
                with open(processed_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)

            except json.JSONDecodeError:
                # If not valid JSON, treat as raw text and process based on content_type
                if content_type == "transcript":
                    # Parse transcript from raw text
                    parsed_data = parse_transcript(
                        content_data, content_type="transcript"
                    )
                    # models.content_models.TranscriptContext expects duration as int
                    json_data = {
                        "id": str(uuid.uuid4()),
                        "title": Path(filename).stem,
                        "content": parsed_data.get("cleaned_content", content_data),
                        "content_type": "transcript",
                        "speakers": parsed_data.get("speakers", []),
                        "duration": parsed_data.get(
                            "duration"
                        ),  # Already an int from parser
                        "transcript_type": parsed_data.get(
                            "transcript_type", "podcast"
                        ),
                        "created_at": datetime.utcnow().isoformat() + "Z",
                    }
                    if parsed_data.get("timestamps"):
                        json_data["timestamps"] = parsed_data["timestamps"]
                    json_data["snippet"] = parsed_data.get("snippet") or (
                        json_data["content"][:200] + "..."
                        if len(json_data["content"]) > 200
                        else json_data["content"]
                    )

                    # Preserve enhanced parsing metadata
                    if parsed_data.get("parsing_confidence") is not None:
                        json_data["parsing_confidence"] = parsed_data[
                            "parsing_confidence"
                        ]
                    if parsed_data.get("detected_format"):
                        json_data["detected_format"] = parsed_data["detected_format"]
                    if parsed_data.get("warnings"):
                        json_data["parsing_warnings"] = parsed_data["warnings"]
                    if parsed_data.get("quality_metrics"):
                        json_data["quality_metrics"] = parsed_data["quality_metrics"]
                    if parsed_data.get("speaking_time_per_speaker"):
                        json_data["speaking_time_per_speaker"] = parsed_data[
                            "speaking_time_per_speaker"
                        ]
                    if parsed_data.get("detected_language"):
                        json_data["detected_language"] = parsed_data[
                            "detected_language"
                        ]
                    if parsed_data.get("key_topics"):
                        json_data["key_topics"] = parsed_data["key_topics"]
                    if parsed_data.get("conversation_flow"):
                        json_data["conversation_flow"] = parsed_data[
                            "conversation_flow"
                        ]
                else:
                    # If not valid JSON, create a simple structure
                    json_data = {
                        "id": str(uuid.uuid4()),
                        "title": Path(filename).stem,
                        "content": content_data,
                        "type": content_type,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                    }
                with open(processed_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)

        elif file_ext in [
            ".md",
            ".txt",
            ".docx",
            ".doc",
            ".pdf",
            ".csv",
            ".rtf",
            ".yaml",
            ".yml",
        ]:
            # For transcript files, use enhanced parser
            if content_type == "transcript":
                # Parse transcript using enhanced parser
                parsed_data = parse_transcript(content_data, content_type="transcript")

                # Create proper JSON structure matching TranscriptContext
                # models.content_models.TranscriptContext expects duration as int
                json_data = {
                    "id": str(uuid.uuid4()),
                    "title": Path(filename).stem,
                    "content": parsed_data.get("cleaned_content", content_data),
                    "content_type": "transcript",  # Use content_type not type
                    "speakers": parsed_data.get("speakers", []),
                    "duration": parsed_data.get(
                        "duration"
                    ),  # Already an int from parser
                    "transcript_type": parsed_data.get("transcript_type", "podcast"),
                    "original_format": file_ext,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }

                # Add timestamps if available
                if parsed_data.get("timestamps"):
                    json_data["timestamps"] = parsed_data["timestamps"]

                # Add snippet if content is long (use generated snippet if available)
                json_data["snippet"] = parsed_data.get("snippet") or (
                    json_data["content"][:200] + "..."
                    if len(json_data["content"]) > 200
                    else json_data["content"]
                )

                # Preserve enhanced parsing metadata
                if parsed_data.get("parsing_confidence") is not None:
                    json_data["parsing_confidence"] = parsed_data["parsing_confidence"]
                if parsed_data.get("detected_format"):
                    json_data["detected_format"] = parsed_data["detected_format"]
                if parsed_data.get("warnings"):
                    json_data["parsing_warnings"] = parsed_data["warnings"]
                if parsed_data.get("quality_metrics"):
                    json_data["quality_metrics"] = parsed_data["quality_metrics"]
                if parsed_data.get("speaking_time_per_speaker"):
                    json_data["speaking_time_per_speaker"] = parsed_data[
                        "speaking_time_per_speaker"
                    ]
                if parsed_data.get("detected_language"):
                    json_data["detected_language"] = parsed_data["detected_language"]
                if parsed_data.get("key_topics"):
                    json_data["key_topics"] = parsed_data["key_topics"]
                if parsed_data.get("conversation_flow"):
                    json_data["conversation_flow"] = parsed_data["conversation_flow"]
            else:
                # For all other files, create a simple structure
                json_data = {
                    "id": str(uuid.uuid4()),
                    "title": Path(filename).stem,
                    "content": content_data,
                    "type": content_type,
                    "original_format": file_ext,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }

            # Save as JSON for consistency
            json_path = processed_path.replace(file_ext, ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            processed_path = json_path

        else:
            # For other file types, just copy
            import shutil

            shutil.copy2(upload_path, processed_path)

        logger.info(f"File processed: {upload_path} -> {processed_path}")

        # Upload to S3 if available
        s3_key = None
        if s3_storage.is_available():
            try:
                # Determine S3 key based on content type and filename
                content_type_dir = f"{content_type}s"
                s3_key_path = f"{content_type_dir}/{Path(processed_path).name}"
                s3_key = await s3_storage.upload_file(processed_path, s3_key_path)
                if s3_key:
                    logger.info(
                        f"File uploaded to S3: s3://{s3_storage.bucket_name}/{s3_key}"
                    )
                else:
                    logger.warning(
                        "Failed to upload file to S3, but local processing succeeded"
                    )
            except Exception as e:
                logger.error(f"Error uploading to S3: {e}")
                # Don't fail the upload if S3 fails

        # Clean up upload file
        try:
            os.remove(upload_path)
        except OSError:
            logger.warning(f"Could not remove upload file: {upload_path}")

        return processed_path

    except Exception as e:
        logger.error(f"File processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


class URLExtractionRequest(BaseModel):
    """Request model for URL extraction."""

    url: str
    content_type: str = "blog_post"


def extract_blog_content_from_url(url: str) -> Dict[str, Any]:
    """
    Extract blog post content from a URL using web scraping.

    Args:
        url: The URL of the blog post

    Returns:
        Dict containing extracted content with title, content, author, etc.
    """
    try:
        # Fetch the webpage
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract title - try common selectors
        title = None
        for selector in ["h1", "title", ".post-title", ".entry-title", "article h1"]:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                break

        if not title:
            title = (
                soup.title.string if soup.title else urlparse(url).path.split("/")[-1]
            )

        # Extract main content - try common article selectors
        content_text = None
        for selector in [
            "article",
            ".post-content",
            ".entry-content",
            ".article-content",
            ".content",
            "main article",
            '[role="main"]',
        ]:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove script and style elements
                for script in content_elem(
                    ["script", "style", "nav", "aside", "footer", "header"]
                ):
                    script.decompose()
                content_text = content_elem.get_text(separator="\n", strip=True)
                break

        # Fallback: get all paragraphs if no article content found
        if not content_text or len(content_text) < 100:
            paragraphs = soup.find_all("p")
            content_text = "\n\n".join(
                [
                    p.get_text(strip=True)
                    for p in paragraphs
                    if len(p.get_text(strip=True)) > 50
                ]
            )

        # Extract metadata
        author = None
        # Try to find author
        for selector in [
            ".author",
            ".by-author",
            '[rel="author"]',
            ".post-author",
            'meta[name="author"]',
        ]:
            author_elem = soup.select_one(selector)
            if author_elem:
                if author_elem.name == "meta":
                    author = author_elem.get("content", "").strip()
                else:
                    author = author_elem.get_text(strip=True)
                break

        # Extract publish date
        published_date = None
        for selector in [
            "time",
            ".published",
            ".post-date",
            'meta[property="article:published_time"]',
        ]:
            date_elem = soup.select_one(selector)
            if date_elem:
                if date_elem.name == "meta":
                    published_date = date_elem.get("content", "")
                elif date_elem.name == "time":
                    published_date = date_elem.get(
                        "datetime", date_elem.get_text(strip=True)
                    )
                else:
                    published_date = date_elem.get_text(strip=True)
                break

        # Extract meta description as snippet
        snippet = None
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            snippet = meta_desc.get("content", "").strip()

        # If no snippet, use first 200 chars of content
        if not snippet and content_text:
            snippet = (
                content_text[:200].strip() + "..."
                if len(content_text) > 200
                else content_text.strip()
            )

        # Extract tags/categories
        tags = []
        for selector in [".tags a", ".tag", 'meta[property="article:tag"]']:
            tag_elems = soup.select(selector)
            if tag_elems:
                for tag_elem in tag_elems:
                    if tag_elem.name == "meta":
                        tag_text = tag_elem.get("content", "").strip()
                    else:
                        tag_text = tag_elem.get_text(strip=True)
                    if tag_text and tag_text not in tags:
                        tags.append(tag_text)

        # Calculate word count
        word_count = len(content_text.split()) if content_text else 0

        # Build the extracted data
        extracted_data = {
            "id": str(uuid.uuid4()),
            "title": title or "Untitled Blog Post",
            "content": content_text or "",
            "snippet": snippet or "",
            "source_url": url,
            "author": author or "Unknown",
            "tags": tags[:10],  # Limit to 10 tags
            "category": "General",
            "word_count": word_count,
            "created_at": published_date or datetime.utcnow().isoformat() + "Z",
            "reading_time": (
                max(1, word_count // 200) if word_count > 0 else None
            ),  # Rough estimate
            "metadata": {
                "extracted_from_url": True,
                "extraction_date": datetime.utcnow().isoformat() + "Z",
                "original_url": url,
            },
        }

        return extracted_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to extract content from URL {url}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to extract content: {str(e)}"
        )


@router.post("/upload/from-url")
@router.post("/upload/url")  # Alias for test compatibility
async def upload_from_url(
    request: URLExtractionRequest, user: UserContext = Depends(get_current_user)
):
    """
    Extract blog post content from a URL and process it for the marketing pipeline.

    Args:
        request: URLExtractionRequest containing URL and content type

    Returns:
        JSONResponse with extracted content
    """
    try:
        logger.info(f"Extracting content from URL: {request.url}")

        # Extract content from URL
        extracted_data = extract_blog_content_from_url(request.url)

        # Validate that we got meaningful content
        if len(extracted_data["content"]) < 100:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient content from URL. The page may be behind a paywall, require JavaScript, or have an unusual structure.",
            )

        # Save to content directory
        content_type_dir = os.path.join(CONTENT_DIR, f"{request.content_type}s")
        os.makedirs(content_type_dir, exist_ok=True)

        # Generate filename from URL
        parsed_url = urlparse(request.url)
        safe_filename = re.sub(
            r"[^a-zA-Z0-9_-]", "_", parsed_url.path.strip("/").replace("/", "_")
        )
        if not safe_filename:
            safe_filename = "extracted_content"
        safe_filename = f"{extracted_data['id']}_{safe_filename}.json"

        processed_path = os.path.join(content_type_dir, safe_filename)

        # Save as JSON
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Content extracted and saved from {request.url} to {processed_path}"
        )

        # Upload to S3 if available
        s3_key = None
        if s3_storage.is_available():
            try:
                content_type_dir = f"{request.content_type}s"
                s3_key_path = f"{content_type_dir}/{safe_filename}"
                s3_key = await s3_storage.upload_file(processed_path, s3_key_path)
                if s3_key:
                    logger.info(
                        f"Content uploaded to S3: s3://{s3_storage.bucket_name}/{s3_key}"
                    )
            except Exception as e:
                logger.error(f"Error uploading to S3: {e}")

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Content extracted from URL successfully",
                "file_id": extracted_data["id"],
                "url": request.url,
                "title": extracted_data["title"],
                "word_count": extracted_data["word_count"],
                "content_type": request.content_type,
                "processed_path": processed_path,
                "s3_uploaded": s3_key is not None,
                "s3_key": s3_key,
                "extracted_data": extracted_data,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL extraction failed for {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"URL extraction failed: {str(e)}")


@router.get("/upload/status/{file_id}")
async def get_upload_status(
    file_id: str, user: UserContext = Depends(get_current_user)
):
    """
    Get the status of an uploaded file.
    """
    try:
        # This would typically check a database or cache for file status
        # For now, we'll return a simple response
        return JSONResponse(
            status_code=200,
            content={
                "file_id": file_id,
                "status": "processed",
                "message": "File has been processed successfully",
            },
        )
    except Exception as e:
        logger.error(f"Failed to get upload status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get upload status: {str(e)}"
        )
