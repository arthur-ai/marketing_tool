"""
Internal Docs API Endpoints.

Endpoints for managing internal documentation configuration and scanning.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field

from ..middleware.keycloak_auth import get_current_user
from ..middleware.rbac import require_roles
from ..models.internal_docs_config import InternalDocsConfig, ScannedDocument
from ..models.scanned_document_db import ScannedDocumentDB
from ..models.user_context import UserContext
from ..services.internal_docs_manager import get_internal_docs_manager
from ..services.internal_docs_scanner import get_internal_docs_scanner
from ..services.scanned_document_db import get_scanned_document_db

logger = logging.getLogger(__name__)

router = APIRouter()


# Request models
class ScanFromUrlRequest(BaseModel):
    """Request model for scanning from base URL."""

    base_url: str = Field(..., description="Base URL to start crawling from")
    max_depth: int = Field(3, ge=1, le=10, description="Maximum crawl depth")
    follow_external: bool = Field(False, description="Whether to follow external links")
    max_pages: int = Field(
        100, ge=1, le=1000, description="Maximum number of pages to scan"
    )
    merge_with_existing: bool = Field(
        True, description="Whether to merge with existing config"
    )


class ScanFromListRequest(BaseModel):
    """Request model for scanning from URL list."""

    urls: List[str] = Field(..., description="List of URLs to scan")
    merge_with_existing: bool = Field(
        True, description="Whether to merge with existing config"
    )


class MergeScanResultsRequest(BaseModel):
    """Request model for merging scan results."""

    scanned_docs: List[Dict[str, Any]] = Field(
        ..., description="List of scanned documents to merge"
    )


class BulkOperationRequest(BaseModel):
    """Request model for bulk operations."""

    urls: List[str] = Field(..., description="List of document URLs")


class BulkCategoryUpdateRequest(BaseModel):
    """Request model for bulk category update."""

    urls: List[str] = Field(..., description="List of document URLs")
    categories: List[str] = Field(
        ..., description="Categories to add (merged with existing)"
    )


@router.get("/config")
async def get_internal_docs_config(user: UserContext = Depends(get_current_user)):
    """
    Get the currently active internal docs configuration.

    Returns:
        InternalDocsConfig or None: The active configuration, or None if no config exists
    """
    try:
        manager = await get_internal_docs_manager()
        config = await manager.get_active_config()

        # Return None instead of 404 to prevent React Query from retrying
        if not config:
            return None

        return config
    except Exception as e:
        logger.error(f"Error getting internal docs config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get internal docs configuration: {str(e)}",
        )


@router.get("/config/{version}", response_model=InternalDocsConfig)
async def get_internal_docs_config_by_version(
    version: str, user: UserContext = Depends(get_current_user)
):
    """
    Get internal docs configuration by version.

    Args:
        version: Version identifier

    Returns:
        InternalDocsConfig: The configuration for the specified version
    """
    try:
        manager = await get_internal_docs_manager()
        config = await manager.get_config_by_version(version)

        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"Internal docs configuration version {version} not found",
            )

        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting internal docs config version {version}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get internal docs configuration version: {str(e)}",
        )


@router.post("/config", response_model=InternalDocsConfig)
async def create_or_update_internal_docs_config(
    request: dict = Body(..., description="Request body with config and set_active"),
    user: UserContext = Depends(require_roles(["admin"])),
):
    """
    Create or update internal docs configuration.

    Only one config is allowed. If a config exists, it will be updated.
    If no config exists, a new one will be created.

    Args:
        request: Dict with 'config' (InternalDocsConfig) and 'set_active' (bool, ignored - always True)

    Returns:
        InternalDocsConfig: The saved configuration
    """
    try:
        config_dict = request.get("config", {})
        # set_active is ignored - always True since there's only one config
        config = InternalDocsConfig(**config_dict)

        manager = await get_internal_docs_manager()
        # save_config will automatically update existing config or create new one
        success = await manager.save_config(config, set_active=True)

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to save internal docs configuration"
            )

        # Return the saved active config (there's only one)
        saved_config = await manager.get_active_config()
        if not saved_config:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve saved configuration"
            )
        return saved_config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving internal docs config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save internal docs configuration: {str(e)}",
        )


@router.get("/versions", response_model=List[str])
async def list_internal_docs_versions(
    user: UserContext = Depends(get_current_user),
):
    """
    List all available internal docs configuration versions.

    Returns:
        List[str]: List of version identifiers
    """
    try:
        manager = await get_internal_docs_manager()
        versions = await manager.list_versions()
        return versions
    except Exception as e:
        logger.error(f"Error listing internal docs versions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list internal docs versions: {str(e)}"
        )


@router.post("/activate/{version}")
async def activate_internal_docs_version(
    version: str, user: UserContext = Depends(require_roles(["admin"]))
):
    """
    Activate a specific internal docs configuration version.

    Args:
        version: Version identifier to activate

    Returns:
        dict: Success message
    """
    try:
        manager = await get_internal_docs_manager()
        success = await manager.activate_version(version)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to activate version {version}. Version may not exist.",
            )

        return {"message": f"Activated internal docs configuration version {version}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating internal docs version {version}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to activate internal docs version: {str(e)}",
        )


@router.post("/scan/url")
async def scan_from_url(
    request: ScanFromUrlRequest, user: UserContext = Depends(require_roles(["admin"]))
):
    """
    Scan internal documentation from a base URL with crawling (runs as background ARQ job).

    Args:
        request: ScanFromUrlRequest with base_url, max_depth, etc.

    Returns:
        dict: Job information
    """
    try:
        from marketing_project.services.job_manager import get_job_manager

        job_manager = get_job_manager()  # Synchronous, not async
        job_id = f"scan_url_{int(datetime.now(timezone.utc).timestamp() * 1000)}"

        # Create job first
        job = await job_manager.create_job(
            job_type="scan_from_url",
            content_id="internal_docs",
            metadata={
                "base_url": request.base_url,
                "max_depth": request.max_depth,
                "follow_external": request.follow_external,
                "max_pages": request.max_pages,
                "merge_with_existing": request.merge_with_existing,
            },
            job_id=job_id,
            user_id=user.user_id,
            user_context=user,
        )

        # Submit to ARQ
        arq_job_id = await job_manager.submit_to_arq(
            job_id,  # positional: job_id for JobManager tracking
            "scan_from_url_job",  # positional: function_name
            request.base_url,  # positional: goes to *args, passed to worker as base_url
            request.max_depth,  # positional: goes to *args, passed to worker as max_depth
            request.follow_external,  # positional: goes to *args, passed to worker as follow_external
            request.max_pages,  # positional: goes to *args, passed to worker as max_pages
            request.merge_with_existing,  # positional: goes to *args, passed to worker as merge_with_existing
            job_id,  # positional: goes to *args, passed to worker as job_id
        )

        return {
            "message": f"Scan job started for {request.base_url}",
            "job_id": job_id,
            "arq_job_id": arq_job_id,
            "base_url": request.base_url,
        }
    except Exception as e:
        logger.error(f"Error starting scan from URL {request.base_url}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start scan from URL: {str(e)}"
        )


@router.post("/scan/list")
async def scan_from_list(
    request: ScanFromListRequest, user: UserContext = Depends(require_roles(["admin"]))
):
    """
    Scan internal documentation from a list of URLs (runs as background ARQ job).

    Args:
        request: ScanFromListRequest with list of URLs

    Returns:
        dict: Job information
    """
    try:
        from marketing_project.services.job_manager import get_job_manager

        job_manager = get_job_manager()  # Synchronous, not async
        job_id = f"scan_list_{int(datetime.now(timezone.utc).timestamp() * 1000)}"

        # Create job first
        job = await job_manager.create_job(
            job_type="scan_from_list",
            content_id="internal_docs",
            metadata={
                "urls_count": len(request.urls),
                "merge_with_existing": request.merge_with_existing,
            },
            job_id=job_id,
            user_id=user.user_id,
            user_context=user,
        )

        # Submit to ARQ
        arq_job_id = await job_manager.submit_to_arq(
            job_id,  # positional: job_id for JobManager tracking
            "scan_from_list_job",  # positional: function_name
            request.urls,  # positional: goes to *args, passed to worker as urls
            request.merge_with_existing,  # positional: goes to *args, passed to worker as merge_with_existing
            job_id,  # positional: goes to *args, passed to worker as job_id
        )

        return {
            "message": f"Scan job started for {len(request.urls)} URLs",
            "job_id": job_id,
            "arq_job_id": arq_job_id,
            "urls_count": len(request.urls),
        }
    except Exception as e:
        logger.error(f"Error starting scan from URL list: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start scan from URL list: {str(e)}"
        )


@router.post("/scan/merge")
async def merge_scan_results(
    request: MergeScanResultsRequest,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """
    Merge scan results with existing configuration.

    Args:
        request: MergeScanResultsRequest with scanned documents

    Returns:
        InternalDocsConfig: Updated configuration
    """
    try:
        manager = await get_internal_docs_manager()
        config = await manager.get_active_config()

        if not config:
            raise HTTPException(
                status_code=404,
                detail="No active internal docs configuration found. Create one first.",
            )

        # Convert dicts to ScannedDocument objects
        scanned_docs = [ScannedDocument(**doc) for doc in request.scanned_docs]

        # Merge results
        config = await manager.merge_scan_results(config, scanned_docs)
        await manager.save_config(config, set_active=True)

        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging scan results: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to merge scan results: {str(e)}"
        )


@router.post("/documents", response_model=InternalDocsConfig)
async def add_document(
    request: dict = Body(
        ..., description="Request body with document and optionally version"
    ),
    user: UserContext = Depends(require_roles(["admin"])),
):
    """
    Add a single document to the configuration.

    Args:
        request: Dict with 'document' (ScannedDocument) and optionally 'version' (str)

    Returns:
        InternalDocsConfig: Updated configuration
    """
    try:
        doc_dict = request.get("document", {})
        version = request.get("version")  # Optional: specify version to update

        doc = ScannedDocument(**doc_dict)

        manager = await get_internal_docs_manager()

        if version:
            config = await manager.get_config_by_version(version)
        else:
            config = await manager.get_active_config()

        if not config:
            raise HTTPException(
                status_code=404,
                detail="No internal docs configuration found. Create one first.",
            )

        config = await manager.add_document(config, doc)
        await manager.save_config(config, set_active=(version is None))

        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@router.delete("/documents/{doc_url:path}", response_model=InternalDocsConfig)
async def remove_document(
    doc_url: str, user: UserContext = Depends(require_roles(["admin"]))
):
    """
    Remove a document from the configuration by URL.

    Args:
        doc_url: URL of document to remove

    Returns:
        InternalDocsConfig: Updated configuration
    """
    try:
        manager = await get_internal_docs_manager()
        config = await manager.get_active_config()

        if not config:
            raise HTTPException(
                status_code=404, detail="No active internal docs configuration found"
            )

        config = await manager.remove_document(config, doc_url)
        await manager.save_config(config, set_active=True)

        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to remove document: {str(e)}"
        )


@router.get("/documents", response_model=List[ScannedDocumentDB])
async def list_scanned_documents(
    active_only: bool = True, user: UserContext = Depends(get_current_user)
):
    """
    List all scanned documents from database.

    Args:
        active_only: Only return active documents (default: True)

    Returns:
        List of ScannedDocumentDB
    """
    try:
        db = get_scanned_document_db()
        if active_only:
            documents = db.get_all_active_documents()
        else:
            # For now, we only have active documents method
            documents = db.get_all_active_documents()
        return documents
    except Exception as e:
        logger.error(f"Error listing scanned documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list scanned documents: {str(e)}"
        )


@router.get("/documents/{url:path}", response_model=ScannedDocumentDB)
async def get_scanned_document(url: str):
    """
    Get a scanned document by URL.

    Args:
        url: Document URL

    Returns:
        ScannedDocumentDB
    """
    try:
        db = get_scanned_document_db()
        document = db.get_document_by_url(url)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found: {url}")
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scanned document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get scanned document: {str(e)}"
        )


@router.get("/documents/search/keywords", response_model=List[ScannedDocumentDB])
async def search_documents_by_keywords(
    keywords: str, limit: int = 50, user: UserContext = Depends(get_current_user)
):
    """
    Search documents by keywords.

    Args:
        keywords: Comma-separated keywords
        limit: Maximum number of results

    Returns:
        List of matching ScannedDocumentDB
    """
    try:
        keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        if not keyword_list:
            raise HTTPException(
                status_code=400, detail="At least one keyword is required"
            )

        db = get_scanned_document_db()
        documents = db.search_by_keywords(keyword_list, limit=limit)
        return documents
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to search documents: {str(e)}"
        )


@router.post("/documents/search/filters", response_model=List[ScannedDocumentDB])
async def search_documents_with_filters(
    filters: Dict[str, Any] = Body(...), limit: int = 50
):
    """
    Search documents with multiple filters.

    Args:
        filters: Dictionary with filter criteria (category, content_type, date_from, date_to, word_count_min, word_count_max, has_internal_links, keywords)
        limit: Maximum number of results

    Returns:
        List of matching ScannedDocumentDB
    """
    try:
        db = get_scanned_document_db()
        documents = db.search_with_filters(filters, limit=limit)
        return documents
    except Exception as e:
        logger.error(f"Error in filtered search: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to search with filters: {str(e)}"
        )


@router.get("/documents/category/{category}", response_model=List[ScannedDocumentDB])
async def get_documents_by_category(
    category: str, user: UserContext = Depends(get_current_user)
):
    """
    Get documents by category.

    Args:
        category: Category name

    Returns:
        List of ScannedDocumentDB in the category
    """
    try:
        db = get_scanned_document_db()
        documents = db.get_documents_by_category(category)
        return documents
    except Exception as e:
        logger.error(f"Error getting documents by category: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get documents by category: {str(e)}"
        )


@router.get("/documents/patterns/anchor-text", response_model=List[str])
async def get_anchor_text_patterns(
    user: UserContext = Depends(get_current_user),
):
    """
    Get all unique anchor text patterns from scanned documents.

    Returns:
        List of anchor text patterns
    """
    try:
        db = get_scanned_document_db()
        patterns = db.get_anchor_text_patterns()
        return patterns
    except Exception as e:
        logger.error(f"Error getting anchor text patterns: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get anchor text patterns: {str(e)}"
        )


@router.get("/documents/pages/commonly-referenced", response_model=List[str])
async def get_commonly_referenced_pages(min_links: int = 2):
    """
    Get pages that are commonly referenced across documents.

    Args:
        min_links: Minimum number of documents that must link to a page

    Returns:
        List of commonly referenced URLs
    """
    try:
        db = get_scanned_document_db()
        pages = db.get_commonly_referenced_pages(min_links=min_links)
        return pages
    except Exception as e:
        logger.error(f"Error getting commonly referenced pages: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get commonly referenced pages: {str(e)}"
        )


@router.get("/documents/stats")
async def get_database_stats(user: UserContext = Depends(get_current_user)):
    """
    Get statistics about the scanned documents database.

    Returns:
        dict: Statistics including total documents, active documents, categories, etc.
    """
    try:
        db = get_scanned_document_db()
        all_docs = db.get_all_active_documents()

        # Calculate statistics
        total_docs = len(all_docs)
        total_words = sum(doc.metadata.word_count or 0 for doc in all_docs)
        total_links = sum(doc.metadata.outbound_link_count for doc in all_docs)

        # Get unique categories
        categories_set = set()
        content_types = {}
        for doc in all_docs:
            categories_set.update(doc.metadata.categories or [])
            content_type = doc.metadata.content_type or "unknown"
            content_types[content_type] = content_types.get(content_type, 0) + 1

        # Get anchor patterns count
        anchor_patterns = db.get_anchor_text_patterns()

        return {
            "total_documents": total_docs,
            "total_words": total_words,
            "total_internal_links": total_links,
            "unique_categories": len(categories_set),
            "categories": sorted(list(categories_set)),
            "content_types": content_types,
            "unique_anchor_patterns": len(anchor_patterns),
            "database_path": db.db_path,
        }
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get database statistics: {str(e)}"
        )


@router.post("/documents/{url:path}/rescan")
async def rescan_document(url: str, user: UserContext = Depends(get_current_user)):
    """
    Re-scan a specific document to update its metadata.

    Args:
        url: Document URL to re-scan

    Returns:
        ScannedDocumentDB: Updated document
    """
    try:
        scanner = await get_internal_docs_scanner()
        # Scan the document (will automatically save to database)
        scanned_doc = await scanner._scan_single_url(url, save_to_db=True)

        if not scanned_doc:
            raise HTTPException(
                status_code=404, detail=f"Failed to scan document: {url}"
            )

        # Get the updated document from database
        db = get_scanned_document_db()
        updated_doc = db.get_document_by_url(url)

        if not updated_doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found in database after scanning: {url}",
            )

        return updated_doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error re-scanning document {url}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to re-scan document: {str(e)}"
        )


@router.get("/documents/search/fulltext", response_model=List[ScannedDocumentDB])
async def full_text_search_documents(
    q: str, limit: int = 50, user: UserContext = Depends(get_current_user)
):
    """
    Full-text search documents.

    Args:
        q: Search query
        limit: Maximum number of results

    Returns:
        List of matching ScannedDocumentDB
    """
    try:
        db = get_scanned_document_db()
        documents = db.full_text_search(q, limit=limit)
        return documents
    except Exception as e:
        logger.error(f"Error in full-text search: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to perform full-text search: {str(e)}"
        )


@router.get("/documents/{url:path}/related", response_model=List[ScannedDocumentDB])
async def get_related_documents(
    url: str, user: UserContext = Depends(get_current_user)
):
    """
    Get related documents for a specific document.

    Args:
        url: Document URL

    Returns:
        List of related ScannedDocumentDB (top 10)
    """
    try:
        db = get_scanned_document_db()
        document = db.get_document_by_url(url)

        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found: {url}")

        # Get related documents (verify they still exist and are active)
        related_urls = document.related_documents or []
        related_docs = []
        for related_url in related_urls[:10]:
            related_doc = db.get_document_by_url(related_url)
            if related_doc and related_doc.is_active:
                related_docs.append(related_doc)

        # If we have fewer than expected, try to recalculate relationships
        if len(related_docs) < len(related_urls[:10]):
            # Recalculate relationships to ensure we have fresh data
            document = db.update_relationships(document)
            db.save_document(document)
            # Get updated related documents
            related_urls = document.related_documents or []
            related_docs = []
            for related_url in related_urls[:10]:
                related_doc = db.get_document_by_url(related_url)
                if related_doc and related_doc.is_active:
                    related_docs.append(related_doc)

        return related_docs
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting related documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get related documents: {str(e)}"
        )


@router.post("/documents/bulk/rescan")
async def bulk_rescan_documents(request: BulkOperationRequest):
    """
    Bulk re-scan documents (runs as background ARQ jobs).

    Args:
        request: BulkOperationRequest with list of URLs

    Returns:
        dict: Job information
    """
    try:
        from marketing_project.services.job_manager import get_job_manager

        job_manager = get_job_manager()  # Synchronous, not async
        job_id = f"bulk_rescan_{int(datetime.now(timezone.utc).timestamp() * 1000)}"

        # Create job first
        job = await job_manager.create_job(
            job_type="bulk_rescan",
            content_id="internal_docs",
            metadata={"urls": request.urls, "urls_count": len(request.urls)},
            job_id=job_id,
            user_id=user.user_id,
            user_context=user,
        )

        # Submit to ARQ
        arq_job_id = await job_manager.submit_to_arq(
            job_id,  # positional: job_id for JobManager tracking
            "bulk_rescan_documents_job",  # positional: function_name
            request.urls,  # positional: goes to *args, passed to worker as urls
            job_id,  # positional: goes to *args, passed to worker as job_id
        )

        return {
            "message": f"Bulk re-scan started for {len(request.urls)} documents",
            "job_id": job_id,
            "arq_job_id": arq_job_id,
            "urls_count": len(request.urls),
        }
    except Exception as e:
        logger.error(f"Error starting bulk re-scan: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start bulk re-scan: {str(e)}"
        )


@router.post("/documents/bulk/delete")
async def bulk_delete_documents(
    request: BulkOperationRequest, user: UserContext = Depends(require_roles(["admin"]))
):
    """
    Bulk delete documents (hard delete).

    Args:
        request: BulkOperationRequest with list of URLs

    Returns:
        dict: Deletion results
    """
    try:
        db = get_scanned_document_db()
        deleted_count = db.bulk_delete_documents(request.urls)

        return {
            "message": f"Deleted {deleted_count} document(s)",
            "deleted_count": deleted_count,
            "requested_count": len(request.urls),
        }
    except Exception as e:
        logger.error(f"Error in bulk delete: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete documents: {str(e)}"
        )


@router.post("/documents/bulk/update-categories")
async def bulk_update_categories(
    request: BulkCategoryUpdateRequest,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """
    Bulk update categories for documents (adds to existing, doesn't replace).

    Args:
        request: BulkCategoryUpdateRequest with URLs and categories

    Returns:
        dict: Update results
    """
    try:
        db = get_scanned_document_db()
        updated_count = db.bulk_update_categories(request.urls, request.categories)

        return {
            "message": f"Updated categories for {updated_count} document(s)",
            "updated_count": updated_count,
            "requested_count": len(request.urls),
        }
    except Exception as e:
        logger.error(f"Error in bulk category update: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update categories: {str(e)}"
        )


@router.post("/documents/bulk/upload")
async def bulk_upload_documents(
    file: UploadFile = File(...),
    format: str = Form("json"),
    user: UserContext = Depends(require_roles(["admin"])),
):
    """
    Bulk upload documents from JSON or CSV file.

    Args:
        file: Uploaded file (JSON or CSV)
        format: File format ('json' or 'csv')

    Returns:
        dict: Upload results with success/failure counts
    """
    try:
        import csv
        import io

        from marketing_project.models.scanned_document_db import (
            ScannedDocumentDB,
            ScannedDocumentMetadata,
        )

        content = await file.read()
        success_count = 0
        failed_count = 0
        errors = []

        db = get_scanned_document_db()
        documents_to_save = []

        if format == "json":
            try:
                data = json.loads(content.decode("utf-8"))
                if not isinstance(data, list):
                    raise ValueError("JSON must be an array of documents")

                for idx, item in enumerate(data):
                    try:
                        # Parse document
                        metadata_dict = item.get("metadata", {})
                        metadata = ScannedDocumentMetadata(**metadata_dict)

                        doc = ScannedDocumentDB(
                            title=item.get("title", "Untitled"),
                            url=item["url"],
                            scanned_at=datetime.now(timezone.utc),
                            metadata=metadata,
                        )

                        # Check if content_text is missing - trigger auto-scan
                        if not metadata.content_text:
                            logger.info(
                                f"Content missing for {doc.url}, triggering auto-scan"
                            )
                            scanner = await get_internal_docs_scanner()
                            # Scan and save to get full metadata
                            scanned_doc = await scanner._scan_single_url(
                                doc.url, save_to_db=True
                            )
                            if scanned_doc:
                                # Get the scanned document with full metadata from database
                                scanned_db_doc = db.get_document_by_url(doc.url)
                                if scanned_db_doc:
                                    # Use the scanned version with full metadata
                                    doc = scanned_db_doc
                                else:
                                    # If scan succeeded but not in DB yet, use what we have
                                    documents_to_save.append(doc)
                            else:
                                # Scan failed, save what we have
                                documents_to_save.append(doc)
                        else:
                            documents_to_save.append(doc)
                    except Exception as e:
                        failed_count += 1
                        errors.append(f"Row {idx + 1}: {str(e)}")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

        elif format == "csv":
            try:
                csv_content = content.decode("utf-8")
                reader = csv.DictReader(io.StringIO(csv_content))

                for idx, row in enumerate(reader):
                    try:
                        # Parse CSV row
                        categories = (
                            [
                                c.strip()
                                for c in row.get("categories", "").split(",")
                                if c.strip()
                            ]
                            if row.get("categories")
                            else []
                        )
                        keywords = (
                            [
                                k.strip()
                                for k in row.get("keywords", "").split(",")
                                if k.strip()
                            ]
                            if row.get("keywords")
                            else []
                        )
                        topics = (
                            [
                                t.strip()
                                for t in row.get("topics", "").split(",")
                                if t.strip()
                            ]
                            if row.get("topics")
                            else []
                        )

                        metadata = ScannedDocumentMetadata(
                            content_text=row.get("content_text", ""),
                            content_summary=row.get("content_summary"),
                            word_count=(
                                int(row.get("word_count", 0))
                                if row.get("word_count")
                                else None
                            ),
                            categories=categories,
                            extracted_keywords=keywords,
                            topics=topics,
                            content_type=row.get("content_type"),
                        )

                        doc = ScannedDocumentDB(
                            title=row.get("title", "Untitled"),
                            url=row["url"],
                            scanned_at=datetime.now(timezone.utc),
                            metadata=metadata,
                        )

                        # Check if content_text is missing - trigger auto-scan
                        if not metadata.content_text:
                            logger.info(
                                f"Content missing for {doc.url}, triggering auto-scan"
                            )
                            scanner = await get_internal_docs_scanner()
                            # Scan and save to get full metadata
                            scanned_doc = await scanner._scan_single_url(
                                doc.url, save_to_db=True
                            )
                            if scanned_doc:
                                # Get the scanned document with full metadata from database
                                scanned_db_doc = db.get_document_by_url(doc.url)
                                if scanned_db_doc:
                                    # Use the scanned version with full metadata
                                    doc = scanned_db_doc
                                else:
                                    # If scan succeeded but not in DB yet, use what we have
                                    documents_to_save.append(doc)
                            else:
                                # Scan failed, save what we have
                                documents_to_save.append(doc)
                        else:
                            documents_to_save.append(doc)
                    except Exception as e:
                        failed_count += 1
                        errors.append(
                            f"Row {idx + 2}: {str(e)}"
                        )  # +2 because of header
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

        # Save all documents
        for doc in documents_to_save:
            try:
                # Calculate relationships
                doc = db.update_relationships(doc)
                db.save_document(doc)
                success_count += 1
            except Exception as e:
                failed_count += 1
                errors.append(f"Failed to save {doc.url}: {str(e)}")

        return {
            "message": f"Upload completed: {success_count} succeeded, {failed_count} failed",
            "success": success_count,
            "failed": failed_count,
            "errors": errors[:20],  # Limit to first 20 errors
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk upload: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload documents: {str(e)}"
        )
