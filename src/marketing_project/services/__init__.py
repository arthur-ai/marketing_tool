"""
Services package for Marketing Project.

This package contains service modules that provide business logic and orchestration
functionality for the marketing project system.

Key Services:
- Pipeline services: FunctionPipeline, StepRetryService
- Content sources: File, API, Database, WebScraping sources
- Management services: JobManager, ApprovalManager, StepResultManager
- Utilities: OCR, Analytics, Content source configuration

Note: Services are typically imported directly from their modules rather than
from this package for better clarity and to avoid circular dependencies.
"""

__all__ = [
    "job_manager",
    "approval_manager",
    "step_result_manager",
    "content_source_factory",
    "content_source_config_loader",
    "file_source",
    "api_source",
    "database_source",
    "web_scraping_source",
    "ocr",
]
