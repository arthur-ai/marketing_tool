"""
API models package for FastAPI endpoints.

This package contains all Pydantic models organized by category for better maintainability.
"""

from .approval_models import (
    ApprovalDecisionRequest,
    ApprovalListItem,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalSettings,
    ApprovalStats,
    PendingApprovalsResponse,
)
from .auth_models import APIKeyAuth, TokenResponse

# Import all models for easy access
from .content_models import (
    BlogPostContext,
    ContentContext,
    ContentType,
    ReleaseNotesContext,
    TranscriptContext,
)
from .processor_models import (
    BlogProcessorRequest,
    BlogProcessorResponse,
    ProcessorResponse,
    ReleaseNotesProcessorRequest,
    ReleaseNotesProcessorResponse,
    TranscriptProcessorRequest,
    TranscriptProcessorResponse,
)
from .request_models import AnalyzeRequest, PipelineRequest, WebhookRequest
from .response_models import (
    APIResponse,
    ContentAnalysisResponse,
    ContentFetchResponse,
    ContentSourceListResponse,
    ContentSourceResponse,
    ErrorResponse,
    HealthResponse,
    PipelineResponse,
    RateLimitResponse,
)
from .validation import validate_api_key_format, validate_content_length

# Re-export everything for backward compatibility
__all__ = [
    # Content models
    "ContentType",
    "ContentContext",
    "BlogPostContext",
    "TranscriptContext",
    "ReleaseNotesContext",
    # Request models
    "AnalyzeRequest",
    "PipelineRequest",
    "WebhookRequest",
    # Response models
    "APIResponse",
    "ErrorResponse",
    "ContentAnalysisResponse",
    "PipelineResponse",
    "HealthResponse",
    "ContentSourceResponse",
    "ContentSourceListResponse",
    "ContentFetchResponse",
    "RateLimitResponse",
    # Processor models
    "BlogProcessorRequest",
    "BlogProcessorResponse",
    "ReleaseNotesProcessorRequest",
    "ReleaseNotesProcessorResponse",
    "TranscriptProcessorRequest",
    "TranscriptProcessorResponse",
    "ProcessorResponse",
    # Approval models
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalListItem",
    "PendingApprovalsResponse",
    "ApprovalDecisionRequest",
    "ApprovalSettings",
    "ApprovalStats",
    # Auth models
    "APIKeyAuth",
    "TokenResponse",
    # Validation helpers
    "validate_content_length",
    "validate_api_key_format",
]
