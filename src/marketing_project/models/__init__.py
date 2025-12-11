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
from .step_models import (
    StepExecutionRequest,
    StepExecutionResponse,
    StepInfo,
    StepListResponse,
    StepRequirementsResponse,
)
from .validation import validate_api_key_format, validate_content_length

# Rebuild models to resolve forward references after all imports
# This ensures Pydantic v2 can properly validate models with forward references
try:
    # Import PipelineConfig to ensure it's available for model rebuild
    from .pipeline_steps import PipelineConfig  # noqa: F401

    StepExecutionRequest.model_rebuild()
except (NameError, AttributeError, ImportError):
    # Model not imported, PipelineConfig not available, or already rebuilt
    pass

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
    # Step execution models
    "StepExecutionRequest",
    "StepExecutionResponse",
    "StepInfo",
    "StepListResponse",
    "StepRequirementsResponse",
    # Validation helpers
    "validate_content_length",
    "validate_api_key_format",
]
