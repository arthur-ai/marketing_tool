"""
API models package for FastAPI endpoints.

This package contains all Pydantic models organized by category for better maintainability.
"""

# Import all models for easy access
from .content_models import (
    ContentType,
    ContentContext,
    BlogPostContext,
    TranscriptContext,
    ReleaseNotesContext
)

from .request_models import (
    AnalyzeRequest,
    PipelineRequest,
    WebhookRequest
)

from .response_models import (
    APIResponse,
    ErrorResponse,
    ContentAnalysisResponse,
    PipelineResponse,
    HealthResponse,
    ContentSourceResponse,
    ContentSourceListResponse,
    ContentFetchResponse,
    RateLimitResponse
)

from .auth_models import (
    APIKeyAuth,
    TokenResponse
)

from .validation import (
    validate_content_length,
    validate_api_key_format
)

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
    
    # Auth models
    "APIKeyAuth",
    "TokenResponse",
    
    # Validation helpers
    "validate_content_length",
    "validate_api_key_format"
]
