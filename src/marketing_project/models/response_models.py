"""
Response models for API endpoints.

This module defines Pydantic models for API responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """Base API response model."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class ErrorResponse(APIResponse):
    """Error response model."""
    success: bool = Field(default=False, description="Always false for errors")
    error_code: Optional[str] = Field(None, description="Error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ContentAnalysisResponse(APIResponse):
    """Response model for content analysis."""
    data: Dict[str, Any] = Field(..., description="Analysis results")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PipelineResponse(APIResponse):
    """Response model for pipeline execution."""
    data: Dict[str, Any] = Field(..., description="Pipeline results")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    checks: Dict[str, bool] = Field(..., description="Health check results")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")


class ContentSourceResponse(BaseModel):
    """Content source response model."""
    name: str = Field(..., description="Source name")
    type: str = Field(..., description="Source type")
    status: str = Field(..., description="Source status")
    healthy: bool = Field(..., description="Whether source is healthy")
    last_check: Optional[datetime] = Field(None, description="Last health check time")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Source metadata")


class ContentSourceListResponse(APIResponse):
    """Response model for listing content sources."""
    sources: List[ContentSourceResponse] = Field(..., description="List of content sources")


class ContentFetchResponse(APIResponse):
    """Response model for content fetching."""
    source_name: str = Field(..., description="Source name")
    success: bool = Field(..., description="Whether fetch was successful")
    total_count: int = Field(..., description="Total items fetched")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Fetched items")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class RateLimitResponse(BaseModel):
    """Rate limit response model."""
    limit: int = Field(..., description="Rate limit per window")
    remaining: int = Field(..., description="Remaining requests in current window")
    reset_time: datetime = Field(..., description="When the rate limit resets")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")
