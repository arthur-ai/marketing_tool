"""
Authentication models for API endpoints.

This module defines Pydantic models for authentication and authorization.
"""

from typing import Optional

from pydantic import BaseModel, Field


class APIKeyAuth(BaseModel):
    """API key authentication model."""

    api_key: str = Field(..., description="API key")


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str = Field(..., description="Access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: Optional[int] = Field(
        None, description="Token expiration time in seconds"
    )
