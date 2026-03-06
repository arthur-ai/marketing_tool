"""
Pydantic schemas for LLM provider credential management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

SUPPORTED_PROVIDERS: List[str] = [
    "openai",
    "anthropic",
    "gemini",
    "vertex_ai",
    "bedrock",
]


class ProviderCredentialsRequest(BaseModel):
    """Request body for creating or updating provider credentials."""

    is_enabled: bool = True
    api_key: Optional[str] = None  # OpenAI / Anthropic / Gemini
    project_id: Optional[str] = None  # Vertex AI project ID
    region: Optional[str] = None  # Vertex AI location / Bedrock region
    vertex_credentials: Optional[Dict[str, Any]] = None  # GCP service account JSON
    aws_bedrock_credentials: Optional[Dict[str, Any]] = None  # AWS credentials dict


class ProviderCredentialsResponse(BaseModel):
    """Response for a single provider (never returns raw credential values)."""

    provider: str
    is_enabled: bool
    has_api_key: bool
    project_id: Optional[str]
    region: Optional[str]
    has_vertex_credentials: bool
    has_aws_credentials: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    model_config = {"from_attributes": True}


class ProviderListResponse(BaseModel):
    """Response for listing all providers."""

    providers: List[ProviderCredentialsResponse]
