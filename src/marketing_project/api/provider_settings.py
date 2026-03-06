"""
API endpoints for LLM provider credential management.

All endpoints require the 'admin' role.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from marketing_project.middleware.rbac import require_roles
from marketing_project.models.provider_models import (
    SUPPORTED_PROVIDERS,
    ProviderCredentialsRequest,
    ProviderCredentialsResponse,
    ProviderListResponse,
)
from marketing_project.models.user_context import UserContext
from marketing_project.services.provider_credential_service import (
    get_provider_credential_service,
)

logger = logging.getLogger("marketing_project.api.provider_settings")

router = APIRouter(prefix="/settings/providers", tags=["Provider Settings"])


class ProviderTestResponse(BaseModel):
    provider: str
    success: bool
    message: str


class ProviderModelsResponse(BaseModel):
    provider: str
    models: List[str]


def _validate_provider(provider: str) -> None:
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider '{provider}'. Supported: {SUPPORTED_PROVIDERS}",
        )


@router.get("", response_model=ProviderListResponse)
async def list_providers(
    user: UserContext = Depends(require_roles(["admin"])),
):
    """List all supported providers with their credential status."""
    svc = get_provider_credential_service()
    providers = await svc.list_all()
    return ProviderListResponse(providers=providers)


@router.get("/{provider}", response_model=ProviderCredentialsResponse)
async def get_provider(
    provider: str,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """Get credential status for a single provider (never returns raw credential values)."""
    _validate_provider(provider)
    svc = get_provider_credential_service()
    record = await svc.get(provider)
    if record is None:
        # Provider is supported but not yet configured — return an empty status
        return ProviderCredentialsResponse(
            provider=provider,
            is_enabled=False,
            has_api_key=False,
            project_id=None,
            region=None,
            has_vertex_credentials=False,
            has_aws_credentials=False,
            created_at=None,
            updated_at=None,
        )
    return ProviderCredentialsResponse(
        provider=provider,
        is_enabled=record.is_enabled,
        has_api_key=bool(record.api_key),
        project_id=record.project_id,
        region=record.region,
        has_vertex_credentials=bool(record.vertex_credentials_json),
        has_aws_credentials=bool(record.aws_bedrock_credentials_json),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.post("/{provider}/test", response_model=ProviderTestResponse)
async def test_provider(
    provider: str,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """Test whether the stored credentials for a provider are valid."""
    _validate_provider(provider)
    svc = get_provider_credential_service()
    try:
        llm_client = await svc.get_llm_client(provider)
        # Send a minimal, cheap prompt to verify the credentials work
        from marketing_project.services.function_pipeline.litellm_client import (
            build_litellm_model,
        )

        test_model_map = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "gemini": "gemini-1.5-flash",
            "vertex_ai": "gemini-1.5-flash",
            "bedrock": "anthropic.claude-3-haiku-20240307-v1:0",
        }
        test_model_name = test_model_map.get(provider)
        litellm_model = build_litellm_model(test_model_name, provider)
        response = await llm_client.acompletion(
            model=litellm_model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=5,
            temperature=0,
        )
        reply = response.choices[0].message.content or ""
        return ProviderTestResponse(
            provider=provider,
            success=True,
            message=f"Connection successful. Model replied: {reply.strip()[:50]}",
        )
    except Exception as exc:
        logger.warning("Provider test failed for '%s': %s", provider, exc)
        return ProviderTestResponse(
            provider=provider,
            success=False,
            message=str(exc)[:300],
        )


@router.get("/{provider}/models", response_model=ProviderModelsResponse)
async def list_provider_models(
    provider: str,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """List models available for a provider from LiteLLM's cost map."""
    _validate_provider(provider)
    svc = get_provider_credential_service()
    llm_client = await svc.get_llm_client(provider)
    try:
        models = llm_client.get_available_models()
    except Exception as exc:
        logger.warning("Failed to list models for provider '%s': %s", provider, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not retrieve models for '{provider}': {exc}",
        ) from exc
    return ProviderModelsResponse(provider=provider, models=models)


@router.put("/{provider}", response_model=ProviderCredentialsResponse)
async def upsert_provider(
    provider: str,
    req: ProviderCredentialsRequest,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """Create or update credentials for a provider."""
    _validate_provider(provider)
    svc = get_provider_credential_service()
    await svc.upsert(provider, req)
    # Re-fetch to return current state
    record = await svc.get(provider)
    if record is None:
        raise HTTPException(status_code=500, detail="Failed to persist credentials")
    return ProviderCredentialsResponse(
        provider=provider,
        is_enabled=record.is_enabled,
        has_api_key=bool(record.api_key),
        project_id=record.project_id,
        region=record.region,
        has_vertex_credentials=bool(record.vertex_credentials_json),
        has_aws_credentials=bool(record.aws_bedrock_credentials_json),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.delete("/{provider}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_provider(
    provider: str,
    user: UserContext = Depends(require_roles(["admin"])),
):
    """Remove stored credentials for a provider."""
    _validate_provider(provider)
    svc = get_provider_credential_service()
    deleted = await svc.delete(provider)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No credentials found for provider '{provider}'",
        )
