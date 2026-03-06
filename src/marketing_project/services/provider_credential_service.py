"""
Provider Credential Service.

Loads and stores encrypted LLM provider credentials from the database,
and constructs an LLMClient for each provider.
"""

import json
import logging
from typing import List, Optional

from sqlalchemy import select

from marketing_project.models.db_models import ProviderCredentialsModel
from marketing_project.models.provider_models import (
    SUPPORTED_PROVIDERS,
    ProviderCredentialsRequest,
    ProviderCredentialsResponse,
)
from marketing_project.services.database import get_database_manager
from marketing_project.services.function_pipeline.litellm_client import LLMClient

logger = logging.getLogger(__name__)


class ProviderCredentialService:
    """DB-backed factory for LLMClient instances."""

    async def upsert(self, provider: str, req: ProviderCredentialsRequest) -> None:
        """Create or update credentials for a provider."""
        db = get_database_manager()
        async with db.get_session() as session:
            result = await session.execute(
                select(ProviderCredentialsModel).where(
                    ProviderCredentialsModel.provider == provider
                )
            )
            record = result.scalar_one_or_none()

            if record is None:
                record = ProviderCredentialsModel(provider=provider)
                session.add(record)

            record.is_enabled = req.is_enabled
            # Only overwrite credential fields when the caller explicitly provides them;
            # a None value means "keep existing" so we never accidentally wipe stored creds.
            if req.api_key is not None:
                record.api_key = req.api_key
            if req.project_id is not None:
                record.project_id = req.project_id
            if req.region is not None:
                record.region = req.region
            if req.vertex_credentials is not None:
                record.vertex_credentials_json = json.dumps(req.vertex_credentials)
            if req.aws_bedrock_credentials is not None:
                record.aws_bedrock_credentials_json = json.dumps(
                    req.aws_bedrock_credentials
                )

    async def get(self, provider: str) -> Optional[ProviderCredentialsModel]:
        """Fetch a provider record from the DB."""
        db = get_database_manager()
        try:
            async with db.get_session() as session:
                result = await session.execute(
                    select(ProviderCredentialsModel).where(
                        ProviderCredentialsModel.provider == provider
                    )
                )
                return result.scalar_one_or_none()
        except Exception as exc:
            # cryptography.fernet.InvalidToken is raised when ENCRYPTION_KEY has been
            # rotated and existing DB values can no longer be decrypted.
            exc_name = type(exc).__name__
            if "InvalidToken" in exc_name:
                logger.error(
                    "Decryption failed for provider '%s' — ENCRYPTION_KEY may have been "
                    "rotated. Re-save credentials via the Providers settings page. Error: %s",
                    provider,
                    exc,
                )
            else:
                logger.warning(
                    "Failed to fetch provider '%s' credentials: %s", provider, exc
                )
            return None

    async def delete(self, provider: str) -> bool:
        """Delete credentials for a provider. Returns True if deleted."""
        db = get_database_manager()
        async with db.get_session() as session:
            result = await session.execute(
                select(ProviderCredentialsModel).where(
                    ProviderCredentialsModel.provider == provider
                )
            )
            record = result.scalar_one_or_none()
            if record is None:
                return False
            session.delete(record)
            return True

    async def list_all(self) -> List[ProviderCredentialsResponse]:
        """Return a ProviderCredentialsResponse for every supported provider.

        Fetches each provider individually so a decryption failure on one
        (e.g. after ENCRYPTION_KEY rotation) does not blank out all others.
        """
        records: dict = {}
        for provider in SUPPORTED_PROVIDERS:
            rec = await self.get(provider)  # already handles InvalidToken per-provider
            if rec is not None:
                records[provider] = rec

        responses = []
        for provider in SUPPORTED_PROVIDERS:
            rec = records.get(provider)
            if rec:
                responses.append(
                    ProviderCredentialsResponse(
                        provider=provider,
                        is_enabled=rec.is_enabled,
                        has_api_key=bool(rec.api_key),
                        project_id=rec.project_id,
                        region=rec.region,
                        has_vertex_credentials=bool(rec.vertex_credentials_json),
                        has_aws_credentials=bool(rec.aws_bedrock_credentials_json),
                        created_at=rec.created_at,
                        updated_at=rec.updated_at,
                    )
                )
            else:
                responses.append(
                    ProviderCredentialsResponse(
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
                )
        return responses

    async def get_llm_client(self, provider: str) -> LLMClient:
        """
        Build an LLMClient with decrypted credentials loaded from the DB.

        Falls back to a credential-less client (uses env vars) if no DB record exists.
        """
        record = await self.get(provider)
        if record is None:
            logger.debug(
                "No DB credentials for provider '%s' — using env-var fallback", provider
            )
            return LLMClient(provider=provider)

        if not record.is_enabled:
            logger.warning(
                "Provider '%s' is disabled — ignoring stored credentials and using env-var fallback",
                provider,
            )
            return LLMClient(provider=provider)

        try:
            vertex_creds = (
                json.loads(record.vertex_credentials_json)
                if record.vertex_credentials_json
                else None
            )
            aws_creds = (
                json.loads(record.aws_bedrock_credentials_json)
                if record.aws_bedrock_credentials_json
                else None
            )
            return LLMClient(
                provider=provider,
                api_key=record.api_key,
                project_id=record.project_id,
                region=record.region,
                vertex_credentials=vertex_creds,
                aws_bedrock_credentials=aws_creds,
            )
        except Exception as exc:
            exc_name = type(exc).__name__
            if "InvalidToken" in exc_name:
                logger.error(
                    "Decryption failed building LLMClient for provider '%s' — "
                    "ENCRYPTION_KEY may have been rotated. Re-save credentials. Error: %s",
                    provider,
                    exc,
                )
            else:
                logger.error(
                    "Failed to build LLMClient for provider '%s': %s", provider, exc
                )
            return LLMClient(provider=provider)


# Singleton
_service: Optional[ProviderCredentialService] = None


def get_provider_credential_service() -> ProviderCredentialService:
    global _service
    if _service is None:
        _service = ProviderCredentialService()
    return _service
