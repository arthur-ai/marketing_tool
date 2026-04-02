"""
Unit tests for ProviderCredentialService and LLMClient.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.provider_models import ProviderCredentialsRequest
from marketing_project.services.function_pipeline.litellm_client import (
    LLMClient,
    build_litellm_model,
    normalize_provider,
)

# ──────────────────────────────────────────────
# normalize_provider / build_litellm_model
# ──────────────────────────────────────────────


class TestNormalizeProvider:
    def test_none_defaults_to_openai(self):
        assert normalize_provider(None) == "openai"

    def test_openai_passthrough(self):
        assert normalize_provider("openai") == "openai"

    def test_aliases(self):
        assert normalize_provider("vertex") == "vertex_ai"
        assert normalize_provider("vertexai") == "vertex_ai"
        assert normalize_provider("google") == "gemini"
        assert normalize_provider("aws_bedrock") == "bedrock"

    def test_case_insensitive(self):
        assert normalize_provider("ANTHROPIC") == "anthropic"
        assert normalize_provider("OpenAI") == "openai"

    def test_unknown_provider_falls_back_to_openai(self):
        # An unrecognised provider string should not be passed through blindly
        result = normalize_provider("some_mystery_provider")
        assert result == "openai"


class TestBuildLitellmModelValidation:
    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            build_litellm_model("", "openai")

    def test_whitespace_only_model_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            build_litellm_model("   ", "anthropic")


class TestBuildLitellmModel:
    def test_openai_no_prefix(self):
        assert build_litellm_model("gpt-4o", "openai") == "gpt-4o"

    def test_anthropic_prefixed(self):
        assert (
            build_litellm_model("claude-3-5-sonnet-latest", "anthropic")
            == "anthropic/claude-3-5-sonnet-latest"
        )

    def test_vertex_ai_prefixed(self):
        assert (
            build_litellm_model("gemini-1.5-pro", "vertex_ai")
            == "vertex_ai/gemini-1.5-pro"
        )

    def test_bedrock_prefixed(self):
        result = build_litellm_model("anthropic.claude-3-haiku", "bedrock")
        assert result == "bedrock/anthropic.claude-3-haiku"


# ──────────────────────────────────────────────
# LLMClient._add_provider_credentials
# ──────────────────────────────────────────────


class TestLLMClientCredentials:
    def test_api_key_injected(self):
        client = LLMClient(provider="openai", api_key="sk-test")
        kwargs = client._add_provider_credentials({})
        assert kwargs["api_key"] == "sk-test"

    def test_no_api_key_not_injected(self):
        client = LLMClient(provider="openai")
        kwargs = client._add_provider_credentials({})
        assert "api_key" not in kwargs

    def test_vertex_ai_credentials_injected(self):
        creds = {"type": "service_account", "project_id": "my-proj"}
        client = LLMClient(
            provider="vertex_ai",
            project_id="my-proj",
            region="us-central1",
            vertex_credentials=creds,
        )
        kwargs = client._add_provider_credentials({})
        assert kwargs["vertex_project"] == "my-proj"
        assert kwargs["vertex_location"] == "us-central1"
        assert kwargs["vertex_credentials"] == creds

    def test_bedrock_credentials_injected(self):
        aws_creds = {
            "aws_access_key_id": "AKIA...",
            "aws_secret_access_key": "secret",
        }
        client = LLMClient(
            provider="bedrock",
            region="us-east-1",
            aws_bedrock_credentials=aws_creds,
        )
        kwargs = client._add_provider_credentials({})
        assert kwargs["aws_access_key_id"] == "AKIA..."
        assert kwargs["aws_secret_access_key"] == "secret"
        assert kwargs["aws_region_name"] == "us-east-1"

    def test_bedrock_session_token_injected(self):
        aws_creds = {
            "aws_access_key_id": "AKIA...",
            "aws_secret_access_key": "secret",
            "aws_session_token": "AQoDYXdz...",
        }
        client = LLMClient(
            provider="bedrock", region="us-east-1", aws_bedrock_credentials=aws_creds
        )
        kwargs = client._add_provider_credentials({})
        assert kwargs["aws_session_token"] == "AQoDYXdz..."

    def test_bedrock_key_without_secret_raises(self):
        client = LLMClient(
            provider="bedrock",
            aws_bedrock_credentials={"aws_access_key_id": "AKIA"},
        )
        with pytest.raises(ValueError, match="must be provided together"):
            client._add_provider_credentials({})


# ──────────────────────────────────────────────
# ProviderCredentialService.upsert — partial update logic
# ──────────────────────────────────────────────


class TestListAll:
    @pytest.mark.asyncio
    async def test_partial_decryption_failure_still_returns_healthy_providers(self):
        """One provider's decryption error must not blank out all others."""
        from marketing_project.models.provider_models import SUPPORTED_PROVIDERS
        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()
        healthy_record = MagicMock()
        healthy_record.is_enabled = True
        healthy_record.api_key = "sk-good"
        healthy_record.project_id = None
        healthy_record.region = None
        healthy_record.vertex_credentials_json = None
        healthy_record.aws_bedrock_credentials_json = None
        healthy_record.created_at = None
        healthy_record.updated_at = None

        call_count = 0

        async def fake_get(provider):
            nonlocal call_count
            call_count += 1
            if provider == "openai":
                return healthy_record
            if provider == "anthropic":
                return None  # simulates InvalidToken → get() returns None
            return None

        with patch.object(svc, "get", side_effect=fake_get):
            responses = await svc.list_all()

        assert call_count == len(SUPPORTED_PROVIDERS)
        openai_resp = next(r for r in responses if r.provider == "openai")
        anthropic_resp = next(r for r in responses if r.provider == "anthropic")
        assert openai_resp.has_api_key is True
        assert (
            anthropic_resp.has_api_key is False
        )  # failed silently, shown as unconfigured


class TestProviderCredentialServiceUpsert:
    """
    Tests for the partial-update behaviour: fields that are None in the request
    must NOT overwrite existing values in the DB record.
    """

    @pytest.mark.asyncio
    async def test_upsert_preserves_existing_api_key_when_not_provided(self):
        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()

        # Simulate an existing DB record
        existing_record = MagicMock()
        existing_record.api_key = "sk-existing-key"
        existing_record.project_id = None
        existing_record.region = None
        existing_record.vertex_credentials_json = None
        existing_record.aws_bedrock_credentials_json = None

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        async def fake_execute(stmt):
            result = MagicMock()
            result.scalar_one_or_none.return_value = existing_record
            return result

        mock_session.execute = fake_execute

        mock_db = MagicMock()
        mock_db.get_session = MagicMock(return_value=mock_session)

        with patch(
            "marketing_project.services.provider_credential_service.get_database_manager",
            return_value=mock_db,
        ):
            req = ProviderCredentialsRequest(is_enabled=True)  # no api_key provided
            await svc.upsert("openai", req)

        # api_key must NOT have been overwritten
        assert existing_record.api_key == "sk-existing-key"

    @pytest.mark.asyncio
    async def test_upsert_updates_api_key_when_provided(self):
        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()

        existing_record = MagicMock()
        existing_record.api_key = "sk-old"

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        async def fake_execute(stmt):
            result = MagicMock()
            result.scalar_one_or_none.return_value = existing_record
            return result

        mock_session.execute = fake_execute
        mock_db = MagicMock()
        mock_db.get_session = MagicMock(return_value=mock_session)

        with patch(
            "marketing_project.services.provider_credential_service.get_database_manager",
            return_value=mock_db,
        ):
            req = ProviderCredentialsRequest(is_enabled=True, api_key="sk-new")
            await svc.upsert("openai", req)

        assert existing_record.api_key == "sk-new"

    @pytest.mark.asyncio
    async def test_upsert_creates_new_record_when_none(self):
        from marketing_project.models.db_models import ProviderCredentialsModel
        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()
        added_records = []

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = MagicMock(side_effect=lambda r: added_records.append(r))

        async def fake_execute(stmt):
            result = MagicMock()
            result.scalar_one_or_none.return_value = None  # no existing record
            return result

        mock_session.execute = fake_execute
        mock_db = MagicMock()
        mock_db.get_session = MagicMock(return_value=mock_session)

        with patch(
            "marketing_project.services.provider_credential_service.get_database_manager",
            return_value=mock_db,
        ):
            req = ProviderCredentialsRequest(is_enabled=True, api_key="sk-brand-new")
            await svc.upsert("anthropic", req)

        assert len(added_records) == 1
        assert added_records[0].provider == "anthropic"
        assert added_records[0].api_key == "sk-brand-new"


# ──────────────────────────────────────────────
# ProviderCredentialService.delete
# ──────────────────────────────────────────────


class TestProviderCredentialServiceGetDecryptionError:
    @pytest.mark.asyncio
    async def test_invalid_token_returns_none_and_logs_error(self, caplog):
        """InvalidToken (wrong ENCRYPTION_KEY) should return None with an actionable error log."""
        import logging

        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()

        class _FakeInvalidToken(Exception):
            pass

        _FakeInvalidToken.__name__ = "InvalidToken"

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.execute = AsyncMock(side_effect=_FakeInvalidToken("bad token"))

        mock_db = MagicMock()
        mock_db.get_session = MagicMock(return_value=mock_session)

        logger_name = "marketing_project.services.provider_credential_service"
        with patch(
            "marketing_project.services.provider_credential_service.get_database_manager",
            return_value=mock_db,
        ):
            with patch.object(
                logging.getLogger(logger_name), "error"
            ) as mock_log_error:
                result = await svc.get("openai")

        assert result is None
        assert mock_log_error.called
        logged_messages = " ".join(str(call) for call in mock_log_error.call_args_list)
        assert "ENCRYPTION_KEY" in logged_messages


class TestProviderCredentialServiceDelete:
    @pytest.mark.asyncio
    async def test_delete_existing_record_returns_true(self):
        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()
        existing_record = MagicMock()
        deleted = []

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.delete = MagicMock(side_effect=lambda r: deleted.append(r))

        async def fake_execute(stmt):
            result = MagicMock()
            result.scalar_one_or_none.return_value = existing_record
            return result

        mock_session.execute = fake_execute
        mock_db = MagicMock()
        mock_db.get_session = MagicMock(return_value=mock_session)

        with patch(
            "marketing_project.services.provider_credential_service.get_database_manager",
            return_value=mock_db,
        ):
            result = await svc.delete("openai")

        assert result is True
        assert len(deleted) == 1
        # Verify session.delete was called synchronously (not awaited)
        mock_session.delete.assert_called_once_with(existing_record)

    @pytest.mark.asyncio
    async def test_delete_missing_record_returns_false(self):
        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        async def fake_execute(stmt):
            result = MagicMock()
            result.scalar_one_or_none.return_value = None
            return result

        mock_session.execute = fake_execute
        mock_db = MagicMock()
        mock_db.get_session = MagicMock(return_value=mock_session)

        with patch(
            "marketing_project.services.provider_credential_service.get_database_manager",
            return_value=mock_db,
        ):
            result = await svc.delete("anthropic")

        assert result is False


# ──────────────────────────────────────────────
# ProviderCredentialService.get_llm_client
# ──────────────────────────────────────────────


class TestGetLLMClient:
    @pytest.mark.asyncio
    async def test_returns_client_without_creds_when_no_record(self):
        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()
        with patch.object(svc, "get", AsyncMock(return_value=None)):
            client = await svc.get_llm_client("openai")
        assert isinstance(client, LLMClient)
        assert client.api_key is None

    @pytest.mark.asyncio
    async def test_returns_client_with_creds_from_record(self):
        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()
        record = MagicMock()
        record.api_key = "sk-decrypted"
        record.project_id = None
        record.region = None
        record.vertex_credentials_json = None
        record.aws_bedrock_credentials_json = None

        with patch.object(svc, "get", AsyncMock(return_value=record)):
            client = await svc.get_llm_client("openai")
        assert isinstance(client, LLMClient)
        assert client.api_key == "sk-decrypted"

    @pytest.mark.asyncio
    async def test_decryption_error_during_client_build_falls_back(self):
        """InvalidToken raised while reading encrypted fields → env-var fallback."""
        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()

        class _FakeInvalidToken(Exception):
            pass

        _FakeInvalidToken.__name__ = "InvalidToken"

        # record.api_key access triggers decryption
        record = MagicMock()
        record.is_enabled = True
        type(record).api_key = property(
            lambda self: (_ for _ in ()).throw(_FakeInvalidToken())
        )

        with patch.object(svc, "get", AsyncMock(return_value=record)):
            client = await svc.get_llm_client("anthropic")

        assert client.api_key is None  # fell back to env-var client

    @pytest.mark.asyncio
    async def test_disabled_provider_returns_credential_less_client(self):
        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()
        record = MagicMock()
        record.is_enabled = False
        record.api_key = "sk-secret"

        with patch.object(svc, "get", AsyncMock(return_value=record)):
            client = await svc.get_llm_client("openai")

        # Credentials must NOT be injected for a disabled provider
        assert client.api_key is None

    @pytest.mark.asyncio
    async def test_parses_vertex_credentials_json(self):
        from marketing_project.services.provider_credential_service import (
            ProviderCredentialService,
        )

        svc = ProviderCredentialService()
        creds = {"type": "service_account", "project_id": "my-proj"}
        record = MagicMock()
        record.api_key = None
        record.project_id = "my-proj"
        record.region = "us-central1"
        record.vertex_credentials_json = json.dumps(creds)
        record.aws_bedrock_credentials_json = None

        with patch.object(svc, "get", AsyncMock(return_value=record)):
            client = await svc.get_llm_client("vertex_ai")
        assert client.vertex_credentials == creds
        assert client.project_id == "my-proj"


# ──────────────────────────────────────────────
# call_llm_structured — response parsing
# ──────────────────────────────────────────────


class TestInjectJsonSchema:
    """Tests for _inject_json_schema — string and list content handling."""

    def test_appends_to_string_system_message(self):
        from pydantic import BaseModel

        from marketing_project.services.function_pipeline.providers import (
            _inject_json_schema,
        )

        class M(BaseModel):
            x: int

        msgs = [{"role": "system", "content": "You are helpful."}]
        result = _inject_json_schema(msgs, M)
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], str)
        assert "IMPORTANT" in result[0]["content"]

    def test_appends_text_block_to_list_content_system_message(self):
        """Vision-style messages must not crash (TypeError: list + str)."""
        from pydantic import BaseModel

        from marketing_project.services.function_pipeline.providers import (
            _inject_json_schema,
        )

        class M(BaseModel):
            x: int

        msgs = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "System instructions."}],
            }
        ]
        result = _inject_json_schema(msgs, M)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert any(
            block.get("type") == "text" and "IMPORTANT" in block.get("text", "")
            for block in content
        )

    def test_prepends_system_message_when_none_present(self):
        from pydantic import BaseModel

        from marketing_project.services.function_pipeline.providers import (
            _inject_json_schema,
        )

        class M(BaseModel):
            x: int

        msgs = [{"role": "user", "content": "hello"}]
        result = _inject_json_schema(msgs, M)
        assert result[0]["role"] == "system"
        assert "IMPORTANT" in result[0]["content"]
        assert result[1]["role"] == "user"


class TestCallLLMStructured:
    """Tests for the core provider-agnostic LLM call function."""

    def _make_response(self, content):
        """Build a minimal litellm-style response object."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        return response

    @pytest.mark.asyncio
    async def test_returns_parsed_result_on_valid_json(self):
        from pydantic import BaseModel

        from marketing_project.services.function_pipeline.providers import (
            call_llm_structured,
        )

        class MyModel(BaseModel):
            name: str
            value: int

        mock_client = MagicMock()
        mock_client.acompletion = AsyncMock(
            return_value=self._make_response('{"name": "test", "value": 42}')
        )

        with patch(
            "marketing_project.services.provider_credential_service.get_provider_credential_service"
        ) as mock_svc_factory:
            mock_svc = AsyncMock()
            mock_svc.get_llm_client = AsyncMock(return_value=mock_client)
            mock_svc_factory.return_value = mock_svc

            result, response = await call_llm_structured(
                messages=[{"role": "user", "content": "hello"}],
                response_model=MyModel,
                model="gpt-4o",
                temperature=0.0,
            )

        assert result.name == "test"
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_raises_value_error_on_empty_content(self):
        from pydantic import BaseModel

        from marketing_project.services.function_pipeline.providers import (
            call_llm_structured,
        )

        class MyModel(BaseModel):
            name: str

        mock_client = MagicMock()
        mock_client.acompletion = AsyncMock(
            return_value=self._make_response(None)  # empty content
        )

        with patch(
            "marketing_project.services.provider_credential_service.get_provider_credential_service"
        ) as mock_svc_factory:
            mock_svc = AsyncMock()
            mock_svc.get_llm_client = AsyncMock(return_value=mock_client)
            mock_svc_factory.return_value = mock_svc

            with pytest.raises(ValueError, match="empty content"):
                await call_llm_structured(
                    messages=[{"role": "user", "content": "hello"}],
                    response_model=MyModel,
                    model="gpt-4o",
                    temperature=0.0,
                )

    @pytest.mark.asyncio
    async def test_raises_value_error_on_invalid_json(self):
        from pydantic import BaseModel

        from marketing_project.services.function_pipeline.providers import (
            call_llm_structured,
        )

        class MyModel(BaseModel):
            name: str

        mock_client = MagicMock()
        mock_client.acompletion = AsyncMock(
            return_value=self._make_response("this is not json at all")
        )

        with patch(
            "marketing_project.services.provider_credential_service.get_provider_credential_service"
        ) as mock_svc_factory:
            mock_svc = AsyncMock()
            mock_svc.get_llm_client = AsyncMock(return_value=mock_client)
            mock_svc_factory.return_value = mock_svc

            with pytest.raises(ValueError, match="invalid JSON"):
                await call_llm_structured(
                    messages=[{"role": "user", "content": "hello"}],
                    response_model=MyModel,
                    model="gpt-4o",
                    temperature=0.0,
                )

    @pytest.mark.asyncio
    async def test_strips_markdown_fences_before_parsing(self):
        from pydantic import BaseModel

        from marketing_project.services.function_pipeline.providers import (
            call_llm_structured,
        )

        class MyModel(BaseModel):
            name: str

        fenced = '```json\n{"name": "fenced"}\n```'
        mock_client = MagicMock()
        mock_client.acompletion = AsyncMock(return_value=self._make_response(fenced))

        with patch(
            "marketing_project.services.provider_credential_service.get_provider_credential_service"
        ) as mock_svc_factory:
            mock_svc = AsyncMock()
            mock_svc.get_llm_client = AsyncMock(return_value=mock_client)
            mock_svc_factory.return_value = mock_svc

            result, _ = await call_llm_structured(
                messages=[{"role": "user", "content": "hello"}],
                response_model=MyModel,
                model="gpt-4o",
                temperature=0.0,
            )

        assert result.name == "fenced"

    @pytest.mark.asyncio
    async def test_raises_value_error_on_schema_mismatch(self):
        from pydantic import BaseModel

        from marketing_project.services.function_pipeline.providers import (
            call_llm_structured,
        )

        class MyModel(BaseModel):
            required_field: int  # LLM won't return this

        mock_client = MagicMock()
        mock_client.acompletion = AsyncMock(
            return_value=self._make_response('{"wrong_field": "oops"}')
        )

        with patch(
            "marketing_project.services.provider_credential_service.get_provider_credential_service"
        ) as mock_svc_factory:
            mock_svc = AsyncMock()
            mock_svc.get_llm_client = AsyncMock(return_value=mock_client)
            mock_svc_factory.return_value = mock_svc

            with pytest.raises(ValueError, match="did not match expected schema"):
                await call_llm_structured(
                    messages=[{"role": "user", "content": "hello"}],
                    response_model=MyModel,
                    model="gpt-4o",
                    temperature=0.0,
                )
