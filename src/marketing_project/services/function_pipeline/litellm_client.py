"""
LLM Client backed by LiteLLM.

Provides a unified async interface for all supported LLM providers:
OpenAI, Anthropic, Google Gemini, Vertex AI, and Amazon Bedrock.

Credentials are injected per-call via _add_provider_credentials() from the
values supplied at construction time (typically loaded from the DB by
ProviderCredentialService).

Pattern mirrors: arthur-engine/genai-engine/src/clients/llm/llm_client.py
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import litellm
from litellm import completion_cost, get_model_cost_map, model_cost_map_url
from litellm.types.utils import ModelResponse

from marketing_project.models.provider_models import SUPPORTED_PROVIDERS

logger = logging.getLogger(
    "marketing_project.services.function_pipeline.litellm_client"
)

# Suppress LiteLLM's verbose default output
litellm.suppress_debug_info = True

# Provider string constants (LiteLLM-native names)
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"
PROVIDER_VERTEX_AI = "vertex_ai"
PROVIDER_BEDROCK = "bedrock"

# Arthur → LiteLLM provider name normalisation map
PROVIDER_ALIAS_MAP: Dict[str, str] = {
    "openai": PROVIDER_OPENAI,
    "anthropic": PROVIDER_ANTHROPIC,
    "gemini": PROVIDER_GEMINI,
    "google": PROVIDER_GEMINI,
    "vertex_ai": PROVIDER_VERTEX_AI,
    "vertexai": PROVIDER_VERTEX_AI,
    "vertex": PROVIDER_VERTEX_AI,
    "bedrock": PROVIDER_BEDROCK,
    "aws_bedrock": PROVIDER_BEDROCK,
}


def normalize_provider(provider: Optional[str]) -> str:
    """Convert Arthur's model_provider string to a canonical LiteLLM provider name."""
    if not provider:
        return PROVIDER_OPENAI
    normalized = PROVIDER_ALIAS_MAP.get(provider.lower(), provider.lower())
    if normalized not in SUPPORTED_PROVIDERS:
        logger.warning(
            "Unknown provider '%s' (normalized: '%s') — falling back to OpenAI. "
            "Supported: %s",
            provider,
            normalized,
            SUPPORTED_PROVIDERS,
        )
        return PROVIDER_OPENAI
    return normalized


def build_litellm_model(model: str, provider: Optional[str]) -> str:
    """
    Construct the LiteLLM model string from (provider, model_name).

    LiteLLM convention:
    - OpenAI: no prefix  → "gpt-4o"
    - Anthropic:         → "anthropic/claude-3-5-sonnet-latest"
    - Gemini:            → "gemini/gemini-1.5-pro"
    - Vertex AI:         → "vertex_ai/gemini-1.5-pro"
    - Bedrock:           → "bedrock/claude-3-sonnet-20240229-v1:0"
    """
    if not model or not model.strip():
        raise ValueError(f"model cannot be empty (provider='{provider}')")
    p = normalize_provider(provider)
    if p == PROVIDER_OPENAI:
        return model  # no prefix for OpenAI
    return f"{p}/{model}"


def _build_supported_models() -> Dict[str, List[str]]:
    """Build {provider: [model_names]} from LiteLLM's live cost map."""
    models: Dict[str, List[str]] = {}
    try:
        for model_name, cost_config in get_model_cost_map(
            url=model_cost_map_url
        ).items():
            if cost_config.get("mode") != "chat":
                continue
            provider = cost_config.get("litellm_provider", "")
            # Normalise all vertex_ai variants to a single key
            if provider.startswith("vertex_ai"):
                provider = PROVIDER_VERTEX_AI
            # Strip "provider/" prefix that LiteLLM sometimes includes in the key
            short_name = model_name.replace(f"{provider}/", "")
            models.setdefault(provider, []).append(short_name)
    except Exception as exc:
        logger.warning("Failed to build supported models map: %s", exc)
    return models


SUPPORTED_TEXT_MODELS: Dict[str, List[str]] = {}
_models_loaded = False
_models_lock = threading.Lock()


def _ensure_models_loaded() -> None:
    """Lazy-load the model map on first access. Safe to call multiple times."""
    global SUPPORTED_TEXT_MODELS, _models_loaded
    if _models_loaded:
        return
    with _models_lock:
        if _models_loaded:
            return
        SUPPORTED_TEXT_MODELS = _build_supported_models()
        _models_loaded = True
        _start_refresh_thread()


def _refresh_models_periodically() -> None:
    global SUPPORTED_TEXT_MODELS
    while True:
        try:
            time.sleep(8 * 60 * 60)  # 8 hours
            SUPPORTED_TEXT_MODELS = _build_supported_models()
            logger.info("Refreshed LiteLLM supported model list")
        except Exception as exc:
            logger.warning("Failed to refresh model list: %s", exc)


def _start_refresh_thread() -> None:
    """Start the background model-refresh daemon. Called once at first model access."""
    t = threading.Thread(target=_refresh_models_periodically, daemon=True)
    t.start()


class LLMClient:
    """
    Async LLM client backed by LiteLLM.

    Credentials are injected at call time via _add_provider_credentials(),
    using values supplied at construction (loaded from DB by ProviderCredentialService).
    """

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        vertex_credentials: Optional[Dict[str, Any]] = None,
        aws_bedrock_credentials: Optional[Dict[str, Any]] = None,
    ):
        self.provider = normalize_provider(provider)
        self.api_key = api_key
        self.project_id = project_id
        self.region = region
        self.vertex_credentials = vertex_credentials
        self.aws_bedrock_credentials = aws_bedrock_credentials

    def _add_provider_credentials(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Inject provider-specific auth credentials into the LiteLLM call kwargs."""
        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.provider == PROVIDER_VERTEX_AI:
            if self.project_id:
                kwargs["vertex_project"] = self.project_id
            if self.region:
                kwargs["vertex_location"] = self.region
            if self.vertex_credentials is not None:
                kwargs["vertex_credentials"] = self.vertex_credentials
            else:
                logger.warning(
                    "Vertex AI: no service account credentials supplied. "
                    "Falling back to application default credentials."
                )

        elif self.provider == PROVIDER_BEDROCK:
            if self.aws_bedrock_credentials:
                creds = self.aws_bedrock_credentials
                for field in (
                    "aws_access_key_id",
                    "aws_secret_access_key",
                    "aws_session_token",
                    "aws_bedrock_runtime_endpoint",
                    "aws_role_name",
                    "aws_session_name",
                ):
                    if creds.get(field):
                        kwargs[field] = creds[field]
            if self.region:
                kwargs["aws_region_name"] = self.region

            has_key = "aws_access_key_id" in kwargs
            has_secret = "aws_secret_access_key" in kwargs
            if has_key != has_secret:
                raise ValueError(
                    "aws_access_key_id and aws_secret_access_key must be provided together"
                )

        return kwargs

    async def acompletion(self, **kwargs: Any) -> ModelResponse:
        """
        Async LLM completion with credential injection and error normalisation.

        Accepts the same kwargs as litellm.acompletion (model, messages,
        temperature, response_format, etc.).
        """
        kwargs = self._add_provider_credentials(kwargs)
        try:
            response = await litellm.acompletion(**kwargs)
            return response
        except litellm.AuthenticationError:
            logger.error("Authentication failed for provider '%s'", self.provider)
            raise
        except litellm.RateLimitError:
            logger.warning("Rate limit hit for provider '%s'", self.provider)
            raise
        except litellm.NotFoundError as exc:
            model = kwargs.get("model", "unknown")
            logger.error(
                "Model '%s' not found for provider '%s': %s", model, self.provider, exc
            )
            raise
        except litellm.BadRequestError:
            logger.error("Bad request to provider '%s'", self.provider)
            raise
        except litellm.ServiceUnavailableError:
            logger.warning("Provider '%s' is temporarily unavailable", self.provider)
            raise
        except litellm.BadGatewayError:
            logger.warning(
                "Bad gateway (502) from provider '%s' — transient, will retry",
                self.provider,
            )
            raise
        except Exception:
            logger.error(
                "Unexpected error calling provider '%s'", self.provider, exc_info=True
            )
            raise

    def get_cost(self, response: ModelResponse) -> Optional[str]:
        """
        Return estimated call cost as a string (e.g. "0.000123").
        Returns None if cost cannot be calculated.
        """
        try:
            cost = completion_cost(response)
            return f"{cost:.6f}" if cost is not None else None
        except Exception as exc:
            logger.debug("Could not calculate cost: %s", exc)
            return None

    def get_available_models(self) -> List[str]:
        """Return models available for this provider from the cached cost map."""
        _ensure_models_loaded()
        return SUPPORTED_TEXT_MODELS.get(self.provider, [])
