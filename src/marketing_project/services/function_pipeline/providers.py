"""
Provider-agnostic LLM call utilities backed by LiteLLM.

All callers should use `call_llm_structured` — it routes to the correct
provider via LiteLLM, injecting DB-backed credentials automatically.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel

from marketing_project.services.function_pipeline.litellm_client import (
    PROVIDER_ANTHROPIC,
    build_litellm_model,
    normalize_provider,
)

logger = logging.getLogger("marketing_project.services.function_pipeline.providers")


def _make_schema_anthropic_safe(schema: dict) -> dict:
    """
    Recursively add 'additionalProperties': false to all object-type nodes.
    Required by Anthropic's API when using output_format with a JSON schema.
    Preserves existing additionalProperties when it is already a schema dict
    (e.g. {type: number} for Dict[str, float] fields).
    """
    schema = dict(schema)
    if schema.get("type") == "object":
        # Don't overwrite if already set to a schema dict (Dict[str, X] pattern)
        if not isinstance(schema.get("additionalProperties"), dict):
            schema["additionalProperties"] = False
        if "properties" in schema:
            schema["properties"] = {
                k: _make_schema_anthropic_safe(v)
                for k, v in schema["properties"].items()
            }
    if "items" in schema:
        schema["items"] = _make_schema_anthropic_safe(schema["items"])
    for key in ("$defs", "definitions"):
        if key in schema:
            schema[key] = {
                k: _make_schema_anthropic_safe(v) for k, v in schema[key].items()
            }
    for key in ("anyOf", "allOf", "oneOf"):
        if key in schema:
            schema[key] = [_make_schema_anthropic_safe(s) for s in schema[key]]
    return schema


def _inject_json_schema(messages: list, response_model: Type[BaseModel]) -> list:
    """
    Append a JSON schema instruction to the system message so all providers
    know to return structured JSON.
    """
    schema_json = json.dumps(response_model.model_json_schema(), indent=2)
    schema_instruction = (
        "\n\nIMPORTANT: Respond with valid JSON only, matching this schema exactly:\n"
        f"```json\n{schema_json}\n```\n"
        "Do not include any text outside the JSON object."
    )

    result = []
    injected = False
    for msg in messages:
        if msg.get("role") == "system" and not injected:
            content = msg["content"]
            if isinstance(content, list):
                # Vision / multi-modal messages use a list of content blocks.
                # Append the schema instruction as an additional text block.
                new_content = content + [
                    {"type": "text", "text": schema_instruction.strip()}
                ]
            else:
                new_content = content + schema_instruction
            result.append({**msg, "content": new_content})
            injected = True
        else:
            result.append(msg)

    if not injected:
        # No system message found — prepend one
        result.insert(0, {"role": "system", "content": schema_instruction.strip()})

    return result


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```", 2)
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            stripped = inner.rsplit("```", 1)[0].strip()
    return stripped


async def call_llm_structured(
    messages: list,
    response_model: Type[BaseModel],
    model: str,
    temperature: float,
    provider: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
    # Legacy parameter — ignored (kept for backward compatibility during transition)
    openai_client: Optional[Any] = None,
) -> Tuple[BaseModel, Any]:
    """
    Provider-agnostic structured LLM call via LiteLLM.

    Credentials are loaded from the DB by ProviderCredentialService.
    `model_config` carries extra kwargs from Arthur (e.g. api_base overrides).

    Args:
        messages: List of {role, content} dicts (OpenAI format).
        response_model: Pydantic model class for output validation.
        model: Model identifier string.
        temperature: Sampling temperature.
        provider: Provider string (e.g. "anthropic", "vertex_ai"). Defaults to "openai".
        model_config: Optional extra LiteLLM kwargs from Arthur prompt config.
        openai_client: Ignored — kept for backward compatibility.

    Returns:
        Tuple of (parsed_result, raw_response).
    """
    from marketing_project.services.provider_credential_service import (
        get_provider_credential_service,
    )

    effective_provider = normalize_provider(provider)
    litellm_model = build_litellm_model(model, effective_provider)

    # Inject JSON schema into system message so all providers know what to return
    messages_with_schema = _inject_json_schema(messages, response_model)

    # Load authenticated client from credential service
    credential_service = get_provider_credential_service()
    llm_client = await credential_service.get_llm_client(effective_provider)

    # Extra kwargs from Arthur prompt config (e.g. api_base override for vLLM)
    extra_kwargs: Dict[str, Any] = dict(model_config or {})
    if effective_provider == PROVIDER_ANTHROPIC:
        # Anthropic requires 'additionalProperties': false on every object schema node.
        # Apply this regardless of whether response_format came from Arthur or our default,
        # since setdefault would be a no-op when Arthur already supplies response_format.
        rf = extra_kwargs.get("response_format")
        if (
            rf
            and rf.get("type") == "json_schema"
            and isinstance(rf.get("json_schema"), dict)
        ):
            # Arthur-supplied schema — patch it in place
            inner = rf["json_schema"]
            if "schema" in inner:
                inner["schema"] = _make_schema_anthropic_safe(inner["schema"])
        else:
            # No schema from Arthur — build one from the Pydantic response model
            safe_schema = _make_schema_anthropic_safe(
                response_model.model_json_schema()
            )
            extra_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": safe_schema},
            }
    else:
        extra_kwargs.setdefault("response_format", {"type": "json_object"})

    response = await llm_client.acompletion(
        model=litellm_model,
        messages=messages_with_schema,
        temperature=temperature,
        **extra_kwargs,
    )

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError(
            f"LLM returned empty content for model '{litellm_model}' "
            f"(provider '{effective_provider}')"
        )
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        logger.error(
            "LLM response was not valid JSON (model=%s, provider=%s): %s\nRaw: %.500s",
            litellm_model,
            effective_provider,
            exc,
            raw,
        )
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
    try:
        parsed_result = response_model.model_validate(data)
    except Exception as exc:
        logger.error(
            "LLM response did not match expected schema (model=%s, provider=%s): %s\nData: %.500s",
            litellm_model,
            effective_provider,
            exc,
            str(data),
        )
        raise ValueError(f"LLM response did not match expected schema: {exc}") from exc
    return parsed_result, response
