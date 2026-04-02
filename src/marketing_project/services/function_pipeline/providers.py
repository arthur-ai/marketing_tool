"""
Provider-agnostic LLM call utilities backed by LiteLLM.

All callers should use `call_llm_structured` — it routes to the correct
provider via LiteLLM, injecting DB-backed credentials automatically.
"""

import json
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

from pydantic import BaseModel

from marketing_project.services.function_pipeline.litellm_client import (
    PROVIDER_ANTHROPIC,
    build_litellm_model,
    normalize_provider,
)

logger = logging.getLogger("marketing_project.services.function_pipeline.providers")


def _apply_field_exclusions(schema: dict, exclude) -> dict:
    """Strip excluded field names from a schema's properties and required list."""
    if not exclude:
        return schema
    schema = dict(schema)
    if "properties" in schema:
        schema["properties"] = {
            k: v for k, v in schema["properties"].items() if k not in exclude
        }
    if "required" in schema:
        schema["required"] = [f for f in schema["required"] if f not in exclude]
    return schema


def _iter_base_models(annotation) -> Iterator[Type[BaseModel]]:
    """Yield every BaseModel subclass found anywhere in a type annotation."""
    if annotation is None:
        return
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        for arg in getattr(annotation, "__args__", ()):
            yield from _iter_base_models(arg)
    elif isinstance(annotation, type):
        try:
            if issubclass(annotation, BaseModel):
                yield annotation
        except TypeError:
            pass


def _collect_nested_exclusions(model: Type[BaseModel]) -> Dict[str, frozenset]:
    """
    Walk the model's field annotations recursively and collect _llm_exclude_fields
    for every nested BaseModel, keyed by class __name__ (which matches $defs keys).
    """
    result: Dict[str, frozenset] = {}
    _walk_for_exclusions(model, result, seen=set())
    return result


def _walk_for_exclusions(
    model: Type[BaseModel], result: Dict[str, frozenset], seen: set
) -> None:
    if id(model) in seen:
        return
    seen.add(id(model))
    for field_info in model.model_fields.values():
        for nested_cls in _iter_base_models(field_info.annotation):
            exclude = getattr(nested_cls, "_llm_exclude_fields", frozenset())
            if exclude:
                result[nested_cls.__name__] = exclude
            _walk_for_exclusions(nested_cls, result, seen)


def _remove_unreferenced_defs(schema: dict) -> dict:
    """
    Drop $defs entries that are no longer $ref-referenced anywhere in the schema.
    This happens when top-level exclusions remove all fields that pointed to a
    nested model — the $def becomes dead code that Anthropic's grammar compiler
    may still try (and fail) to compile.
    """
    if "$defs" not in schema:
        return schema
    # Serialise everything except $defs to find active $ref strings
    non_defs = {k: v for k, v in schema.items() if k != "$defs"}
    non_defs_str = json.dumps(non_defs)
    # Also check cross-references within $defs themselves
    defs_str = json.dumps(schema["$defs"])
    referenced = {
        def_name
        for def_name in schema["$defs"]
        if f'"$ref": "#/$defs/{def_name}"' in non_defs_str
        or f'"$ref": "#/$defs/{def_name}"' in defs_str
    }
    if len(referenced) == len(schema["$defs"]):
        return schema  # nothing to prune
    schema = dict(schema)
    pruned = {k: v for k, v in schema["$defs"].items() if k in referenced}
    schema["$defs"] = pruned if pruned else schema.pop("$defs", {})
    if not schema.get("$defs"):
        schema.pop("$defs", None)
    return schema


def _get_llm_schema(response_model: Type[BaseModel]) -> dict:
    """
    Return the JSON schema for a Pydantic model, excluding any fields listed in
    the model's `_llm_exclude_fields` class variable. This lets us keep rich
    fields on the model for downstream use while staying within Anthropic's
    grammar compilation limits (~16 Optional fields per schema).

    Also applies _llm_exclude_fields from nested BaseModel classes to their
    corresponding $defs entries, so the grammar size reduction applies to the
    full schema tree (not just the top-level model). Unreferenced $defs are
    pruned after exclusions to avoid Anthropic compiling dead grammar rules.
    """
    schema = response_model.model_json_schema()

    # Apply top-level exclusions
    top_exclude = getattr(response_model, "_llm_exclude_fields", frozenset())
    schema = _apply_field_exclusions(schema, top_exclude)

    # Apply nested model exclusions to their $defs entries
    if "$defs" in schema:
        nested_excludes = _collect_nested_exclusions(response_model)
        if nested_excludes:
            defs = dict(schema["$defs"])
            for def_name, def_exclude in nested_excludes.items():
                if def_name in defs:
                    defs[def_name] = _apply_field_exclusions(
                        defs[def_name], def_exclude
                    )
            schema = dict(schema)
            schema["$defs"] = defs

    # Drop $defs no longer referenced after exclusions
    schema = _remove_unreferenced_defs(schema)

    return schema


def _make_schema_anthropic_safe(schema: dict) -> dict:
    """
    Recursively sanitise a JSON schema for Anthropic's structured output API:
    1. Add 'additionalProperties': false to all object-type nodes (required by Anthropic).
    2. Strip numeric/string constraint keywords (minimum, maximum, minLength, etc.) that
       Anthropic rejects and that cause grammar compilation errors (LiteLLM issues #21097,
       #21366). Pydantic ge/le constraints generate these keys; validation is done
       client-side via model_validate() after the LLM call.
    """
    # Strip constraint keywords Anthropic rejects at the current level
    _ANTHROPIC_UNSUPPORTED_CONSTRAINTS = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
        "multipleOf",
    }
    schema = {
        k: v for k, v in schema.items() if k not in _ANTHROPIC_UNSUPPORTED_CONSTRAINTS
    }

    if schema.get("type") == "object":
        # Always set additionalProperties: false — Anthropic rejects any dict value
        # (e.g. Dict[str, str] produces {"additionalProperties": {"type": "string"}}).
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
    schema_json = json.dumps(_get_llm_schema(response_model), indent=2)
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
        # Always build response_format from the Pydantic model for Anthropic calls.
        # Arthur-supplied schemas are not used here because Arthur strips nested
        # object `properties` on write (UP-4007), which causes Anthropic's structured
        # output mode to enforce empty objects for all nested fields.
        safe_schema = _make_schema_anthropic_safe(_get_llm_schema(response_model))
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
        timeout=300,
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
