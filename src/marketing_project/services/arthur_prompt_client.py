"""
Arthur AI prompt client.

Fetches versioned prompt messages (system + user) from Arthur's prompt management API
for each pipeline step, so prompts can be updated in Arthur without redeploying the app.

Write operations (save_arthur_prompt, add_arthur_prompt_tag) are used by Claude when
prompt changes are agreed on — they create new versions in Arthur without redeployment.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("marketing_project.services.arthur_prompt_client")


@dataclass
class ArthurPromptResult:
    """Result from fetching a prompt from Arthur, including model routing info."""

    system_content: str
    user_template: str
    model_name: Optional[str] = None  # e.g. "claude-3-5-sonnet-latest"
    model_provider: Optional[str] = None  # e.g. "anthropic" or "openai"
    model_config: Optional[dict] = (
        None  # Extra LLM kwargs from Arthur (e.g. api_base for vLLM)
    )


# Maps internal pipeline step names to Arthur prompt names.
# Step 0 has no suffix; all other steps have the _prompt suffix.
ARTHUR_PROMPT_NAMES: dict[str, str] = {
    "blog_post_preprocessing_approval": "blog_post_preprocessing_approval",
    "seo_keywords": "seo_keywords_prompt",
    "marketing_brief": "marketing_brief_prompt",
    "article_generation": "article_generation_prompt",
    "seo_optimization": "seo_optimization_prompt",
    "suggested_links": "suggested_links_prompt",
    "content_formatting": "content_formatting_prompt",
    "brand_kit": "brand_kit_prompt",
    "competitor_research_analysis": "competitor_research_analysis",
    "competitor_research_summary": "competitor_research_summary",
    # Social media pipeline
    "social_media_marketing_brief": "social_media_marketing_brief",
    "social_media_marketing_brief_linkedin": "social_media_marketing_brief_linkedin",
    "social_media_marketing_brief_hackernews": "social_media_marketing_brief_hackernews",
    "social_media_marketing_brief_email": "social_media_marketing_brief_email",
    "social_media_angle_hook": "social_media_angle_hook",
    "social_media_angle_hook_linkedin": "social_media_angle_hook_linkedin",
    "social_media_angle_hook_hackernews": "social_media_angle_hook_hackernews",
    "social_media_angle_hook_email": "social_media_angle_hook_email",
    "social_media_post_generation": "social_media_post_generation",
    "social_media_post_generation_linkedin": "social_media_post_generation_linkedin",
    "social_media_post_generation_hackernews": "social_media_post_generation_hackernews",
    "social_media_post_generation_email": "social_media_post_generation_email",
    # Transcript pipeline
    "transcript_preprocessing_approval": "transcript_preprocessing_approval",
    "transcript_content_extraction": "transcript_content_extraction",
    "transcript_duration_extraction": "transcript_duration_extraction",
    "transcript_speakers_extraction": "transcript_speakers_extraction",
}


async def fetch_arthur_prompt(step_name: str) -> Optional[ArthurPromptResult]:
    """
    Fetch the production-tagged system and user message templates for a pipeline step.

    Reads ARTHUR_BASE_URL, ARTHUR_API_KEY, and ARTHUR_TASK_ID from environment.
    Returns None silently if any of those are missing or the step is unknown,
    so callers can fall back to local Jinja2 templates without raising.

    Args:
        step_name: Internal pipeline step name (e.g. "seo_keywords").

    Returns:
        Tuple of (system_content, user_template) where user_template is a
        Jinja2 template string with {{ variable }} placeholders, or None if
        Arthur is not configured or the fetch fails.
    """
    arthur_base_url = os.getenv("ARTHUR_BASE_URL")
    arthur_api_key = os.getenv("ARTHUR_API_KEY")
    arthur_task_id = os.getenv("ARTHUR_TASK_ID")

    if not all([arthur_base_url, arthur_api_key, arthur_task_id]):
        return None

    prompt_name = ARTHUR_PROMPT_NAMES.get(step_name)
    if not prompt_name:
        return None

    url = (
        f"{arthur_base_url}/api/v1/tasks/{arthur_task_id}"
        f"/prompts/{prompt_name}/versions/production"
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {arthur_api_key}"},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

        messages = data.get("messages", [])
        if len(messages) < 2:
            logger.warning(
                "Arthur prompt '%s' returned fewer than 2 messages", prompt_name
            )
            return None

        system_content = messages[0].get("content", "")
        # Support both 2-message (system + user) and 3-message (system + assistant + user) formats.
        # The user template is always the last message.
        user_template = messages[-1].get("content", "")

        if not system_content or not user_template:
            logger.warning(
                "Arthur prompt '%s' has an empty system or user message", prompt_name
            )
            return None

        logger.debug("Fetched Arthur prompt for step '%s'", step_name)
        return ArthurPromptResult(
            system_content=system_content,
            user_template=user_template,
            model_name=data.get("model_name"),
            model_provider=data.get("model_provider"),
            model_config=data.get("config") or None,
        )

    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Arthur prompt fetch failed for step '%s': HTTP %s — %s",
            step_name,
            exc.response.status_code,
            exc.response.text[:200],
        )
        return None
    except Exception as exc:
        logger.warning("Arthur prompt fetch failed for step '%s': %s", step_name, exc)
        return None


async def save_arthur_prompt(
    step_name: str,
    messages: List[Dict[str, Any]],
    model_name: str,
    model_provider: str,
    config: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Save a new version of a prompt to Arthur Engine.

    Each call creates a new auto-incremented version.  The prompt is identified
    by its step_name, which is resolved to an Arthur prompt name via ARTHUR_PROMPT_NAMES.

    Args:
        step_name:      Internal pipeline step name (e.g. "seo_keywords").
        messages:       OpenAI-format message list:
                        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        model_name:     LLM model, e.g. "gpt-4o".
        model_provider: One of: openai, anthropic, gemini, bedrock, vertex_ai.
        config:         Optional dict with temperature, max_tokens, etc.
        tools:          Optional OpenAI function-calling tool definitions.

    Returns:
        The saved AgenticPrompt object from Arthur, or None on failure.
    """
    arthur_base_url = os.getenv("ARTHUR_BASE_URL")
    arthur_api_key = os.getenv("ARTHUR_API_KEY")
    arthur_task_id = os.getenv("ARTHUR_TASK_ID")

    if not all([arthur_base_url, arthur_api_key, arthur_task_id]):
        logger.warning("save_arthur_prompt: Arthur env vars not configured")
        return None

    prompt_name = ARTHUR_PROMPT_NAMES.get(step_name)
    if not prompt_name:
        logger.warning("save_arthur_prompt: unknown step_name '%s'", step_name)
        return None

    url = f"{arthur_base_url}/api/v1/tasks/{arthur_task_id}/prompts/{prompt_name}"
    body: Dict[str, Any] = {
        "messages": messages,
        "model_name": model_name,
        "model_provider": model_provider,
    }
    if config:
        body["config"] = config
    if tools:
        body["tools"] = tools

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {arthur_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
            response.raise_for_status()
            result = response.json()

        logger.info(
            "Saved Arthur prompt '%s' as version %s (step: %s)",
            prompt_name,
            result.get("version"),
            step_name,
        )
        return result

    except httpx.HTTPStatusError as exc:
        logger.error(
            "save_arthur_prompt failed for '%s': HTTP %s — %s",
            prompt_name,
            exc.response.status_code,
            exc.response.text[:300],
        )
        return None
    except Exception as exc:
        logger.error("save_arthur_prompt failed for '%s': %s", prompt_name, exc)
        return None


async def add_arthur_prompt_tag(
    step_name: str,
    version: int,
    tag: str,
) -> bool:
    """
    Tag a specific prompt version in Arthur (e.g. mark version 3 as 'production').

    Args:
        step_name: Internal pipeline step name.
        version:   The version number to tag.
        tag:       Tag string, e.g. "production" or "staging".

    Returns:
        True on success, False on failure.
    """
    arthur_base_url = os.getenv("ARTHUR_BASE_URL")
    arthur_api_key = os.getenv("ARTHUR_API_KEY")
    arthur_task_id = os.getenv("ARTHUR_TASK_ID")

    if not all([arthur_base_url, arthur_api_key, arthur_task_id]):
        logger.warning("add_arthur_prompt_tag: Arthur env vars not configured")
        return False

    prompt_name = ARTHUR_PROMPT_NAMES.get(step_name)
    if not prompt_name:
        logger.warning("add_arthur_prompt_tag: unknown step_name '%s'", step_name)
        return False

    url = (
        f"{arthur_base_url}/api/v1/tasks/{arthur_task_id}"
        f"/prompts/{prompt_name}/versions/{version}/tags"
    )
    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(
                url,
                json={"tag": tag},
                headers={
                    "Authorization": f"Bearer {arthur_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            response.raise_for_status()
        logger.info("Tagged Arthur prompt '%s' v%s as '%s'", prompt_name, version, tag)
        return True
    except httpx.HTTPStatusError as exc:
        logger.error(
            "add_arthur_prompt_tag failed for '%s' v%s: HTTP %s — %s",
            prompt_name,
            version,
            exc.response.status_code,
            exc.response.text[:200],
        )
        return False
    except Exception as exc:
        logger.error("add_arthur_prompt_tag failed for '%s': %s", prompt_name, exc)
        return False


async def list_arthur_prompts() -> Optional[List[Dict[str, Any]]]:
    """
    Return all agentic prompts registered under the configured Arthur task.

    Returns:
        List of prompt objects from Arthur, or None if Arthur is not configured.
    """
    arthur_base_url = os.getenv("ARTHUR_BASE_URL")
    arthur_api_key = os.getenv("ARTHUR_API_KEY")
    arthur_task_id = os.getenv("ARTHUR_TASK_ID")

    if not all([arthur_base_url, arthur_api_key, arthur_task_id]):
        return None

    url = f"{arthur_base_url}/api/v1/tasks/{arthur_task_id}/prompts"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {arthur_api_key}"},
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        logger.warning("list_arthur_prompts failed: %s", exc)
        return None


async def list_arthur_prompt_versions(step_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    List all versions of a prompt by step name.

    Returns:
        List of version objects, or None if not configured or step unknown.
    """
    arthur_base_url = os.getenv("ARTHUR_BASE_URL")
    arthur_api_key = os.getenv("ARTHUR_API_KEY")
    arthur_task_id = os.getenv("ARTHUR_TASK_ID")

    if not all([arthur_base_url, arthur_api_key, arthur_task_id]):
        return None

    prompt_name = ARTHUR_PROMPT_NAMES.get(step_name)
    if not prompt_name:
        logger.warning("list_arthur_prompt_versions: unknown step_name '%s'", step_name)
        return None

    url = f"{arthur_base_url}/api/v1/tasks/{arthur_task_id}/prompts/{prompt_name}/versions"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {arthur_api_key}"},
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        logger.warning(
            "list_arthur_prompt_versions failed for '%s': %s", step_name, exc
        )
        return None
