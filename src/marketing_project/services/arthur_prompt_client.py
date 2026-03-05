"""
Arthur AI prompt client.

Fetches versioned prompt messages (system + user) from Arthur's prompt management API
for each pipeline step, so prompts can be updated in Arthur without redeploying the app.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

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
