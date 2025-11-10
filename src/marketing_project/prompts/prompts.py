"""
Prompt template management for Marketing Project.

This module provides a centralized system for loading and managing Jinja2 prompt templates
with support for multiple languages and template versioning. Templates are preloaded on
module import for fast access.

Key Features:
- Template versioning (defaults to v1)
- Multi-language support with fallback to 'en'
- Template caching for performance
- Preloading of all templates on import

Usage:
    from marketing_project.prompts.prompts import TEMPLATES, get_template, has_template

    # Check if template exists
    if has_template('en', 'seo_keywords_agent_instructions'):
        template = get_template('en', 'seo_keywords_agent_instructions')
        result = template.render(context={})

    # Or use preloaded templates
    template_key = ('en', 'seo_keywords_agent_instructions')
    if template_key in TEMPLATES:
        result = TEMPLATES[template_key].render(context={})
"""

import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template

# 1) Which version?
TEMPLATE_VERSION = os.getenv("TEMPLATE_VERSION", "v1")

# 2) Base prompts directory: .../prompts/v1
_THIS_DIR = Path(__file__).parent
_BASE_DIR = _THIS_DIR / TEMPLATE_VERSION

# 3) Caches
_envs: dict[str, Environment] = {}
TEMPLATES: dict[tuple[str, str], Template] = {}


def get_env(lang: str) -> Environment:
    """
    Return a Jinja2 Environment for `lang` (e.g. 'en'), falling back to 'en'.
    """
    if lang not in _envs:
        lang_dir = _BASE_DIR / lang
        if not lang_dir.is_dir():
            lang_dir = _BASE_DIR / "en"
        _envs[lang] = Environment(
            loader=FileSystemLoader(str(lang_dir)), autoescape=True
        )
    return _envs[lang]


def list_templates(lang: str) -> list[str]:
    """
    List all .j2 filenames for a given lang (or 'en' fallback).
    """
    lang_dir = _BASE_DIR / lang
    if not lang_dir.is_dir():
        lang_dir = _BASE_DIR / "en"
    return [p.name for p in lang_dir.glob("*.j2")]


def get_template(lang: str, name: str) -> Template:
    """
    Get a template by language and name, loading it if not already cached.

    Args:
        lang: Language code (e.g., 'en')
        name: Template name without extension (e.g., 'seo_keywords_agent_instructions')

    Returns:
        Jinja2 Template object

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    template_key = (lang, name)

    # Check if already loaded
    if template_key in TEMPLATES:
        return TEMPLATES[template_key]

    # Load on demand
    env = get_env(lang)
    template = env.get_template(f"{name}.j2")
    TEMPLATES[template_key] = template
    return template


def has_template(lang: str, name: str) -> bool:
    """
    Check if a template exists for the given language and name.

    Args:
        lang: Language code (e.g., 'en')
        name: Template name without extension (e.g., 'seo_keywords_agent_instructions')

    Returns:
        True if template exists, False otherwise
    """
    template_key = (lang, name)

    # Check if already loaded
    if template_key in TEMPLATES:
        return True

    # Check if file exists
    lang_dir = _BASE_DIR / lang
    if not lang_dir.is_dir():
        lang_dir = _BASE_DIR / "en"  # fallback to en

    template_path = lang_dir / f"{name}.j2"
    return template_path.exists()


def _load_all_templates():
    """
    Scan every lang folder under prompts/v1 and preload each template into TEMPLATES.
    Keyed by (lang, name_without_ext).
    """
    if not _BASE_DIR.exists():
        return
    for lang_dir in _BASE_DIR.iterdir():
        if not lang_dir.is_dir():
            continue
        env = get_env(lang_dir.name)  # ensure env is cached
        for tpl_path in lang_dir.glob("*.j2"):
            name = tpl_path.stem  # e.g. 'seo_keywords_agent_instructions'
            TEMPLATES[(lang_dir.name, name)] = env.get_template(tpl_path.name)


_load_all_templates()

__all__ = [
    "TEMPLATES",
    "get_env",
    "get_template",
    "has_template",
    "list_templates",
    "TEMPLATE_VERSION",
]
