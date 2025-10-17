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

# Cache one Environment per language
_envs: dict[str, Environment] = {}


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
