"""
Prompts package for Marketing Project.

This package contains Jinja2 template management for agent prompts and instructions.
Templates are organized by version and language (e.g., prompts/v1/en/).

Main Module:
- prompts.prompts: Template loading and management utilities
"""

from .prompts import (
    TEMPLATE_VERSION,
    TEMPLATES,
    get_env,
    get_template,
    has_template,
    list_templates,
)

__all__ = [
    "TEMPLATES",
    "TEMPLATE_VERSION",
    "get_env",
    "get_template",
    "has_template",
    "list_templates",
]
