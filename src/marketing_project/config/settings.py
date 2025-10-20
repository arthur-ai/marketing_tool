"""
Application settings and configuration.

This module provides centralized configuration to avoid circular imports.
"""

import os

import yaml


def get_prompts_dir() -> str:
    """
    Get the prompts directory path based on the template version.

    Returns:
        str: Absolute path to the prompts directory
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_version = os.getenv("TEMPLATE_VERSION", "v1")
    return os.path.join(base_dir, "prompts", template_version)


# Module-level constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(BASE_DIR, "config", "pipeline.yml")
TEMPLATE_VERSION = os.getenv("TEMPLATE_VERSION", "v1")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts", TEMPLATE_VERSION)

# Load pipeline specification
with open(SPEC_PATH) as f:
    PIPELINE_SPEC = yaml.safe_load(f)
