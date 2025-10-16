"""
Content analysis plugin for Marketing Project.

This plugin provides general content analysis functions that work across
all content types for routing and processing decisions.
"""

from . import tasks
from .tasks import analyze_content_for_pipeline

__all__ = ["tasks", "analyze_content_for_pipeline"]
