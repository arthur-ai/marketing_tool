"""
Core modules for Marketing Project.

This package contains the core domain models, content sources, parsers, and utilities
that form the foundation of the marketing project system.

Key Modules:
- models: Domain models for content types (BlogPostContext, TranscriptContext, etc.)
- content_sources: Abstract base classes and configuration for content sources
- parsers: Content parsing utilities for different content types
- utils: Utility functions for content conversion and validation

Usage:
    from marketing_project.core import (
        BlogPostContext,
        TranscriptContext,
        ContentSource,
        ContentSourceManager,
        convert_dict_to_content_context,
    )
"""

from .content_sources import ContentSource, ContentSourceManager
from .models import (
    AppContext,
    BaseContentContext,
    BlogPostContext,
    ContentContext,
    EmailContext,
    ReleaseNotesContext,
    TranscriptContext,
)
from .utils import convert_dict_to_content_context

__all__ = [
    # Models
    "BaseContentContext",
    "TranscriptContext",
    "BlogPostContext",
    "ReleaseNotesContext",
    "ContentContext",
    "AppContext",
    "EmailContext",
    # Content Sources
    "ContentSource",
    "ContentSourceManager",
    # Utils
    "convert_dict_to_content_context",
]
