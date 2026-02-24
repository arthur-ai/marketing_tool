"""
Backward-compatibility shim for design_kit API.

This module re-exports from brand_kit.
"""

from marketing_project.api.brand_kit import router

__all__ = ["router"]
