"""
Backward-compatibility shim for design_kit tasks.

This module re-exports from brand_kit plugin.
"""

from marketing_project.plugins.brand_kit.tasks import BrandKitPlugin
from marketing_project.plugins.brand_kit.tasks import BrandKitPlugin as DesignKitPlugin
from marketing_project.plugins.brand_kit.tasks import ContentAnalysis

__all__ = ["DesignKitPlugin", "BrandKitPlugin", "ContentAnalysis"]
