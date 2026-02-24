"""
Backward-compatibility shim for design_kit_config.

This module re-exports from brand_kit_config.
"""

from marketing_project.models.brand_kit_config import BrandKitConfig
from marketing_project.models.brand_kit_config import BrandKitConfig as DesignKitConfig
from marketing_project.models.brand_kit_config import ContentTypeConfig

__all__ = ["DesignKitConfig", "BrandKitConfig", "ContentTypeConfig"]
