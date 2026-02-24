"""
Backward-compatibility shim for design_kit_manager.

This module re-exports from brand_kit_manager.
"""

from marketing_project.services.brand_kit_manager import BrandKitManager
from marketing_project.services.brand_kit_manager import (
    BrandKitManager as DesignKitManager,
)
from marketing_project.services.brand_kit_manager import get_brand_kit_manager
from marketing_project.services.brand_kit_manager import (
    get_brand_kit_manager as get_design_kit_manager,
)

__all__ = [
    "DesignKitManager",
    "BrandKitManager",
    "get_design_kit_manager",
    "get_brand_kit_manager",
]
