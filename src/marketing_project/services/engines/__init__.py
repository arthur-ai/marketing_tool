"""
Generic engine framework for composable processing.

This module provides a framework for creating engines that can be mixed and matched
to process different fields of a result model using different strategies (LLM, local, etc.).
"""

from marketing_project.services.engines.base import Engine
from marketing_project.services.engines.composer import EngineComposer
from marketing_project.services.engines.registry import EngineRegistry, get_engine

__all__ = [
    "Engine",
    "EngineRegistry",
    "get_engine",
    "EngineComposer",
]
