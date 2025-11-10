"""
Pipeline plugins package.

This package contains all pipeline step plugins and supporting infrastructure.
"""

# Import key classes for convenience
from marketing_project.plugins.base import PipelineStepPlugin
from marketing_project.plugins.context_utils import ContextTransformer
from marketing_project.plugins.dependency_graph import DependencyGraph
from marketing_project.plugins.registry import PluginRegistry, get_plugin_registry

__all__ = [
    "PipelineStepPlugin",
    "get_plugin_registry",
    "PluginRegistry",
    "ContextTransformer",
    "DependencyGraph",
]
