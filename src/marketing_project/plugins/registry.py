"""
Plugin registry for pipeline steps.

This module provides a registry system for discovering and loading pipeline step plugins.
"""

import logging
from importlib import import_module
from typing import Dict, Optional, Type

from marketing_project.plugins.base import PipelineStepPlugin
from marketing_project.plugins.dependency_graph import DependencyGraph

logger = logging.getLogger("marketing_project.plugins.registry")


class PluginRegistry:
    """
    Registry for pipeline step plugins.

    Automatically discovers and loads plugins from the plugins directory.
    """

    def __init__(self):
        self._plugins: Dict[str, PipelineStepPlugin] = {}
        self._initialized = False
        self._validated = False

    def register(self, plugin: PipelineStepPlugin):
        """
        Register a plugin instance.

        Args:
            plugin: Plugin instance to register
        """
        if not isinstance(plugin, PipelineStepPlugin):
            raise TypeError(
                f"Plugin must be an instance of PipelineStepPlugin, got {type(plugin)}"
            )

        step_name = plugin.step_name
        if step_name in self._plugins:
            logger.warning(
                f"Plugin for step '{step_name}' already registered, overwriting"
            )

        self._plugins[step_name] = plugin
        logger.debug(f"Registered plugin: {step_name} (step {plugin.step_number})")

    def get_plugin(self, step_name: str) -> Optional[PipelineStepPlugin]:
        """
        Get a plugin by step name.

        Args:
            step_name: Name of the step (e.g., 'seo_keywords')

        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(step_name)

    def get_plugin_by_number(self, step_number: int) -> Optional[PipelineStepPlugin]:
        """
        Get a plugin by step number.

        Args:
            step_number: Step number (1-8)

        Returns:
            Plugin instance or None if not found
        """
        for plugin in self._plugins.values():
            if plugin.step_number == step_number:
                return plugin
        return None

    def get_all_plugins(self) -> Dict[str, PipelineStepPlugin]:
        """
        Get all registered plugins.

        Returns:
            Dictionary mapping step names to plugin instances
        """
        return self._plugins.copy()

    def get_plugins_in_order(self) -> list[PipelineStepPlugin]:
        """
        Get all plugins sorted by step number.

        Returns:
            List of plugins in execution order
        """
        return sorted(self._plugins.values(), key=lambda p: p.step_number)

    def auto_discover(self):
        """
        Automatically discover and register plugins from the plugins directory.

        This method attempts to import known plugin modules and register them.
        """
        if self._initialized:
            return

        # List of known plugin modules to import
        # Note: DesignKitPlugin is not included as it's not part of the content pipeline
        # It's used separately for generating DesignKitConfig
        plugin_modules = [
            (
                "marketing_project.plugins.transcript_preprocessing_approval",
                "TranscriptPreprocessingApprovalPlugin",
            ),
            (
                "marketing_project.plugins.blog_post_preprocessing_approval",
                "BlogPostPreprocessingApprovalPlugin",
            ),
            ("marketing_project.plugins.seo_keywords", "SEOKeywordsPlugin"),
            ("marketing_project.plugins.marketing_brief", "MarketingBriefPlugin"),
            ("marketing_project.plugins.article_generation", "ArticleGenerationPlugin"),
            ("marketing_project.plugins.seo_optimization", "SEOOptimizationPlugin"),
            ("marketing_project.plugins.suggested_links", "SuggestedLinksPlugin"),
            ("marketing_project.plugins.content_formatting", "ContentFormattingPlugin"),
        ]

        for module_path, class_name in plugin_modules:
            try:
                module = import_module(module_path)
                plugin_class = getattr(module, class_name, None)

                if plugin_class and issubclass(plugin_class, PipelineStepPlugin):
                    plugin_instance = plugin_class()
                    self.register(plugin_instance)
                    logger.info(f"Auto-discovered plugin: {plugin_instance.step_name}")
                else:
                    logger.debug(
                        f"Plugin class {class_name} not found in {module_path}"
                    )
            except ImportError as e:
                logger.debug(f"Could not import plugin from {module_path}: {e}")
            except Exception as e:
                logger.warning(f"Error loading plugin from {module_path}: {e}")

        self._initialized = True

    def validate_dependencies(self) -> tuple[bool, list[str]]:
        """
        Validate plugin dependencies using dependency graph.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if self._validated:
            return True, []

        plugins = list(self._plugins.values())
        if not plugins:
            return True, []

        graph = DependencyGraph(plugins)
        is_valid, errors = graph.validate()

        if is_valid:
            self._validated = True
            logger.info("Plugin dependency validation passed")
        else:
            logger.error("Plugin dependency validation failed:")
            for error in errors:
                logger.error(f"  - {error}")

        return is_valid, errors


# Global registry instance
_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.

    Returns:
        PluginRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
        _registry.auto_discover()
        # Validate dependencies on first access
        is_valid, errors = _registry.validate_dependencies()
        if not is_valid:
            logger.warning(
                f"Plugin dependency validation found {len(errors)} issue(s). "
                "Pipeline may fail at runtime."
            )
    return _registry


def reset_registry():
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry = None
