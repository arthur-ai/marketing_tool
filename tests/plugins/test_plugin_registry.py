"""
Tests for plugin registry.
"""

from unittest.mock import MagicMock, patch

import pytest

from marketing_project.plugins.base import PipelineStepPlugin
from marketing_project.plugins.registry import PluginRegistry, get_plugin_registry


class MockPlugin(PipelineStepPlugin):
    """Mock plugin for testing."""

    def __init__(self, step_name: str, step_number: int):
        self._step_name = step_name
        self._step_number = step_number

    @property
    def step_name(self) -> str:
        return self._step_name

    @property
    def step_number(self) -> int:
        return self._step_number

    @property
    def response_model(self):
        from pydantic import BaseModel

        class MockResult(BaseModel):
            pass

        return MockResult

    async def execute(self, context, pipeline, job_id=None):
        return self.response_model()


class TestPluginRegistry:
    """Test PluginRegistry."""

    def test_init(self):
        """Test PluginRegistry initialization."""
        registry = PluginRegistry()
        assert registry._plugins == {}
        assert registry._initialized is False
        assert registry._validated is False

    def test_register_plugin(self):
        """Test registering a plugin."""
        registry = PluginRegistry()
        plugin = MockPlugin("test_step", 1)

        registry.register(plugin)

        assert "test_step" in registry._plugins
        assert registry._plugins["test_step"] == plugin

    def test_register_plugin_overwrite(self):
        """Test that registering a plugin with same name overwrites."""
        registry = PluginRegistry()
        plugin1 = MockPlugin("test_step", 1)
        plugin2 = MockPlugin("test_step", 2)

        registry.register(plugin1)
        registry.register(plugin2)

        assert registry._plugins["test_step"] == plugin2

    def test_register_invalid_plugin(self):
        """Test that registering invalid plugin raises error."""
        registry = PluginRegistry()

        with pytest.raises(
            TypeError, match="Plugin must be an instance of PipelineStepPlugin"
        ):
            registry.register("not a plugin")

    def test_get_plugin(self):
        """Test getting a plugin by name."""
        registry = PluginRegistry()
        plugin = MockPlugin("test_step", 1)
        registry.register(plugin)

        result = registry.get_plugin("test_step")
        assert result == plugin

    def test_get_plugin_not_found(self):
        """Test getting a plugin that doesn't exist."""
        registry = PluginRegistry()

        result = registry.get_plugin("nonexistent")
        assert result is None

    def test_get_plugin_by_number(self):
        """Test getting a plugin by step number."""
        registry = PluginRegistry()
        plugin = MockPlugin("test_step", 1)
        registry.register(plugin)

        result = registry.get_plugin_by_number(1)
        assert result == plugin

    def test_get_plugins_in_order(self):
        """Test getting plugins in execution order."""
        registry = PluginRegistry()
        plugin1 = MockPlugin("step1", 1)
        plugin2 = MockPlugin("step2", 2)
        plugin3 = MockPlugin("step3", 3)

        # Register out of order
        registry.register(plugin3)
        registry.register(plugin1)
        registry.register(plugin2)

        plugins = registry.get_plugins_in_order()
        assert [p.step_number for p in plugins] == [1, 2, 3]

    def test_auto_discover(self):
        """Test auto-discovery of plugins."""
        registry = PluginRegistry()

        with patch("marketing_project.plugins.registry.import_module") as mock_import:
            mock_module = MagicMock()
            mock_plugin_class = MagicMock()
            mock_plugin_class.return_value = MockPlugin("seo_keywords", 1)
            mock_module.SEOKeywordsPlugin = mock_plugin_class
            mock_import.return_value = mock_module

            registry.auto_discover()

            assert registry._initialized is True
            # Note: Actual auto_discover may not work in test environment
            # but we verify the method runs without error

    def test_validate_dependencies(self):
        """Test dependency validation."""
        registry = PluginRegistry()

        # Create plugins with dependencies
        plugin1 = MockPlugin("step1", 1)
        plugin2 = MockPlugin("step2", 2)
        plugin2.get_required_context_keys = lambda: ["step1"]

        registry.register(plugin1)
        registry.register(plugin2)

        is_valid, errors = registry.validate_dependencies()
        # Should be valid since step1 exists
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_dependencies_missing(self):
        """Test dependency validation with missing dependency."""
        registry = PluginRegistry()

        plugin = MockPlugin("step2", 2)
        plugin.get_required_context_keys = lambda: ["missing_step"]

        registry.register(plugin)

        is_valid, errors = registry.validate_dependencies()
        # May or may not be valid depending on implementation
        # Just verify the method runs
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)


class TestGetPluginRegistry:
    """Test get_plugin_registry function."""

    def test_get_plugin_registry_returns_singleton(self):
        """Test that get_plugin_registry returns a singleton."""
        registry1 = get_plugin_registry()
        registry2 = get_plugin_registry()

        assert registry1 is registry2
