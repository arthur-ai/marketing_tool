"""
Tests for plugin registry service.
"""

from unittest.mock import MagicMock

import pytest

from marketing_project.plugins.base import PipelineStepPlugin
from marketing_project.plugins.registry import PluginRegistry, get_plugin_registry


class TestPlugin(PipelineStepPlugin):
    """Test plugin implementation."""

    step_name = "test_step"
    step_number = 1

    @property
    def response_model(self):
        from marketing_project.models.pipeline_steps import SEOKeywordsResult

        return SEOKeywordsResult

    def get_required_context_keys(self):
        return ["input_content"]

    def validate_context(self, context):
        return "input_content" in context

    async def execute(self, context, pipeline=None, job_id=None):
        from marketing_project.models.pipeline_steps import SEOKeywordsResult

        return SEOKeywordsResult(
            main_keyword="test",
            primary_keywords=["test1"],
            confidence_score=0.9,
        )


@pytest.fixture
def plugin_registry():
    """Create a PluginRegistry instance."""
    return PluginRegistry()


def test_register(plugin_registry):
    """Test register method."""
    plugin = TestPlugin()
    plugin_registry.register(plugin)

    assert plugin_registry.get_plugin("test_step") is not None


def test_get_plugin(plugin_registry):
    """Test get_plugin method."""
    plugin = TestPlugin()
    plugin_registry.register(plugin)

    retrieved = plugin_registry.get_plugin("test_step")

    assert retrieved is not None
    assert retrieved.step_name == "test_step"


def test_get_plugin_by_number(plugin_registry):
    """Test get_plugin_by_number method."""
    plugin = TestPlugin()
    plugin_registry.register(plugin)

    retrieved = plugin_registry.get_plugin_by_number(1)

    assert retrieved is not None
    assert retrieved.step_number == 1


def test_get_all_plugins(plugin_registry):
    """Test get_all_plugins method."""
    plugin = TestPlugin()
    plugin_registry.register(plugin)

    plugins = plugin_registry.get_all_plugins()

    assert isinstance(plugins, dict)
    assert "test_step" in plugins


def test_get_plugins_in_order(plugin_registry):
    """Test get_plugins_in_order method."""
    plugin = TestPlugin()
    plugin_registry.register(plugin)

    plugins = plugin_registry.get_plugins_in_order()

    assert isinstance(plugins, list)
    assert len(plugins) >= 1


def test_validate_dependencies(plugin_registry):
    """Test validate_dependencies method."""
    plugin = TestPlugin()
    plugin_registry.register(plugin)

    valid, errors = plugin_registry.validate_dependencies()

    assert isinstance(valid, bool)
    assert isinstance(errors, list)


def test_get_plugin_registry_singleton():
    """Test that get_plugin_registry returns a singleton."""
    registry1 = get_plugin_registry()
    registry2 = get_plugin_registry()

    assert registry1 is registry2
