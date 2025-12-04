"""
Tests for the generic engine framework.
"""

from typing import Any, Dict, Optional

import pytest

from marketing_project.services.engines.base import Engine
from marketing_project.services.engines.composer import EngineComposer
from marketing_project.services.engines.registry import (
    EngineRegistry,
    get_engine,
    get_registry,
    register_engine,
)


class MockEngine(Engine):
    """Mock engine for testing."""

    def __init__(self, engine_type: str):
        self.engine_type = engine_type
        self.supported_operations = {"test_op"}

    def supports_operation(self, operation: str) -> bool:
        return operation in self.supported_operations

    async def execute(
        self,
        operation: str,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        pipeline: Optional[Any] = None,
    ) -> Any:
        return f"{self.engine_type}:{operation}:{inputs.get('value', 'default')}"


class TestEngineRegistry:
    """Tests for EngineRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving engines."""
        registry = EngineRegistry()
        engine = MockEngine("test")

        registry.register("test", engine)
        assert registry.get("test") == engine
        assert registry.has("test") is True

    def test_get_nonexistent(self):
        """Test getting non-existent engine."""
        registry = EngineRegistry()
        assert registry.get("nonexistent") is None
        assert registry.has("nonexistent") is False

    def test_list_types(self):
        """Test listing engine types."""
        registry = EngineRegistry()
        registry.register("engine1", MockEngine("engine1"))
        registry.register("engine2", MockEngine("engine2"))

        types = registry.list_types()
        assert "engine1" in types
        assert "engine2" in types

    def test_global_registry(self):
        """Test global registry functions."""
        engine = MockEngine("global_test")
        register_engine("global_test", engine)

        retrieved = get_engine("global_test")
        assert retrieved == engine

        registry = get_registry()
        assert registry.has("global_test") is True


class TestEngineComposer:
    """Tests for EngineComposer."""

    @pytest.mark.asyncio
    async def test_default_engine(self):
        """Test composer with default engine only."""
        engine1 = MockEngine("engine1")
        register_engine("engine1", engine1)

        composer = EngineComposer(default_engine_type="engine1")
        result = await composer.execute_operation(
            "field1", "test_op", {"value": "test"}, {}, None
        )

        assert result == "engine1:test_op:test"

    @pytest.mark.asyncio
    async def test_field_override(self):
        """Test composer with field override."""
        engine1 = MockEngine("engine1")
        engine2 = MockEngine("engine2")
        register_engine("engine1", engine1)
        register_engine("engine2", engine2)

        composer = EngineComposer(
            default_engine_type="engine1",
            field_overrides={"field2": "engine2"},
        )

        # Default engine
        result1 = await composer.execute_operation(
            "field1", "test_op", {"value": "test1"}, {}, None
        )
        assert result1 == "engine1:test_op:test1"

        # Override engine
        result2 = await composer.execute_operation(
            "field2", "test_op", {"value": "test2"}, {}, None
        )
        assert result2 == "engine2:test_op:test2"

    @pytest.mark.asyncio
    async def test_get_engine_type_for_field(self):
        """Test getting engine type for a field."""
        composer = EngineComposer(
            default_engine_type="engine1",
            field_overrides={"field2": "engine2"},
        )

        assert composer.get_engine_type_for_field("field1") == "engine1"
        assert composer.get_engine_type_for_field("field2") == "engine2"

    @pytest.mark.asyncio
    async def test_missing_engine(self):
        """Test composer with missing engine."""
        composer = EngineComposer(default_engine_type="nonexistent")

        with pytest.raises(ValueError, match="No engine found"):
            await composer.execute_operation("field1", "test_op", {}, {}, None)
