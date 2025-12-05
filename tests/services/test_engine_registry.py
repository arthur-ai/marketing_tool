"""
Tests for engine registry service.
"""

import pytest

from marketing_project.services.engines.base import Engine
from marketing_project.services.engines.registry import (
    EngineRegistry,
    get_engine,
    register_engine,
)


@pytest.fixture
def engine_registry():
    """Create an EngineRegistry instance."""
    return EngineRegistry()


class TestEngine(Engine):
    """Test engine implementation."""

    def supports_operation(self, operation: str) -> bool:
        return operation == "test"

    async def execute(self, operation: str, inputs: dict, context: dict, pipeline=None):
        return {"result": "test"}


def test_register(engine_registry):
    """Test register method."""
    engine = TestEngine()
    engine_registry.register("test_engine", engine)

    assert engine_registry.has("test_engine") is True


def test_get(engine_registry):
    """Test get method."""
    engine = TestEngine()
    engine_registry.register("test_engine", engine)

    retrieved = engine_registry.get("test_engine")

    assert retrieved is not None
    assert retrieved == engine


def test_get_not_found(engine_registry):
    """Test get with non-existent engine."""
    engine = engine_registry.get("non_existent")

    assert engine is None


def test_has(engine_registry):
    """Test has method."""
    engine = TestEngine()
    engine_registry.register("test_engine", engine)

    assert engine_registry.has("test_engine") is True
    assert engine_registry.has("non_existent") is False


def test_list_types(engine_registry):
    """Test list_types method."""
    engine = TestEngine()
    engine_registry.register("test_engine", engine)

    types = engine_registry.list_types()

    assert isinstance(types, list)
    assert "test_engine" in types


def test_get_engine_global():
    """Test get_engine global function."""
    engine = TestEngine()
    register_engine("test_engine_global", engine)

    retrieved = get_engine("test_engine_global")

    assert retrieved is not None


def test_register_engine_global():
    """Test register_engine global function."""
    engine = TestEngine()
    register_engine("test_engine_global2", engine)

    retrieved = get_engine("test_engine_global2")

    assert retrieved is not None
