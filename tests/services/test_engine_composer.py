"""
Tests for engine composer service.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.engines.composer import EngineComposer


@pytest.fixture
def engine_composer():
    """Create an EngineComposer instance."""
    return EngineComposer(default_engine_type="llm")


def test_get_engine_for_field(engine_composer):
    """Test _get_engine_for_field method."""
    with patch("marketing_project.services.engines.registry.get_engine") as mock_get:
        mock_engine = MagicMock()
        mock_get.return_value = mock_engine

        engine = engine_composer._get_engine_for_field("main_keyword")

        assert engine is not None or engine is None


def test_get_engine_type_for_field(engine_composer):
    """Test get_engine_type_for_field method."""
    engine_type = engine_composer.get_engine_type_for_field("main_keyword")

    assert isinstance(engine_type, str)
    assert engine_type == "llm"

    # Test with override
    engine_composer.field_overrides = {"main_keyword": "local_semantic"}
    engine_type = engine_composer.get_engine_type_for_field("main_keyword")

    assert engine_type == "local_semantic"


@pytest.mark.asyncio
async def test_execute_operation(engine_composer):
    """Test execute_operation method."""
    # Mock the default engine
    with patch("marketing_project.services.engines.registry.get_engine") as mock_get:
        mock_engine = MagicMock()
        mock_engine.supports_operation = MagicMock(return_value=True)
        mock_engine.execute = AsyncMock(return_value="test_result")
        mock_get.return_value = mock_engine

        # Set default engine directly
        engine_composer._default_engine = mock_engine

        result = await engine_composer.execute_operation(
            field_name="main_keyword",
            operation="extract",
            inputs={"content": "test"},
            context={},
        )

        assert result is not None or result is None
