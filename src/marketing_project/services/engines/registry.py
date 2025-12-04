"""
Engine registry for managing and retrieving engines.

The registry provides a central place to register and retrieve engines by type.
"""

import logging
from typing import Dict, Optional

from marketing_project.services.engines.base import Engine

logger = logging.getLogger(__name__)


class EngineRegistry:
    """
    Registry for managing engines.

    Engines are registered by type name (e.g., 'llm', 'local_semantic')
    and can be retrieved later for use in composers.
    """

    def __init__(self):
        """Initialize the registry."""
        self._engines: Dict[str, Engine] = {}

    def register(self, engine_type: str, engine: Engine) -> None:
        """
        Register an engine with the registry.

        Args:
            engine_type: Type identifier for the engine (e.g., 'llm', 'local_semantic')
            engine: Engine instance to register
        """
        if engine_type in self._engines:
            logger.warning(
                f"Engine type '{engine_type}' already registered. Overwriting."
            )
        self._engines[engine_type] = engine
        logger.debug(f"Registered engine type: {engine_type}")

    def get(self, engine_type: str) -> Optional[Engine]:
        """
        Get an engine by type.

        Args:
            engine_type: Type identifier for the engine

        Returns:
            Engine instance if found, None otherwise
        """
        return self._engines.get(engine_type)

    def has(self, engine_type: str) -> bool:
        """
        Check if an engine type is registered.

        Args:
            engine_type: Type identifier to check

        Returns:
            True if registered, False otherwise
        """
        return engine_type in self._engines

    def list_types(self) -> list[str]:
        """
        List all registered engine types.

        Returns:
            List of engine type identifiers
        """
        return list(self._engines.keys())


# Global registry instance
_registry = EngineRegistry()


def get_engine(engine_type: str) -> Optional[Engine]:
    """
    Get an engine from the global registry.

    Args:
        engine_type: Type identifier for the engine

    Returns:
        Engine instance if found, None otherwise
    """
    return _registry.get(engine_type)


def register_engine(engine_type: str, engine: Engine) -> None:
    """
    Register an engine with the global registry.

    Args:
        engine_type: Type identifier for the engine
        engine: Engine instance to register
    """
    _registry.register(engine_type, engine)


def get_registry() -> EngineRegistry:
    """
    Get the global registry instance.

    Returns:
        The global EngineRegistry instance
    """
    return _registry
