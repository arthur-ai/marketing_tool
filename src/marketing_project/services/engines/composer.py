"""
Engine composer for orchestrating multiple engines.

The composer allows mixing and matching different engines for different fields,
enabling per-field engine selection with default + override configuration.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.services.engines.base import Engine
from marketing_project.services.engines.registry import get_engine

logger = logging.getLogger(__name__)


class EngineComposer:
    """
    Composes multiple engines to populate different fields of a result model.

    Supports:
    - Default engine for all fields
    - Field-level overrides to use different engines for specific fields
    - Mix-and-match: some fields from LLM, others from local processing
    """

    def __init__(
        self,
        default_engine_type: str,
        field_overrides: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the composer.

        Args:
            default_engine_type: Default engine type to use (e.g., 'llm')
            field_overrides: Optional dict mapping field names to engine types
        """
        self.default_engine_type = default_engine_type
        self.field_overrides = field_overrides or {}
        self._default_engine: Optional[Engine] = None
        self._override_engines: Dict[str, Engine] = {}

    def _get_engine(self, engine_type: str) -> Optional[Engine]:
        """
        Get an engine instance, caching it for reuse.

        Args:
            engine_type: Type identifier for the engine

        Returns:
            Engine instance if found, None otherwise
        """
        engine = get_engine(engine_type)
        if engine is None:
            logger.error(f"Engine type '{engine_type}' not found in registry")
        return engine

    def _get_engine_for_field(self, field_name: str) -> Optional[Engine]:
        """
        Get the appropriate engine for a given field.

        Checks field overrides first, then falls back to default engine.

        Args:
            field_name: Name of the field to get engine for

        Returns:
            Engine instance if found, None otherwise
        """
        # Check for field override
        if field_name in self.field_overrides:
            override_type = self.field_overrides[field_name]
            if override_type not in self._override_engines:
                engine = self._get_engine(override_type)
                if engine:
                    self._override_engines[override_type] = engine
                return engine
            return self._override_engines[override_type]

        # Use default engine
        if self._default_engine is None:
            self._default_engine = self._get_engine(self.default_engine_type)
        return self._default_engine

    async def execute_operation(
        self,
        field_name: str,
        operation: str,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        pipeline: Optional[Any] = None,
    ) -> Any:
        """
        Execute an operation for a specific field using the appropriate engine.

        Args:
            field_name: Name of the field this operation is for
            operation: Name of the operation to execute
            inputs: Input data for the operation
            context: Execution context
            pipeline: Optional pipeline instance (for LLM engines)

        Returns:
            Result of the operation

        Raises:
            ValueError: If engine is not found or operation fails
        """
        engine = self._get_engine_for_field(field_name)
        if engine is None:
            raise ValueError(
                f"No engine found for field '{field_name}'. "
                f"Default: {self.default_engine_type}, "
                f"Override: {self.field_overrides.get(field_name)}"
            )

        if not engine.supports_operation(operation):
            raise ValueError(
                f"Engine for field '{field_name}' does not support operation '{operation}'"
            )

        try:
            result = await engine.execute(operation, inputs, context, pipeline)
            return result
        except Exception as e:
            logger.error(
                f"Error executing operation '{operation}' for field '{field_name}': {e}"
            )
            raise

    def get_engine_type_for_field(self, field_name: str) -> str:
        """
        Get the engine type that will be used for a given field.

        Args:
            field_name: Name of the field

        Returns:
            Engine type identifier
        """
        return self.field_overrides.get(field_name, self.default_engine_type)
