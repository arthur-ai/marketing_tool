"""
Base engine interface for the generic engine framework.

Engines provide a standardized way to execute operations that can be mixed and matched
across different fields of a result model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Engine(ABC):
    """
    Abstract base class for all engines.

    Engines provide a standardized interface for executing operations.
    Different engines can implement different strategies (LLM, local processing, etc.)
    for the same operations.
    """

    @abstractmethod
    async def execute(
        self,
        operation: str,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        pipeline: Optional[Any] = None,
    ) -> Any:
        """
        Execute an operation with the given inputs and context.

        Args:
            operation: Name of the operation to execute (e.g., 'extract_main_keyword')
            inputs: Input data for the operation (e.g., content, keywords)
            context: Execution context (e.g., job_id, step_name)
            pipeline: Optional pipeline instance (for LLM engines)

        Returns:
            Result of the operation (type depends on operation)

        Raises:
            NotImplementedError: If the operation is not supported by this engine
        """
        pass

    def supports_operation(self, operation: str) -> bool:
        """
        Check if this engine supports a given operation.

        Args:
            operation: Name of the operation to check

        Returns:
            True if the operation is supported, False otherwise
        """
        # Default implementation: engines should override this
        # to provide a list of supported operations
        return True
