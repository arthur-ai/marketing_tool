"""
LLM-based SEO keywords extraction engine.

This engine wraps the existing LLM-based keyword extraction logic.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import SEOKeywordsResult
from marketing_project.plugins.base import PipelineStepPlugin
from marketing_project.services.engines.base import Engine

logger = logging.getLogger(__name__)


class LLMSEOKeywordsEngine(Engine):
    """
    LLM-based engine for SEO keywords extraction.

    This engine uses the existing LLM-based extraction via the plugin's
    _execute_step method, returning a complete SEOKeywordsResult.
    """

    def __init__(self, plugin: PipelineStepPlugin):
        """
        Initialize the LLM engine.

        Args:
            plugin: The SEOKeywordsPlugin instance to use for LLM calls
        """
        self.plugin = plugin

    def supports_operation(self, operation: str) -> bool:
        """
        Check if this engine supports an operation.

        The LLM engine supports 'extract_all' which returns a complete result.

        Args:
            operation: Name of the operation

        Returns:
            True if operation is 'extract_all', False otherwise
        """
        return operation == "extract_all"

    async def execute(
        self,
        operation: str,
        inputs: Dict[str, Any],
        context: Dict[str, Any],
        pipeline: Optional[Any] = None,
    ) -> SEOKeywordsResult:
        """
        Execute the LLM-based extraction.

        Args:
            operation: Must be 'extract_all'
            inputs: Should contain 'content' dict
            context: Execution context
            pipeline: FunctionPipeline instance (required)

        Returns:
            Complete SEOKeywordsResult from LLM

        Raises:
            ValueError: If operation is not 'extract_all' or pipeline is missing
        """
        if operation != "extract_all":
            raise ValueError(
                f"LLM engine only supports 'extract_all' operation, got '{operation}'"
            )

        if pipeline is None:
            raise ValueError("LLM engine requires pipeline instance")

        # Use the plugin's _execute_step method to get LLM result
        # This maintains backward compatibility with existing LLM extraction
        result = await self.plugin._execute_step(context, pipeline)
        return result
