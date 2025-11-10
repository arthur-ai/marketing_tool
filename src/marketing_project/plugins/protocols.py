"""
Protocol definitions for plugin interfaces.

This module defines the Protocol interfaces that plugins use to interact with
the pipeline, ensuring type safety and clear contracts.
"""

from typing import Any, Dict, Optional, Protocol

from pydantic import BaseModel


class PipelineProtocol(Protocol):
    """
    Protocol defining the interface that FunctionPipeline must implement
    for plugin interaction.

    This protocol ensures type safety when plugins interact with the pipeline,
    providing autocomplete and type checking without tight coupling.
    """

    def _get_user_prompt(self, step_name: str, context: Dict[str, Any]) -> str:
        """
        Load user prompt from template and render with context.

        Args:
            step_name: Name of the step (e.g., "seo_keywords")
            context: Context variables for template rendering

        Returns:
            Rendered user prompt string
        """
        ...

    def _get_system_instruction(
        self, agent_name: str, context: Optional[Dict] = None
    ) -> str:
        """
        Load system instruction from template.

        Args:
            agent_name: Name of the agent (e.g., "seo_keywords")
            context: Optional context variables for template rendering

        Returns:
            System instruction string
        """
        ...

    async def _call_function(
        self,
        prompt: str,
        system_instruction: str,
        response_model: type[BaseModel],
        step_name: str,
        step_number: int,
        context: Optional[Dict] = None,
        max_retries: int = 2,
        job_id: Optional[str] = None,
    ) -> BaseModel:
        """
        Call OpenAI with structured output.

        Args:
            prompt: User prompt with content to process
            system_instruction: System instructions for this step
            response_model: Pydantic model defining expected output structure
            step_name: Name of the current step
            step_number: Step sequence number
            context: Additional context from previous steps
            max_retries: Maximum number of retry attempts
            job_id: Optional job ID for approval tracking

        Returns:
            Instance of response_model with structured data
        """
        ...
