"""
Base plugin interface for pipeline steps.

This module defines the standard interface that all pipeline step plugins must implement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from marketing_project.plugins.context_utils import ContextTransformer

if TYPE_CHECKING:
    from marketing_project.plugins.protocols import PipelineProtocol

T = TypeVar("T", bound=BaseModel)


class PipelineStepPlugin(ABC):
    """
    Base class for all pipeline step plugins.

    Each plugin handles a specific step in the content pipeline and provides
    a standardized interface for execution.
    """

    @property
    @abstractmethod
    def step_name(self) -> str:
        """Return the name of this pipeline step (e.g., 'seo_keywords')."""
        pass

    @property
    @abstractmethod
    def step_number(self) -> int:
        """Return the step number in the pipeline (1-8)."""
        pass

    @property
    @abstractmethod
    def response_model(self) -> type[BaseModel]:
        """Return the Pydantic model class for this step's output."""
        pass

    @abstractmethod
    async def execute(
        self,
        context: Dict[str, Any],
        pipeline: "PipelineProtocol",  # FunctionPipeline instance
        job_id: Optional[str] = None,
    ) -> BaseModel:
        """
        Execute this pipeline step.

        Args:
            context: Accumulated context from previous steps and input content
            pipeline: FunctionPipeline instance for making API calls
            job_id: Optional job ID for tracking

        Returns:
            Pydantic model instance with step results
        """
        pass

    def get_required_context_keys(self) -> list[str]:
        """
        Return list of context keys required from previous steps.

        Override this method to specify dependencies.
        """
        return []

    def validate_context(self, context: Dict[str, Any]) -> bool:
        """
        Validate that required context is available.

        Args:
            context: Context dictionary to validate

        Returns:
            True if context is valid, False otherwise
        """
        required = self.get_required_context_keys()
        return all(key in context for key in required)

    def _get_context_model(
        self,
        context: Dict[str, Any],
        key: str,
        model_class: Type[T],
        default: Optional[Any] = None,
    ) -> Optional[T]:
        """
        Get a value from context and ensure it's a Pydantic model instance.

        Helper method that uses ContextTransformer to convert dicts to models.

        Args:
            context: Context dictionary
            key: Key to look up in context
            model_class: The Pydantic model class to convert to
            default: Default value if key not found (None if not provided)

        Returns:
            Instance of model_class or None if key not found and no default
        """
        return ContextTransformer.get_context_model(context, key, model_class, default)

    def _ensure_model(
        self, value: Union[Dict[str, Any], BaseModel], model_class: Type[T]
    ) -> T:
        """
        Ensure a value is a Pydantic model instance, converting from dict if needed.

        Helper method that uses ContextTransformer for conversion.

        Args:
            value: Either a dict or a Pydantic model instance
            model_class: The Pydantic model class to convert to

        Returns:
            Instance of model_class
        """
        return ContextTransformer.ensure_model(value, model_class)

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the context dictionary for prompt template rendering.

        Plugins should override this method to customize how context is prepared
        for their specific prompt templates. The default implementation prepares
        the context for template rendering.

        Args:
            context: Raw context dictionary

        Returns:
            Context dictionary ready for template rendering
        """
        return ContextTransformer.prepare_template_context(context)

    async def _execute_step(
        self,
        context: Dict[str, Any],
        pipeline: "PipelineProtocol",
        job_id: Optional[str] = None,
    ) -> BaseModel:
        """
        Execute the step using the common execution pattern.

        This method handles the common execution flow:
        1. Build prompt context
        2. Get user prompt from template
        3. Get system instruction
        4. Call pipeline's _call_function
        5. Return result

        Plugins can override _build_prompt_context() to customize prompt context.

        Args:
            context: Accumulated context from previous steps
            pipeline: Pipeline instance for making API calls
            job_id: Optional job ID for tracking

        Returns:
            Pydantic model instance with step results
        """
        # Build prompt context (plugins can override _build_prompt_context)
        prompt_context = self._build_prompt_context(context)

        # Get user prompt from template
        prompt = pipeline._get_user_prompt(self.step_name, prompt_context)

        # Get system instruction
        system_instruction = pipeline._get_system_instruction(self.step_name)

        # Get execution step number from context if available (dynamic numbering)
        # Otherwise fall back to static step_number (for backward compatibility)
        execution_step_number = context.get("_execution_step_number", self.step_number)

        # Execute the step using pipeline's _call_function
        result = await pipeline._call_function(
            prompt=prompt,
            system_instruction=system_instruction,
            response_model=self.response_model,
            step_name=self.step_name,
            step_number=execution_step_number,
            context=context,
            job_id=job_id,
        )

        return result
