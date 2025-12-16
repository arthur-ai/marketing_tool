"""
Base plugin interface for pipeline steps.

This module defines the standard interface that all pipeline step plugins must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from marketing_project.models.pipeline_steps import PipelineStepConfig
from marketing_project.plugins.context_utils import ContextTransformer

if TYPE_CHECKING:
    from marketing_project.plugins.protocols import PipelineProtocol

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class PipelineStepPlugin(ABC):
    """
    Base class for all pipeline step plugins.

    Each plugin handles a specific step in the content pipeline and provides
    a standardized interface for execution.
    """

    def __init__(self):
        """Initialize the plugin with optional model configuration."""
        self._model_config: Optional[PipelineStepConfig] = None

    @property
    def model_config(self) -> Optional[PipelineStepConfig]:
        """
        Get the model configuration for this step.

        Returns:
            PipelineStepConfig if set, None otherwise
        """
        return self._model_config

    @model_config.setter
    def model_config(self, value: Optional[PipelineStepConfig]):
        """Set the model configuration for this step."""
        self._model_config = value

    def get_model_config(
        self,
        default_model: str = "gpt-5.1",
        default_temperature: float = 0.7,
        default_max_retries: int = 2,
    ) -> PipelineStepConfig:
        """
        Get model configuration for this step with defaults applied.

        Args:
            default_model: Default model to use if not configured
            default_temperature: Default temperature to use if not configured
            default_max_retries: Default max retries to use if not configured

        Returns:
            PipelineStepConfig with step-specific values or defaults
        """
        if self._model_config:
            # Merge with defaults for any None values
            return PipelineStepConfig(
                step_name=self.step_name,
                model=self._model_config.model or default_model,
                temperature=(
                    self._model_config.temperature
                    if self._model_config.temperature is not None
                    else default_temperature
                ),
                max_retries=(
                    self._model_config.max_retries
                    if self._model_config.max_retries is not None
                    else default_max_retries
                ),
            )
        return PipelineStepConfig(
            step_name=self.step_name,
            model=default_model,
            temperature=default_temperature,
            max_retries=default_max_retries,
        )

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
        Validate that required context is available and has correct data types.

        Args:
            context: Context dictionary to validate

        Returns:
            True if context is valid, False otherwise
        """
        required = self.get_required_context_keys()
        return all(key in context for key in required)

    def validate_context_detailed(
        self, context: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate context with detailed error messages.

        Args:
            context: Context dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        required = self.get_required_context_keys()

        # Check presence
        for key in required:
            if key not in context:
                errors.append(f"Missing required context key: '{key}'")
                continue

            value = context[key]

            # Check data quality
            if value is None:
                errors.append(f"Context key '{key}' is None (required)")
            elif isinstance(value, str) and not value.strip():
                errors.append(f"Context key '{key}' is empty string (required)")
            elif isinstance(value, (list, dict)) and len(value) == 0:
                errors.append(
                    f"Context key '{key}' is empty {type(value).__name__} (required)"
                )

        return len(errors) == 0, errors

    def auto_fix_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to auto-fix common context validation issues.

        Args:
            context: Context dictionary to fix

        Returns:
            Fixed context dictionary
        """
        fixed_context = context.copy()
        required = self.get_required_context_keys()

        for key in required:
            if key not in fixed_context:
                # Try to infer from similar keys
                if key == "seo_keywords" and "keywords" in fixed_context:
                    fixed_context[key] = fixed_context["keywords"]
                    logger.debug(f"Auto-fixed: mapped 'keywords' to '{key}'")
                elif key == "content_type" and "type" in fixed_context:
                    fixed_context[key] = fixed_context["type"]
                    logger.debug(f"Auto-fixed: mapped 'type' to '{key}'")
                # Add more auto-fix rules as needed

        return fixed_context

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

    async def get_context_from_registry(
        self,
        job_id: str,
        required_keys: List[str],
        execution_context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get context from context registry for specified keys (on-demand loading).

        This method allows plugins to request specific context keys from the registry
        instead of receiving full context, enabling efficient context management.

        Args:
            job_id: Job identifier
            required_keys: List of context keys to retrieve
            execution_context_id: Optional execution context ID

        Returns:
            Dictionary with requested context keys
        """
        try:
            from marketing_project.services.context_registry import get_context_registry

            context_registry = get_context_registry()
            return await context_registry.query_context(
                job_id=job_id,
                keys=required_keys,
                execution_context_id=execution_context_id,
            )
        except Exception as e:
            logger.warning(f"Failed to get context from registry: {e}")
            return {}

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

        # Get system instruction (pass prompt_context for platform-specific templates)
        system_instruction = pipeline._get_system_instruction(
            self.step_name, context=prompt_context
        )

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

        # If result is ApprovalRequiredSentinel, propagate it up
        from marketing_project.processors.approval_helper import (
            ApprovalRequiredSentinel,
        )

        if isinstance(result, ApprovalRequiredSentinel):
            return result

        return result
