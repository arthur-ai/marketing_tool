"""
Context transformation utilities for pipeline plugins.

This module provides utilities for converting between dicts and Pydantic models,
preparing context for template rendering, and managing context transformations.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel

logger = logging.getLogger("marketing_project.plugins.context_utils")

T = TypeVar("T", bound=BaseModel)


class ContextTransformer:
    """
    Utility class for transforming context between different representations.

    Provides methods for converting dicts to Pydantic models, preparing context
    for template rendering, and managing context transformations.
    """

    @staticmethod
    def ensure_model(
        value: Union[Dict[str, Any], BaseModel], model_class: Type[T]
    ) -> T:
        """
        Ensure a value is a Pydantic model instance, converting from dict if needed.

        Args:
            value: Either a dict or a Pydantic model instance
            model_class: The Pydantic model class to convert to

        Returns:
            Instance of model_class

        Raises:
            ValueError: If value cannot be converted to model_class
        """
        if isinstance(value, model_class):
            return value
        elif isinstance(value, dict):
            try:
                return model_class(**value)
            except Exception as e:
                logger.error(f"Failed to convert dict to {model_class.__name__}: {e}")
                raise ValueError(
                    f"Cannot convert dict to {model_class.__name__}: {e}"
                ) from e
        else:
            raise ValueError(
                f"Expected dict or {model_class.__name__}, got {type(value).__name__}"
            )

    @staticmethod
    def get_context_model(
        context: Dict[str, Any],
        key: str,
        model_class: Type[T],
        default: Optional[Any] = None,
    ) -> Optional[T]:
        """
        Get a value from context and ensure it's a Pydantic model instance.

        Args:
            context: Context dictionary
            key: Key to look up in context
            model_class: The Pydantic model class to convert to
            default: Default value if key not found (None if not provided)

        Returns:
            Instance of model_class or None if key not found and no default
        """
        value = context.get(key, default)
        if value is None:
            return None
        return ContextTransformer.ensure_model(value, model_class)

    @staticmethod
    def prepare_template_context(context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context for Jinja2 template rendering.

        Converts Pydantic models to dicts and handles special cases like
        content truncation.

        Args:
            context: Raw context dictionary

        Returns:
            Context dictionary ready for template rendering
        """
        template_context = {}

        for key, value in context.items():
            if hasattr(value, "model_dump"):
                # Pydantic model - convert to dict
                template_context[key] = value.model_dump()
            elif hasattr(value, "__dict__") and not isinstance(
                value, (str, int, float, bool, list, dict, type(None))
            ):
                # Other objects with __dict__ - convert to dict
                template_context[key] = (
                    value.__dict__ if hasattr(value, "__dict__") else value
                )
            else:
                # Primitive types, dicts, lists - use as-is
                template_context[key] = value

        return template_context

    @staticmethod
    def merge_contexts(*contexts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple context dictionaries, with later contexts overriding earlier ones.

        Args:
            *contexts: Variable number of context dictionaries to merge

        Returns:
            Merged context dictionary
        """
        merged = {}
        for context in contexts:
            merged.update(context)
        return merged
