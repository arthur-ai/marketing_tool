"""
Validation helpers for API models.

This module contains validation functions used across the API models.
"""


def validate_content_length(content: str) -> bool:
    """
    Validate content length is within acceptable limits.

    Args:
        content: Content string to validate

    Returns:
        bool: True if content length is valid, False otherwise
    """
    if not content:
        return False
    min_length = 10  # Minimum content length
    max_length = 100000  # Maximum content length
    return min_length <= len(content) <= max_length


def validate_api_key_format(api_key: str) -> bool:
    """
    Validate API key format.

    Args:
        api_key: API key string to validate

    Returns:
        bool: True if API key format is valid, False otherwise
    """
    if not api_key:
        return False
    # Basic format validation - should be at least 32 characters
    return len(api_key) >= 32 and api_key.replace("-", "").replace("_", "").isalnum()
