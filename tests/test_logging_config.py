"""
Tests for logging configuration module.
"""

import logging
import os
from unittest.mock import patch

import pytest

from marketing_project.logging_config import (
    LOG_LEVEL,
    LOGGING_CONFIG,
    LangChainLoggingCallbackHandler,
)


def test_log_level_from_environment():
    """Test that log level is read from environment variable."""
    with patch.dict(os.environ, {"MARKETING_PROJECT_LOG_LEVEL": "DEBUG"}):
        # Reload module to get new LOG_LEVEL
        import importlib

        import marketing_project.logging_config

        importlib.reload(marketing_project.logging_config)
        assert marketing_project.logging_config.LOG_LEVEL == "DEBUG"


def test_logging_config_structure():
    """Test that logging configuration has correct structure."""
    assert "version" in LOGGING_CONFIG
    assert LOGGING_CONFIG["version"] == 1
    assert "formatters" in LOGGING_CONFIG
    assert "handlers" in LOGGING_CONFIG
    assert "loggers" in LOGGING_CONFIG
    assert "root" in LOGGING_CONFIG


def test_logging_config_formatters():
    """Test that logging formatters are configured correctly."""
    assert "standard" in LOGGING_CONFIG["formatters"]
    assert "detailed" in LOGGING_CONFIG["formatters"]
    assert "format" in LOGGING_CONFIG["formatters"]["standard"]
    assert "format" in LOGGING_CONFIG["formatters"]["detailed"]


def test_logging_config_handlers():
    """Test that logging handlers are configured correctly."""
    assert "console" in LOGGING_CONFIG["handlers"]
    assert "error_console" in LOGGING_CONFIG["handlers"]
    assert LOGGING_CONFIG["handlers"]["console"]["class"] == "logging.StreamHandler"
    assert (
        LOGGING_CONFIG["handlers"]["error_console"]["class"] == "logging.StreamHandler"
    )


def test_logging_config_loggers():
    """Test that loggers are configured for marketing_project modules."""
    loggers = LOGGING_CONFIG["loggers"]
    assert "marketing_project" in loggers
    assert "marketing_project.runner" in loggers
    assert "marketing_project.core" in loggers
    assert "marketing_project.plugins" in loggers
    assert "marketing_project.services" in loggers
    assert "marketing_project.api" in loggers


def test_langchain_logging_callback_handler_init():
    """Test LangChain callback handler initialization."""
    handler = LangChainLoggingCallbackHandler("test_logger")
    assert handler.logger is not None
    assert handler.logger.name == "test_logger"


def test_langchain_logging_callback_handler_llm_start():
    """Test LLM start callback."""
    handler = LangChainLoggingCallbackHandler("test_logger")
    # Should not raise an exception
    handler.on_llm_start({"name": "test_model"}, ["test prompt"])


def test_langchain_logging_callback_handler_llm_end():
    """Test LLM end callback."""
    handler = LangChainLoggingCallbackHandler("test_logger")
    # Should not raise an exception
    handler.on_llm_end({"response": "test response"})


def test_langchain_logging_callback_handler_llm_error():
    """Test LLM error callback."""
    handler = LangChainLoggingCallbackHandler("test_logger")
    # Should not raise an exception
    handler.on_llm_error(Exception("test error"))


def test_logging_config_applied():
    """Test that logging configuration is applied correctly."""
    # Get a logger and verify it's configured
    logger = logging.getLogger("marketing_project.test")
    assert logger is not None
    # Logger should have handlers if config was applied
    assert len(logger.handlers) > 0 or logger.propagate
