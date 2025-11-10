"""
Logging configuration for Marketing Project.

This module sets up PEP 282-compliant logging for the application,
directing all logs to stdout/stderr for Docker container logging.

Usage:
    Import this module at the entry point of your application (e.g., main.py or runner.py) before any other imports that use logging.
"""

import logging
import logging.config
import os
import sys

from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.base import BaseCallbackHandler

# --- Log Level Setup ---
LOG_LEVEL = os.getenv("MARKETING_PROJECT_LOG_LEVEL", "INFO").upper()


# --- Logging Configuration Dictionary ---
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "level": LOG_LEVEL,
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "standard",
        },
        "error_console": {
            "level": "ERROR",
            "class": "logging.StreamHandler",
            "stream": sys.stderr,
            "formatter": "detailed",
        },
    },
    "loggers": {
        # Marketing Project modules - all use console
        "marketing_project": {
            "handlers": ["console", "error_console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.runner": {
            "handlers": ["console", "error_console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.agents": {
            "handlers": ["console", "error_console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.core": {
            "handlers": ["console", "error_console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.plugins": {
            "handlers": ["console", "error_console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.services": {
            "handlers": ["console", "error_console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.api": {
            "handlers": ["console", "error_console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.middleware": {
            "handlers": ["console", "error_console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.processors": {
            "handlers": ["console", "error_console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        # LangChain - reduce noise in production
        "langchain": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "langchain_openai": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "langchain_core": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        # FastAPI/Uvicorn
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "fastapi": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console", "error_console"],
        "level": "INFO",
    },
}


# --- LangChain Callback Handler for Logging ---
class LangChainLoggingCallbackHandler(BaseCallbackHandler):
    def __init__(self, logger_name="langchain"):
        self.logger = logging.getLogger(logger_name)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.logger.info(f"[LLM Start] {serialized.get('name')} | Prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        self.logger.info(f"[LLM End] Response: {response}")

    def on_llm_error(self, error, **kwargs):
        self.logger.error(f"[LLM Error] {error}")


# --- Apply Logging Configuration ---
logging.config.dictConfig(LOGGING_CONFIG)
