"""
Logging configuration for Marketing Project.

This module sets up PEP 282-compliant logging for the application, directing logs from different modules to separate files and not to the terminal/console.

Usage:
    Import this module at the entry point of your application (e.g., main.py or runner.py) before any other imports that use logging.
"""

import datetime
import logging
import logging.config
import os

from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.base import BaseCallbackHandler

# --- Directory and Log Level Setup ---
LOG_DIR = os.getenv("MARKETING_PROJECT_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_LEVEL = os.getenv("MARKETING_PROJECT_LOG_LEVEL", "DEBUG").upper()

# --- Timestamped Log File Helper ---
now_str = datetime.datetime.now().strftime("%Y-%m-%d")


def log_path(filename):
    return os.path.join(LOG_DIR, f"{filename}_{now_str}.log")


# --- Logging Configuration Dictionary ---
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "runner_file": {
            "level": LOG_LEVEL,
            "class": "logging.FileHandler",
            "filename": log_path("runner"),
            "formatter": "standard",
            "encoding": "utf8",
        },
        "agents_file": {
            "level": LOG_LEVEL,
            "class": "logging.FileHandler",
            "filename": log_path("agents"),
            "formatter": "standard",
            "encoding": "utf8",
        },
        "core_file": {
            "level": LOG_LEVEL,
            "class": "logging.FileHandler",
            "filename": log_path("core"),
            "formatter": "standard",
            "encoding": "utf8",
        },
        "plugins_file": {
            "level": LOG_LEVEL,
            "class": "logging.FileHandler",
            "filename": log_path("plugins"),
            "formatter": "standard",
            "encoding": "utf8",
        },
        "services_file": {
            "level": LOG_LEVEL,
            "class": "logging.FileHandler",
            "filename": log_path("services"),
            "formatter": "standard",
            "encoding": "utf8",
        },
        "langchain_file": {
            "level": LOG_LEVEL,
            "class": "logging.FileHandler",
            "filename": log_path("agents"),
            "formatter": "standard",
            "encoding": "utf8",
        },
    },
    "loggers": {
        "marketing_project.runner": {
            "handlers": ["runner_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.agents": {
            "handlers": ["agents_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.core": {
            "handlers": ["core_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.plugins": {
            "handlers": ["plugins_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "marketing_project.services": {
            "handlers": ["services_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "langchain": {
            "handlers": ["langchain_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "langchain_openai": {
            "handlers": ["langchain_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    },
    "root": {
        "handlers": [],  # No console handler
        "level": "WARNING",
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
