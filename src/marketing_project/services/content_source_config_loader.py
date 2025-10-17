"""
Content source configuration loader for Marketing Project.

This module loads content source configurations from YAML files and environment variables.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from marketing_project.core.content_sources import (
    APISourceConfig,
    ContentSourceType,
    DatabaseSourceConfig,
    FileSourceConfig,
    RSSSourceConfig,
    SocialMediaSourceConfig,
    WebhookSourceConfig,
    WebScrapingSourceConfig,
)

logger = logging.getLogger("marketing_project.services.content_source_config_loader")


class ContentSourceConfigLoader:
    """Loads content source configurations from various sources."""

    def __init__(self, config_file: Optional[str] = None):
        # Default to the config file in the marketing_project package
        if config_file is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            config_file = os.path.join(base_dir, "config", "pipeline.yml")
        self.config_file = config_file
        self.env_prefix = "CONTENT_SOURCE_"

    def load_configs(self) -> List[Dict[str, Any]]:
        """Load content source configurations from YAML and environment."""
        configs = []

        # Load from YAML file
        yaml_configs = self._load_from_yaml()
        configs.extend(yaml_configs)

        # Load from environment variables
        env_configs = self._load_from_environment()
        configs.extend(env_configs)

        return configs

    def _load_from_yaml(self) -> List[Dict[str, Any]]:
        """Load configurations from YAML file."""
        configs = []

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    data = yaml.safe_load(f)

                content_sources = data.get("content_sources", {})
                if content_sources.get("enabled", True):
                    default_sources = content_sources.get("default_sources", [])

                    for source_config in default_sources:
                        if source_config.get("enabled", True):
                            config = self._convert_yaml_to_config(source_config)
                            if config:
                                configs.append(config)

        except Exception as e:
            logger.error(f"Failed to load YAML configuration: {e}")

        return configs

    def _load_from_environment(self) -> List[Dict[str, Any]]:
        """Load configurations from environment variables."""
        configs = []

        # File source from environment
        content_dir = os.getenv("CONTENT_DIR")
        if content_dir:
            config = {
                "name": "env_file_source",
                "type": "file",
                "enabled": True,
                "config": {"file_paths": [content_dir], "watch_directory": True},
            }
            configs.append(config)

        # API source from environment
        api_url = os.getenv("CONTENT_API_URL")
        if api_url:
            config = {
                "name": "env_api_source",
                "type": "api",
                "enabled": True,
                "config": {
                    "base_url": api_url,
                    "endpoints": ["/content"],
                    "auth_type": "api_key",
                    "auth_config": {
                        "key_name": "X-API-Key",
                        "key_value": os.getenv("CONTENT_API_KEY", ""),
                    },
                },
            }
            configs.append(config)

        # Database source from environment
        db_url = os.getenv("CONTENT_DATABASE_URL")
        if db_url:
            config = {
                "name": "env_database_source",
                "type": "database",
                "enabled": True,
                "config": {
                    "connection_string": db_url,
                    "query": "SELECT * FROM content",
                },
            }
            configs.append(config)

        return configs

    def _convert_yaml_to_config(
        self, yaml_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Convert YAML configuration to internal format."""
        try:
            source_type = yaml_config.get("type", "").lower()
            config_data = yaml_config.get("config", {})

            # Substitute environment variables
            config_data = self._substitute_env_vars(config_data)

            return {
                "name": yaml_config.get("name"),
                "type": source_type,
                "enabled": yaml_config.get("enabled", True),
                "config": config_data,
            }

        except Exception as e:
            logger.error(f"Failed to convert YAML config: {e}")
            return None

    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration."""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif (
            isinstance(config, str) and config.startswith("${") and config.endswith("}")
        ):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config

    def create_source_configs(self) -> List[Any]:
        """Create actual source configuration objects."""
        configs = self.load_configs()
        source_configs = []

        for config in configs:
            try:
                source_type = config.get("type")
                config_data = config.get("config", {})

                if source_type == "file":
                    source_config = FileSourceConfig(
                        name=config["name"],
                        source_type=ContentSourceType.FILE,
                        **config_data,
                    )
                elif source_type == "api":
                    source_config = APISourceConfig(
                        name=config["name"],
                        source_type=ContentSourceType.API,
                        **config_data,
                    )
                elif source_type == "database":
                    source_config = DatabaseSourceConfig(
                        name=config["name"],
                        source_type=ContentSourceType.DATABASE,
                        **config_data,
                    )
                elif source_type == "web_scraping":
                    source_config = WebScrapingSourceConfig(
                        name=config["name"],
                        source_type=ContentSourceType.WEB_SCRAPING,
                        **config_data,
                    )
                elif source_type == "webhook":
                    source_config = WebhookSourceConfig(
                        name=config["name"],
                        source_type=ContentSourceType.WEBHOOK,
                        **config_data,
                    )
                elif source_type == "rss":
                    source_config = RSSSourceConfig(
                        name=config["name"],
                        source_type=ContentSourceType.RSS,
                        **config_data,
                    )
                elif source_type == "social_media":
                    source_config = SocialMediaSourceConfig(
                        name=config["name"],
                        source_type=ContentSourceType.SOCIAL_MEDIA,
                        **config_data,
                    )
                else:
                    logger.warning(f"Unknown source type: {source_type}")
                    continue

                source_configs.append(source_config)

            except Exception as e:
                logger.error(
                    f"Failed to create source config for {config.get('name')}: {e}"
                )
                continue

        return source_configs
