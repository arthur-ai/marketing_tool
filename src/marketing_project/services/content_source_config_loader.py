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
    S3SourceConfig,
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
            logger.info(
                f"Loading content source configuration from: {self.config_file}"
            )

            if not os.path.exists(self.config_file):
                logger.warning(f"Configuration file not found: {self.config_file}")
                return configs

            with open(self.config_file, "r") as f:
                data = yaml.safe_load(f)

            content_sources = data.get("content_sources", {})
            logger.info(
                f"Content sources section found: enabled={content_sources.get('enabled', True)}"
            )

            if content_sources.get("enabled", True):
                default_sources = content_sources.get("default_sources", [])
                logger.info(
                    f"Found {len(default_sources)} default content sources in configuration"
                )

                # Check if we're in local deployment (no AWS_S3_BUCKET set)
                aws_s3_bucket = os.getenv("AWS_S3_BUCKET")
                is_local_deployment = not aws_s3_bucket

                if is_local_deployment:
                    logger.info(
                        "AWS_S3_BUCKET not set - detected local deployment. "
                        "Auto-enabling local_content source and disabling s3_content source."
                    )

                for source_config in default_sources:
                    source_name = source_config.get("name", "unknown")
                    source_enabled = source_config.get("enabled", True)

                    # Auto-enable local_content for local deployments
                    if is_local_deployment and source_name == "local_content":
                        source_enabled = True
                        logger.info(
                            f"Auto-enabling '{source_name}' for local deployment"
                        )
                    # Auto-disable s3_content for local deployments
                    elif is_local_deployment and source_name == "s3_content":
                        source_enabled = False
                        logger.info(
                            f"Auto-disabling '{source_name}' for local deployment (no AWS_S3_BUCKET)"
                        )

                    logger.info(
                        f"Processing source '{source_name}': enabled={source_enabled}"
                    )

                    if source_enabled:
                        config = self._convert_yaml_to_config(source_config)
                        if config:
                            configs.append(config)
                            logger.info(
                                f"Successfully loaded configuration for source '{source_name}'"
                            )
                        else:
                            logger.warning(
                                f"Failed to convert configuration for source '{source_name}'"
                            )
                    else:
                        logger.info(f"Skipping disabled source '{source_name}'")

            logger.info(f"Loaded {len(configs)} enabled content sources from YAML")

        except Exception as e:
            logger.error(f"Failed to load YAML configuration: {e}", exc_info=True)

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
            source_name = yaml_config.get("name", "unknown")
            source_type = yaml_config.get("type", "").lower()
            config_data = yaml_config.get("config", {})

            logger.debug(f"Converting config for '{source_name}' (type: {source_type})")
            logger.debug(f"  Original config data: {config_data}")

            # Substitute environment variables
            config_data = self._substitute_env_vars(config_data)

            logger.debug(f"  After env substitution: {config_data}")

            return {
                "name": source_name,
                "type": source_type,
                "enabled": yaml_config.get("enabled", True),
                "config": config_data,
            }

        except Exception as e:
            logger.error(f"Failed to convert YAML config: {e}", exc_info=True)
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
        logger.info("Creating source configuration objects...")
        configs = self.load_configs()
        source_configs = []

        logger.info(f"Processing {len(configs)} configurations")

        for config in configs:
            try:
                source_name = config.get("name", "unknown")
                source_type = config.get("type")
                config_data = config.get("config", {})

                logger.info(f"Creating {source_type} source config for '{source_name}'")

                if source_type == "file":
                    source_config = FileSourceConfig(
                        name=config["name"],
                        source_type=ContentSourceType.FILE,
                        **config_data,
                    )
                    logger.info(f"  File paths: {source_config.file_paths}")
                    logger.info(f"  File patterns: {source_config.file_patterns}")
                    logger.info(f"  Watch directory: {source_config.watch_directory}")
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
                elif source_type == "s3":
                    source_config = S3SourceConfig(
                        name=config["name"],
                        source_type=ContentSourceType.S3,
                        **config_data,
                    )
                    logger.info(f"  Bucket: {source_config.bucket_name or 'from env'}")
                    logger.info(f"  Prefix: {source_config.prefix}")
                    logger.info(f"  File patterns: {source_config.file_patterns}")
                else:
                    logger.warning(f"Unknown source type: {source_type}")
                    continue

                source_configs.append(source_config)
                logger.info(f"Successfully created source config for '{source_name}'")

            except Exception as e:
                logger.error(
                    f"Failed to create source config for {config.get('name')}: {e}",
                    exc_info=True,
                )
                continue

        logger.info(f"Created {len(source_configs)} source configuration objects")
        return source_configs
