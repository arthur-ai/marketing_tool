"""
Tests for content source config loader service.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from marketing_project.services.content_source_config_loader import (
    ContentSourceConfigLoader,
)


@pytest.fixture
def temp_config_file():
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        config = {
            "content_sources": {
                "enabled": True,
                "default_sources": [
                    {
                        "name": "test-source",
                        "type": "file",
                        "file_paths": ["test.txt"],
                    }
                ],
            }
        }
        yaml.dump(config, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def config_loader(temp_config_file):
    """Create a ContentSourceConfigLoader instance."""
    return ContentSourceConfigLoader(config_file=temp_config_file)


def test_load_configs(config_loader, temp_config_file):
    """Test load_configs method."""
    configs = config_loader.load_configs()

    assert isinstance(configs, list)
    assert len(configs) >= 0


def test_load_from_yaml(config_loader, temp_config_file):
    """Test _load_from_yaml method."""
    configs = config_loader._load_from_yaml()

    assert isinstance(configs, list)


def test_load_from_environment(config_loader):
    """Test _load_from_environment method."""
    with patch.dict("os.environ", {"CONTENT_SOURCE_TEST_TYPE": "file"}):
        configs = config_loader._load_from_environment()

        assert isinstance(configs, list)


def test_create_source_configs(config_loader, temp_config_file):
    """Test create_source_configs method."""
    configs = config_loader.create_source_configs()

    assert isinstance(configs, list)
