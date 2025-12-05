"""
Tests for context registry service.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.services.context_registry import (
    ContextReference,
    ContextRegistry,
    get_context_registry,
)


@pytest.fixture
def temp_registry_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def context_registry(temp_registry_dir):
    """Create a ContextRegistry instance for testing."""
    return ContextRegistry(base_dir=temp_registry_dir, enable_compression=False)


def test_context_reference_to_dict():
    """Test ContextReference.to_dict method."""
    ref = ContextReference(
        job_id="job-1",
        step_name="seo_keywords",
        step_number=1,
        execution_context_id="0",
        timestamp="2024-01-01T00:00:00Z",
        compressed=False,
    )

    data = ref.to_dict()

    assert data["job_id"] == "job-1"
    assert data["step_name"] == "seo_keywords"
    assert data["step_number"] == 1


def test_context_reference_from_dict():
    """Test ContextReference.from_dict method."""
    data = {
        "job_id": "job-1",
        "step_name": "seo_keywords",
        "step_number": 1,
        "execution_context_id": "0",
        "timestamp": "2024-01-01T00:00:00Z",
        "compressed": False,
    }

    ref = ContextReference.from_dict(data)

    assert ref.job_id == "job-1"
    assert ref.step_name == "seo_keywords"


def test_context_registry_initialization(temp_registry_dir):
    """Test ContextRegistry initialization."""
    registry = ContextRegistry(base_dir=temp_registry_dir)

    assert registry.base_dir == Path(temp_registry_dir)
    assert registry.base_dir.exists()


@pytest.mark.asyncio
async def test_register_step_output(context_registry):
    """Test registering step output."""
    context_data = {"seo_keywords": {"main_keyword": "test"}}

    ref = await context_registry.register_step_output(
        job_id="job-1",
        step_name="seo_keywords",
        step_number=1,
        execution_context_id="0",
        output_data=context_data,
    )

    assert ref is not None
    assert ref.job_id == "job-1"
    assert ref.step_name == "seo_keywords"


@pytest.mark.asyncio
async def test_resolve_context(context_registry):
    """Test resolving context."""
    context_data = {"seo_keywords": {"main_keyword": "test"}}

    ref = await context_registry.register_step_output(
        job_id="job-1",
        step_name="seo_keywords",
        step_number=1,
        execution_context_id="0",
        output_data=context_data,
    )

    loaded_data = await context_registry.resolve_context(ref)

    assert loaded_data is not None
    assert isinstance(loaded_data, dict)


@pytest.mark.asyncio
async def test_get_full_history(context_registry):
    """Test getting full history for a job."""
    await context_registry.register_step_output(
        "job-1", "seo_keywords", 1, "0", {"data": "test1"}
    )
    await context_registry.register_step_output(
        "job-1", "marketing_brief", 2, "0", {"data": "test2"}
    )

    history = await context_registry.get_full_history("job-1")

    assert isinstance(history, dict)


@pytest.mark.asyncio
async def test_get_context(context_registry):
    """Test getting context."""
    context_data = {"data": "test"}
    ref = await context_registry.register_step_output(
        "job-1", "seo_keywords", 1, "0", context_data
    )

    loaded = await context_registry.get_context(
        "job-1", "seo_keywords", execution_context_id="0"
    )

    assert loaded is not None


def test_get_context_registry_singleton():
    """Test that get_context_registry returns a singleton."""
    registry1 = get_context_registry()
    registry2 = get_context_registry()

    assert registry1 is registry2
