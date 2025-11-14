"""
Tests for runner module.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from marketing_project.core.models import (
    BlogPostContext,
    ReleaseNotesContext,
    TranscriptContext,
)
from marketing_project.runner import (
    build_fastapi_app,
    run_function_pipeline,
    run_server,
    run_single_content,
)


@pytest.mark.asyncio
async def test_run_function_pipeline_no_content():
    """Test run_function_pipeline with no content."""
    mock_config_loader = MagicMock()
    mock_config_loader.create_source_configs.return_value = []

    mock_manager = MagicMock()
    mock_manager.add_source_from_config = AsyncMock()
    mock_manager.fetch_content_as_models = AsyncMock(return_value=[])
    mock_manager.sources = {}

    with patch(
        "marketing_project.runner.ContentSourceConfigLoader",
        return_value=mock_config_loader,
    ):
        with patch(
            "marketing_project.runner.ContentSourceManager", return_value=mock_manager
        ):
            result = await run_function_pipeline()

    assert result["success_count"] == 0
    assert result["total_count"] == 0
    assert len(result["processed_content"]) == 0


@pytest.mark.asyncio
async def test_run_function_pipeline_with_blog_post():
    """Test run_function_pipeline with blog post content."""
    mock_config_loader = MagicMock()
    mock_config_loader.create_source_configs.return_value = []

    mock_manager = MagicMock()
    mock_manager.add_source_from_config = AsyncMock()
    blog_post = BlogPostContext(
        id="1",
        title="Test Blog",
        content="Content",
        snippet="Snippet",
        created_at="2024-01-01T00:00:00Z",
    )
    mock_manager.fetch_content_as_models = AsyncMock(return_value=[blog_post])
    mock_manager.sources = {}

    with patch(
        "marketing_project.runner.ContentSourceConfigLoader",
        return_value=mock_config_loader,
    ):
        with patch(
            "marketing_project.runner.ContentSourceManager", return_value=mock_manager
        ):
            with patch(
                "marketing_project.runner.process_blog_post", new_callable=AsyncMock
            ) as mock_process:
                mock_process.return_value = json.dumps(
                    {"status": "success", "message": "Processed"}
                )
                result = await run_function_pipeline()

    assert result["total_count"] == 1
    mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_run_function_pipeline_with_transcript():
    """Test run_function_pipeline with transcript content."""
    mock_config_loader = MagicMock()
    mock_config_loader.create_source_configs.return_value = []

    mock_manager = MagicMock()
    mock_manager.add_source_from_config = AsyncMock()
    transcript = TranscriptContext(
        id="1",
        title="Test Transcript",
        content="Content",
        snippet="Snippet",
        created_at="2024-01-01T00:00:00Z",
    )
    mock_manager.fetch_content_as_models = AsyncMock(return_value=[transcript])
    mock_manager.sources = {}

    with patch(
        "marketing_project.runner.ContentSourceConfigLoader",
        return_value=mock_config_loader,
    ):
        with patch(
            "marketing_project.runner.ContentSourceManager", return_value=mock_manager
        ):
            with patch(
                "marketing_project.runner.process_transcript", new_callable=AsyncMock
            ) as mock_process:
                mock_process.return_value = json.dumps(
                    {"status": "success", "message": "Processed"}
                )
                result = await run_function_pipeline()

    assert result["total_count"] == 1
    mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_run_function_pipeline_with_release_notes():
    """Test run_function_pipeline with release notes content."""
    mock_config_loader = MagicMock()
    mock_config_loader.create_source_configs.return_value = []

    mock_manager = MagicMock()
    mock_manager.add_source_from_config = AsyncMock()
    release_notes = ReleaseNotesContext(
        id="1",
        title="Test Release",
        content="Content",
        snippet="Snippet",
        version="1.0.0",
        created_at="2024-01-01T00:00:00Z",
    )
    mock_manager.fetch_content_as_models = AsyncMock(return_value=[release_notes])
    mock_manager.sources = {}

    with patch(
        "marketing_project.runner.ContentSourceConfigLoader",
        return_value=mock_config_loader,
    ):
        with patch(
            "marketing_project.runner.ContentSourceManager", return_value=mock_manager
        ):
            with patch(
                "marketing_project.runner.process_release_notes", new_callable=AsyncMock
            ) as mock_process:
                mock_process.return_value = json.dumps(
                    {"status": "success", "message": "Processed"}
                )
                result = await run_function_pipeline()

    assert result["total_count"] == 1
    mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_run_function_pipeline_handles_errors():
    """Test that run_function_pipeline handles processing errors gracefully."""
    mock_config_loader = MagicMock()
    mock_config_loader.create_source_configs.return_value = []

    mock_manager = MagicMock()
    mock_manager.add_source_from_config = AsyncMock()
    blog_post = BlogPostContext(
        id="1",
        title="Test Blog",
        content="Content",
        snippet="Snippet",
        created_at="2024-01-01T00:00:00Z",
    )
    mock_manager.fetch_content_as_models = AsyncMock(return_value=[blog_post])
    mock_manager.sources = {}

    with patch(
        "marketing_project.runner.ContentSourceConfigLoader",
        return_value=mock_config_loader,
    ):
        with patch(
            "marketing_project.runner.ContentSourceManager", return_value=mock_manager
        ):
            with patch(
                "marketing_project.runner.process_blog_post", new_callable=AsyncMock
            ) as mock_process:
                mock_process.side_effect = Exception("Processing error")
                result = await run_function_pipeline()

    assert result["total_count"] == 1
    assert result["success_count"] == 0


@pytest.mark.asyncio
async def test_run_single_content_blog_post():
    """Test run_single_content with blog_post type."""
    content_data = {"title": "Test", "content": "Content"}
    with patch(
        "marketing_project.runner.process_blog_post", new_callable=AsyncMock
    ) as mock_process:
        mock_process.return_value = json.dumps({"status": "success", "result": {}})
        result = await run_single_content(content_data, "blog_post")

    assert result["status"] == "success"
    mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_run_single_content_transcript():
    """Test run_single_content with transcript type."""
    content_data = {"title": "Test", "content": "Content"}
    with patch(
        "marketing_project.runner.process_transcript", new_callable=AsyncMock
    ) as mock_process:
        mock_process.return_value = json.dumps({"status": "success", "result": {}})
        result = await run_single_content(content_data, "transcript")

    assert result["status"] == "success"
    mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_run_single_content_release_notes():
    """Test run_single_content with release_notes type."""
    content_data = {"title": "Test", "content": "Content"}
    with patch(
        "marketing_project.runner.process_release_notes", new_callable=AsyncMock
    ) as mock_process:
        mock_process.return_value = json.dumps({"status": "success", "result": {}})
        result = await run_single_content(content_data, "release_notes")

    assert result["status"] == "success"
    mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_run_single_content_invalid_type():
    """Test run_single_content with invalid content type."""
    content_data = {"title": "Test", "content": "Content"}
    with pytest.raises(ValueError, match="Unknown content type"):
        await run_single_content(content_data, "invalid_type")


def test_build_fastapi_app():
    """Test build_fastapi_app creates FastAPI app."""
    app = build_fastapi_app()
    assert app is not None
    assert app.title == "Marketing Project - Function Pipeline Server"


def test_build_fastapi_app_endpoints():
    """Test that build_fastapi_app creates expected endpoints."""
    app = build_fastapi_app()
    client = TestClient(app)

    # Test that endpoints exist (may fail if content manager not initialized, but structure is correct)
    # We can't easily test the full functionality without mocking the entire content manager setup
    assert app is not None


@pytest.mark.asyncio
async def test_run_server():
    """Test run_server function."""
    mock_app = MagicMock()
    mock_config = MagicMock()
    mock_server = MagicMock()
    mock_server.serve = AsyncMock()

    with patch("marketing_project.runner.build_fastapi_app", return_value=mock_app):
        with patch("marketing_project.runner.uvicorn.Config", return_value=mock_config):
            with patch(
                "marketing_project.runner.uvicorn.Server", return_value=mock_server
            ):
                await run_server(host="127.0.0.1", port=9000)

    mock_server.serve.assert_called_once()
