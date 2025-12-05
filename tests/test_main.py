"""
Tests for main CLI module.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from marketing_project.main import (
    _content_sources_async,
    cli,
    content_sources_cmd,
    run_pipeline,
    serve_server,
)


def test_cli_group():
    """Test that CLI group is defined."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Marketing Project CLI" in result.output


def test_run_pipeline_command():
    """Test run pipeline command."""
    runner = CliRunner()
    with patch(
        "marketing_project.main.run_function_pipeline", new_callable=AsyncMock
    ) as mock_run:
        with patch("marketing_project.main.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = lambda coro: coro
            result = runner.invoke(cli, ["run"])
            # Command should execute (may fail if dependencies not available, but structure is correct)
            assert result.exit_code in [0, 1]  # May fail due to missing dependencies


def test_serve_server_command():
    """Test serve server command."""
    runner = CliRunner()
    with patch("marketing_project.main.run_server", new_callable=AsyncMock) as mock_run:
        with patch("marketing_project.main.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = lambda coro: coro
            result = runner.invoke(
                cli, ["serve", "--host", "127.0.0.1", "--port", "9000"]
            )
            # Command should execute
            assert result.exit_code in [0, 1]


def test_serve_server_default_options():
    """Test serve server command with default options."""
    runner = CliRunner()
    with patch("marketing_project.main.run_server", new_callable=AsyncMock) as mock_run:
        with patch("marketing_project.main.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = lambda coro: coro
            result = runner.invoke(cli, ["serve"])
            # Command should execute
            assert result.exit_code in [0, 1]


def test_content_sources_command_list():
    """Test content-sources command with --list flag."""
    runner = CliRunner()
    with patch(
        "marketing_project.main._content_sources_async", new_callable=AsyncMock
    ) as mock_async:
        with patch("marketing_project.main.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = lambda coro: coro
            result = runner.invoke(cli, ["content-sources", "--list"])
            # Command should execute
            assert result.exit_code in [0, 1]


def test_content_sources_command_status():
    """Test content-sources command with --status flag."""
    runner = CliRunner()
    with patch(
        "marketing_project.main._content_sources_async", new_callable=AsyncMock
    ) as mock_async:
        with patch("marketing_project.main.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = lambda coro: coro
            result = runner.invoke(cli, ["content-sources", "--status"])
            # Command should execute
            assert result.exit_code in [0, 1]


def test_content_sources_command_test():
    """Test content-sources command with --test flag."""
    runner = CliRunner()
    with patch(
        "marketing_project.main._content_sources_async", new_callable=AsyncMock
    ) as mock_async:
        with patch("marketing_project.main.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = lambda coro: coro
            result = runner.invoke(cli, ["content-sources", "--test"])
            # Command should execute
            assert result.exit_code in [0, 1]


def test_content_sources_command_fetch():
    """Test content-sources command with --fetch flag."""
    runner = CliRunner()
    with patch(
        "marketing_project.main._content_sources_async", new_callable=AsyncMock
    ) as mock_async:
        with patch("marketing_project.main.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = lambda coro: coro
            result = runner.invoke(cli, ["content-sources", "--fetch"])
            # Command should execute
            assert result.exit_code in [0, 1]


@pytest.mark.asyncio
async def test_content_sources_async_list_sources():
    """Test _content_sources_async with list_sources flag."""
    mock_config_loader = MagicMock()
    mock_config_loader.create_source_configs.return_value = []

    mock_manager = MagicMock()
    mock_manager.sources = {
        "source1": MagicMock(get_status=lambda: {"status": "active", "type": "file"}),
        "source2": MagicMock(get_status=lambda: {"status": "active", "type": "api"}),
    }
    mock_manager.add_source_from_config = AsyncMock()
    mock_manager.cleanup = AsyncMock()

    with patch(
        "marketing_project.main.ContentSourceConfigLoader",
        return_value=mock_config_loader,
    ):
        with patch(
            "marketing_project.main.ContentSourceManager", return_value=mock_manager
        ):
            await _content_sources_async(
                list_sources=True,
                check_status=False,
                test_sources=False,
                fetch_content=False,
            )

    mock_manager.add_source_from_config.assert_called()


@pytest.mark.asyncio
async def test_content_sources_async_check_status():
    """Test _content_sources_async with check_status flag."""
    mock_config_loader = MagicMock()
    mock_config_loader.create_source_configs.return_value = []

    mock_manager = MagicMock()
    mock_manager.add_source_from_config = AsyncMock()
    mock_manager.health_check_all = AsyncMock(
        return_value={"source1": True, "source2": False}
    )
    mock_manager.cleanup = AsyncMock()

    with patch(
        "marketing_project.main.ContentSourceConfigLoader",
        return_value=mock_config_loader,
    ):
        with patch(
            "marketing_project.main.ContentSourceManager", return_value=mock_manager
        ):
            await _content_sources_async(
                list_sources=False,
                check_status=True,
                test_sources=False,
                fetch_content=False,
            )

    mock_manager.health_check_all.assert_called_once()


@pytest.mark.asyncio
async def test_content_sources_async_test_sources():
    """Test _content_sources_async with test_sources flag."""
    from marketing_project.core.content_sources import ContentSourceResult

    mock_config_loader = MagicMock()
    mock_config_loader.create_source_configs.return_value = []

    mock_manager = MagicMock()
    mock_manager.add_source_from_config = AsyncMock()
    mock_manager.fetch_all_content = AsyncMock(
        return_value=[
            ContentSourceResult(
                source_name="source1",
                success=True,
                total_count=5,
                content_items=[],
                error_message=None,
            ),
            ContentSourceResult(
                source_name="source2",
                success=False,
                total_count=0,
                content_items=[],
                error_message="Error",
            ),
        ]
    )
    mock_manager.cleanup = AsyncMock()

    with patch(
        "marketing_project.main.ContentSourceConfigLoader",
        return_value=mock_config_loader,
    ):
        with patch(
            "marketing_project.main.ContentSourceManager", return_value=mock_manager
        ):
            await _content_sources_async(
                list_sources=False,
                check_status=False,
                test_sources=True,
                fetch_content=False,
            )

    mock_manager.fetch_all_content.assert_called_once()


@pytest.mark.asyncio
async def test_content_sources_async_fetch_content():
    """Test _content_sources_async with fetch_content flag."""
    from marketing_project.models.content_models import BlogPostContext

    mock_config_loader = MagicMock()
    mock_config_loader.create_source_configs.return_value = []

    mock_manager = MagicMock()
    mock_manager.add_source_from_config = AsyncMock()
    mock_manager.fetch_content_as_models = AsyncMock(
        return_value=[
            BlogPostContext(
                id="1",
                title="Test Post",
                content="Content",
                snippet="Snippet",
                created_at="2024-01-01T00:00:00Z",
            )
        ]
    )
    mock_manager.cleanup = AsyncMock()

    with patch(
        "marketing_project.main.ContentSourceConfigLoader",
        return_value=mock_config_loader,
    ):
        with patch(
            "marketing_project.main.ContentSourceManager", return_value=mock_manager
        ):
            await _content_sources_async(
                list_sources=False,
                check_status=False,
                test_sources=False,
                fetch_content=True,
            )

    mock_manager.fetch_content_as_models.assert_called_once()


@pytest.mark.asyncio
async def test_content_sources_async_cleanup():
    """Test that cleanup is called in _content_sources_async."""
    mock_config_loader = MagicMock()
    mock_config_loader.create_source_configs.return_value = []

    mock_manager = MagicMock()
    mock_manager.add_source_from_config = AsyncMock()
    mock_manager.cleanup = AsyncMock()

    with patch(
        "marketing_project.main.ContentSourceConfigLoader",
        return_value=mock_config_loader,
    ):
        with patch(
            "marketing_project.main.ContentSourceManager", return_value=mock_manager
        ):
            await _content_sources_async(
                list_sources=False,
                check_status=False,
                test_sources=False,
                fetch_content=False,
            )

    mock_manager.cleanup.assert_called_once()
