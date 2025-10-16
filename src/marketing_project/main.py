import asyncio
import logging
import os

import click
import yaml
from dotenv import find_dotenv, load_dotenv

# Activate logging setup before any other imports that use logging
import marketing_project.logging_config

# Initialize logger
logger = logging.getLogger("marketing_project.runner")

from marketing_project.runner import (
    run_marketing_project_pipeline,
    run_marketing_project_server,
)
from marketing_project.services.content_source_config_loader import (
    ContentSourceConfigLoader,
)
from marketing_project.services.content_source_factory import ContentSourceManager

# # Load .env variables
dotenv_path = find_dotenv()

# Force override of any existing env vars
load_dotenv(dotenv_path=dotenv_path, override=True, verbose=True)


@click.group()
def cli():
    """Marketing Project CLI"""
    pass


@cli.command("run")
@click.option("--lang", default="en", help="Prompt language (default: en)")
@click.option("--prompts-dir", default=None, help="Prompt templates directory")
def run_pipeline(lang, prompts_dir):
    """Run the marketing project content processing pipeline asynchronously."""
    asyncio.run(_run_pipeline_async(lang, prompts_dir))


async def _run_pipeline_async(lang, prompts_dir):
    # Default prompts_dir logic
    if not prompts_dir:
        base = os.path.dirname(__file__)
        prompts_dir = os.path.abspath(
            os.path.join(base, "prompts", os.getenv("TEMPLATE_VERSION", "v1"))
        )
    await run_marketing_project_pipeline(prompts_dir=prompts_dir, lang=lang)


@cli.command("serve")
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
@click.option("--lang", default="en", help="Prompt language (default: en)")
@click.option("--prompts-dir", default=None, help="Prompt templates directory")
def serve_server(host, port, lang, prompts_dir):
    """Serve the marketing project content processing API as an async HTTP (FastAPI) server."""
    asyncio.run(_run_server_async(host, port, lang, prompts_dir))


async def _run_server_async(host, port, lang, prompts_dir):
    if not prompts_dir:
        base = os.path.dirname(__file__)
        prompts_dir = os.path.abspath(
            os.path.join(base, "prompts", os.getenv("TEMPLATE_VERSION", "v1"))
        )
    await run_marketing_project_server(
        host=host, port=port, prompts_dir=prompts_dir, lang=lang
    )


@cli.command("content-sources")
@click.option(
    "--list", "list_sources", is_flag=True, help="List all configured content sources"
)
@click.option(
    "--status", "check_status", is_flag=True, help="Check status of all content sources"
)
@click.option("--test", "test_sources", is_flag=True, help="Test all content sources")
@click.option(
    "--fetch", "fetch_content", is_flag=True, help="Fetch content from all sources"
)
@click.option("--lang", default="en", help="Prompt language (default: en)")
@click.option("--prompts-dir", default=None, help="Prompt templates directory")
def content_sources_cmd(
    list_sources, check_status, test_sources, fetch_content, lang, prompts_dir
):
    """Manage content sources"""
    asyncio.run(
        _content_sources_async(
            list_sources, check_status, test_sources, fetch_content, lang, prompts_dir
        )
    )


async def _content_sources_async(
    list_sources, check_status, test_sources, fetch_content, lang, prompts_dir
):
    """Handle content sources commands asynchronously."""

    # Default prompts_dir logic
    if not prompts_dir:
        base = os.path.dirname(__file__)
        prompts_dir = os.path.abspath(
            os.path.join(base, "prompts", os.getenv("TEMPLATE_VERSION", "v1"))
        )

    # Load configurations
    config_loader = ContentSourceConfigLoader()
    source_configs = config_loader.create_source_configs()

    # Create manager and add sources
    manager = ContentSourceManager()
    for config in source_configs:
        await manager.add_source_from_config(config)

    if list_sources:
        logger.info("Configured Content Sources:")
        for name, source in manager.sources.items():
            status = source.get_status()
            logger.info(f"  - {name}: {status['status']} ({status['type']})")

    if check_status:
        logger.info("Content Source Health Check:")
        health_status = await manager.health_check_all()
        for name, is_healthy in health_status.items():
            status = "✓" if is_healthy else "✗"
            logger.info(f"  {status} {name}")

    if test_sources:
        logger.info("Testing Content Sources:")
        results = await manager.fetch_all_content(limit_per_source=1)
        for result in results:
            status = "✓" if result.success else "✗"
            logger.info(f"  {status} {result.source_name}: {result.total_count} items")
            if not result.success:
                logger.error(f"    Error: {result.error_message}")

    if fetch_content:
        logger.info("Fetching Content:")
        content_models = await manager.fetch_content_as_models()
        logger.info(f"Total content items: {len(content_models)}")
        for model in content_models:
            logger.info(f"  - {model.title} ({model.__class__.__name__})")

    await manager.cleanup()


if __name__ == "__main__":
    cli()
