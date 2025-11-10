"""
Simplified Marketing Project Runner

This CLI tool demonstrates the function-based pipeline for content processing.
Updated to use the new simplified processors with FunctionPipeline.
"""

import asyncio
import json
import logging
from pathlib import Path

# For HTTP serving (optional)
import uvicorn
from fastapi import BackgroundTasks, FastAPI

from marketing_project.core.models import (
    BlogPostContext,
    ReleaseNotesContext,
    TranscriptContext,
)
from marketing_project.processors import (
    process_blog_post,
    process_release_notes,
    process_transcript,
)
from marketing_project.services.content_source_config_loader import (
    ContentSourceConfigLoader,
)
from marketing_project.services.content_source_factory import ContentSourceManager

# Initialize logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("marketing_project.runner")


async def run_function_pipeline():
    """
    Run the simplified function-based pipeline for content processing.

    This demonstrates the new architecture:
    1. Fetch content from sources
    2. Route to appropriate processor (blog, transcript, release notes)
    3. Process through FunctionPipeline (7 AI steps)
    4. Return structured results
    """
    logger.info("=" * 80)
    logger.info("Marketing Project - Function Pipeline Runner")
    logger.info("=" * 80)

    # Initialize content source manager
    content_manager = ContentSourceManager()

    # Load and add content sources from configuration
    config_loader = ContentSourceConfigLoader()
    source_configs = config_loader.create_source_configs()

    for config in source_configs:
        await content_manager.add_source_from_config(config)

    logger.info(f"Initialized {len(content_manager.sources)} content sources")

    # Fetch content from all sources as ContentContext models
    content_models = await content_manager.fetch_content_as_models()
    logger.info(f"Fetched {len(content_models)} content items")

    # Process content through the appropriate processor
    results = []
    for content_context in content_models:
        try:
            content_type = content_context.__class__.__name__.replace(
                "Context", ""
            ).lower()
            logger.info(f"\n{'─' * 80}")
            logger.info(
                f"Processing {content_type}: {content_context.title or 'Untitled'}"
            )
            logger.info(f"{'─' * 80}")

            # Convert Pydantic model to JSON string
            content_json = content_context.model_dump_json()

            # Route to appropriate processor
            if isinstance(content_context, BlogPostContext):
                result_json = await process_blog_post(content_json)
            elif isinstance(content_context, TranscriptContext):
                result_json = await process_transcript(content_json)
            elif isinstance(content_context, ReleaseNotesContext):
                result_json = await process_release_notes(content_json)
            else:
                logger.warning(f"Unknown content type: {type(content_context)}")
                continue

            # Parse result
            result = json.loads(result_json)

            if result.get("status") == "success":
                logger.info(f"✅ Success: {result.get('message')}")
                results.append(result)
            else:
                logger.error(f"❌ Error: {result.get('message')}")

        except Exception as e:
            logger.error(f"Error processing content: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

    logger.info(f"\n{'=' * 80}")
    logger.info(
        f"Pipeline Complete: {len(results)}/{len(content_models)} items processed successfully"
    )
    logger.info(f"={'=' * 80}\n")

    return {
        "content_manager": content_manager,
        "processed_content": results,
        "success_count": len(results),
        "total_count": len(content_models),
    }


async def run_single_content(content_data: dict, content_type: str):
    """
    Process a single content item through the function pipeline.

    Args:
        content_data: Dictionary with content fields
        content_type: One of "blog_post", "transcript", "release_notes"

    Returns:
        Processing result dictionary
    """
    logger.info(f"Processing single {content_type} item")

    # Convert to JSON string
    content_json = json.dumps(content_data)

    # Route to appropriate processor
    if content_type == "blog_post":
        result_json = await process_blog_post(content_json)
    elif content_type == "transcript":
        result_json = await process_transcript(content_json)
    elif content_type == "release_notes":
        result_json = await process_release_notes(content_json)
    else:
        raise ValueError(f"Unknown content type: {content_type}")

    # Parse and return result
    return json.loads(result_json)


# Optional: Async FastAPI server for HTTP/webhook deployment
def build_fastapi_app():
    """Build FastAPI application with simplified endpoints."""
    app = FastAPI(title="Marketing Project - Function Pipeline Server")

    # Global content manager for webhook endpoints
    content_manager = None

    @app.on_event("startup")
    async def startup_event():
        nonlocal content_manager

        logger.info("Initializing content sources...")
        content_manager = ContentSourceManager()
        config_loader = ContentSourceConfigLoader()
        source_configs = config_loader.create_source_configs()

        for config in source_configs:
            await content_manager.add_source_from_config(config)

        logger.info(f"Initialized {len(content_manager.sources)} content sources")

    @app.on_event("shutdown")
    async def shutdown_event():
        nonlocal content_manager
        if content_manager:
            await content_manager.cleanup()

    @app.post("/run")
    async def run_pipeline_endpoint(background: BackgroundTasks):
        """Run the complete pipeline on all content sources."""
        background.add_task(run_function_pipeline)
        return {"status": "accepted", "message": "Pipeline started in background"}

    @app.post("/process/blog")
    async def process_blog_endpoint(content: dict):
        """Process a single blog post."""
        result = await run_single_content(content, "blog_post")
        return result

    @app.post("/process/transcript")
    async def process_transcript_endpoint(content: dict):
        """Process a single transcript."""
        result = await run_single_content(content, "transcript")
        return result

    @app.post("/process/release-notes")
    async def process_release_notes_endpoint(content: dict):
        """Process single release notes."""
        result = await run_single_content(content, "release_notes")
        return result

    @app.get("/content-sources")
    async def list_content_sources():
        """List all configured content sources."""
        nonlocal content_manager
        if not content_manager:
            return {"error": "Content manager not initialized"}

        sources = []
        for name, source in content_manager.sources.items():
            status = source.get_status()
            sources.append(
                {
                    "name": name,
                    "type": status["type"],
                    "status": status["status"],
                    "enabled": status["enabled"],
                }
            )

        return {"sources": sources}

    @app.get("/content-sources/{source_name}/status")
    async def get_source_status(source_name: str):
        """Get status of a specific content source."""
        nonlocal content_manager
        if not content_manager:
            return {"error": "Content manager not initialized"}

        if source_name not in content_manager.sources:
            return {"error": f"Source '{source_name}' not found"}

        source = content_manager.sources[source_name]
        is_healthy = await source.health_check()
        status = source.get_status()

        return {"name": source_name, "healthy": is_healthy, "status": status}

    @app.post("/content-sources/{source_name}/fetch")
    async def fetch_source_content(source_name: str, limit: int = 10):
        """Fetch content from a specific source."""
        nonlocal content_manager
        if not content_manager:
            return {"error": "Content manager not initialized"}

        if source_name not in content_manager.sources:
            return {"error": f"Source '{source_name}' not found"}

        source = content_manager.sources[source_name]
        result = await source.fetch_content(limit)

        return {
            "source_name": result.source_name,
            "success": result.success,
            "total_count": result.total_count,
            "content_items": result.content_items,
            "error_message": result.error_message,
        }

    return app


async def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the FastAPI server."""
    app = build_fastapi_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


# CLI entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run as HTTP server
        host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080
        logger.info(f"Starting server on {host}:{port}")
        asyncio.run(run_server(host, port))
    else:
        # Run pipeline once
        asyncio.run(run_function_pipeline())
