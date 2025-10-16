import json
import logging
import os
from datetime import datetime

import uvicorn

# For HTTP serving (optional)
from fastapi import BackgroundTasks, FastAPI

from marketing_project.agents.article_generation_agent import (
    get_article_generation_agent,
)
from marketing_project.agents.blog_agent import get_blog_agent
from marketing_project.agents.content_formatting_agent import (
    get_content_formatting_agent,
)
from marketing_project.agents.content_pipeline_agent import get_content_pipeline_agent
from marketing_project.agents.design_kit_agent import get_design_kit_agent
from marketing_project.agents.internal_docs_agent import get_internal_docs_agent
from marketing_project.agents.marketing_agent import get_marketing_orchestrator_agent
from marketing_project.agents.marketing_brief_agent import get_marketing_brief_agent
from marketing_project.agents.releasenotes_agent import get_releasenotes_agent
from marketing_project.agents.seo_keywords_agent import get_seo_keywords_agent
from marketing_project.agents.seo_optimization_agent import get_seo_optimization_agent
from marketing_project.agents.transcripts_agent import get_transcripts_agent
from marketing_project.core.models import (
    AppContext,
    BlogPostContext,
    ReleaseNotesContext,
    TranscriptContext,
)
from marketing_project.services.content_source_config_loader import (
    ContentSourceConfigLoader,
)
from marketing_project.services.content_source_factory import ContentSourceManager

# Initialize logger
logger = logging.getLogger("marketing_project.runner")


async def run_marketing_project_pipeline(prompts_dir, lang):
    """
    Run the Marketing Project pipeline for content processing.

    This pipeline processes different types of content:
    - Transcripts (podcasts, videos, meetings)
    - Blog posts (articles, tutorials)
    - Release notes (software releases, updates)
    """
    # Set up specialized agents and orchestrator
    transcripts_agent = await get_transcripts_agent(prompts_dir, lang)
    blog_agent = await get_blog_agent(prompts_dir, lang)
    releasenotes_agent = await get_releasenotes_agent(prompts_dir, lang)
    orchestrator_agent = await get_marketing_orchestrator_agent(
        prompts_dir, lang, transcripts_agent, blog_agent, releasenotes_agent
    )

    # Content processing pipeline
    # In practice, content would come from your actual sources:
    # - Content management systems
    # - APIs (YouTube, podcast platforms, etc.)
    # - File uploads
    # - Web scraping
    # - Database queries

    # Content source integration

    # Initialize content source manager
    content_manager = ContentSourceManager()

    # Load and add content sources from configuration
    config_loader = ContentSourceConfigLoader()
    source_configs = config_loader.create_source_configs()

    for config in source_configs:
        await content_manager.add_source_from_config(config)

    # Fetch content from all sources as ContentContext models
    content_models = await content_manager.fetch_content_as_models()

    # Process content through the pipeline
    processed_content = []
    for content_context in content_models:
        try:
            # Process through orchestrator (which routes to appropriate agents)
            app_context = {
                "content": content_context,
                "labels": {},
                "content_type": content_context.__class__.__name__.replace(
                    "Context", ""
                ).lower(),
            }

            # Let the orchestrator handle routing to the appropriate agent
            processed = await orchestrator_agent.run_async(app_context)
            processed_content.append(processed)

        except Exception as e:
            logger.error(f"Error processing content item: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

    logger.info(
        "Marketing Project pipeline initialized and ready for content processing"
    )
    logger.info("Architecture:")
    logger.info(
        "- Content Sources: Fetch content from various sources (files, APIs, databases, etc.)"
    )
    logger.info(
        "- Orchestrator Agent: Central routing point that directs content to specialized agents"
    )
    logger.info("  - Transcripts Agent: For podcasts, videos, meetings")
    logger.info("  - Blog Agent: For articles, tutorials, written content")
    logger.info("  - Release Notes Agent: For software releases, updates")
    logger.info(f"Content sources: {len(content_manager.sources)} configured")
    logger.info(f"Content processed: {len(processed_content)} items")

    return {
        "agents": {
            "transcripts_agent": transcripts_agent,
            "blog_agent": blog_agent,
            "releasenotes_agent": releasenotes_agent,
            "orchestrator_agent": orchestrator_agent,
        },
        "content_manager": content_manager,
        "processed_content": processed_content,
    }


async def run_content_analysis_pipeline(prompts_dir, lang):
    """
    Run the new Content Analysis Pipeline for comprehensive content processing.

    This pipeline follows a 7-step workflow:
    1. AnalyzeContent → 2. ExtractSEOKeywords → 3. GenerateMarketingBrief →
    4. GenerateArticle → 5. OptimizeSEO → 6. SuggestInternalDocs → 7. FormatContent
    """
    # Set up all specialized agents for the new pipeline
    seo_keywords_agent = await get_seo_keywords_agent(prompts_dir, lang)
    marketing_brief_agent = await get_marketing_brief_agent(prompts_dir, lang)
    article_generation_agent = await get_article_generation_agent(prompts_dir, lang)
    seo_optimization_agent = await get_seo_optimization_agent(prompts_dir, lang)
    internal_docs_agent = await get_internal_docs_agent(prompts_dir, lang)
    content_formatting_agent = await get_content_formatting_agent(prompts_dir, lang)
    design_kit_agent = await get_design_kit_agent(prompts_dir, lang)

    # Create the main content pipeline orchestrator
    content_pipeline_agent = await get_content_pipeline_agent(
        prompts_dir,
        lang,
        seo_keywords_agent=seo_keywords_agent,
        marketing_brief_agent=marketing_brief_agent,
        article_generation_agent=article_generation_agent,
        seo_optimization_agent=seo_optimization_agent,
        internal_docs_agent=internal_docs_agent,
        content_formatting_agent=content_formatting_agent,
        design_kit_agent=design_kit_agent,
    )

    # Content source integration for content analysis pipeline

    # Initialize content source manager
    content_manager = ContentSourceManager()

    # Load and add content sources from configuration
    config_loader = ContentSourceConfigLoader()
    source_configs = config_loader.create_source_configs()

    for config in source_configs:
        await content_manager.add_source_from_config(config)

    # Fetch content from all sources as ContentContext models
    content_models = await content_manager.fetch_content_as_models()

    # Process content through the content analysis pipeline
    processed_content = []
    for content_context in content_models:
        try:
            # Process through content pipeline orchestrator (which manages the 8-step workflow)
            app_context = {
                "content": content_context,
                "labels": {},
                "content_type": content_context.__class__.__name__.replace(
                    "Context", ""
                ).lower(),
            }

            # Let the content pipeline orchestrator handle the complete workflow
            processed = await content_pipeline_agent.run_async(app_context)
            processed_content.append(processed)

        except Exception as e:
            logger.error(f"Error processing content item in analysis pipeline: {e}")
            continue

    logger.info("Content Analysis Pipeline initialized and ready for processing")
    logger.info("Architecture:")
    logger.info(
        "- Content Sources: Fetch content from various sources (files, APIs, databases, etc.)"
    )
    logger.info("- Content Pipeline Orchestrator: Manages the complete 8-step workflow")
    logger.info("  - SEO Keywords Agent: Extract and analyze SEO keywords")
    logger.info("  - Marketing Brief Agent: Generate marketing briefs and strategy")
    logger.info("  - Article Generation Agent: Create high-quality articles")
    logger.info("  - SEO Optimization Agent: Apply comprehensive SEO optimizations")
    logger.info(
        "  - Internal Docs Agent: Suggest internal documents and cross-references"
    )
    logger.info("  - Content Formatting Agent: Format and finalize content")
    logger.info(
        "  - Design Kit Agent: Apply professional design templates and visual enhancements"
    )
    logger.info(f"Content sources: {len(content_manager.sources)} configured")
    logger.info(f"Content processed: {len(processed_content)} items")

    return {
        "content_pipeline_agent": content_pipeline_agent,
        "content_manager": content_manager,
        "processed_content": processed_content,
    }


# Optional: Async FastAPI server for MCP/K8s deployment
def build_fastapi_app(prompts_dir, lang):
    app = FastAPI(title="Marketing Project Server")

    # Global content manager for webhook endpoints
    content_manager = None

    @app.on_event("startup")
    async def startup_event():
        nonlocal content_manager

        content_manager = ContentSourceManager()
        config_loader = ContentSourceConfigLoader()
        source_configs = config_loader.create_source_configs()

        for config in source_configs:
            await content_manager.add_source_from_config(config)

    @app.on_event("shutdown")
    async def shutdown_event():
        nonlocal content_manager
        if content_manager:
            await content_manager.cleanup()

    @app.post("/run")
    async def run_pipeline_endpoint(background: BackgroundTasks):
        background.add_task(run_marketing_project_pipeline, prompts_dir, lang)
        return {"status": "accepted"}

    @app.post("/webhook/{source_name}")
    async def webhook_endpoint(source_name: str, request: dict):
        """Receive webhook content for a specific source."""
        nonlocal content_manager
        if not content_manager:
            return {"error": "Content manager not initialized"}

        # Find webhook source
        webhook_source = None
        for name, source in content_manager.sources.items():
            if (
                hasattr(source, "config")
                and source.config.source_type.value == "webhook"
                and name == source_name
            ):
                webhook_source = source
                break

        if not webhook_source:
            return {"error": f"Webhook source '{source_name}' not found"}

        # Process webhook
        signature = request.headers.get(
            "X-Signature", request.headers.get("X-Hub-Signature")
        )
        success = await webhook_source.receive_webhook(request.json, signature)

        return {"status": "received" if success else "rejected"}

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


async def run_marketing_project_server(host, port, prompts_dir, lang):
    app = build_fastapi_app(prompts_dir, lang)
    uvicorn.run(app, host=host, port=port)
