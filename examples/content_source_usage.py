#!/usr/bin/env python3
"""
Example usage of the content source system with proper ContentContext models.

This example demonstrates how to use the content source factory to fetch
and process content as proper Pydantic models.
"""

import asyncio
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from marketing_project.core.content_sources import ContentSourceType, FileSourceConfig
from marketing_project.core.models import ContentContext
from marketing_project.models.content_models import (
    BlogPostContext,
    ReleaseNotesContext,
    TranscriptContext,
)
from marketing_project.services.content_source_factory import ContentSourceManager


async def main():
    """Demonstrate content source usage with proper models."""

    # Create content manager
    manager = ContentSourceManager()

    # Add a file source
    file_config = FileSourceConfig(
        name="example_files",
        source_type=ContentSourceType.FILE,
        file_paths=["content/"],
        file_patterns=["content/**/*.json", "content/**/*.md"],
        watch_directory=False,
    )

    success = await manager.add_source_from_config(file_config)
    print(f"Added file source: {success}")

    # Fetch content as ContentContext models
    print("\n=== Fetching Content as Models ===")
    content_models = await manager.fetch_content_as_models()

    print(f"Found {len(content_models)} content items:")
    for i, model in enumerate(content_models, 1):
        print(f"\n{i}. {model.__class__.__name__}")
        print(f"   ID: {model.id}")
        print(f"   Title: {model.title}")
        print(f"   Type: {type(model).__name__}")

        # Show type-specific fields
        if isinstance(model, TranscriptContext):
            print(f"   Speakers: {model.speakers}")
            print(f"   Duration: {model.duration}")
            print(f"   Transcript Type: {model.transcript_type}")
        elif isinstance(model, BlogPostContext):
            print(f"   Author: {model.author}")
            print(f"   Tags: {model.tags}")
            print(f"   Category: {model.category}")
        elif isinstance(model, ReleaseNotesContext):
            print(f"   Version: {model.version}")
            print(f"   Changes: {len(model.changes)} items")
            print(f"   Features: {len(model.features)} items")

    # Search content models
    print("\n=== Searching Content Models ===")
    search_results = await manager.search_content_models("marketing")
    print(f"Found {len(search_results)} items containing 'marketing'")

    for model in search_results:
        print(f"- {model.title} ({model.__class__.__name__})")

    # Get content by type
    print("\n=== Content by Type ===")
    blog_posts = await manager.get_content_models_by_type("blog_post")
    print(f"Blog posts: {len(blog_posts)}")

    transcripts = await manager.get_content_models_by_type("transcript")
    print(f"Transcripts: {len(transcripts)}")

    release_notes = await manager.get_content_models_by_type("release_notes")
    print(f"Release notes: {len(release_notes)}")

    # Show source statistics
    print("\n=== Source Statistics ===")
    stats = await manager.get_source_statistics()
    print(f"Total sources: {stats['total_sources']}")
    print(f"Active sources: {stats['active_sources']}")
    print(f"Error sources: {stats['error_sources']}")
    print(f"Total content items: {stats['total_content_items']}")

    # Cleanup
    await manager.cleanup()
    print("\n=== Cleanup Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
