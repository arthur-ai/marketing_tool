"""
Comprehensive tests for core utility functions.
"""

import pytest

from marketing_project.core.utils import (
    convert_dict_to_content_context,
    create_standard_task_result,
    ensure_content_context,
    extract_content_metadata_for_pipeline,
    merge_task_results,
    validate_content_for_processing,
)


def test_convert_dict_to_content_context_blog_post():
    """Test convert_dict_to_content_context for blog post."""
    data = {
        "id": "test-1",
        "title": "Test Blog",
        "content": "Test content",
        "author": "Test Author",
        "tags": ["test", "blog"],
    }

    context = convert_dict_to_content_context(data)

    assert context.id == "test-1"
    assert context.title == "Test Blog"
    assert hasattr(context, "author")


def test_convert_dict_to_content_context_transcript():
    """Test convert_dict_to_content_context for transcript."""
    data = {
        "id": "test-1",
        "title": "Podcast",
        "content": "Speaker 1: Hello",
        "speakers": ["Speaker 1", "Speaker 2"],
        "duration": "30:00",
    }

    context = convert_dict_to_content_context(data)

    assert context.id == "test-1"
    assert hasattr(context, "speakers")
    assert hasattr(context, "duration")


def test_convert_dict_to_content_context_release_notes():
    """Test convert_dict_to_content_context for release notes."""
    data = {
        "id": "test-1",
        "title": "Release v1.0",
        "content": "Release notes",
        "version": "1.0.0",
        "changes": ["Feature 1", "Feature 2"],
    }

    context = convert_dict_to_content_context(data)

    assert context.id == "test-1"
    assert hasattr(context, "version")
    assert hasattr(context, "changes")


def test_convert_dict_to_content_context_missing_id():
    """Test convert_dict_to_content_context with missing id."""
    data = {"title": "Test"}

    with pytest.raises(ValueError, match="Missing required field: id"):
        convert_dict_to_content_context(data)


def test_ensure_content_context():
    """Test ensure_content_context function."""
    from marketing_project.models.content_models import BlogPostContext

    blog_post = BlogPostContext(id="test-1", title="Test")
    context = ensure_content_context(blog_post)

    assert context.id == "test-1"


def test_create_standard_task_result():
    """Test create_standard_task_result function."""
    result = create_standard_task_result(
        success=True,
        data={"test": "data"},
        task_name="test_task",
    )

    assert result["success"] is True
    assert result["data"]["test"] == "data"
    assert result["task_name"] == "test_task"


def test_validate_content_for_processing():
    """Test validate_content_for_processing function."""
    from marketing_project.models.content_models import BlogPostContext

    content = BlogPostContext(id="test-1", title="Test", content="Content")
    validation = validate_content_for_processing(content)

    assert validation["is_valid"] is True
    assert "issues" in validation
    assert "warnings" in validation


def test_extract_content_metadata_for_pipeline():
    """Test extract_content_metadata_for_pipeline function."""
    from marketing_project.models.content_models import BlogPostContext

    content = BlogPostContext(
        id="test-1",
        title="Test Blog",
        content="Test content",
        author="Test Author",
    )
    metadata = extract_content_metadata_for_pipeline(content)

    assert metadata["id"] == "test-1"
    assert metadata["title"] == "Test Blog"
    assert "author" in metadata


def test_merge_task_results():
    """Test merge_task_results function."""
    results = [
        create_standard_task_result(
            success=True,
            data={"main_keyword": "test1"},
            task_name="seo_keywords",
        ),
        create_standard_task_result(
            success=True,
            data={"target_audience": "developers"},
            task_name="marketing_brief",
        ),
    ]

    merged = merge_task_results(results)

    assert merged["success"] is True
    assert "data" in merged
    assert "seo_keywords" in merged["data"] or "main_keyword" in str(merged)
