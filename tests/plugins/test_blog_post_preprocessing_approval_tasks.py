"""
Tests for BlogPostPreprocessingApprovalPlugin.

Covers all codepaths in execute() plus schema complexity regression test
verifying the AnthropicException: Schema is too complex fix.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.models.pipeline_steps import (
    BlogPostPreprocessingApprovalLLMResult,
    BlogPostPreprocessingApprovalResult,
)
from marketing_project.plugins.blog_post_preprocessing_approval.tasks import (
    BlogPostPreprocessingApprovalPlugin,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plugin():
    return BlogPostPreprocessingApprovalPlugin()


@pytest.fixture
def mock_pipeline():
    p = MagicMock()
    p._call_function = AsyncMock()
    return p


def _make_llm_result(**kwargs):
    defaults = dict(
        is_valid=True,
        title_validated=True,
        content_validated=True,
        author_validated=True,
        category_validated=True,
        tags_validated=True,
        validation_issues=[],
        author=None,
        category=None,
        tags=[],
        requires_approval=False,
        approval_suggestions=[],
        confidence_score=None,
    )
    defaults.update(kwargs)
    return BlogPostPreprocessingApprovalLLMResult(**defaults)


# ---------------------------------------------------------------------------
# Plugin properties
# ---------------------------------------------------------------------------


def test_step_name(plugin):
    assert plugin.step_name == "blog_post_preprocessing_approval"


def test_step_number(plugin):
    assert plugin.step_number == 1


def test_response_model_is_llm_base(plugin):
    assert plugin.response_model is BlogPostPreprocessingApprovalLLMResult


def test_required_context_keys(plugin):
    assert "input_content" in plugin.get_required_context_keys()


# ---------------------------------------------------------------------------
# REGRESSION: Schema complexity — this is the entire point of the refactor
# ---------------------------------------------------------------------------


def test_llm_result_schema_complexity():
    """Anthropic rejects schemas with too many properties/anyOf patterns.

    Before this fix BlogPostPreprocessingApprovalResult had 21 fields and
    10 anyOf patterns, causing AnthropicException: Schema is too complex.
    The LLM base class must stay small enough for Anthropic to compile.
    """
    schema = BlogPostPreprocessingApprovalLLMResult.model_json_schema()
    props = schema.get("properties", {})
    anyof_count = sum(1 for v in props.values() if "anyOf" in v)

    assert len(props) <= 13, (
        f"LLM schema has {len(props)} properties — Anthropic will reject it. "
        "Check that no required fields were added to BlogPostPreprocessingApprovalLLMResult."
    )
    assert anyof_count <= 3, (
        f"LLM schema has {anyof_count} anyOf patterns — Anthropic may reject it. "
        "Keep Optional fields on the LLM base class to a minimum."
    )


def test_full_result_schema_has_enrichment_fields():
    """Full result schema should contain enrichment fields not in LLM base."""
    llm_schema = BlogPostPreprocessingApprovalLLMResult.model_json_schema()
    full_schema = BlogPostPreprocessingApprovalResult.model_json_schema()

    llm_props = set(llm_schema.get("properties", {}).keys())
    full_props = set(full_schema.get("properties", {}).keys())

    enrichment_fields = {
        "word_count",
        "reading_time",
        "content_summary",
        "sentiment_score",
    }
    assert enrichment_fields.issubset(
        full_props
    ), "Full result missing enrichment fields"
    assert not enrichment_fields.issubset(
        llm_props
    ), "Enrichment fields leaked into LLM schema"


# ---------------------------------------------------------------------------
# Model upgrade round-trip
# ---------------------------------------------------------------------------


def test_model_upgrade_roundtrip():
    """BlogPostPreprocessingApprovalResult(**llm_result.model_dump()) must preserve all base fields."""
    llm_result = _make_llm_result(
        author="Jane Doe",
        category="Engineering",
        tags=["python", "api"],
        requires_approval=True,
        confidence_score=0.85,
    )
    full = BlogPostPreprocessingApprovalResult(**llm_result.model_dump())

    assert full.is_valid == llm_result.is_valid
    assert full.author == "Jane Doe"
    assert full.category == "Engineering"
    assert full.tags == ["python", "api"]
    assert full.requires_approval is True
    assert full.confidence_score == 0.85
    # Enrichment fields default correctly
    assert full.word_count is None
    assert full.reading_time is None
    assert full.content_summary is None
    assert full.overall_sentiment is None
    assert full.headings == []


# ---------------------------------------------------------------------------
# execute() — skip path (non-blog_post content)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_skips_non_blog_post(plugin, mock_pipeline):
    """For non-blog_post content types, execute() returns a valid result without calling LLM."""
    context = {
        "content_type": "transcript",
        "input_content": {"title": "A talk", "content": "Hello world"},
    }
    result = await plugin.execute(context, mock_pipeline, "job-1")

    mock_pipeline._call_function.assert_not_called()
    assert isinstance(result, BlogPostPreprocessingApprovalResult)
    assert result.is_valid is True
    assert result.requires_approval is False
    assert result.confidence_score == 1.0


# ---------------------------------------------------------------------------
# execute() — sentinel propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_propagates_sentinel(plugin, mock_pipeline):
    """When _execute_step returns ApprovalRequiredSentinel, it is returned unchanged.

    The processors package __init__ triggers an sklearn import that fails on
    NumPy 2.x in the test environment, so we mock sys.modules to avoid it.
    """
    import sys
    from types import ModuleType

    class _FakeSentinel:
        pass

    fake_module = ModuleType("marketing_project.processors.approval_helper")
    fake_module.ApprovalRequiredSentinel = _FakeSentinel

    sentinel = _FakeSentinel()
    with patch.dict(
        sys.modules,
        {
            "marketing_project.processors": MagicMock(),
            "marketing_project.processors.approval_helper": fake_module,
        },
    ):
        with patch.object(plugin, "_execute_step", return_value=sentinel):
            context = {
                "content_type": "blog_post",
                "input_content": {"title": "T", "content": "C"},
            }
            result = await plugin.execute(context, mock_pipeline, "job-1")

    assert result is sentinel


# ---------------------------------------------------------------------------
# execute() — programmatic computation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_computes_word_count(plugin, mock_pipeline):
    """word_count is computed from content when LLM leaves it None."""
    llm_result = _make_llm_result()
    content = "one two three four five"  # 5 words

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": content},
        }
        result = await plugin.execute(context, mock_pipeline)

    assert result.word_count == 5


@pytest.mark.asyncio
async def test_execute_computes_reading_time(plugin, mock_pipeline):
    """reading_time is computed as round(word_count / 200, 1)."""
    llm_result = _make_llm_result()
    # 400 words → 2.0 minutes
    content = " ".join(["word"] * 400)

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": content},
        }
        result = await plugin.execute(context, mock_pipeline)

    assert result.word_count == 400
    assert result.reading_time == 2.0


@pytest.mark.asyncio
async def test_execute_content_summary_long(plugin, mock_pipeline):
    """content_summary truncates at 500 chars with '...' suffix."""
    llm_result = _make_llm_result()
    content = "x" * 600

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": content},
        }
        result = await plugin.execute(context, mock_pipeline)

    assert result.content_summary == "x" * 500 + "..."


@pytest.mark.asyncio
async def test_execute_content_summary_short(plugin, mock_pipeline):
    """content_summary equals content_str when content is ≤ 500 chars."""
    llm_result = _make_llm_result()
    content = "Short content."

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": content},
        }
        result = await plugin.execute(context, mock_pipeline)

    assert result.content_summary == "Short content."


@pytest.mark.asyncio
async def test_execute_confidence_score_defaults_to_one(plugin, mock_pipeline):
    """confidence_score defaults to 1.0 when LLM returns None."""
    llm_result = _make_llm_result(confidence_score=None)

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": "Hello world"},
        }
        result = await plugin.execute(context, mock_pipeline)

    assert result.confidence_score == 1.0


@pytest.mark.asyncio
async def test_execute_confidence_score_preserved_when_set(plugin, mock_pipeline):
    """confidence_score from LLM is preserved if not None."""
    llm_result = _make_llm_result(confidence_score=0.72)

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": "Hello world"},
        }
        result = await plugin.execute(context, mock_pipeline)

    assert result.confidence_score == 0.72


@pytest.mark.asyncio
async def test_execute_empty_content_skips_computation(plugin, mock_pipeline):
    """Empty content string: word_count/reading_time/content_summary stay None."""
    llm_result = _make_llm_result()

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": ""},
        }
        result = await plugin.execute(context, mock_pipeline)

    assert result.word_count is None
    assert result.reading_time is None
    assert result.content_summary is None
    assert result.confidence_score == 1.0  # still gets default


# ---------------------------------------------------------------------------
# execute() — merge back into input_content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_merges_author_when_missing(plugin, mock_pipeline):
    """Author extracted by LLM is merged into input_content when missing."""
    llm_result = _make_llm_result(author="Alice Smith")

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": "Hello"},
        }
        result = await plugin.execute(context, mock_pipeline)

    assert result.author == "Alice Smith"
    assert context["input_content"]["author"] == "Alice Smith"


@pytest.mark.asyncio
async def test_execute_does_not_overwrite_existing_author(plugin, mock_pipeline):
    """Author in input_content is preserved even if LLM extracted a different one."""
    llm_result = _make_llm_result(author="LLM Author")

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {
                "title": "T",
                "content": "Hello",
                "author": "Original Author",
            },
        }
        await plugin.execute(context, mock_pipeline)

    assert context["input_content"]["author"] == "Original Author"


@pytest.mark.asyncio
async def test_execute_merges_category(plugin, mock_pipeline):
    """Category extracted by LLM is merged into input_content when missing."""
    llm_result = _make_llm_result(category="Engineering")

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": "Hello"},
        }
        result = await plugin.execute(context, mock_pipeline)

    assert result.category == "Engineering"
    assert context["input_content"]["category"] == "Engineering"


@pytest.mark.asyncio
async def test_execute_merges_tags_when_input_empty(plugin, mock_pipeline):
    """Tags extracted by LLM are merged when input_content.tags is empty."""
    llm_result = _make_llm_result(tags=["python", "api"])

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": "Hello", "tags": []},
        }
        result = await plugin.execute(context, mock_pipeline)

    assert result.tags == ["python", "api"]
    assert context["input_content"]["tags"] == ["python", "api"]


@pytest.mark.asyncio
async def test_execute_merges_word_count_to_input_content(plugin, mock_pipeline):
    """word_count is written back to input_content."""
    llm_result = _make_llm_result()
    content = " ".join(["word"] * 100)

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": content},
        }
        await plugin.execute(context, mock_pipeline)

    assert context["input_content"]["word_count"] == 100


@pytest.mark.asyncio
async def test_execute_merges_reading_time_to_input_content(plugin, mock_pipeline):
    """reading_time is written back to input_content."""
    llm_result = _make_llm_result()
    content = " ".join(["word"] * 200)

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": content},
        }
        await plugin.execute(context, mock_pipeline)

    assert context["input_content"]["reading_time"] == 1.0


@pytest.mark.asyncio
async def test_execute_sets_snippet_from_content_summary(plugin, mock_pipeline):
    """content_summary is written as input_content['snippet'] when snippet missing."""
    llm_result = _make_llm_result()
    content = "A" * 300

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": {"title": "T", "content": content},
        }
        await plugin.execute(context, mock_pipeline)

    snippet = context["input_content"].get("snippet", "")
    assert len(snippet) <= 200
    assert snippet  # non-empty


@pytest.mark.asyncio
async def test_execute_input_content_not_dict_does_not_raise(plugin, mock_pipeline):
    """If input_content is None, merge block is skipped without AttributeError."""
    llm_result = _make_llm_result()

    with patch.object(plugin, "_execute_step", return_value=llm_result):
        context = {
            "content_type": "blog_post",
            "input_content": None,
        }
        result = await plugin.execute(context, mock_pipeline)

    assert isinstance(result, BlogPostPreprocessingApprovalResult)
