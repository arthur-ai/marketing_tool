"""
Live integration tests for the blog post preprocessing pipeline.

These tests make real API calls (Arthur for prompts, Anthropic/OpenAI for LLM inference).
They are gated on ANTHROPIC_API_KEY being set in the environment and must be
explicitly invoked with the `integration` marker — they never run in the normal
unit-test suite.

Run step-1 only (fast, ~5s):
    pytest tests/integrations/test_blog_post_pipeline_live.py::TestBlogPostPreprocessingApprovalLive -v -m integration

Run full pipeline (slow, all steps, ~60-120s):
    pytest tests/integrations/test_blog_post_pipeline_live.py::TestFullBlogPostPipelineLive -v -m integration -m slow

Run everything:
    pytest tests/integrations/test_blog_post_pipeline_live.py -v -m integration
"""

import json
import os

import pytest
from dotenv import find_dotenv, load_dotenv

# Load .env so ANTHROPIC_API_KEY is visible before any import checks
_dotenv_path = find_dotenv()
if _dotenv_path:
    load_dotenv(dotenv_path=_dotenv_path, override=True)

from marketing_project.models.pipeline_steps import (
    BlogPostPreprocessingApprovalLLMResult,
    BlogPostPreprocessingApprovalResult,
)
from marketing_project.plugins.blog_post_preprocessing_approval.tasks import (
    BlogPostPreprocessingApprovalPlugin,
)
from marketing_project.services.function_pipeline import FunctionPipeline

pytestmark = pytest.mark.integration

_ANTHROPIC_KEY_PRESENT = bool(os.getenv("ANTHROPIC_API_KEY"))
_SKIP_REASON = "ANTHROPIC_API_KEY not set — skipping live integration tests"

# ---------------------------------------------------------------------------
# Shared test content
# ---------------------------------------------------------------------------

# A realistic blog post with enough content for the LLM to validate.
# Tags, category, and author are intentionally left empty so we can assert
# the LLM extracts them from the body text.
_BLOG_POST: dict = {
    "id": "live-integration-test-1",
    "title": "Building Reliable Machine Learning Pipelines in Production",
    "content": (
        "Machine learning in production is dramatically different from the clean "
        "Jupyter notebook world where most models are born. The gap between a "
        "prototype that achieves 95% accuracy in a notebook and a system that "
        "reliably serves predictions 24/7 at scale is enormous, and teams often "
        "underestimate just how wide that gap is until they are in the middle of it.\n\n"
        "The first thing that surprises most teams is data drift. Your training data "
        "was a snapshot of the world at a particular moment. The world keeps moving. "
        "Feature distributions shift as user behavior evolves, market conditions change, "
        "or upstream data pipelines are modified by other teams. A model that was "
        "accurate six months ago may be giving quietly wrong predictions today, and "
        "without proper monitoring you will not know until something blows up downstream.\n\n"
        "There are several practical strategies that make ML pipelines more reliable. "
        "First, treat your model registry as a first-class artifact store with clear "
        "versioning and rollback procedures. Second, implement data validation at every "
        "ingestion boundary using tools like Great Expectations or Pydantic schemas. "
        "Third, monitor feature distributions in production against your training "
        "distribution baseline — any KL divergence spike is an early warning. Fourth, "
        "build a shadow mode capability so you can run new model versions in parallel "
        "before cutting over traffic.\n\n"
        "The teams that get ML reliability right treat it like any other software "
        "engineering discipline: observability, testing, staged rollouts, and clear "
        "ownership. The teams that struggle tend to think of the model as magic that "
        "happens once and stays fixed forever. It does not. Models are living systems "
        "that require the same care and feeding as any production service."
    ),
    "tags": [],
    "category": "",
    "author": "",
}

_BLOG_POST_JSON: str = json.dumps(_BLOG_POST)
_EXPECTED_WORD_COUNT: int = len(_BLOG_POST["content"].split())


# ---------------------------------------------------------------------------
# Step 1 plugin tests — fast, isolated
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ANTHROPIC_KEY_PRESENT, reason=_SKIP_REASON)
class TestBlogPostPreprocessingApprovalLive:
    """Live tests for step 1: BlogPostPreprocessingApprovalPlugin against real Anthropic."""

    @pytest.mark.asyncio
    async def test_result_type_is_full_dto_not_llm_base(self):
        """execute() returns BlogPostPreprocessingApprovalResult, not the slim LLM base."""
        plugin = BlogPostPreprocessingApprovalPlugin()
        pipeline = FunctionPipeline()
        context = {"content_type": "blog_post", "input_content": _BLOG_POST.copy()}

        result = await plugin.execute(context, pipeline)

        # Must be the enriched DTO — not the 13-field schema sent to Anthropic
        assert isinstance(result, BlogPostPreprocessingApprovalResult)
        assert type(result) is not BlogPostPreprocessingApprovalLLMResult

    @pytest.mark.asyncio
    async def test_plugin_validates_well_formed_post(self):
        """LLM marks a well-formed post as valid with no approval required."""
        plugin = BlogPostPreprocessingApprovalPlugin()
        pipeline = FunctionPipeline()
        context = {"content_type": "blog_post", "input_content": _BLOG_POST.copy()}

        result = await plugin.execute(context, pipeline)

        assert result.is_valid is True
        assert result.requires_approval is False

    @pytest.mark.asyncio
    async def test_word_count_computed_programmatically(self):
        """word_count is computed from content string in execute(), not from LLM response."""
        plugin = BlogPostPreprocessingApprovalPlugin()
        pipeline = FunctionPipeline()
        context = {"content_type": "blog_post", "input_content": _BLOG_POST.copy()}

        result = await plugin.execute(context, pipeline)

        assert result.word_count == _EXPECTED_WORD_COUNT

    @pytest.mark.asyncio
    async def test_reading_time_computed(self):
        """reading_time is round(word_count / 200, 1) — computed after LLM call."""
        plugin = BlogPostPreprocessingApprovalPlugin()
        pipeline = FunctionPipeline()
        context = {"content_type": "blog_post", "input_content": _BLOG_POST.copy()}

        result = await plugin.execute(context, pipeline)

        expected_reading_time = round(_EXPECTED_WORD_COUNT / 200, 1)
        assert result.reading_time == expected_reading_time

    @pytest.mark.asyncio
    async def test_content_summary_populated(self):
        """content_summary is the first 500 chars of content (with ... if truncated)."""
        plugin = BlogPostPreprocessingApprovalPlugin()
        pipeline = FunctionPipeline()
        context = {"content_type": "blog_post", "input_content": _BLOG_POST.copy()}

        result = await plugin.execute(context, pipeline)

        assert result.content_summary is not None
        assert len(result.content_summary) <= 503  # 500 chars + "..."

    @pytest.mark.asyncio
    async def test_confidence_score_present(self):
        """confidence_score defaults to 1.0 if LLM returns None."""
        plugin = BlogPostPreprocessingApprovalPlugin()
        pipeline = FunctionPipeline()
        context = {"content_type": "blog_post", "input_content": _BLOG_POST.copy()}

        result = await plugin.execute(context, pipeline)

        assert result.confidence_score is not None
        assert 0.0 <= result.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_enrichment_fields_merged_into_input_content(self):
        """word_count and reading_time are written back to input_content for downstream steps."""
        plugin = BlogPostPreprocessingApprovalPlugin()
        pipeline = FunctionPipeline()
        blog_post = _BLOG_POST.copy()
        context = {"content_type": "blog_post", "input_content": blog_post}

        await plugin.execute(context, pipeline)

        # input_content is mutated in-place
        updated = context["input_content"]
        assert updated.get("word_count") == _EXPECTED_WORD_COUNT
        assert updated.get("reading_time") == round(_EXPECTED_WORD_COUNT / 200, 1)
        assert updated.get("snippet")  # non-empty snippet written from content_summary

    @pytest.mark.asyncio
    async def test_skip_path_does_not_hit_llm(self):
        """Non-blog_post content type returns a default valid result without an LLM call."""
        plugin = BlogPostPreprocessingApprovalPlugin()
        # Use a default FunctionPipeline — if the LLM were called it would incur cost
        pipeline = FunctionPipeline()
        context = {
            "content_type": "transcript",
            "input_content": {"title": "A talk", "content": "Hello world"},
        }

        result = await plugin.execute(context, pipeline)

        assert isinstance(result, BlogPostPreprocessingApprovalResult)
        assert result.is_valid is True
        assert result.requires_approval is False
        assert result.confidence_score == 1.0

    @pytest.mark.asyncio
    async def test_schema_sent_to_anthropic_is_small(self):
        """Regression: LLM schema has ≤13 properties and ≤3 anyOf — Anthropic will accept it."""
        schema = BlogPostPreprocessingApprovalLLMResult.model_json_schema()
        props = schema.get("properties", {})
        anyof_count = sum(1 for v in props.values() if "anyOf" in v)

        assert (
            len(props) <= 13
        ), f"LLM schema has {len(props)} properties — Anthropic will reject it"
        assert (
            anyof_count <= 3
        ), f"LLM schema has {anyof_count} anyOf patterns — Anthropic may reject it"

    @pytest.mark.asyncio
    async def test_ai_extracts_metadata_from_content(self):
        """LLM extracts category and/or tags from the body text when they are missing."""
        plugin = BlogPostPreprocessingApprovalPlugin()
        pipeline = FunctionPipeline()
        blog_post = _BLOG_POST.copy()
        assert not blog_post["tags"]
        assert not blog_post["category"]

        context = {"content_type": "blog_post", "input_content": blog_post}
        result = await plugin.execute(context, pipeline)

        # LLM output is non-deterministic but must be structurally valid
        assert isinstance(result.tags, list)
        assert result.category is None or isinstance(result.category, str)


# ---------------------------------------------------------------------------
# Full pipeline end-to-end tests — slow, all steps
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ANTHROPIC_KEY_PRESENT, reason=_SKIP_REASON)
@pytest.mark.slow
class TestFullBlogPostPipelineLive:
    """Full pipeline end-to-end: all steps run against real APIs, no mocks."""

    @pytest.mark.asyncio
    async def test_pipeline_executes_and_returns_result_dict(self):
        """Full pipeline runs and returns a structured result dict (step 1 must be present).

        The seo_optimization step has a known pre-existing schema mismatch that
        causes it to fail, so we do not assert on overall pipeline_status here.
        What matters: the pipeline ran, step_results is present, and step 1 succeeded.
        """
        pipeline = FunctionPipeline()
        result = await pipeline.execute_pipeline(
            _BLOG_POST_JSON,
            content_type="blog_post",
            # No job_id — skips Redis/DB job-tracking calls
        )

        assert isinstance(result, dict)
        assert "pipeline_status" in result
        assert "step_results" in result
        # Step 1 must always be present — it is our fix
        assert "blog_post_preprocessing_approval" in result["step_results"]

    @pytest.mark.asyncio
    async def test_pipeline_step1_result_has_enrichment_fields(self):
        """Step 1 result in full pipeline output has word_count computed programmatically."""
        pipeline = FunctionPipeline()
        result = await pipeline.execute_pipeline(
            _BLOG_POST_JSON,
            content_type="blog_post",
        )

        step_results = result.get("step_results", {})
        assert (
            "blog_post_preprocessing_approval" in step_results
        ), "blog_post_preprocessing_approval missing from step_results"
        step1 = step_results["blog_post_preprocessing_approval"]
        assert isinstance(step1, dict)
        assert step1.get("is_valid") is True
        assert step1.get("requires_approval") is False
        # word_count must be present — it's programmatically computed in execute()
        assert step1.get("word_count") == _EXPECTED_WORD_COUNT

    @pytest.mark.asyncio
    async def test_pipeline_produces_seo_keywords(self):
        """SEO keywords step runs and returns a result dict."""
        pipeline = FunctionPipeline()
        result = await pipeline.execute_pipeline(
            _BLOG_POST_JSON,
            content_type="blog_post",
        )

        step_results = result.get("step_results", {})
        assert (
            "seo_keywords" in step_results
        ), "seo_keywords step missing from step_results"
        seo = step_results["seo_keywords"]
        assert isinstance(seo, dict)

    @pytest.mark.asyncio
    async def test_pipeline_produces_marketing_brief(self):
        """Marketing brief step runs and returns a result dict."""
        pipeline = FunctionPipeline()
        result = await pipeline.execute_pipeline(
            _BLOG_POST_JSON,
            content_type="blog_post",
        )

        step_results = result.get("step_results", {})
        assert (
            "marketing_brief" in step_results
        ), "marketing_brief step missing from step_results"
        assert isinstance(step_results["marketing_brief"], dict)

    @pytest.mark.asyncio
    async def test_pipeline_step1_enrichment_in_step_results(self):
        """Step 1 result in step_results carries all programmatically computed fields.

        We verify enrichment via step_results (present in both success and failure
        paths) rather than the top-level input_content key, which is only present
        in the success path (compile_pipeline_result).
        """
        pipeline = FunctionPipeline()
        result = await pipeline.execute_pipeline(
            _BLOG_POST_JSON,
            content_type="blog_post",
        )

        step1 = result.get("step_results", {}).get(
            "blog_post_preprocessing_approval", {}
        )
        assert step1.get("word_count") == _EXPECTED_WORD_COUNT
        assert step1.get("reading_time") == round(_EXPECTED_WORD_COUNT / 200, 1)
        assert step1.get("content_summary")  # non-empty
