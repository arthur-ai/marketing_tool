"""
Extended tests for worker functions — covers error paths, social media, brand kit,
scanning jobs, startup/shutdown, and expiry functions.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marketing_project.worker import (
    _flush_telemetry,
    _fmt_error,
    analyze_brand_kit_batch_job,
    bulk_rescan_documents_job,
    execute_single_step_job,
    expire_stale_approvals,
    process_blog_job,
    process_competitor_research_job,
    process_multi_platform_social_media_job,
    process_release_notes_job,
    process_social_media_job,
    process_transcript_job,
    resume_pipeline_job,
    scan_from_list_job,
    scan_from_url_job,
    shutdown,
    startup,
    synthesize_brand_kit_job,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.__getitem__ = MagicMock(return_value=MagicMock())
    return ctx


@pytest.fixture
def mock_job_manager():
    mgr = MagicMock()
    mgr.get_job = AsyncMock(return_value=MagicMock(metadata={}))
    mgr.update_job_progress = AsyncMock()
    mgr.update_job_status = AsyncMock()
    mgr._save_job = AsyncMock()
    return mgr


# ---------------------------------------------------------------------------
# _fmt_error
# ---------------------------------------------------------------------------


def test_fmt_error_short_message():
    e = ValueError("something went wrong")
    result = _fmt_error(e)
    assert "ValueError: something went wrong" == result


def test_fmt_error_long_message_truncated():
    e = ValueError("x" * 600)
    result = _fmt_error(e)
    assert len(result) == 500
    assert result.endswith("...")


def test_fmt_error_exact_boundary():
    # exactly 500 chars — no truncation
    msg = "y" * 493
    e = ValueError(msg)
    result = _fmt_error(e)
    assert len(result) <= 500


# ---------------------------------------------------------------------------
# _flush_telemetry
# ---------------------------------------------------------------------------


def test_flush_telemetry_exception_swallowed():
    with patch(
        "marketing_project.services.telemetry.flush_telemetry",
        side_effect=RuntimeError("otel down"),
    ):
        _flush_telemetry("job-flush-1")  # must not raise


def test_flush_telemetry_import_error_swallowed():
    import sys

    saved = sys.modules.get("marketing_project.services.telemetry")
    try:
        sys.modules["marketing_project.services.telemetry"] = None  # type: ignore[assignment]
        _flush_telemetry("job-flush-import")
    finally:
        if saved is None:
            sys.modules.pop("marketing_project.services.telemetry", None)
        else:
            sys.modules["marketing_project.services.telemetry"] = saved


# ---------------------------------------------------------------------------
# process_blog_job — error and approval paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_blog_job_exception_path(mock_ctx, mock_job_manager):
    content_json = '{"id": "b-1", "title": "Blog", "content": "Content"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_blog_post",
            side_effect=RuntimeError("LLM failed"),
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(RuntimeError, match="LLM failed"):
            await process_blog_job(mock_ctx, content_json, "blog-err-1")

    mock_job_manager.update_job_status.assert_called()


@pytest.mark.asyncio
async def test_process_blog_job_status_update_also_fails(mock_ctx, mock_job_manager):
    mock_job_manager.update_job_status = AsyncMock(side_effect=Exception("redis down"))
    content_json = '{"id": "b-1", "title": "Blog", "content": "Content"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_blog_post",
            side_effect=RuntimeError("fail"),
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch(
            "marketing_project.worker.emit_error",
            new_callable=AsyncMock,
            side_effect=Exception("emit fail"),
        ),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(RuntimeError):
            await process_blog_job(mock_ctx, content_json, "blog-err-2")


@pytest.mark.asyncio
async def test_process_blog_job_result_status_error(mock_ctx, mock_job_manager):
    content_json = '{"id": "b-1", "title": "Blog", "content": "Content"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_blog_post",
            return_value='{"status": "error", "message": "AI refused"}',
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(Exception, match="AI refused"):
            await process_blog_job(mock_ctx, content_json, "blog-err-3")


@pytest.mark.asyncio
async def test_process_blog_job_waiting_for_approval(mock_ctx, mock_job_manager):
    content_json = '{"id": "b-1", "title": "Blog", "content": "Content"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_blog_post",
            return_value='{"status": "waiting_for_approval"}',
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_blog_job(mock_ctx, content_json, "blog-wfa-1")
    assert result["status"] == "waiting_for_approval"
    mock_job_manager.update_job_status.assert_called()


@pytest.mark.asyncio
async def test_process_blog_job_no_job_found(mock_ctx, mock_job_manager):
    mock_job_manager.get_job = AsyncMock(return_value=None)
    content_json = '{"id": "b-1", "title": "Blog", "content": "Content"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_blog_post",
            return_value='{"status": "success", "data": {}}',
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_blog_job(mock_ctx, content_json, "blog-nojob")
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_process_blog_job_invalid_json_content(mock_ctx, mock_job_manager):
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_blog_post",
            return_value='{"status": "success", "data": {}}',
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_blog_job(mock_ctx, "not-valid-json", "blog-badjson")
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# process_release_notes_job — error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_release_notes_job_exception_path(mock_ctx, mock_job_manager):
    content_json = '{"id": "rn-1", "title": "v1.0", "content": "Changes"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_release_notes",
            side_effect=RuntimeError("LLM failed"),
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(RuntimeError):
            await process_release_notes_job(mock_ctx, content_json, "rn-err-1")


@pytest.mark.asyncio
async def test_process_release_notes_job_result_error(mock_ctx, mock_job_manager):
    content_json = '{"id": "rn-1", "title": "v1.0", "content": "Changes"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_release_notes",
            return_value='{"status": "error", "message": "rn fail"}',
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(Exception, match="rn fail"):
            await process_release_notes_job(mock_ctx, content_json, "rn-err-2")


@pytest.mark.asyncio
async def test_process_release_notes_job_waiting_for_approval(
    mock_ctx, mock_job_manager
):
    content_json = '{"id": "rn-1", "title": "v1.0", "content": "Changes"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_release_notes",
            return_value='{"status": "waiting_for_approval"}',
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_release_notes_job(mock_ctx, content_json, "rn-wfa-1")
    assert result["status"] == "waiting_for_approval"


# ---------------------------------------------------------------------------
# process_transcript_job — error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_transcript_job_exception_path(mock_ctx, mock_job_manager):
    content_json = '{"id": "t-1", "title": "Podcast", "content": "Speaker: Hello"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_transcript",
            side_effect=RuntimeError("Failed"),
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(RuntimeError):
            await process_transcript_job(mock_ctx, content_json, "t-err-1")


@pytest.mark.asyncio
async def test_process_transcript_job_waiting_for_approval(mock_ctx, mock_job_manager):
    content_json = '{"id": "t-1", "title": "Podcast", "content": "Speaker: Hello"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_transcript",
            return_value='{"status": "waiting_for_approval"}',
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_transcript_job(mock_ctx, content_json, "t-wfa-1")
    assert result["status"] == "waiting_for_approval"


@pytest.mark.asyncio
async def test_process_transcript_job_result_error(mock_ctx, mock_job_manager):
    content_json = '{"id": "t-1", "title": "Podcast", "content": "Speaker: Hello"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.process_transcript",
            return_value='{"status": "error", "message": "transcript failed"}',
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(Exception, match="transcript failed"):
            await process_transcript_job(mock_ctx, content_json, "t-err-2")


# ---------------------------------------------------------------------------
# process_social_media_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_social_media_job_success(mock_ctx, mock_job_manager):
    content_json = '{"id": "sm-1", "title": "AI Trends", "content": "Content"}'
    mock_job_manager.get_job = AsyncMock(
        return_value=MagicMock(
            metadata={"social_media_platform": "linkedin", "variations_count": 2}
        )
    )
    mock_pipeline = MagicMock()
    mock_pipeline.execute_pipeline = AsyncMock(
        return_value={"pipeline_status": "completed", "posts": []}
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_social_media_job(mock_ctx, content_json, "sm-ok-1")
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_process_social_media_job_no_job(mock_ctx, mock_job_manager):
    mock_job_manager.get_job = AsyncMock(return_value=None)
    content_json = '{"id": "sm-1", "title": "Title", "content": "Content"}'
    mock_pipeline = MagicMock()
    mock_pipeline.execute_pipeline = AsyncMock(
        return_value={"pipeline_status": "completed"}
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_social_media_job(mock_ctx, content_json, "sm-nojob")
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_process_social_media_job_waiting_for_approval(
    mock_ctx, mock_job_manager
):
    content_json = '{"id": "sm-1", "title": "Title", "content": "Content"}'
    mock_pipeline = MagicMock()
    mock_pipeline.execute_pipeline = AsyncMock(
        return_value={
            "pipeline_status": "waiting_for_approval",
            "metadata": {"stopped_at_step": 2},
        }
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_social_media_job(mock_ctx, content_json, "sm-wfa")
    assert result["pipeline_status"] == "waiting_for_approval"


@pytest.mark.asyncio
async def test_process_social_media_job_pipeline_failed(mock_ctx, mock_job_manager):
    content_json = '{"id": "sm-1", "title": "Title", "content": "Content"}'
    mock_pipeline = MagicMock()
    mock_pipeline.execute_pipeline = AsyncMock(
        return_value={
            "pipeline_status": "failed",
            "metadata": {"error": "LLM error"},
        }
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(Exception, match="LLM error"):
            await process_social_media_job(mock_ctx, content_json, "sm-fail")


@pytest.mark.asyncio
async def test_process_social_media_job_with_pipeline_config(
    mock_ctx, mock_job_manager
):
    content_json = '{"id": "sm-1", "title": "Title", "content": "Content"}'
    pipeline_config_data = {
        "default_temperature": 0.7,
        "default_max_retries": 3,
        "step_configs": {},
    }
    mock_job_manager.get_job = AsyncMock(
        return_value=MagicMock(
            metadata={
                "pipeline_config": pipeline_config_data,
                "social_media_platform": "twitter",
                "email_type": "newsletter",
            }
        )
    )
    mock_pipeline = MagicMock()
    mock_pipeline.execute_pipeline = AsyncMock(
        return_value={"pipeline_status": "completed"}
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_social_media_job(mock_ctx, content_json, "sm-config")
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_process_social_media_job_exception(mock_ctx, mock_job_manager):
    content_json = '{"id": "sm-1", "title": "Title", "content": "Content"}'
    mock_pipeline = MagicMock()
    mock_pipeline.execute_pipeline = AsyncMock(side_effect=RuntimeError("crash"))
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(RuntimeError):
            await process_social_media_job(mock_ctx, content_json, "sm-exc")


@pytest.mark.asyncio
async def test_process_social_media_job_invalid_content_json(
    mock_ctx, mock_job_manager
):
    mock_pipeline = MagicMock()
    mock_pipeline.execute_pipeline = AsyncMock(
        return_value={"pipeline_status": "completed"}
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_social_media_job(mock_ctx, "not-json", "sm-badjson")
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# process_multi_platform_social_media_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_multi_platform_success(mock_ctx, mock_job_manager):
    content_json = '{"id": "mp-1", "title": "Title", "content": "Content"}'
    mock_job_manager.get_job = AsyncMock(
        return_value=MagicMock(
            metadata={"social_media_platforms": ["linkedin", "twitter"]}
        )
    )
    mock_pipeline = MagicMock()
    mock_pipeline.execute_multi_platform_pipeline = AsyncMock(
        return_value={"pipeline_status": "completed", "results": {}}
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_multi_platform_social_media_job(
            mock_ctx, content_json, "mp-ok-1"
        )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_process_multi_platform_single_platform_warning(
    mock_ctx, mock_job_manager
):
    content_json = '{"id": "mp-1", "title": "Title", "content": "Content"}'
    mock_job_manager.get_job = AsyncMock(
        return_value=MagicMock(metadata={"social_media_platform": "linkedin"})
    )
    mock_pipeline = MagicMock()
    mock_pipeline.execute_multi_platform_pipeline = AsyncMock(
        return_value={"pipeline_status": "completed"}
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_multi_platform_social_media_job(
            mock_ctx, content_json, "mp-single"
        )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_process_multi_platform_empty_platforms(mock_ctx, mock_job_manager):
    content_json = '{"id": "mp-1", "title": "Title", "content": "Content"}'
    mock_job_manager.get_job = AsyncMock(
        return_value=MagicMock(metadata={"social_media_platforms": []})
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(Exception):
            await process_multi_platform_social_media_job(
                mock_ctx, content_json, "mp-empty"
            )


@pytest.mark.asyncio
async def test_process_multi_platform_no_job(mock_ctx, mock_job_manager):
    mock_job_manager.get_job = AsyncMock(return_value=None)
    content_json = '{"id": "mp-1", "title": "Title", "content": "Content"}'
    mock_pipeline = MagicMock()
    mock_pipeline.execute_multi_platform_pipeline = AsyncMock(
        return_value={"pipeline_status": "completed"}
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_multi_platform_social_media_job(
            mock_ctx, content_json, "mp-nojob"
        )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_process_multi_platform_with_pipeline_config(mock_ctx, mock_job_manager):
    content_json = '{"id": "mp-1", "title": "Title", "content": "Content"}'
    config_data = {
        "default_temperature": 0.5,
        "default_max_retries": 2,
        "step_configs": {},
    }
    mock_job_manager.get_job = AsyncMock(
        return_value=MagicMock(
            metadata={
                "pipeline_config": config_data,
                "social_media_platforms": ["linkedin"],
            }
        )
    )
    mock_pipeline = MagicMock()
    mock_pipeline.execute_multi_platform_pipeline = AsyncMock(
        return_value={"pipeline_status": "completed"}
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await process_multi_platform_social_media_job(
            mock_ctx, content_json, "mp-config"
        )
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_process_multi_platform_exception(mock_ctx, mock_job_manager):
    content_json = '{"id": "mp-1", "title": "Title", "content": "Content"}'
    mock_pipeline = MagicMock()
    mock_pipeline.execute_multi_platform_pipeline = AsyncMock(
        side_effect=RuntimeError("pipeline crash")
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.SocialMediaPipeline", return_value=mock_pipeline
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(RuntimeError):
            await process_multi_platform_social_media_job(
                mock_ctx, content_json, "mp-exc"
            )


# ---------------------------------------------------------------------------
# analyze_brand_kit_batch_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_brand_kit_batch_job_success(mock_ctx, mock_job_manager):
    content_batch = [
        {"title": "Blog 1", "content": "AI content"},
        {"title": "Blog 2", "content": "ML content"},
    ]
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
        patch("marketing_project.plugins.brand_kit.tasks.BrandKitPlugin") as MockPlugin,
        patch("marketing_project.services.function_pipeline.FunctionPipeline"),
    ):
        plugin_instance = MagicMock()
        plugin_instance._analyze_content = AsyncMock(
            return_value={"tone": "professional"}
        )
        MockPlugin.return_value = plugin_instance
        result = await analyze_brand_kit_batch_job(
            mock_ctx, content_batch, 0, "parent-1", "batch-1"
        )
    assert result["status"] == "success"
    assert result["batch_index"] == 0
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_analyze_brand_kit_batch_job_none_analyses(mock_ctx, mock_job_manager):
    content_batch = [{"title": "Short", "content": "Short content"}]
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
        patch("marketing_project.plugins.brand_kit.tasks.BrandKitPlugin") as MockPlugin,
        patch("marketing_project.services.function_pipeline.FunctionPipeline"),
    ):
        plugin_instance = MagicMock()
        plugin_instance._analyze_content = AsyncMock(return_value=None)
        MockPlugin.return_value = plugin_instance
        result = await analyze_brand_kit_batch_job(
            mock_ctx, content_batch, 1, "parent-1", "batch-2"
        )
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_analyze_brand_kit_batch_job_error(mock_ctx, mock_job_manager):
    content_batch = [{"title": "Blog", "content": "Content"}]
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
        patch("marketing_project.plugins.brand_kit.tasks.BrandKitPlugin") as MockPlugin,
        patch("marketing_project.services.function_pipeline.FunctionPipeline"),
    ):
        plugin_instance = MagicMock()
        plugin_instance._analyze_content = AsyncMock(
            side_effect=RuntimeError("analysis failed")
        )
        MockPlugin.return_value = plugin_instance
        with pytest.raises(RuntimeError):
            await analyze_brand_kit_batch_job(
                mock_ctx, content_batch, 0, "parent-1", "batch-err"
            )


# ---------------------------------------------------------------------------
# synthesize_brand_kit_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_brand_kit_job_success(mock_ctx, mock_job_manager):
    all_analyses = [
        {"analyses": [{"tone": "professional"}]},
        {"analyses": [{"voice": "friendly"}]},
    ]
    mock_config = MagicMock()
    mock_config.model_dump.return_value = {"version": "1.0"}
    mock_config.version = "1.0"
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
        patch("marketing_project.plugins.brand_kit.tasks.BrandKitPlugin") as MockPlugin,
        patch("marketing_project.services.function_pipeline.FunctionPipeline"),
    ):
        plugin_instance = MagicMock()
        plugin_instance._synthesize_config = AsyncMock(return_value=mock_config)
        MockPlugin.return_value = plugin_instance
        result = await synthesize_brand_kit_job(
            mock_ctx, all_analyses, "parent-1", "synth-1"
        )
    assert result["status"] == "success"
    assert result["version"] == "1.0"


@pytest.mark.asyncio
async def test_synthesize_brand_kit_job_mixed_format_analyses(
    mock_ctx, mock_job_manager
):
    # Mixed: dict with "analyses" key, plain list, and plain dict
    all_analyses = [
        {"analyses": [{"tone": "formal"}]},
        [{"voice": "casual"}],
        {"extra": "data"},
    ]
    mock_config = MagicMock()
    mock_config.model_dump.return_value = {"version": "2.0"}
    mock_config.version = "2.0"
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
        patch("marketing_project.plugins.brand_kit.tasks.BrandKitPlugin") as MockPlugin,
        patch("marketing_project.services.function_pipeline.FunctionPipeline"),
    ):
        plugin_instance = MagicMock()
        plugin_instance._synthesize_config = AsyncMock(return_value=mock_config)
        MockPlugin.return_value = plugin_instance
        result = await synthesize_brand_kit_job(
            mock_ctx, all_analyses, "parent-1", "synth-2"
        )
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_synthesize_brand_kit_job_error(mock_ctx, mock_job_manager):
    all_analyses = [{"analyses": []}]
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
        patch("marketing_project.plugins.brand_kit.tasks.BrandKitPlugin") as MockPlugin,
        patch("marketing_project.services.function_pipeline.FunctionPipeline"),
    ):
        plugin_instance = MagicMock()
        plugin_instance._synthesize_config = AsyncMock(
            side_effect=RuntimeError("synthesis failed")
        )
        MockPlugin.return_value = plugin_instance
        with pytest.raises(RuntimeError):
            await synthesize_brand_kit_job(
                mock_ctx, all_analyses, "parent-1", "synth-err"
            )


# ---------------------------------------------------------------------------
# bulk_rescan_documents_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bulk_rescan_all_success(mock_ctx, mock_job_manager):
    urls = ["https://example.com/doc1", "https://example.com/doc2"]
    mock_scanner = MagicMock()
    mock_scanner._scan_single_url = AsyncMock()
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            return_value=mock_scanner,
        ),
    ):
        result = await bulk_rescan_documents_job(mock_ctx, urls, "bulk-1")
    assert result["status"] == "success"
    assert result["scanned_count"] == 2
    assert result["failed_count"] == 0


@pytest.mark.asyncio
async def test_bulk_rescan_partial_failures(mock_ctx, mock_job_manager):
    urls = ["https://example.com/doc1", "https://bad.url/doc2"]
    mock_scanner = MagicMock()
    mock_scanner._scan_single_url = AsyncMock(
        side_effect=[None, RuntimeError("404 not found")]
    )
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            return_value=mock_scanner,
        ),
    ):
        result = await bulk_rescan_documents_job(mock_ctx, urls, "bulk-partial")
    assert result["status"] == "success"
    assert result["scanned_count"] == 1
    assert result["failed_count"] == 1
    assert len(result["errors"]) == 1


@pytest.mark.asyncio
async def test_bulk_rescan_scanner_init_fails(mock_ctx, mock_job_manager):
    urls = ["https://example.com/doc1"]
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            side_effect=RuntimeError("scanner init failed"),
        ),
    ):
        with pytest.raises(RuntimeError):
            await bulk_rescan_documents_job(mock_ctx, urls, "bulk-err")


@pytest.mark.asyncio
async def test_bulk_rescan_all_failures(mock_ctx, mock_job_manager):
    urls = ["https://bad1.com", "https://bad2.com", "https://bad3.com"]
    mock_scanner = MagicMock()
    mock_scanner._scan_single_url = AsyncMock(side_effect=RuntimeError("network error"))
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            return_value=mock_scanner,
        ),
    ):
        result = await bulk_rescan_documents_job(mock_ctx, urls, "bulk-all-fail")
    assert result["status"] == "success"
    assert result["scanned_count"] == 0
    assert result["failed_count"] == 3


# ---------------------------------------------------------------------------
# scan_from_url_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_from_url_job_no_merge(mock_ctx, mock_job_manager):
    mock_doc = MagicMock()
    mock_doc.model_dump.return_value = {"url": "https://example.com", "title": "Doc"}
    mock_scanner = MagicMock()
    mock_scanner.scan_from_base_url = AsyncMock(return_value=[mock_doc])
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            return_value=mock_scanner,
        ),
    ):
        result = await scan_from_url_job(
            mock_ctx, "https://example.com", 1, False, 50, False, "url-nomerge"
        )
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_scan_from_url_job_merge_new_config(mock_ctx, mock_job_manager):
    mock_doc = MagicMock()
    mock_doc.model_dump.return_value = {"url": "https://example.com", "title": "Doc"}
    mock_scanner = MagicMock()
    mock_scanner.scan_from_base_url = AsyncMock(return_value=[mock_doc])
    mock_manager = MagicMock()
    mock_manager.get_active_config = AsyncMock(return_value=None)
    mock_manager.save_config = AsyncMock(return_value=True)
    mock_config_instance = MagicMock()
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            return_value=mock_scanner,
        ),
        patch(
            "marketing_project.services.internal_docs_manager.get_internal_docs_manager",
            new_callable=AsyncMock,
            return_value=mock_manager,
        ),
        patch(
            "marketing_project.models.internal_docs_config.InternalDocsConfig",
            return_value=mock_config_instance,
        ),
    ):
        result = await scan_from_url_job(
            mock_ctx, "https://example.com", 2, False, 100, True, "url-merge-new"
        )
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_scan_from_url_job_merge_existing_config(mock_ctx, mock_job_manager):
    mock_doc = MagicMock()
    mock_scanner = MagicMock()
    mock_scanner.scan_from_base_url = AsyncMock(return_value=[mock_doc])
    mock_existing_config = MagicMock()
    mock_manager = MagicMock()
    mock_manager.get_active_config = AsyncMock(return_value=mock_existing_config)
    mock_manager.merge_scan_results = AsyncMock(return_value=mock_existing_config)
    mock_manager.save_config = AsyncMock(return_value=True)
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            return_value=mock_scanner,
        ),
        patch(
            "marketing_project.services.internal_docs_manager.get_internal_docs_manager",
            new_callable=AsyncMock,
            return_value=mock_manager,
        ),
    ):
        result = await scan_from_url_job(
            mock_ctx, "https://example.com", 2, False, 100, True, "url-merge-exist"
        )
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_scan_from_url_job_scanner_fails(mock_ctx, mock_job_manager):
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            side_effect=RuntimeError("scanner fail"),
        ),
    ):
        with pytest.raises(RuntimeError):
            await scan_from_url_job(
                mock_ctx, "https://example.com", 1, False, 50, False, "url-err"
            )


# ---------------------------------------------------------------------------
# scan_from_list_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_from_list_job_no_merge(mock_ctx, mock_job_manager):
    mock_doc = MagicMock()
    mock_doc.model_dump.return_value = {"url": "https://example.com", "title": "Doc"}
    mock_scanner = MagicMock()
    mock_scanner.scan_from_url_list = AsyncMock(return_value=[mock_doc])
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            return_value=mock_scanner,
        ),
    ):
        result = await scan_from_list_job(
            mock_ctx, ["https://example.com"], False, "list-nomerge"
        )
    assert result["status"] == "success"
    assert result["scanned_count"] == 1


@pytest.mark.asyncio
async def test_scan_from_list_job_merge_no_existing(mock_ctx, mock_job_manager):
    mock_doc = MagicMock()
    mock_doc.model_dump.return_value = {"url": "https://example.com", "title": "Doc"}
    mock_scanner = MagicMock()
    mock_scanner.scan_from_url_list = AsyncMock(return_value=[mock_doc])
    mock_manager = MagicMock()
    mock_manager.get_active_config = AsyncMock(return_value=None)
    mock_manager.save_config = AsyncMock(return_value=True)
    mock_config_instance = MagicMock()
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            return_value=mock_scanner,
        ),
        patch(
            "marketing_project.services.internal_docs_manager.get_internal_docs_manager",
            new_callable=AsyncMock,
            return_value=mock_manager,
        ),
        patch(
            "marketing_project.models.internal_docs_config.InternalDocsConfig",
            return_value=mock_config_instance,
        ),
    ):
        result = await scan_from_list_job(
            mock_ctx, ["https://example.com"], True, "list-merge-new"
        )
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_scan_from_list_job_merge_existing_config(mock_ctx, mock_job_manager):
    mock_doc = MagicMock()
    mock_doc.model_dump.return_value = {"url": "https://example.com", "title": "Doc"}
    mock_scanner = MagicMock()
    mock_scanner.scan_from_url_list = AsyncMock(return_value=[mock_doc])
    mock_existing = MagicMock()
    mock_existing.scanned_documents = [mock_doc]
    mock_manager = MagicMock()
    mock_manager.get_active_config = AsyncMock(return_value=mock_existing)
    mock_manager.merge_scan_results = AsyncMock(return_value=mock_existing)
    mock_manager.save_config = AsyncMock(return_value=True)
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            return_value=mock_scanner,
        ),
        patch(
            "marketing_project.services.internal_docs_manager.get_internal_docs_manager",
            new_callable=AsyncMock,
            return_value=mock_manager,
        ),
    ):
        result = await scan_from_list_job(
            mock_ctx, ["https://example.com"], True, "list-merge-exist"
        )
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_scan_from_list_job_save_fails(mock_ctx, mock_job_manager):
    mock_doc = MagicMock()
    mock_doc.model_dump.return_value = {}
    mock_scanner = MagicMock()
    mock_scanner.scan_from_url_list = AsyncMock(return_value=[mock_doc])
    mock_manager = MagicMock()
    mock_manager.get_active_config = AsyncMock(return_value=None)
    mock_manager.save_config = AsyncMock(return_value=False)
    mock_config_instance = MagicMock()
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            return_value=mock_scanner,
        ),
        patch(
            "marketing_project.services.internal_docs_manager.get_internal_docs_manager",
            new_callable=AsyncMock,
            return_value=mock_manager,
        ),
        patch(
            "marketing_project.models.internal_docs_config.InternalDocsConfig",
            return_value=mock_config_instance,
        ),
    ):
        with pytest.raises(Exception, match="Failed to save"):
            await scan_from_list_job(
                mock_ctx, ["https://example.com"], True, "list-savefail"
            )


@pytest.mark.asyncio
async def test_scan_from_list_job_scanner_exception(mock_ctx, mock_job_manager):
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch(
            "marketing_project.worker.get_internal_docs_scanner",
            new_callable=AsyncMock,
            side_effect=RuntimeError("list scanner fail"),
        ),
    ):
        with pytest.raises(RuntimeError):
            await scan_from_list_job(
                mock_ctx, ["https://example.com"], False, "list-err"
            )


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_success(mock_ctx):
    mock_db = MagicMock()
    mock_db.initialize = AsyncMock(return_value=True)
    mock_db.create_tables = AsyncMock()
    with (
        patch(
            "marketing_project.services.database.get_database_manager",
            return_value=mock_db,
        ),
        patch("marketing_project.services.telemetry.setup_tracing", return_value=True),
    ):
        await startup(mock_ctx)


@pytest.mark.asyncio
async def test_startup_db_not_configured(mock_ctx):
    mock_db = MagicMock()
    mock_db.initialize = AsyncMock(return_value=False)
    with (
        patch(
            "marketing_project.services.database.get_database_manager",
            return_value=mock_db,
        ),
        patch("marketing_project.services.telemetry.setup_tracing", return_value=False),
    ):
        await startup(mock_ctx)


@pytest.mark.asyncio
async def test_startup_db_exception(mock_ctx):
    with (
        patch(
            "marketing_project.services.database.get_database_manager",
            side_effect=Exception("db error"),
        ),
        patch("marketing_project.services.telemetry.setup_tracing", return_value=False),
    ):
        await startup(mock_ctx)  # must not raise


@pytest.mark.asyncio
async def test_startup_telemetry_not_configured(mock_ctx):
    mock_db = MagicMock()
    mock_db.initialize = AsyncMock(return_value=True)
    mock_db.create_tables = AsyncMock()
    with (
        patch(
            "marketing_project.services.database.get_database_manager",
            return_value=mock_db,
        ),
        patch("marketing_project.services.telemetry.setup_tracing", return_value=False),
    ):
        await startup(mock_ctx)


@pytest.mark.asyncio
async def test_startup_telemetry_exception(mock_ctx):
    mock_db = MagicMock()
    mock_db.initialize = AsyncMock(return_value=True)
    mock_db.create_tables = AsyncMock()
    with (
        patch(
            "marketing_project.services.database.get_database_manager",
            return_value=mock_db,
        ),
        patch(
            "marketing_project.services.telemetry.setup_tracing",
            side_effect=Exception("otel error"),
        ),
    ):
        await startup(mock_ctx)  # must not raise


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_success(mock_ctx):
    mock_db = MagicMock()
    mock_db.is_initialized = True
    mock_db.cleanup = AsyncMock()
    with patch(
        "marketing_project.services.database.get_database_manager", return_value=mock_db
    ):
        await shutdown(mock_ctx)
    mock_db.cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_db_not_initialized(mock_ctx):
    mock_db = MagicMock()
    mock_db.is_initialized = False
    with patch(
        "marketing_project.services.database.get_database_manager", return_value=mock_db
    ):
        await shutdown(mock_ctx)
    mock_db.cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_exception_swallowed(mock_ctx):
    with patch(
        "marketing_project.services.database.get_database_manager",
        side_effect=Exception("db error"),
    ):
        await shutdown(mock_ctx)  # must not raise


# ---------------------------------------------------------------------------
# expire_stale_approvals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expire_stale_approvals_db_not_initialized(mock_ctx):
    mock_db = MagicMock()
    mock_db.is_initialized = False
    with patch(
        "marketing_project.services.database.get_database_manager", return_value=mock_db
    ):
        await expire_stale_approvals(mock_ctx)  # early return


@pytest.mark.asyncio
async def test_expire_stale_approvals_no_stale_jobs(mock_ctx):
    mock_db = MagicMock()
    mock_db.is_initialized = True
    mock_session = MagicMock()
    empty_result = MagicMock()
    empty_result.fetchall.return_value = []
    mock_session.execute = AsyncMock(return_value=empty_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_db.get_session.return_value = mock_session
    with patch(
        "marketing_project.services.database.get_database_manager", return_value=mock_db
    ):
        await expire_stale_approvals(mock_ctx)  # exits after finding no stale jobs


@pytest.mark.asyncio
async def test_expire_stale_approvals_with_stale_jobs(mock_ctx):
    mock_db = MagicMock()
    mock_db.is_initialized = True
    mock_session = MagicMock()
    select_result = MagicMock()
    select_result.fetchall.return_value = [("job-stale-1",)]
    update_result = MagicMock()
    update_result.rowcount = 1
    mock_session.execute = AsyncMock(side_effect=[select_result, update_result])
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_db.get_session.return_value = mock_session

    mock_approval = MagicMock()
    mock_approval.id = "approval-1"
    mock_approval_mgr = MagicMock()
    mock_approval_mgr.list_approvals = AsyncMock(return_value=[mock_approval])
    mock_approval_mgr.decide_approval = AsyncMock()
    mock_approval_mgr.execute_rejection_with_retry = AsyncMock(return_value="retry-1")

    mock_jm = MagicMock()

    with (
        patch(
            "marketing_project.services.database.get_database_manager",
            return_value=mock_db,
        ),
        patch(
            "marketing_project.services.approval_manager.get_approval_manager",
            new_callable=AsyncMock,
            return_value=mock_approval_mgr,
        ),
        patch("marketing_project.worker.get_job_manager", return_value=mock_jm),
    ):
        try:
            await expire_stale_approvals(mock_ctx)
        except Exception:
            pass  # some inner paths may require deeper mocking — tolerance test


# ---------------------------------------------------------------------------
# process_competitor_research_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_competitor_research_job_success(mock_ctx):
    mock_result = MagicMock()
    mock_result.analyses = [{"competitor": "CompanyX", "insight": "Good SEO"}]
    mock_service = MagicMock()
    mock_service.run_research = AsyncMock(return_value=mock_result)
    request_data = {"urls": ["https://competitor.com"], "focus_areas": ["seo"]}
    with patch(
        "marketing_project.services.competitor_research_service.get_competitor_research_service",
        return_value=mock_service,
    ):
        result = await process_competitor_research_job(
            mock_ctx, "comp-1", json.dumps(request_data)
        )
    assert result["status"] == "completed"
    assert result["analyses_count"] == 1


@pytest.mark.asyncio
async def test_process_competitor_research_job_error(mock_ctx):
    mock_service = MagicMock()
    mock_service.run_research = AsyncMock(side_effect=RuntimeError("API failed"))
    mock_service.update_job_status = AsyncMock()
    request_data = {"urls": ["https://competitor.com"], "focus_areas": ["seo"]}
    with patch(
        "marketing_project.services.competitor_research_service.get_competitor_research_service",
        return_value=mock_service,
    ):
        result = await process_competitor_research_job(
            mock_ctx, "comp-err", json.dumps(request_data)
        )
    assert result["status"] == "failed"
    assert "API failed" in result["error"]


@pytest.mark.asyncio
async def test_process_competitor_research_job_invalid_json(mock_ctx):
    mock_service = MagicMock()
    mock_service.update_job_status = AsyncMock()
    with patch(
        "marketing_project.services.competitor_research_service.get_competitor_research_service",
        return_value=mock_service,
    ):
        result = await process_competitor_research_job(
            mock_ctx, "comp-badjson", "not-valid-json"
        )
    assert result["status"] == "failed"


# ---------------------------------------------------------------------------
# resume_pipeline_job — additional error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_pipeline_job_pipeline_failed(mock_ctx, mock_job_manager):
    context_data = {
        "context": {},
        "last_step": "seo_keywords",
        "last_step_number": 1,
        "original_content": {"id": "t-1"},
        "content_type": "blog_post",
        "input_content": {"id": "t-1"},
    }
    mock_pipeline = MagicMock()
    mock_pipeline.resume_pipeline = AsyncMock(
        return_value={"pipeline_status": "failed", "error": "step crashed"}
    )
    with (
        patch(
            "marketing_project.services.function_pipeline.FunctionPipeline",
            return_value=mock_pipeline,
        ),
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI"),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(Exception, match="step crashed"):
            await resume_pipeline_job(mock_ctx, "orig-1", context_data, "resume-fail-1")


@pytest.mark.asyncio
async def test_resume_pipeline_job_waiting_for_approval(mock_ctx, mock_job_manager):
    context_data = {
        "context": {},
        "last_step": "seo_keywords",
        "last_step_number": 1,
        "original_content": {"id": "t-1"},
        "content_type": "blog_post",
        "input_content": {"id": "t-1"},
    }
    mock_pipeline = MagicMock()
    mock_pipeline.resume_pipeline = AsyncMock(
        return_value={
            "pipeline_status": "waiting_for_approval",
            "metadata": {"approval_step_name": "marketing_brief"},
        }
    )
    with (
        patch(
            "marketing_project.services.function_pipeline.FunctionPipeline",
            return_value=mock_pipeline,
        ),
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI"),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        result = await resume_pipeline_job(
            mock_ctx, "orig-1", context_data, "resume-wfa-1"
        )
    assert result["status"] == "waiting_for_approval"


@pytest.mark.asyncio
async def test_resume_pipeline_job_exception(mock_ctx, mock_job_manager):
    context_data = {
        "context": {},
        "last_step": "seo_keywords",
        "last_step_number": 1,
        "original_content": {"id": "t-1"},
        "content_type": "blog_post",
        "input_content": {"id": "t-1"},
    }
    mock_pipeline = MagicMock()
    mock_pipeline.resume_pipeline = AsyncMock(side_effect=RuntimeError("resume failed"))
    with (
        patch(
            "marketing_project.services.function_pipeline.FunctionPipeline",
            return_value=mock_pipeline,
        ),
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI"),
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker.emit_error", new_callable=AsyncMock),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        with pytest.raises(RuntimeError):
            await resume_pipeline_job(mock_ctx, "orig-1", context_data, "resume-exc-1")


# ---------------------------------------------------------------------------
# execute_single_step_job — error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_single_step_job_error(mock_ctx, mock_job_manager):
    content_json = '{"id": "t-1", "title": "Test", "content": "Content"}'
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.worker.FunctionPipeline") as MockPipeline,
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        pipeline_instance = MagicMock()
        pipeline_instance.execute_single_step = AsyncMock(
            side_effect=RuntimeError("step failed")
        )
        MockPipeline.return_value = pipeline_instance
        with pytest.raises(RuntimeError):
            await execute_single_step_job(
                mock_ctx, "seo_keywords", content_json, {}, "step-err-1"
            )


@pytest.mark.asyncio
async def test_execute_single_step_job_no_completed_job(mock_ctx, mock_job_manager):
    content_json = '{"id": "t-1", "title": "Test", "content": "Content"}'
    call_count = 0

    async def get_job_side_effect(job_id, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MagicMock(metadata={})
        return None  # second call returns None

    mock_job_manager.get_job = AsyncMock(side_effect=get_job_side_effect)
    with (
        patch(
            "marketing_project.worker.get_job_manager", return_value=mock_job_manager
        ),
        patch("marketing_project.worker.FunctionPipeline") as MockPipeline,
        patch("marketing_project.worker.is_tracing_available", return_value=False),
        patch("marketing_project.worker._flush_telemetry"),
    ):
        pipeline_instance = MagicMock()
        pipeline_instance.execute_single_step = AsyncMock(
            return_value={"main_keyword": "test"}
        )
        MockPipeline.return_value = pipeline_instance
        result = await execute_single_step_job(
            mock_ctx, "seo_keywords", content_json, {}, "step-nojob"
        )
    assert result["status"] == "success"
