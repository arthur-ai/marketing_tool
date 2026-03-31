"""
Unit tests for tracing utility functions.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_get_current_traceparent_returns_none_when_tracing_unavailable():
    """Returns None when OpenTelemetry is not installed."""
    with patch(
        "marketing_project.services.function_pipeline.tracing._tracing_available",
        False,
    ):
        from marketing_project.services.function_pipeline.tracing import (
            get_current_traceparent,
        )

        result = get_current_traceparent()
        assert result is None


def test_get_current_traceparent_returns_traceparent_string():
    """Returns W3C traceparent string when a span is active."""
    expected = "00-aabbccddeeff00112233445566778899-0011223344556677-01"

    mock_propagator = MagicMock()
    mock_propagator.inject.side_effect = lambda carrier, **_: carrier.update(
        {"traceparent": expected}
    )

    with patch(
        "marketing_project.services.function_pipeline.tracing._tracing_available",
        True,
    ):
        with patch(
            "marketing_project.services.function_pipeline.tracing.TraceContextTextMapPropagator",
            return_value=mock_propagator,
        ):
            from marketing_project.services.function_pipeline.tracing import (
                get_current_traceparent,
            )

            result = get_current_traceparent()
            assert result == expected


def test_get_current_traceparent_returns_none_when_no_active_span():
    """Returns None when no span is active (propagator injects nothing)."""
    mock_propagator = MagicMock()
    mock_propagator.inject.side_effect = lambda carrier, **_: None  # injects nothing

    with patch(
        "marketing_project.services.function_pipeline.tracing._tracing_available",
        True,
    ):
        with patch(
            "marketing_project.services.function_pipeline.tracing.TraceContextTextMapPropagator",
            return_value=mock_propagator,
        ):
            from marketing_project.services.function_pipeline.tracing import (
                get_current_traceparent,
            )

            result = get_current_traceparent()
            assert result is None


def test_get_current_traceparent_returns_none_on_exception():
    """Returns None if propagation raises unexpectedly."""
    with patch(
        "marketing_project.services.function_pipeline.tracing._tracing_available",
        True,
    ):
        with patch(
            "marketing_project.services.function_pipeline.tracing.TraceContextTextMapPropagator",
            side_effect=RuntimeError("oops"),
        ):
            from marketing_project.services.function_pipeline.tracing import (
                get_current_traceparent,
            )

            result = get_current_traceparent()
            assert result is None


# ─── _SpanHolder infrastructure tests ─────────────────────────────────────────


def test_create_span_returns_none_when_tracing_unavailable():
    """create_span() returns None gracefully when OTel is not installed."""
    with patch(
        "marketing_project.services.function_pipeline.tracing._tracing_available",
        False,
    ):
        from marketing_project.services.function_pipeline.tracing import create_span

        result = create_span("test.span")
        assert result is None


def test_create_span_returns_span_holder_and_sets_attributes():
    """create_span() returns a _SpanHolder and sets provided attributes on the span."""
    from marketing_project.services.function_pipeline.tracing import (
        _SpanHolder,
        create_span,
    )

    mock_span = MagicMock()
    mock_token = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_span

    with patch(
        "marketing_project.services.function_pipeline.tracing._tracing_available",
        True,
    ):
        with patch(
            "marketing_project.services.function_pipeline.tracing.trace"
        ) as mock_trace_mod:
            with patch(
                "marketing_project.services.function_pipeline.tracing.context_api"
            ) as mock_ctx_api:
                mock_trace_mod.get_tracer.return_value = mock_tracer
                mock_trace_mod.get_tracer_provider.return_value = MagicMock(
                    __class__=type("RealProvider", (), {})
                )
                mock_trace_mod.SpanKind.INTERNAL = "INTERNAL"
                mock_trace_mod.set_span_in_context.return_value = MagicMock()
                mock_ctx_api.attach.return_value = mock_token
                # Mark span as non-noop so the warning branch is not taken
                mock_span.__class__.__name__ = "Span"

                holder = create_span("job.pipeline", attributes={"job.id": "j1"})

    assert isinstance(holder, _SpanHolder)
    assert holder.span is mock_span
    assert holder.token is mock_token
    mock_span.set_attribute.assert_called_with("job.id", "j1")


def test_close_span_calls_detach_before_end():
    """close_span() must call context_api.detach() before span.end() to restore parent context."""
    from marketing_project.services.function_pipeline.tracing import (
        _SpanHolder,
        close_span,
    )

    call_order = []

    mock_span = MagicMock()
    mock_span.end.side_effect = lambda: call_order.append("end")
    mock_token = MagicMock()
    holder = _SpanHolder(mock_span, mock_token)

    with patch(
        "marketing_project.services.function_pipeline.tracing._tracing_available",
        True,
    ):
        with patch(
            "marketing_project.services.function_pipeline.tracing.context_api"
        ) as mock_ctx_api:
            mock_ctx_api.detach.side_effect = lambda t: call_order.append("detach")

            close_span(holder)

    assert call_order == ["detach", "end"], (
        "detach() must be called before end() so the parent span is restored "
        "as current before this span is ended"
    )


def test_close_span_with_none_holder_is_noop():
    """close_span(None) must not raise."""
    from marketing_project.services.function_pipeline.tracing import close_span

    with patch(
        "marketing_project.services.function_pipeline.tracing._tracing_available",
        True,
    ):
        close_span(None)  # must not raise


def test_create_job_root_span_uses_current_context():
    """create_job_root_span always reads parent context from context_api.get_current().

    Callers that need to restore a saved traceparent must call context_api.attach()
    *before* invoking create_job_root_span — not via a kwarg.
    """
    from marketing_project.services.function_pipeline.tracing import (
        _SpanHolder,
        create_job_root_span,
    )

    mock_span = MagicMock()
    mock_token = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_span
    sentinel_ctx = object()  # what context_api.get_current() will return

    with patch(
        "marketing_project.services.function_pipeline.tracing._tracing_available",
        True,
    ):
        with patch(
            "marketing_project.services.function_pipeline.tracing.trace"
        ) as mock_trace_mod:
            with patch(
                "marketing_project.services.function_pipeline.tracing.context_api"
            ) as mock_ctx_api:
                mock_trace_mod.get_tracer.return_value = mock_tracer
                mock_trace_mod.SpanKind.INTERNAL = "INTERNAL"
                mock_trace_mod.set_span_in_context.return_value = MagicMock()
                mock_ctx_api.get_current.return_value = sentinel_ctx
                mock_ctx_api.attach.return_value = mock_token

                holder = create_job_root_span(
                    job_id="job-123",
                    job_type="resume_pipeline",
                    input_value={},
                )

    assert isinstance(holder, _SpanHolder)
    # Span must be started with the context returned by get_current()
    mock_tracer.start_span.assert_called_once()
    _, start_kwargs = mock_tracer.start_span.call_args
    assert start_kwargs.get("context") is sentinel_ctx, (
        "create_job_root_span must use context_api.get_current() — callers attach() "
        "the parent context before calling this function"
    )


# ─── pipeline.py integration: __job_root_traceparent__ propagation ────────────


@pytest.mark.asyncio
async def test_execute_pipeline_stores_job_root_traceparent_in_context():
    """execute_pipeline captures get_current_traceparent() right after job_root_span
    is created and stores it as pipeline_context['__job_root_traceparent__'] before
    any step plugin runs.  This ensures the approval helper always uses the job-root
    traceparent, never a step-span traceparent, when saving pipeline context.
    """
    from marketing_project.services.function_pipeline import FunctionPipeline

    expected_traceparent = "00-aabbccddeeff00112233445566778899-0011223344556677-01"
    captured_contexts: list = []

    # _execute_step_with_plugin receives pipeline_context — capture it then stop.
    async def _capture_and_stop(step_name, pipeline_context, **kwargs):
        captured_contexts.append(dict(pipeline_context))
        raise RuntimeError("stop after first step")

    content_json = json.dumps(
        {"id": "t1", "title": "Test", "content": "Body", "snippet": "S"}
    )

    mock_plugin = MagicMock()
    mock_plugin.step_name = "seo_keywords"
    mock_plugin.step_number = 1
    mock_plugin.is_optional = False

    mock_registry = MagicMock()
    mock_registry.validate_dependencies.return_value = (True, [])
    mock_registry.get_plugins_in_order.return_value = [mock_plugin]

    with patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI"):
        with patch(
            "marketing_project.services.function_pipeline.pipeline.is_tracing_available",
            return_value=True,
        ):
            with patch(
                "marketing_project.services.function_pipeline.pipeline.create_job_root_span",
                return_value=MagicMock(),  # truthy → traceparent capture path runs
            ):
                with patch(
                    "marketing_project.services.function_pipeline.pipeline.get_current_traceparent",
                    return_value=expected_traceparent,
                ):
                    with patch(
                        "marketing_project.services.function_pipeline.pipeline.get_plugin_registry",
                        return_value=mock_registry,
                    ):
                        with patch(
                            "marketing_project.services.function_pipeline.pipeline.filter_active_plugins",
                            return_value=[mock_plugin],
                        ):
                            with patch(
                                "marketing_project.services.function_pipeline.pipeline.update_job_progress",
                                new_callable=AsyncMock,
                            ):
                                pipeline = FunctionPipeline()
                                pipeline._execute_step_with_plugin = _capture_and_stop  # type: ignore[method-assign]

                                try:
                                    await pipeline.execute_pipeline(
                                        content_json,
                                        job_id="job-tp-integration",
                                    )
                                except (RuntimeError, Exception):
                                    pass  # expected — we stopped after capturing

    assert captured_contexts, "Plugin was never called — test setup is wrong"
    ctx = captured_contexts[0]
    assert ctx.get("__job_root_traceparent__") == expected_traceparent, (
        "pipeline_context must contain '__job_root_traceparent__' equal to the "
        "value returned by get_current_traceparent() at job_root_span creation time, "
        "so the approval helper links resume jobs to the correct parent span"
    )


@pytest.mark.asyncio
async def test_resume_pipeline_propagates_job_root_traceparent_for_second_approval():
    """resume_pipeline() must re-inject __job_root_traceparent__ into the rebuilt
    pipeline_context so a second approval gate in a multi-approval workflow still
    uses the original job.pipeline span, not the step-span traceparent.
    """
    from marketing_project.services.function_pipeline import FunctionPipeline

    original_traceparent = "00-aabbccddeeff00112233445566778899-0011223344556677-01"
    captured_contexts: list = []

    async def _capture_and_stop(step_name, pipeline_context, **kwargs):
        captured_contexts.append(dict(pipeline_context))
        raise RuntimeError("stop after first step")

    context_data = {
        "context": {"seo_keywords": {"main_keyword": "test"}},
        "last_step": "seo_keywords",
        "last_step_number": 1,
        "original_content": {"id": "t1", "title": "Test", "content": "Body"},
        "content_type": "blog_post",
        # Stored by approval_manager.save_pipeline_context when the first gate fired
        "traceparent": original_traceparent,
    }

    mock_plugin = MagicMock()
    mock_plugin.step_name = "marketing_brief"
    mock_plugin.step_number = 2
    mock_plugin.is_optional = False

    mock_registry = MagicMock()
    mock_registry.validate_dependencies.return_value = (True, [])
    mock_registry.get_plugins_in_order.return_value = [
        MagicMock(step_name="seo_keywords", step_number=1),
        mock_plugin,
    ]

    with patch("marketing_project.services.function_pipeline.pipeline.AsyncOpenAI"):
        with patch(
            "marketing_project.services.function_pipeline.pipeline.get_plugin_registry",
            return_value=mock_registry,
        ):
            with patch(
                "marketing_project.services.function_pipeline.pipeline.load_pipeline_configs",
                new_callable=AsyncMock,
                return_value={"internal_docs_config": None, "brand_kit_config": None},
            ):
                with patch(
                    "marketing_project.services.function_pipeline.pipeline.filter_active_plugins",
                    return_value=[mock_plugin],
                ):
                    with patch(
                        "marketing_project.services.function_pipeline.pipeline.update_job_progress",
                        new_callable=AsyncMock,
                    ):
                        pipeline = FunctionPipeline()
                        pipeline._execute_step_with_plugin = _capture_and_stop  # type: ignore[method-assign]

                        try:
                            await pipeline.resume_pipeline(
                                context_data,
                                job_id="job-tp-resume",
                            )
                        except (RuntimeError, Exception):
                            pass  # expected — we stopped after capturing

    assert captured_contexts, "Plugin was never called — test setup is wrong"
    ctx = captured_contexts[0]
    assert ctx.get("__job_root_traceparent__") == original_traceparent, (
        "resume_pipeline must re-inject __job_root_traceparent__ from context_data "
        "so a second approval gate links to the original job.pipeline span, "
        "not to a step-span traceparent from the resumed execution"
    )
