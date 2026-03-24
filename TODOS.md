# TODOS

## OTel context propagation at ARQ worker boundary

**What:** Attach the extracted OTel context at the ARQ worker entry point (e.g., a thin wrapper around each worker function or an ARQ `on_startup`/job prologue hook) rather than passing it through application-layer plumbing (`__job_root_traceparent__` in `pipeline_context`).

**Why:** The current approach (manual traceparent string through Redis → `TraceContextTextMapPropagator.extract()` → `parent_context` kwarg → stored in pipeline_context dict) works and is tested, but it's 3 callsites of plumbing that could be replaced by a single `context_api.attach()` at job entry. All child coroutines would inherit the context via `contextvars` without any application-layer awareness.

**Pros:** Eliminates `__job_root_traceparent__` key from pipeline_context, removes `parent_context` kwarg from `create_job_root_span`, and future worker functions get correct span parentage for free.

**Cons:** Requires modifying `WorkerSettings` and wrapping each worker function — touches more files than the current fix and must be tested carefully with ARQ's async task lifecycle.

**Context:** Deferred during `fix/telemetry-span-hierarchy` because the current approach solves the production debugging incident and is fully tested. The architectural improvement is real but not urgent. Relevant files: `worker.py` (`WorkerSettings`), `tracing.py` (`create_job_root_span`).

**Effort:** M (human: ~4h / CC: ~15min)
**Priority:** P3
**Depends on:** None
