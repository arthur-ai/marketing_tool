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

## Guided Onboarding

### Move quick-start configs to DB

**What:** Add a `onboarding_examples` table (or JSON config table) for the 3 quick-start job templates shown above the create form on the job creation page.

**Why:** Currently frontend constants — every example change requires a code deploy. DB-backed lets admins add, edit, or remove examples without redeploying. Directly supports autonomous adoption: the marketing team can curate examples that match current priorities.

**Pros:** Zero-deploy content updates for examples; admins can tune quick-start configs as they learn what the team actually needs.

**Cons:** Requires a manual SQL migration (`IF NOT EXISTS` pattern), a new settings endpoint, and a small admin UI. Adds a DB table for what could remain a JSON file.

**Context:** Deferred from the Full Adoption Fix PR (quick-start feature ships as frontend constants in v1). The DB migration is the right long-term architecture per the design doc. Relevant files: `models/db_models.py`, `api/routes.py`, settings page in marketing-frontend.

**Effort:** M (human: ~1 day / CC: ~20min)
**Priority:** P3
**Depends on:** Quick-start feature shipping in Full Adoption Fix PR

## Design System

### Create DESIGN.md

**What:** Run `/design-consultation` to produce a `DESIGN.md` design system document for the marketing frontend — covering aesthetic, typography, color palette, spacing scale, component patterns, and motion guidelines.

**Why:** The frontend currently has two conflicting design systems: MUI (`@mui/material`) on the dashboard page and Tailwind + shadcn/ui everywhere else. Without a documented design system, new components default to whoever wrote them last. This is the root cause of the dashboard's AI-slop patterns (purple gradient hero, icon-in-colored-circle card grids). DESIGN.md gives future implementers a single source of truth.

**Pros:** Eliminates design fragmentation; prevents AI slop patterns from recurring; allows `/design-review` and `/plan-design-review` to calibrate against explicit decisions rather than universal principles.

**Cons:** Requires a design consultation session (~1-2 hours); introduces a document that must be maintained as the product evolves.

**Context:** Flagged during the Full Adoption Fix `/plan-design-review` (2026-03-24). The review found no DESIGN.md in either `marketing_tool/` or `marketing-frontend/`. All design decisions for this PR were made from universal principles instead. The `/design-consultation` skill produces a full system. Relevant files: `marketing-frontend/` root.

**Effort:** M (human: ~2h / CC: ~20min with `/design-consultation`)
**Priority:** P3
**Depends on:** Full Adoption Fix PR shipping (so the new components exist to inform the system)

## Trust Signals

### Quality score analytics storage

**What:** Persist quality metrics to PostgreSQL after computation: `job_id`, `word_count`, `reading_level`, `has_headings`, `keyword_match_pct`, `profound_personas_used` (list), `computed_at`.

**Why:** Enables trend tracking once the marketing team is using the tool autonomously: is content quality improving over time? Are keywords matching? Which content types score highest? The quality endpoint already computes this data — storing it costs one DB write per quality request.

**Pros:** Long-term visibility into output quality; could drive VP reporting; no additional LLM cost.

**Cons:** Requires a new DB table, migration, and a write path in the quality endpoint. Adds latency to the quality request (one async INSERT).

**Context:** Deferred from the Full Adoption Fix PR (quality endpoint computes on-demand in v1, no storage). Relevant files: `models/db_models.py`, `api/jobs.py` (quality endpoint), `services/analytics_service.py`.

**Effort:** S (human: ~2h / CC: ~10min)
**Priority:** P3
**Depends on:** Quality endpoint shipping in Full Adoption Fix PR
