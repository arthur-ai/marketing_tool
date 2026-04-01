# Changelog

All notable changes to this project will be documented in this file.

## [0.1.2.0] - 2026-04-01

### Fixed
- `SEOKeywordsComposer.compose_result()` was silently discarding all LLM keyword output: `isinstance(llm_result, SEOKeywordsResult)` returned `False` for `SEOKeywordsLLMResult` instances (parent class is not an instance of child), causing `_build_result({})` to run with an empty dict and return `main_keyword="keyword"` for every pipeline run. Fixed by checking `isinstance(llm_result, SEOKeywordsLLMResult)` first, then upgrading to `SEOKeywordsResult` via `model_validate()`.
- `_execute_keyword_steps` return type annotation in `SEOKeywordsPlugin` corrected from `SEOKeywordsLLMResult` to `SEOKeywordsResult`.

### Tests
- Regression test added: `test_compose_result_upgrades_llm_base_to_full_result` verifies the LLM base type upgrade path and confirms real keywords are preserved (not the `"keyword"` default).

## [0.1.1.0] - 2026-03-31

### Added
- Test coverage raised from 58.5% to 70.1%, clearing the CI gate (`--cov-fail-under=70`)
- `.coveragerc` excluding 8 infra files (S3, OTEL, NLP model downloads, web scraping) that require real external dependencies — not unit-testable
- 500+ new tests across `worker.py`, `step_result_manager.py`, `job_manager.py`, `approvals.py`, `internal_docs.py`, `scanned_document_db.py`, and `context_registry.py`
- Anthropic schema grammar limit fixes: `_llm_exclude_fields` on five Pydantic models to stay under the ~16 Optional-field compilation limit; nested `$defs` exclusions and dead-def pruning in `_get_llm_schema`
- `litellm.InternalServerError` now caught and re-raised (triggers retry logic) instead of falling through to the generic exception handler

### Fixed
- `synthesize_brand_kit_job` `finally` block used undefined `job_id` — changed to `synthesis_job_id` (bug would cause `NameError` when the job failed or timed out)
- Anthropic schema constraint keywords (`minimum`, `maximum`, `minLength`, etc.) stripped in `_make_schema_anthropic_safe` to prevent grammar compilation errors; Pydantic `ge`/`le` validation runs client-side after the LLM call

## [0.1.0.0] - 2026-03-31

### Added
- Quality scores API: `GET /api/v1/analytics/quality-scores` returns persisted per-job quality metrics with filtering by job ID and pagination
- Onboarding examples API: full CRUD under `/api/v1/onboarding-examples` — public read endpoint for active examples and admin endpoints for managing the full catalog
- Quality metrics (Flesch-Kincaid grade, word count, heading presence, keyword match %) now persist to PostgreSQL after each job completes — available for trend analysis via the quality scores API
- SSE job progress streaming improvements in worker and job manager

### Fixed
- Test suite: updated all test mocks and assertions to match LiteLLM/Anthropic provider refactor — 1357 tests now pass with 0 failures
- Integration tests: skip conditions now correctly check for all required API keys (OPENAI, ARTHUR, ANTHROPIC) before running live pipeline tests
- Social media pipeline: empty `platforms` list now raises `ValueError` immediately rather than silently completing with no output
- Quality score background tasks now hold strong references to prevent garbage collection before they finish writing to the database
- Onboarding examples admin list endpoint now correctly applies `limit`/`offset` to the query instead of fetching the full table
- All onboarding example API handlers now return proper HTTP 500 responses instead of leaking SQLAlchemy tracebacks on DB errors
- `update_job_status` now logs a WARNING when the job cannot be found in any store, making silent no-ops visible in logs

### Changed
- `QualityScoresListResponse` and `QualityScoreRecord` Pydantic models added to analytics module for the new quality scores list endpoint
- Admin onboarding list endpoint max `limit` capped at 1000 (down from 5000) to prevent oversized JSONB responses
