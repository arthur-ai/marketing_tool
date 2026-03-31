# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0.0] - 2026-03-31

### Added
- Quality scores API: `GET /api/v1/analytics/quality-scores` returns persisted per-job quality metrics with filtering by job ID and pagination
- Onboarding examples API: full CRUD under `/api/v1/onboarding-examples` — public read endpoint for active examples and admin endpoints for managing the full catalog
- Job quality endpoint now persists computed metrics (Flesch-Kincaid grade, word count, heading presence, keyword match %) to PostgreSQL via upsert for trend tracking
- SSE job progress streaming improvements in worker and job manager

### Fixed
- Test suite: updated all test mocks and assertions to match LiteLLM/Anthropic provider refactor — 1357 tests now pass with 0 failures
- Integration tests: skip conditions now correctly check for all required API keys (OPENAI, ARTHUR, ANTHROPIC) before running live pipeline tests
- Social media pipeline: empty `platforms` list now raises `ValueError` immediately rather than silently completing with no output
- Background task tracking: `asyncio.create_task` for quality score persistence now holds a strong reference to prevent GC before the task completes
- Onboarding examples admin list endpoint now correctly applies `limit`/`offset` to the query instead of fetching the full table
- All onboarding example API handlers now return proper HTTP 500 responses instead of leaking SQLAlchemy tracebacks on DB errors
- `update_job_status` now logs a WARNING when the job cannot be found in any store, making silent no-ops visible in logs

### Changed
- `QualityScoresListResponse` and `QualityScoreRecord` Pydantic models added to analytics module for the new quality scores list endpoint
- Admin onboarding list endpoint max `limit` capped at 1000 (down from 5000) to prevent oversized JSONB responses
