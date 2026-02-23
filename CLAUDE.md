# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
pip install -r requirements-dev.txt && pip install -e .
```

### Running the server
```bash
# Development (hot-reload via Docker)
make dev-up
make dev-logs

# Direct uvicorn
uvicorn marketing_project.server:app --host 0.0.0.0 --port 8000 --reload --reload-dir src/

# ARQ background worker (required for job processing)
arq marketing_project.worker.WorkerSettings
```

### Testing
```bash
# All tests with coverage
pytest tests/ -v --cov=src --cov-report=xml --cov-report=html --cov-fail-under=70

# Unit tests only
make test-unit   # pytest tests/ -v --cov=src/marketing_project

# Integration tests only
make test-integration   # pytest tests/integrations/ -v -m integration

# Single test file
pytest tests/api/test_approvals_comprehensive.py -v

# Single test by name
pytest tests/services/test_job_manager.py::test_enqueue_job -v
```

### Linting and Formatting
```bash
make lint       # black --check, isort --check, flake8
make format     # auto-fix with black + isort
make security   # bandit -r src/
```

### Dependency management
```bash
make pip-compile       # regenerate requirements.txt from requirements.in
make pip-compile-dev   # regenerate requirements-dev.txt from requirements-dev.in
```

## Architecture

### Request Flow

```
HTTP Request
  → Middleware stack (TrustedHost → CORS → ErrorHandling → Logging → RequestID)
  → APIRouter (/api/v1/*)
  → Keycloak JWT auth dependency (get_current_user)
  → RBAC dependency (require_roles / require_any_role)
  → Route handler (api/*.py)
  → JobManager.enqueue_job() → ARQ Redis Queue
  → ARQ Worker → FunctionPipeline.execute_pipeline()
  → Plugins run sequentially → results saved to filesystem + PostgreSQL
```

### Key Source Files

- **`server.py`** — FastAPI app factory, lifespan management (DB init, telemetry, resource cleanup), registers all routers via `register_routes()`
- **`api/routes.py`** — Central registry for 16 sub-routers under `/api/v1/`
- **`worker.py`** — ARQ `WorkerSettings`; defines background functions `process_blog_job`, `process_release_notes_job`, `process_transcript_job`
- **`services/function_pipeline/pipeline.py`** — Core pipeline orchestrator using `AsyncOpenAI` with structured outputs (Pydantic JSON mode); emits OpenTelemetry spans per step
- **`services/job_manager.py`** — ARQ-based async job manager; states: `PENDING → QUEUED → PROCESSING → COMPLETED/FAILED`; persists in Redis (TTL) and PostgreSQL (`JobModel`)
- **`middleware/keycloak_auth.py`** — RS256/ES256 JWT validation via Keycloak JWKS endpoint; handles issuer mismatch between `localhost` and `host.docker.internal`; `PUBLIC_PATHS` bypasses auth
- **`middleware/rbac.py`** — RBAC dependency factories; reads roles from `UserContext.roles` (JWT's `realm_access.roles` + `resource_access.*.roles`)
- **`config/settings.py`** — Loads `.env` at module import; also parses `config/pipeline.yml` into `PIPELINE_SPEC`

### Plugin System

Pipeline steps are `PipelineStepPlugin` subclasses in `plugins/`. `PluginRegistry` auto-discovers all plugins at startup. Each plugin declares `step_name`, `step_number`, and implements `execute()`. `DependencyGraph` validates ordering. Pipeline steps are configured in `config/pipeline.yml`.

### Data Persistence

- **PostgreSQL**: Long-term job storage (`JobModel`), approval settings, design kit config, internal docs config, pipeline settings — all via SQLAlchemy async
- **Redis**: ARQ job queue, short-TTL job state cache, circuit breaker for resilience
- **Filesystem**: Step results saved under `data/step_results/` per job ID

### Authentication

All endpoints require `Authorization: Bearer <keycloak_jwt>` except `PUBLIC_PATHS` (`/api/v1/health`, `/api/v1/ready`, `/docs`, `/redoc`, `/openapi.json`). The `get_current_user` dependency injects a `UserContext` into every protected handler. Use `tests/utils/keycloak_test_helpers.py` to generate test JWTs and mock JWKS in tests.

### Testing Patterns

- `asyncio_mode = auto` in `pytest.ini` — all `async def` test functions run automatically
- `tests/conftest.py` provides 15+ shared fixtures (content models, `FunctionPipeline` with mocked OpenAI, `JobManager` with mocked Redis, plugin registries)
- Mock strategy: `unittest.mock.AsyncMock`/`MagicMock`/`patch` for OpenAI, Redis, DB managers — avoid real network/DB calls in unit tests
- Test markers: `unit`, `integration`, `performance`, `slow`, `plugin`, `e2e`
- Many domains have sibling `_comprehensive` and `_extended` test files for edge case coverage

### Environment Variables

Copy `env.example` to `.env`. Required variables:
- `OPENAI_API_KEY` — LLM inference
- `DATABASE_URL` — PostgreSQL connection string
- `REDIS_URL` / `REDIS_HOST` + `REDIS_PORT` — job queue and caching
- `KEYCLOAK_SERVER_URL`, `KEYCLOAK_REALM`, `KEYCLOAK_CLIENT_ID`, `KEYCLOAK_CLIENT_SECRET` — authentication

Telemetry (optional): `ARTHUR_API_KEY`, `ARTHUR_TASK_ID`, `ARTHUR_BASE_URL` for Arthur AI observability; `OTEL_EXPORT_CONSOLE=true` for local span logging.
