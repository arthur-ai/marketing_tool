# ARQ Job Queue Setup

## Overview

The Marketing Project now uses **ARQ (Async Redis Queue)** for distributed background job processing. This allows the API to respond immediately while jobs process asynchronously in separate worker processes.

## Architecture

```
┌─────────────────────┐    ┌─────────────────────┐
│   FastAPI API       │    │  ARQ Workers (x2)   │
│  (Returns job IDs)  │    │  (Process jobs)     │
│                     │    │                     │
│  POST /process/blog │    │  - Worker 1         │
│  → job_id: abc-123  │    │  - Worker 2         │
└──────────┬──────────┘    └──────────┬──────────┘
           │                          │
           └────────┬─────────────────┘
                    │
           ┌────────▼────────┐
           │  Redis Queue    │
           │  (Job storage)  │
           └─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/ibrahim/Documents/Github/marketing_tool
pip install -r requirements.txt
```

### 2. Start with Docker Compose (Recommended)

```bash
cd deploy/docker
docker-compose -f docker-compose.dev.yml up
```

This starts:
- **API** on http://localhost:8000
- **2 ARQ Workers** (auto-scales background processing)
- **Redis** on localhost:6379

### 3. Manual Setup (Without Docker)

**Terminal 1 - Start Redis:**
```bash
redis-server
```

**Terminal 2 - Start API:**
```bash
cd /Users/ibrahim/Documents/Github/marketing_tool
python -m src.marketing_project.server
```

**Terminal 3 - Start Worker:**
```bash
cd /Users/ibrahim/Documents/Github/marketing_tool
arq marketing_project.worker.WorkerSettings
```

## How It Works

### Submit a Job

```bash
curl -X POST http://localhost:8000/api/v1/process/blog \
  -H "Content-Type: application/json" \
  -d '{
    "content": {
      "id": "blog-123",
      "title": "Test Blog",
      "content": "Test content",
      "snippet": "Test snippet",
      "author": "Test Author",
      "tags": ["test"],
      "category": "testing"
    }
  }'
```

**Response (Immediate):**
```json
{
  "success": true,
  "message": "Blog post submitted for processing",
  "job_id": "abc-123-def-456",
  "content_id": "blog-123",
  "status_url": "/api/v1/jobs/abc-123-def-456/status"
}
```

### Check Job Status

```bash
curl http://localhost:8000/api/v1/jobs/abc-123-def-456/status
```

**Response:**
```json
{
  "success": true,
  "message": "Job abc-123-def-456 status: processing",
  "job_id": "abc-123-def-456",
  "status": "processing",
  "progress": 45,
  "current_step": "Generating marketing brief"
}
```

### Get Job Result

```bash
curl http://localhost:8000/api/v1/jobs/abc-123-def-456/result
```

## Configuration

### Environment Variables

```bash
# Redis Connection
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DATABASE=0
REDIS_PASSWORD=  # Optional

# ARQ Worker Settings
ARQ_MAX_JOBS=10           # Max concurrent jobs per worker
ARQ_JOB_TIMEOUT=600       # Job timeout in seconds (10 minutes)
```

### Scaling Workers

**Docker Compose:**
```yaml
worker:
  deploy:
    replicas: 5  # Change to any number
```

**Manual:**
```bash
# Start multiple workers in separate terminals
arq marketing_project.worker.WorkerSettings  # Worker 1
arq marketing_project.worker.WorkerSettings  # Worker 2
arq marketing_project.worker.WorkerSettings  # Worker 3
```

## Monitoring

### View Worker Logs

**Docker:**
```bash
docker-compose -f docker-compose.dev.yml logs -f worker
```

**Manual:**
Workers log to stdout with INFO level by default.

### ARQ Dashboard (Optional)

Install ARQ dashboard for web UI:
```bash
pip install arq-dashboard
arq-dashboard --redis redis://localhost:6379
```

Visit: http://localhost:8000 (different port if API is running)

## Job Lifecycle

1. **PENDING** → Job created, not yet queued
2. **QUEUED** → Submitted to ARQ, waiting for worker
3. **PROCESSING** → Worker is executing the job
4. **COMPLETED** → Job finished successfully
5. **FAILED** → Job encountered an error
6. **CANCELLED** → Job was manually cancelled

## API Endpoints

### Job Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/jobs` | List all jobs |
| `GET` | `/api/v1/jobs/{job_id}` | Get job details |
| `GET` | `/api/v1/jobs/{job_id}/status` | Get job status (lightweight) |
| `GET` | `/api/v1/jobs/{job_id}/result` | Get job result (only if completed) |
| `DELETE` | `/api/v1/jobs/{job_id}` | Cancel job |

### Content Processing

| Method | Endpoint | Returns |
|--------|----------|---------|
| `POST` | `/api/v1/process/blog` | Job ID |
| `POST` | `/api/v1/process/release-notes` | Job ID |
| `POST` | `/api/v1/process/transcript` | Job ID |

## Troubleshooting

### Worker Not Starting

```bash
# Check if Redis is running
redis-cli ping  # Should return "PONG"

# Check worker logs
arq marketing_project.worker.WorkerSettings --verbose
```

### Jobs Stuck in QUEUED

- Verify workers are running: `docker-compose ps` or check processes
- Check Redis connection: `redis-cli ping`
- Check worker logs for errors

### Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/Users/ibrahim/Documents/Github/marketing_tool/src
arq marketing_project.worker.WorkerSettings
```

## Production Recommendations

### 1. Database-Backed Job Storage

Current implementation stores jobs in memory. For production:

```python
# Replace in-memory dict with database
class JobManager:
    def __init__(self, db_session):
        self.db = db_session  # SQLAlchemy or MongoDB
```

### 2. Job Result Storage

Configure ARQ to store results in Redis:

```python
class WorkerSettings:
    keep_result = 86400  # 24 hours
```

### 3. Monitoring & Alerting

- Use ARQ's `on_job_start`, `on_job_end` hooks
- Send metrics to Prometheus/Datadog
- Set up alerts for failed jobs

### 4. Graceful Shutdown

Workers handle SIGTERM gracefully and finish current jobs.

## Comparison: Before vs After

### Before (asyncio.create_task)
```python
✅ Simple setup
❌ Jobs lost on restart
❌ All in one process
❌ Can't scale workers
```

### After (ARQ)
```python
✅ Distributed workers
✅ Job persistence in Redis
✅ Horizontal scaling
✅ Separate API/worker processes
✅ Production-ready
```

## Next Steps

1. **Test the integration**: Submit a job and check status
2. **Monitor performance**: Watch worker logs
3. **Scale if needed**: Add more workers in docker-compose
4. **Add monitoring**: Integrate Prometheus metrics
5. **Database migration**: Move job storage to database for production

## Resources

- [ARQ Documentation](https://arq-docs.helpmanual.io/)
- [Redis Documentation](https://redis.io/docs/)
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
