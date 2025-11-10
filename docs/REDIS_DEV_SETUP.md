# Redis Development Setup Guide

This guide explains how to set up and run the new centralized Redis configuration in development.

## Quick Start

### Using Docker Compose (Recommended - Easiest)

**If you're using docker-compose, you don't need to do anything!** Just rebuild and start:

```bash
cd deploy/docker
docker-compose -f docker-compose.dev.yml build  # Rebuild to get new dependencies
docker-compose -f docker-compose.dev.yml up
```

That's it! Docker-compose handles:
- ✅ Installing `tenacity` (from requirements.txt)
- ✅ Starting Redis container
- ✅ Setting Redis connection variables
- ✅ Starting API and worker with correct config

**No `.env` changes needed** - all defaults work perfectly for development.

---

### Manual Setup (If Not Using Docker Compose)

### 1. Install New Dependencies

The new Redis implementation requires `tenacity` for retry logic:

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Copy `env.example` to `.env` if you haven't already:

```bash
cp env.example .env
```

**Minimum required for development** (defaults work for local Redis):

```bash
# Basic Redis connection (defaults work for local Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DATABASE=0
# REDIS_PASSWORD=  # Not needed for local Redis without auth
# REDIS_SSL=false  # Not needed for local Redis
```

**Optional: Customize Redis behavior** (all have sensible defaults):

```bash
# Connection pool settings
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=5.0
REDIS_CONNECT_TIMEOUT=5.0

# Retry configuration
REDIS_RETRY_ATTEMPTS=3
REDIS_RETRY_BACKOFF_MIN=2
REDIS_RETRY_BACKOFF_MAX=10

# Circuit breaker (prevents cascading failures)
REDIS_CIRCUIT_FAILURE_THRESHOLD=5
REDIS_CIRCUIT_RECOVERY_TIMEOUT=60

# CloudWatch metrics (not needed for dev)
ENABLE_CLOUDWATCH_METRICS=false
```

### 3. Start Redis

#### Option A: Using Docker (Recommended)

```bash
# Start Redis container
docker run -d \
  --name redis-dev \
  -p 6379:6379 \
  redis:7-alpine

# Or use docker-compose
cd deploy/docker
docker-compose up -d redis
```

#### Option B: Using Homebrew (macOS)

```bash
brew install redis
brew services start redis
```

#### Option C: Using apt (Linux)

```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
```

#### Option D: Using Docker Compose (Full Stack)

```bash
cd deploy/docker
docker-compose -f docker-compose.dev.yml up -d
```

This starts:
- Redis on port 6379
- API server on port 8000 (with hot reload)
- ARQ worker (background job processor)

### 4. Verify Redis is Running

```bash
# Test Redis connection
redis-cli ping
# Should return: PONG

# Or using Docker
docker exec redis-dev redis-cli ping
```

### 5. Start the Application

```bash
# Start API server
python -m marketing_project.server

# In another terminal, start ARQ worker
arq marketing_project.worker.WorkerSettings
```

### 6. Verify Everything Works

1. **Check health endpoint:**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

   Should return:
   ```json
   {
     "status": "healthy",
     "checks": {
       "redis_healthy": true
     },
     "redis": {
       "healthy": true,
       "circuit_breaker_state": "closed"
     }
   }
   ```

2. **Check Redis connection in logs:**
   Look for:
   ```
   Created Redis connection pool: localhost:6379 (db=0, max_connections=50, ssl=False, ...)
   ```

3. **Test a job:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/process/blog \
     -H "Content-Type: application/json" \
     -d '{"content": {"title": "Test", "content": "Test content"}}'
   ```

## What Changed?

### Before (Old Pattern)
- Each service created its own Redis connection
- No connection pooling
- No retry logic
- No circuit breaker
- No health monitoring

### After (New Pattern)
- **Centralized RedisManager** - All services use the same connection pool
- **Connection pooling** - Reuses connections efficiently
- **Automatic retries** - Handles transient failures
- **Circuit breaker** - Prevents cascading failures
- **Health monitoring** - Periodic health checks
- **Better error messages** - More context for debugging

## Troubleshooting

### Redis Connection Failed

**Error:** `Failed to connect to Redis: Connection refused`

**Solution:**
1. Make sure Redis is running: `redis-cli ping`
2. Check `REDIS_HOST` and `REDIS_PORT` in `.env`
3. For Docker, use `redis` as hostname (not `localhost`)

### Circuit Breaker is Open

**Error:** `Circuit breaker is open`

**Solution:**
- This means Redis has been failing repeatedly
- Check Redis is running and accessible
- Wait for recovery timeout (default: 60 seconds)
- Check logs for the underlying Redis errors

### SSL Certificate Errors

**Error:** `SSL certificate verification failed`

**Solution:**
- For local development, set `REDIS_SSL_VALIDATE=false` in `.env`
- For production (ElastiCache), ensure certificates are properly configured

### Port Already in Use

**Error:** `Address already in use` on port 6379

**Solution:**
```bash
# Find what's using the port
lsof -i :6379  # macOS/Linux
netstat -ano | findstr :6379  # Windows

# Stop the conflicting service or use a different port
```

## Development Tips

### 1. Monitor Redis Operations

Enable debug logging to see Redis operations:

```bash
LOG_LEVEL=DEBUG python -m marketing_project.server
```

### 2. Test Circuit Breaker

To test the circuit breaker behavior:

```bash
# Stop Redis
docker stop redis-dev

# Make some API calls - they should fail gracefully
# After 5 failures, circuit breaker opens

# Start Redis again
docker start redis-dev

# Wait 60 seconds for recovery timeout
# Circuit breaker should move to half-open, then close
```

### 3. Check Connection Pool Usage

The health endpoint shows pool information:

```bash
curl http://localhost:8000/api/v1/health | jq .redis
```

### 4. View Redis Data

```bash
# Connect to Redis CLI
redis-cli

# List all keys
KEYS *

# View a specific key
GET job:some-job-id

# Monitor commands in real-time
MONITOR
```

## Migration Notes

### No Breaking Changes

The new RedisManager is **backward compatible**. Existing code continues to work, but now benefits from:
- Connection pooling
- Automatic retries
- Circuit breaker protection
- Better error handling

### Services Updated

All these services now use RedisManager:
- ✅ `JobManager` - Job state persistence
- ✅ `ApprovalManager` - Approval requests
- ✅ `DesignKitManager` - Design kit configuration
- ✅ `InternalDocsManager` - Internal docs configuration
- ✅ `AnalyticsService` - Analytics caching

### ARQ Worker

The ARQ worker uses its own Redis connection (required by ARQ), but uses the same environment variables for consistency.

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Start Redis: `docker-compose up -d redis` or `brew services start redis`
3. ✅ Verify: `curl http://localhost:8000/api/v1/health`
4. ✅ Start developing!

## Need Help?

- Check logs: `docker-compose logs api` or `docker-compose logs worker`
- Test Redis: `redis-cli ping`
- Check health: `curl http://localhost:8000/api/v1/health`
- Review environment: `cat .env | grep REDIS`
