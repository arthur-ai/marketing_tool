# Docker Setup for Marketing Project

This directory contains Docker configurations for both development and production environments.

## 🚀 Quick Start (Development)

**Recommended: Use the Makefile commands from the project root!**

```bash
# From project root
make dev-up      # Start development with hot-reload
make dev-logs    # View logs
make dev-down    # Stop
```

See [Full Command Reference](#-command-reference) below.

---

## 📋 Table of Contents

- [Development Mode (Hot Reload)](#-development-mode-hot-reload)
- [Production Mode](#-production-mode)
- [Command Reference](#-command-reference)
- [File Structure](#-file-structure)
- [Troubleshooting](#-troubleshooting)
- [Optional Services](#-optional-services)

---

## 🔧 Development Mode (Hot Reload)

**Best for active development** - Code changes reflect immediately without rebuilding!

### Features

- ✅ **Hot Reload** - Uvicorn automatically reloads on file changes
- ✅ **Volume Mounts** - Source code mounted from host
- ✅ **Debug Logging** - Verbose output for troubleshooting
- ✅ **Fast Iteration** - No rebuilds needed for code changes
- ✅ **Redis + Workers** - Background job processing (Phase 2)

### Using Makefile (Recommended)

```bash
# Start development environment
make dev-up

# View logs (in another terminal)
make dev-logs

# Restart (rarely needed)
make dev-restart

# Stop
make dev-down

# Rebuild (only if dependencies changed)
make dev-build

# Open shell in container
make dev-shell
```

### Using Docker Compose Directly

```bash
# Start
docker-compose -f deploy/docker/docker-compose.dev.yml up -d

# View logs
docker-compose -f deploy/docker/docker-compose.dev.yml logs -f api

# Stop
docker-compose -f deploy/docker/docker-compose.dev.yml down
```

### Development Workflow

1. **Start once**: `make dev-up`
2. **Edit code**: Make changes in `src/`
3. **Watch reload**: Check logs with `make dev-logs`
4. **Test**: Visit http://localhost:8000/docs
5. **Done**: `make dev-down`

### When to Restart?

- ✅ **NO restart** for Python code changes (auto-reloads)
- ⚠️ **Restart needed** for:
  - Adding packages to `requirements.txt` → `make dev-build`
  - Changing Docker config → `make dev-build`
  - Changing environment variables → `make dev-restart`

### Mounted Volumes

The following are mounted for hot-reload:
- `../../src` → `/app/src` - Source code
- `../../requirements.txt` → `/app/requirements.txt`
- `../../setup.py` → `/app/setup.py`
- `../../pyproject.toml` → `/app/pyproject.toml`
- `./logs` → `/app/logs`
- `./uploads` → `/app/uploads`
- `./content` → `/app/content`

---

## 🏭 Production Mode

**Best for testing production configuration** - Source code is copied into the image.

> **⚠️ Phase 2 Requirement:** Redis and ARQ workers are now **required** for job management and background processing. The production compose file includes these by default.

### Using Makefile (Recommended)

```bash
# Start production environment
make prod-up

# View logs
make prod-logs

# Restart
make prod-restart

# Stop
make prod-down

# Rebuild
make prod-build
```

### Using Docker Compose Directly

```bash
# Start
docker-compose -f deploy/docker/docker-compose.yml up -d

# View logs
docker-compose -f deploy/docker/docker-compose.yml logs -f api

# Stop
docker-compose -f deploy/docker/docker-compose.yml down
```

### Production Features

- ✅ Source code copied into image
- ✅ No hot reload (stability)
- ✅ Optimized for production workloads
- ✅ Health checks enabled
- ✅ Redis with AOF persistence (Phase 2)
- ✅ ARQ workers for background jobs (Phase 2)
- ✅ Job state persistence across restarts (Phase 2)

---

## 📋 Command Reference

### Development Commands

| Command | Description |
|---------|-------------|
| `make dev-up` | Start development with hot-reload |
| `make dev-down` | Stop development |
| `make dev-restart` | Restart containers |
| `make dev-logs` | View logs (follow mode) |
| `make dev-build` | Rebuild containers |
| `make dev-shell` | Open shell in container |

### Production Commands

| Command | Description |
|---------|-------------|
| `make prod-up` | Start production |
| `make prod-down` | Stop production |
| `make prod-restart` | Restart containers |
| `make prod-logs` | View logs (follow mode) |
| `make prod-build` | Rebuild containers |

### Utility Commands

| Command | Description |
|---------|-------------|
| `make clean` | Clean Python cache files |
| `make docker-clean` | Clean all Docker resources |
| `make help` | Show all available commands |

---

## 📁 File Structure

```
deploy/docker/
├── Dockerfile                # Production Dockerfile
├── Dockerfile.dev            # Development Dockerfile (with hot-reload)
├── docker-compose.yml        # Production compose file
├── docker-compose.dev.yml    # Development compose file
├── docker-compose.*.yml      # Optional service configs (postgres, mongo, redis)
├── logs/                     # Application logs
├── uploads/                  # Uploaded files
└── content/                  # Content storage
```

---

## 🔧 Troubleshooting

### Changes not reflecting?

```bash
# Check if uvicorn detected the change
make dev-logs

# If not, restart
make dev-restart
```

### Port already in use?

```bash
# Stop all containers
make dev-down
make prod-down

# Or kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Permission errors?

```bash
# Fix ownership (container runs as appuser)
sudo chown -R $USER:$USER deploy/docker/logs deploy/docker/uploads deploy/docker/content
```

### Need to rebuild everything?

```bash
# Clean up and rebuild
make docker-clean
make dev-build
```

### Clear Python cache

```bash
make clean

# Or manually
find src -type d -name __pycache__ -exec rm -rf {} +
find src -type f -name "*.pyc" -delete
```

### Container won't start?

```bash
# Check logs
make dev-logs

# Check if port is available
netstat -an | grep 8000

# Remove old containers
docker-compose -f deploy/docker/docker-compose.dev.yml down -v
make dev-build
```

---

## 🌐 API Endpoints

Once running, access:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health
- **API Base**: http://localhost:8000/api/v1/
- **Job Status**: http://localhost:8000/api/v1/jobs
- **Redis**: localhost:6379 (if you need to connect directly)

---

## 🔌 Optional Services

### Method 1: Using Separate Compose Files

The project provides modular compose files for different services:

```bash
# With PostgreSQL
docker-compose -f deploy/docker/docker-compose.yml -f deploy/docker/docker-compose.postgres.yml up -d

# With MongoDB
docker-compose -f deploy/docker/docker-compose.yml -f deploy/docker/docker-compose.mongodb.yml up -d

# With Redis
docker-compose -f deploy/docker/docker-compose.yml -f deploy/docker/docker-compose.redis.yml up -d

# With Everything (Full Stack)
docker-compose -f deploy/docker/docker-compose.yml -f deploy/docker/docker-compose.full.yml up -d
```

### Method 2: Uncommenting Services

Alternatively, uncomment the relevant service in your main compose file:

```yaml
# Uncomment in docker-compose.yml or docker-compose.dev.yml
postgres:
  image: postgres:15-alpine
  ports:
    - "5432:5432"
  environment:
    - POSTGRES_DB=marketing_project
    - POSTGRES_USER=postgres
    - POSTGRES_PASSWORD=password
  volumes:
    - postgres_data:/var/lib/postgresql/data
```

Don't forget to uncomment the corresponding volume at the bottom of the file.

### Available Services

| Service | Port | Purpose | Compose File |
|---------|------|---------|--------------|
| PostgreSQL | 5432 | Relational database | `docker-compose.postgres.yml` |
| MongoDB | 27017 | Document database | `docker-compose.mongodb.yml` |
| Redis | 6379 | Cache & message broker | `docker-compose.redis.yml` |
| All Services | - | Full stack | `docker-compose.full.yml` |

### Service Configuration

The API automatically detects which services are available:

- **SQLite**: Always available (default, no external dependencies)
- **PostgreSQL**: Used if `POSTGRES_URL` environment variable is set
- **MongoDB**: Used if `MONGODB_URL` environment variable is set
- **Redis**: Used if `REDIS_URL` environment variable is set

---

## 🎯 Quick Example

```bash
# 1. Start development
make dev-up

# 2. In another terminal, watch logs
make dev-logs

# 3. Edit code in src/marketing_project/
# 4. Save file
# 5. Watch logs - uvicorn reloads automatically!

# 6. Test your changes
curl http://localhost:8000/api/v1/health

# 7. When done
make dev-down
```

---

## 💡 Best Practices

1. **Use dev mode for development** - Hot reload saves time
2. **Keep logs running** in a separate terminal
3. **Only rebuild** when dependencies change
4. **Use prod mode** to test production behavior before deploying
5. **Clean regularly** - Run `make clean` and `make docker-clean` periodically

---

## 🆘 Getting Help

- Run `make help` for all available commands
- Check logs with `make dev-logs` or `make prod-logs`
- See project README.md for general setup
- See CONTRIBUTING.md for development guidelines

---

**Happy Coding! 🚀**
