# Docker Compose Configurations

This project provides multiple Docker Compose configurations to suit different development needs. Choose the one that fits your requirements.

## Quick Start (Recommended)

**Just the API with SQLite (no external dependencies):**
```bash
docker-compose up
```
- ✅ Fastest startup
- ✅ No external dependencies
- ✅ Perfect for development and testing
- ✅ Uses SQLite for data storage

## Optional Services

### With PostgreSQL
```bash
docker-compose -f docker-compose.yml -f docker-compose.postgres.yml up
```
- Adds PostgreSQL database
- Useful if you need relational data storage

### With MongoDB
```bash
docker-compose -f docker-compose.yml -f docker-compose.mongodb.yml up
```
- Adds MongoDB database
- Useful for document-based storage

### With Redis
```bash
docker-compose -f docker-compose.yml -f docker-compose.redis.yml up
```
- Adds Redis for caching
- Useful for performance optimization

### With Everything (Full Stack)
```bash
docker-compose -f docker-compose.yml -f docker-compose.full.yml up
```
- PostgreSQL + MongoDB + Redis
- Use only if you need all services

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | Main application |
| PostgreSQL | 5432 | Database (if enabled) |
| MongoDB | 27017 | Document database (if enabled) |
| Redis | 6379 | Cache (if enabled) |

## Environment Variables

The API automatically detects which services are available and configures itself accordingly:

- **SQLite**: Always available (default)
- **PostgreSQL**: Used if `POSTGRES_URL` is set
- **MongoDB**: Used if `MONGODB_URL` is set
- **Redis**: Used if `REDIS_URL` is set

## Development Tips

1. **Start simple**: Use the basic `docker-compose up` first
2. **Add services as needed**: Only enable what you actually use
3. **Check logs**: `docker-compose logs api` to see application logs
4. **Health checks**: Visit `http://localhost:8000/api/v1/health`

## Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

## Troubleshooting

- **Port conflicts**: Make sure ports 8000, 5432, 27017, 6379 are available
- **Permission issues**: Check file permissions in `./logs` and `./uploads` directories
- **Database connection**: Ensure the database service is healthy before the API starts
