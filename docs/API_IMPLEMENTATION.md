# FastAPI Implementation Guide

## Overview

This document describes the comprehensive FastAPI implementation for the Marketing Project. The implementation includes production-ready features such as authentication, rate limiting, CORS, error handling, and comprehensive API endpoints.

## Architecture

### Components

1. **API Models** (`api_models.py`) - Pydantic models for request/response validation
2. **Middleware** (`middleware/`) - Authentication, CORS, rate limiting, logging, error handling
3. **API Endpoints** (`api_endpoints.py`) - Main API endpoints with proper validation
4. **Server Configuration** (`server.py`) - FastAPI app with comprehensive middleware stack

### Middleware Stack

The middleware is applied in the following order (last added is first executed):

1. **TrustedHostMiddleware** - Host validation
2. **CORS Middleware** - Cross-origin resource sharing
3. **Authentication Middleware** - API key authentication
4. **Rate Limiting Middleware** - Request rate limiting
5. **Error Handling Middleware** - Global exception handling
6. **Logging Middleware** - Request/response logging
7. **Request ID Middleware** - Request tracking

## API Endpoints

### Authentication Required Endpoints

All endpoints except health checks require API key authentication via `X-API-Key` header.

#### Content Analysis
- `POST /api/v1/analyze` - Analyze content for marketing pipeline processing

#### Pipeline Execution
- `POST /api/v1/pipeline` - Run the complete marketing pipeline on content

#### Content Sources
- `GET /api/v1/content-sources` - List all configured content sources
- `GET /api/v1/content-sources/{source_name}/status` - Get status of a specific content source
- `POST /api/v1/content-sources/{source_name}/fetch` - Fetch content from a specific source

#### Health Checks
- `GET /api/v1/health` - API health check (requires auth)
- `GET /api/v1/ready` - API readiness check (requires auth)

### Legacy Endpoints (No Auth Required)

- `GET /health` - Legacy health check
- `GET /ready` - Legacy readiness check
- `POST /run` - Legacy pipeline execution (deprecated)

## Authentication

### API Key Authentication

The API uses API key authentication with the following features:

- **Header**: `X-API-Key` or `X-Api-Key`
- **Format**: Minimum 32 characters, alphanumeric with hyphens/underscores
- **Roles**: `admin`, `user`, `viewer`
- **Permissions**: Role-based access control

### Environment Variables

```bash
# Primary API key (admin role)
API_KEY=your_api_key_here_32_chars_minimum

# Additional API keys
API_KEY_1=additional_api_key_1
API_KEY_1_ROLE=user
API_KEY_2=additional_api_key_2
API_KEY_2_ROLE=viewer
```

### Role Permissions

- **admin**: `read`, `write`, `delete`, `admin`
- **user**: `read`, `write`
- **viewer**: `read`

## Rate Limiting

### Configuration

- **Default**: 100 requests per minute
- **Burst Limit**: 20 requests per second
- **Scope**: Per IP and per user
- **Headers**: Rate limit information in response headers

### Environment Variables

```bash
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST_LIMIT=20
```

### Response Headers

- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: When the rate limit resets
- `Retry-After`: Seconds to wait before retry (when rate limited)

## CORS Configuration

### Default Origins

- `http://localhost:3000` (React dev server)
- `http://localhost:8080` (Vue dev server)
- `http://localhost:4200` (Angular dev server)

### Environment Variables

```bash
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:4200
CORS_ALLOW_CREDENTIALS=true
```

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "message": "Error description",
  "error_code": "ERROR_CODE",
  "error_details": {
    "additional_info": "value"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Common Error Codes

- `MISSING_API_KEY` - API key not provided
- `INVALID_API_KEY` - Invalid API key
- `RATE_LIMIT_EXCEEDED` - Rate limit exceeded
- `VALIDATION_ERROR` - Request validation failed
- `INTERNAL_SERVER_ERROR` - Server error

## Request/Response Models

### Content Types

- `BlogPostContext` - Blog post content
- `TranscriptContext` - Transcript content
- `ReleaseNotesContext` - Release notes content

### Example Request

```json
{
  "content": {
    "id": "example-1",
    "title": "Marketing Automation Guide",
    "content": "Marketing automation is a powerful tool...",
    "author": "John Doe",
    "tags": ["marketing", "automation"],
    "category": "tutorial",
    "content_type": "blog_post"
  }
}
```

### Example Response

```json
{
  "success": true,
  "message": "Content analysis completed successfully",
  "data": {
    "content_quality": {
      "word_count": 1500,
      "readability_score": 75
    },
    "seo_potential": {
      "has_keywords": true,
      "title_optimization": "good"
    }
  },
  "metadata": {
    "content_id": "example-1",
    "analysis_timestamp": 1640995200.0
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Logging

### Request Logging

Each request is logged with:
- Request ID
- Method and URL
- Client IP
- User role
- Processing time
- Response status

### Log Format

```
INFO: Request abc123: POST /api/v1/analyze - 0.234s
```

## Running the API

### Development

```bash
# Set environment variables
export API_KEY="your-api-key-here-32-chars-minimum"

# Run the server
python -m marketing_project.server
```

### Production

```bash
# Using uvicorn directly
uvicorn marketing_project.server:app --host 0.0.0.0 --port 8000 --workers 4

# Using gunicorn
gunicorn marketing_project.server:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Testing

### Test Script

Run the included test script:

```bash
python test_api.py
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# API health check (requires auth)
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/health

# Content analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "content": {
      "id": "test-1",
      "title": "Test Article",
      "content": "This is a test article.",
      "content_type": "blog_post"
    }
  }'
```

## OpenAPI Documentation

The API includes comprehensive OpenAPI documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## Security Features

1. **API Key Authentication** - Secure API access
2. **Rate Limiting** - Prevent abuse
3. **CORS Protection** - Control cross-origin access
4. **Input Validation** - Pydantic model validation
5. **Error Handling** - Secure error responses
6. **Request Logging** - Audit trail
7. **Host Validation** - Trusted host middleware

## Performance Features

1. **Async Processing** - Non-blocking operations
2. **Background Tasks** - Long-running operations
3. **Request ID Tracking** - Request correlation
4. **Response Caching** - Rate limit headers
5. **Memory Management** - Automatic cleanup

## Monitoring

### Health Checks

- `/health` - Basic health check
- `/ready` - Readiness check
- `/api/v1/health` - API health check
- `/api/v1/ready` - API readiness check

### Metrics

- Request processing time
- Rate limit usage
- Error rates
- Authentication success/failure

## Deployment

### Docker

```dockerfile
FROM python:3.12-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "-m", "marketing_project.server"]
```

### Kubernetes

The API is ready for Kubernetes deployment with:
- Health check endpoints
- Resource limits
- Environment variable configuration
- Secret management

## Migration from Legacy API

The new API maintains backward compatibility with legacy endpoints:

- `/health` → `/api/v1/health`
- `/ready` → `/api/v1/ready`
- `/run` → `/api/v1/pipeline`

Legacy endpoints are deprecated but still functional.
