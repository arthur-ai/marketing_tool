# Marketing Project 🤖📈

[![CI Status](https://github.com/your-org/marketing-project/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/marketing-project/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/your-org/marketing-project/main)](https://codecov.io/gh/your-org/marketing-project)
[![Security](https://img.shields.io/badge/security-enterprise--grade-green.svg)](https://github.com/your-org/marketing-project/security)
[![Performance](https://img.shields.io/badge/performance-optimized-blue.svg)](https://github.com/your-org/marketing-project/performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-ready marketing agentic project with enterprise security, performance monitoring, and comprehensive API endpoints.

## ✨ Features

### 🔒 **Enterprise Security**
- **API Key Authentication** with role-based access control
- **Advanced Rate Limiting** with attack detection and IP whitelisting
- **Input Validation** preventing SQL injection, XSS, and command injection
- **Security Audit Logging** with comprehensive event tracking
- **Content Sanitization** and validation

### 🚀 **High Performance**
- **Real-time Performance Monitoring** with metrics collection
- **Intelligent Caching** with TTL and LRU eviction
- **Connection Pooling** for database optimization
- **Query Optimization** for SQL and MongoDB
- **Load Testing Framework** with multiple test scenarios

### 🗄️ **Database Support**
- **SQL Databases**: SQLite, PostgreSQL, MySQL
- **NoSQL Databases**: MongoDB, Redis
- **Content Sources**: File, API, Web Scraping, Webhook
- **Health Monitoring** and connection management

### 🌐 **Production Ready**
- **Docker & Kubernetes** deployment with HPA and monitoring
- **CI/CD Pipeline** with security scanning and performance testing
- **Comprehensive Testing** with 456+ tests
- **API Documentation** with Swagger/OpenAPI
- **Monitoring & Observability** with Prometheus and Grafana

## 🚀 Quick Start

### Using pip (Recommended)

```bash
# Clone
git clone https://github.com/your-org/marketing-project.git
cd marketing-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt

# Install the project in development mode
pip install -e .

# Set up environment
cp env.example .env
# Fill .env with your secrets (especially OPENAI_API_KEY)

# Run the marketing project
python -m src.marketing_project.main run

# Start the server
python -m src.marketing_project.main serve

# Run tests
pytest

# Run security and database tests
python test_security_and_database.py

# Run load tests
python run_load_test.py --url http://localhost:8000 --test basic
```

## 🔌 API Endpoints

The Marketing Project provides a comprehensive REST API with enterprise-grade security and performance monitoring.

### **Core Endpoints**

| Method | Endpoint | Description | Authentication |
|--------|----------|-------------|----------------|
| `GET` | `/api/v1/health` | Health check for Kubernetes probes | None |
| `GET` | `/api/v1/ready` | Readiness check for Kubernetes probes | None |
| `POST` | `/api/v1/analyze` | Analyze content for marketing insights | API Key |
| `POST` | `/api/v1/pipeline` | Run complete marketing pipeline | API Key |
| `GET` | `/api/v1/content-sources` | List all content sources | API Key |
| `GET` | `/api/v1/content-sources/{name}/status` | Get source status | API Key |
| `POST` | `/api/v1/content-sources/{name}/fetch` | Fetch content from source | API Key |

### **Performance & Monitoring Endpoints**

| Method | Endpoint | Description | Authentication |
|--------|----------|-------------|----------------|
| `GET` | `/api/v1/performance/summary` | Get performance summary and metrics | API Key |
| `GET` | `/api/v1/performance/endpoints` | Get performance metrics by endpoint | API Key |
| `GET` | `/api/v1/performance/slow-requests` | Get requests slower than threshold | API Key |
| `GET` | `/api/v1/performance/error-requests` | Get error requests from last hour | API Key |

### **Security & Audit Endpoints**

| Method | Endpoint | Description | Authentication |
|--------|----------|-------------|----------------|
| `GET` | `/api/v1/security/audit` | Get security audit logs | API Key |
| `GET` | `/api/v1/security/stats` | Get security statistics and metrics | API Key |

### **Cache Management Endpoints**

| Method | Endpoint | Description | Authentication |
|--------|----------|-------------|----------------|
| `GET` | `/api/v1/cache/stats` | Get cache statistics and performance | API Key |
| `POST` | `/api/v1/cache/clear` | Clear all cache entries | Admin API Key |

### **System & Database Endpoints**

| Method | Endpoint | Description | Authentication |
|--------|----------|-------------|----------------|
| `GET` | `/api/v1/database/status` | Get database connection status | API Key |
| `GET` | `/api/v1/system/info` | Get system information and configuration | API Key |

### **Authentication**

All protected endpoints require an API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{"content": {"id": "test", "title": "Test", "content": "Test content", "type": "blog_post"}}' \
     http://localhost:8000/api/v1/analyze
```

### **API Documentation**

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## 🧩 Agents & Extensions

Drop new agents into `agents/` and workflows into `plugins/` with `@task` decorator. Manage sequences via `config/pipeline.yml`.

## 🔒 Security Features

### **Authentication & Authorization**
- **API Key Authentication**: Secure API key validation with role-based access
- **Rate Limiting**: Advanced rate limiting with IP and user-based limits
- **Attack Detection**: Automatic detection and blocking of suspicious patterns
- **IP Whitelisting**: Configurable IP whitelist for trusted sources

### **Input Validation**
- **SQL Injection Prevention**: Pattern detection and query sanitization
- **XSS Protection**: Script injection detection and content sanitization
- **Command Injection Prevention**: Shell command pattern blocking
- **Content Validation**: Comprehensive input validation and sanitization

### **Security Monitoring**
- **Audit Logging**: Comprehensive security event logging
- **Risk Scoring**: Automatic risk assessment for requests
- **Anomaly Detection**: Detection of unusual access patterns
- **Security Alerts**: Real-time security event notifications

## 🚀 Performance & Monitoring

### **Performance Monitoring**
- **Real-time Metrics**: Request/response time, memory, CPU usage
- **Performance Dashboards**: Built-in performance monitoring
- **Load Testing**: Comprehensive load testing framework
- **Query Optimization**: Automatic database query optimization

### **Caching & Optimization**
- **Intelligent Caching**: LRU cache with TTL support
- **Connection Pooling**: Database connection optimization
- **Query Caching**: Automatic query result caching
- **Response Compression**: Automatic response compression

### **Load Testing**
```bash
# Basic load test
python run_load_test.py --url http://localhost:8000 --test basic

# Stress test
python run_load_test.py --url http://localhost:8000 --test stress

# Comprehensive test
python run_load_test.py --url http://localhost:8000 --test all
```

## 🐳 Deployment

### **Docker**
```bash
# Build and run with Docker Compose
docker-compose up -d

# Build production image
docker build -t marketing-project-api:latest .
```

### **Kubernetes**
```bash
# Deploy to Kubernetes
kubectl apply -k k8s/

# Check deployment status
kubectl get pods -n marketing-project
```

### **Environment Configuration**
Copy `env.example` to `.env` and configure:
- API keys and authentication
- Database connections
- Security settings
- Performance monitoring
- Logging configuration

## 🌐 Internationalization

Templates live under `prompts/${TEMPLATE_VERSION}/{en,fr,...}/`. Set `TEMPLATE_VERSION=v1` in your `.env`.

## 📁 Project Structure

- `src/marketing_project/` - Main source code
  - `agents/` - Agent implementations
  - `core/` - Core models and utilities
  - `plugins/` - Extensible plugin system
  - `services/` - External service integrations
  - `prompts/` - Template system
- `tests/` - Test suite
- `config/` - Configuration files
- `docs/` - Documentation
- `k8/` - Kubernetes deployment files

## 🧩 Architecture

This project follows this architecture:
- **Agent-based design** - Modular, extensible agents
- **Plugin system** - Easy to add new functionality
- **Configuration-driven** - YAML-based pipeline configuration
- **Multi-locale support** - Internationalization ready
- **Comprehensive testing** - pytest with async support
- **Docker & K8s ready** - Production deployment ready
- **Modern Python tooling** - pip for dependency management

## 📦 Dependency Management

```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Add a new dependency
pip install package-name
# Then update requirements.txt manually

# Update dependencies
pip install --upgrade -r requirements.txt

# Show installed packages
pip list

# Freeze current environment
pip freeze > requirements-current.txt
```


## 🚀 Kubernetes Deployment

The project includes complete Kubernetes manifests for production deployment:

```bash
# Deploy to Kubernetes
kubectl apply -k k8/

# Or deploy individual components
kubectl apply -f k8/namespace.yml
kubectl apply -f k8/configmap.yml
kubectl apply -f k8/deployment.yml
kubectl apply -f k8/service.yml
kubectl apply -f k8/ingress.yml
kubectl apply -f k8/hpa.yml
kubectl apply -f k8/cronjob.yml
```

### Features:
- **Auto-scaling** - HPA based on CPU and memory usage
- **Health checks** - Liveness and readiness probes
- **TLS termination** - Secure HTTPS access
- **Resource limits** - Prevents resource exhaustion
- **Scheduled execution** - CronJob for automated processing
- **Monitoring ready** - Metrics endpoint and structured logging

See [`k8/README.md`](k8/README.md) for detailed deployment instructions.

## 📚 Documentation

See [`docs/`](docs/) for architecture diagrams and API reference.

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
