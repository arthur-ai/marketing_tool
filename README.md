# Marketing Project 🤖📈

[![CI Status](https://github.com/your-org/marketing-project/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/marketing-project/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/your-org/marketing-project/main)](https://codecov.io/gh/your-org/marketing-project)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A marketing agentic project with extensible agents, plugins, and multi-locale support.

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
```

## 🧩 Agents & Extensions

Drop new agents into `agents/` and workflows into `plugins/` with `@task` decorator. Manage sequences via `config/pipeline.yml`.

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
