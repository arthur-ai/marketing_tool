# Marketing Project Makefile
.PHONY: help install install-dev test test-unit test-integration lint format clean build run serve docker-build docker-run deploy-staging deploy-production dev-up dev-down dev-restart dev-logs dev-build prod-up prod-down prod-restart prod-logs prod-build docker-clean pip-compile pip-compile-dev pip-compile-all

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  dev-up           Start development environment (hot-reload enabled)"
	@echo "  dev-down         Stop development environment"
	@echo "  dev-restart      Restart development containers"
	@echo "  dev-logs         View development logs (follow mode)"
	@echo "  dev-build        Rebuild development containers"
	@echo "  dev-shell        Open shell in development container"
	@echo ""
	@echo "Production:"
	@echo "  prod-up          Start production environment"
	@echo "  prod-down        Stop production environment"
	@echo "  prod-restart     Restart production containers"
	@echo "  prod-logs        View production logs (follow mode)"
	@echo "  prod-build       Rebuild production containers"
	@echo ""
	@echo "Local Development:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  run              Run the application locally"
	@echo "  serve            Start the FastAPI server locally"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint             Run all linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  security         Run security checks"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean up temporary files"
	@echo "  docker-clean     Clean up Docker resources"
	@echo "  build            Build the package"
	@echo "  pip-compile      Regenerate requirements.txt with Python 3.13"
	@echo "  pip-compile-dev  Regenerate requirements-dev.txt with Python 3.13"
	@echo "  pip-compile-all  Regenerate both requirements files with Python 3.13"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-staging   Deploy to staging environment"
	@echo "  deploy-production Deploy to production environment"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

# Testing
test: test-unit test-integration

test-unit:
	pytest tests/ -v --cov=src/marketing_project --cov-report=xml --cov-report=html

test-integration:
	pytest tests/integrations/ -v -m integration

# Linting and formatting
lint: lint-black lint-isort lint-flake8

lint-black:
	black --check --diff src/ tests/

lint-isort:
	isort --check-only --diff src/ tests/

lint-flake8:
	flake8 src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Security checks
security: security-bandit

security-bandit:
	bandit -r src/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name "htmlcov" -delete
	rm -f .coverage coverage.xml

# Build
build:
	python setup.py sdist bdist_wheel

# Run application
run:
	marketing-project run

serve:
	marketing-project serve

# Docker Development operations
dev-up:
	@echo "Starting development environment with hot-reload..."
	docker-compose -f deploy/docker/docker-compose.dev.yml up -d
	@echo "Development environment started! Code changes will reload automatically."
	@echo "API available at: http://localhost:8000"
	@echo "View logs: make dev-logs"

dev-down:
	@echo "Stopping development environment..."
	docker-compose -f deploy/docker/docker-compose.dev.yml down

dev-restart:
	@echo "Restarting development containers..."
	docker-compose -f deploy/docker/docker-compose.dev.yml restart

dev-logs:
	@echo "Following development logs (Ctrl+C to exit)..."
	docker-compose -f deploy/docker/docker-compose.dev.yml logs -f api

dev-build:
	@echo "Building development containers..."
	docker-compose -f deploy/docker/docker-compose.dev.yml build --no-cache
	docker-compose -f deploy/docker/docker-compose.dev.yml up -d

dev-shell:
	@echo "Opening shell in development container..."
	docker-compose -f deploy/docker/docker-compose.dev.yml exec api /bin/bash

# Docker Production operations
prod-up:
	@echo "Starting production environment..."
	docker-compose -f deploy/docker/docker-compose.yml up -d
	@echo "Production environment started!"
	@echo "API available at: http://localhost:8000"

prod-down:
	@echo "Stopping production environment..."
	docker-compose -f deploy/docker/docker-compose.yml down

prod-restart:
	@echo "Restarting production containers..."
	docker-compose -f deploy/docker/docker-compose.yml restart

prod-logs:
	@echo "Following production logs (Ctrl+C to exit)..."
	docker-compose -f deploy/docker/docker-compose.yml logs -f api

prod-build:
	@echo "Building production containers..."
	docker-compose -f deploy/docker/docker-compose.yml build --no-cache
	docker-compose -f deploy/docker/docker-compose.yml up -d

# Legacy Docker operations (for backwards compatibility)
docker-build:
	docker build -f deploy/docker/Dockerfile -t marketing-project:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env marketing-project:latest

# Docker cleanup
docker-clean:
	@echo "Cleaning up Docker resources..."
	docker-compose -f deploy/docker/docker-compose.dev.yml down -v
	docker-compose -f deploy/docker/docker-compose.yml down -v
	@echo "Removing dangling images..."
	docker image prune -f
	@echo "Docker cleanup complete!"

# Deployment (requires kubectl and proper context)
deploy-staging:
	kubectl set image deployment/marketing-project-server marketing-project=registry.gitlab.com/your-group/arthur-marketing-generator:latest -n marketing-project
	kubectl rollout status deployment/marketing-project-server -n marketing-project

deploy-production:
	kubectl set image deployment/marketing-project-server marketing-project=registry.gitlab.com/your-group/arthur-marketing-generator:latest -n marketing-project
	kubectl rollout status deployment/marketing-project-server -n marketing-project

# Development helpers
dev-setup: install-dev
	@echo "Setting up development environment..."
	@echo "Creating logs directory..."
	@mkdir -p logs
	@echo "Copying environment file..."
	@cp env.example .env
	@echo "Development environment ready!"

# CI/CD helpers
ci-test: install-dev test lint security

ci-build: docker-build

ci-deploy: deploy-staging

# Dependency management
pip-compile:
	@echo "Regenerating requirements.txt with Python 3.13..."
	docker run --rm -v "$(PWD):/workspace" -w /workspace python:3.13-slim bash -c "pip install --no-cache-dir pip-tools && pip-compile --upgrade requirements.in"

pip-compile-dev:
	@echo "Regenerating requirements-dev.txt with Python 3.13..."
	docker run --rm -v "$(PWD):/workspace" -w /workspace python:3.13-slim bash -c "pip install --no-cache-dir pip-tools && pip-compile --upgrade requirements-dev.in"

pip-compile-all: pip-compile pip-compile-dev
	@echo "All requirements files regenerated with Python 3.13!"

# GitHub Actions helpers
gh-format: format
	@echo "Code formatted. Commit and push to trigger GitHub Actions."

gh-test: test lint security
	@echo "All checks passed. Ready for GitHub Actions."

gh-release:
	@echo "To create a release:"
	@echo "1. Update version in setup.py"
	@echo "2. Create and push a tag: git tag v1.0.0 && git push origin v1.0.0"
	@echo "3. GitHub Actions will automatically create a release"
