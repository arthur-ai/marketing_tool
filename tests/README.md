# Marketing Project Test Suite

This folder contains comprehensive tests for the function-based pipeline, processors, plugins, and integrations of the Marketing Project.

## Structure

- `api/` - Tests for API endpoints (health, system, content, processors, core)
- `processors/` - Tests for content processors (blog, transcript, release notes)
- `plugins/` - Tests for pipeline step plugins (SEO keywords, marketing brief, article generation, etc.)
- `services/` - Tests for service layer (function pipeline, job manager, Redis manager)
- `worker/` - Tests for ARQ worker functions
- `integrations/` - End-to-end integration tests for complete pipeline flows
- `integration/` - Integration tests for Redis and other external services
- `middleware/` - Tests for authentication and RBAC middleware
- `models/` - Tests for user context and other models
- `utils/` - Test utilities and helpers (JWT generation, etc.)
- `conftest.py` - Global test fixtures and mocks
- `requirements-test.txt` - All requirements to run these tests

## Running the Tests

### Install Dependencies
```bash
pip install -r requirements-test.txt
```

### Run All Tests
```bash
python -m pytest tests/
```

### Run Specific Test Categories

**Processor Tests:**
```bash
# All processor tests
python -m pytest tests/processors/

# Specific processor
python -m pytest tests/processors/test_blog_processor.py
```

**Plugin Tests:**
```bash
# All plugin tests
python -m pytest tests/plugins/

# Specific plugin
python -m pytest tests/plugins/test_seo_keywords_plugin.py

# Pattern matching
python -m pytest tests/plugins/ -k "seo"
```

**Service Tests:**
```bash
# All service tests
python -m pytest tests/services/

# Function pipeline tests
python -m pytest tests/services/test_function_pipeline.py
```

**Integration Tests:**
```bash
# End-to-end tests
python -m pytest tests/integrations/

# Redis integration tests
python -m pytest tests/integration/
```

### Advanced Options

**With Coverage:**
```bash
python -m pytest tests/ --cov=src/marketing_project --cov-report=html --cov-report=term
```

**In Parallel:**
```bash
# Install pytest-xdist first: pip install pytest-xdist
python -m pytest tests/ -n auto
```

**Verbose Output:**
```bash
python -m pytest tests/ -v
```

**Debug Mode:**
```bash
python -m pytest tests/ --pdb
```

## Test Coverage

### Processors
- **Blog Processor** - Blog post processing through function pipeline
- **Transcript Processor** - Transcript processing through function pipeline
- **Release Notes Processor** - Release notes processing through function pipeline

### Plugins (Pipeline Steps)
- **SEO Keywords Plugin** - Keyword extraction and analysis
- **Marketing Brief Plugin** - Marketing brief generation
- **Article Generation Plugin** - Article content generation
- **SEO Optimization Plugin** - SEO optimization
- **Suggested Links Plugin** - Internal link suggestions
- **Content Formatting Plugin** - Final content formatting
- **Content Analysis** - Content type analysis and metadata extraction

### Services
- **Function Pipeline** - Complete pipeline execution and orchestration
- **Job Manager** - Background job tracking and ARQ integration
- **Redis Manager** - Redis connection management and resilience

### Workers
- **ARQ Worker Functions** - Background job processing (blog, transcript, release notes)

### Authentication & Authorization
- **Keycloak Authentication** - JWT token validation and user context extraction
- **RBAC Middleware** - Role-based access control testing
- **User Context Models** - User context model validation
- **Integration Tests** - End-to-end authentication flow testing

Each test suite includes:
- ✅ Unit tests for core functionality
- ✅ Error handling and edge cases
- ✅ Integration tests for complete workflows
- ✅ Mocking for external dependencies

## Test Categories

**Unit Tests:**
```bash
python -m pytest tests/ -k "not integration"
```

**Integration Tests:**
```bash
python -m pytest tests/ -k "integration"
```

**Error Handling Tests:**
```bash
python -m pytest tests/ -k "error"
```

**Authentication Tests:**
```bash
# Run all authentication-related tests
python -m pytest tests/middleware/test_keycloak_auth.py tests/middleware/test_rbac.py tests/models/test_user_context.py

# Run integration tests for authentication
python -m pytest tests/integration/test_keycloak_integration.py -v
```

## Continuous Integration

The test suite is designed for CI environments:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python -m pytest tests/ --cov=src/marketing_project --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```
