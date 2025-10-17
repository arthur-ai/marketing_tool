# Marketing Project Test Suite

This folder contains comprehensive tests for all agents, plugins, integrations, and the main CLI of the Marketing Project.

## Structure

- `agents/` - Tests for each agent factory (async, covers prompt loading/config)
- `plugins/` - Comprehensive tests for all plugins (unit tests, integration tests, edge cases)
- `integrations/` - Integration tests for full pipeline runs
- `test_main.py` - CLI/entrypoint coverage
- `conftest.py` - Global test fixtures and LLM mocking
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

**Plugin Tests:**
```bash
# All plugin tests
python -m pytest tests/plugins/

# Specific plugin
python -m pytest tests/plugins/test_article_generation.py

# Pattern matching
python -m pytest tests/plugins/ -k "seo"
```

**Agent Tests:**
```bash
python -m pytest tests/agents/
```

**Integration Tests:**
```bash
python -m pytest tests/integrations/
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

## Plugin Test Coverage

The plugin tests provide comprehensive coverage for all 10 plugins:

- **Article Generation** - Content creation and optimization
- **Blog Posts** - Blog post processing and routing
- **Content Analysis** - Content type analysis and metadata extraction
- **Content Formatting** - Formatting rules and visual elements
- **Internal Docs** - Internal documentation processing
- **Marketing Brief** - Marketing brief analysis and processing
- **Release Notes** - Release notes processing and routing
- **SEO Keywords** - Keyword extraction and optimization
- **SEO Optimization** - SEO analysis and recommendations
- **Transcripts** - Transcript processing and routing

Each plugin includes:
- ✅ Unit tests for all functions
- ✅ Integration tests between plugins
- ✅ Edge case and error handling tests
- ✅ Performance and memory efficiency tests

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
