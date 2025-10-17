# Testing Guide

This guide explains the different testing options available for PRs and development.

## 🚀 **Quick PR Testing (Recommended)**

For most PRs, use the **PR Checks** workflow which runs:
- ✅ Quick unit tests (fails fast)
- ✅ Code formatting checks
- ✅ Security scans
- ✅ Docker build test
- ✅ CLI smoke tests

**Trigger**: Automatically on every PR

## 📋 **Available Testing Workflows**

### 1. **PR Checks** (`.github/workflows/pr-checks.yml`)
**Best for**: Most PRs, quick feedback
- Quick unit tests (max 3 failures)
- Linting (black, isort, flake8, mypy)
- Security scan (safety, bandit)
- Docker build test
- CLI smoke tests

### 2. **CI Pipeline** (`.github/workflows/ci.yml`)
**Best for**: Comprehensive testing
- Full unit test suite with coverage
- Integration tests
- All code quality checks
- Security scanning
- Docker image building

### 3. **Performance Tests** (`.github/workflows/performance.yml`)
**Best for**: Performance-critical changes
- Benchmark tests
- Memory usage tests
- Load testing with Locust
- Performance regression detection


## 🛠️ **Local Testing**

### Quick Tests
```bash
# Run quick tests locally
make test

# Run with coverage
make test-unit

# Run specific test file
pytest tests/test_specific.py -v

# Run specific test function
pytest tests/test_specific.py::test_function -v
```

### Code Quality
```bash
# Check formatting
make lint

# Auto-format code
make format

# Run security checks
make security
```

### Docker Testing
```bash
# Build Docker image
make docker-build

# Test Docker image
make docker-run

# Test specific commands
docker run --rm marketing-project:latest --help
```

## 🎯 **Choosing the Right Tests**

### **For Small Changes** (docs, config, minor fixes)
- Use **PR Checks** only
- Fast feedback, minimal resource usage

### **For Feature Changes** (new functionality)
- Use **PR Checks** + **CI Pipeline**
- Comprehensive testing with coverage

### **For Performance Changes** (optimizations, algorithms)
- Use **PR Checks** + **Performance Tests**
- Ensure no performance regressions


## ⚡ **Fast Testing Tips**

### 1. **Use pytest markers**
```python
@pytest.mark.quick
def test_quick_function():
    pass

@pytest.mark.slow
def test_slow_function():
    pass
```

### 2. **Run specific test categories**
```bash
# Run only quick tests
pytest -m quick

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration
```

### 3. **Use pytest-xdist for parallel testing**
```bash
# Run tests in parallel
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

### 4. **Use pytest-cov for coverage**
```bash
# Run with coverage
pytest --cov=src/marketing_project --cov-report=html

# Run with coverage and fail if below threshold
pytest --cov=src/marketing_project --cov-fail-under=80
```

## 🔧 **Customizing Tests**

### **Add Custom Test Markers**
```python
# In conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark integration tests")
    config.addinivalue_line("markers", "slow: mark slow tests")
    config.addinivalue_line("markers", "quick: mark quick tests")
```

### **Skip Tests Conditionally**
```python
import pytest
import os

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
def test_requires_api():
    pass
```

### **Use Fixtures for Setup**
```python
@pytest.fixture
def sample_content():
    return {"title": "Test", "content": "Test content"}

def test_process_content(sample_content):
    result = process_content(sample_content)
    assert result is not None
```

## 📊 **Test Reports**

### **Coverage Reports**
- HTML coverage report: `htmlcov/index.html`
- XML coverage report: `coverage.xml`
- Terminal coverage: `pytest --cov=src/marketing_project`

### **Performance Reports**
- Benchmark results: `pytest --benchmark-only`
- Memory profiling: `python -m memory_profiler script.py`

### **Security Reports**
- Safety report: `safety check --json`
- Bandit report: `bandit -r src/ -f json`

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Tests failing locally but passing in CI**
   - Check environment variables
   - Ensure all dependencies are installed
   - Check Python version compatibility

2. **Slow tests**
   - Use `pytest -m quick` for fast feedback
   - Mark slow tests with `@pytest.mark.slow`
   - Use `pytest --maxfail=1` to fail fast

3. **Flaky tests**
   - Add retries: `pytest --maxfail=1 --tb=short`
   - Use `pytest-rerunfailures` for automatic retries
   - Check for race conditions

4. **Memory issues**
   - Use `pytest --maxfail=1` to limit memory usage
   - Check for memory leaks in tests
   - Use `pytest-xdist` for parallel execution

### **Debug Commands**
```bash
# Run with verbose output
pytest -v -s

# Run with debug output
pytest --log-cli-level=DEBUG

# Run specific test with debug
pytest tests/test_specific.py::test_function -v -s --log-cli-level=DEBUG

# Run with coverage and debug
pytest --cov=src/marketing_project --cov-report=html -v -s
```

## 📈 **Best Practices**

1. **Write Fast Tests**
   - Keep unit tests under 1 second
   - Use mocks for external dependencies
   - Mark slow tests appropriately

2. **Use Descriptive Test Names**
   ```python
   def test_user_can_login_with_valid_credentials():
       pass

   def test_user_cannot_login_with_invalid_credentials():
       pass
   ```

3. **Test Edge Cases**
   - Empty inputs
   - Invalid inputs
   - Boundary conditions
   - Error conditions

4. **Use Fixtures for Common Setup**
   - Database connections
   - Test data
   - Mock objects

5. **Keep Tests Independent**
   - Each test should be able to run alone
   - Don't rely on test execution order
   - Clean up after each test

6. **Use Appropriate Assertions**
   - Be specific about what you're testing
   - Use descriptive assertion messages
   - Test one thing per test

## 🎯 **Summary**

- **Quick PRs**: Use PR Checks workflow
- **Feature PRs**: Use PR Checks + CI Pipeline
- **Performance PRs**: Use PR Checks + Performance Tests
- **Local Development**: Use `make test`, `make lint`, `make format`
