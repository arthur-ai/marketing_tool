# Contributing to Marketing Tool

First off, thank you for considering contributing! 🎉

## How to Contribute

### 1. Fork and Clone
1. Fork the repo
2. Clone your fork: `git clone https://github.com/your-username/marketing_tool.git`
3. Add upstream remote: `git remote add upstream https://github.com/arthur-ai/marketing_tool.git`

### 2. Set Up Development Environment
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in editable mode
pip install -e .

# Install pre-commit hooks (REQUIRED)
pre-commit install
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/my-new-feature
```

### 4. Make Your Changes
- Write clear, concise code following our coding guidelines
- Add tests for new features under `tests/`
- Update documentation as needed

### 5. Run Tests and Checks
```bash
# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run pre-commit hooks manually (optional - they run automatically on commit)
pre-commit run --all-files
```

### 6. Commit Your Changes
```bash
git add .
git commit -m "feat: add awesome new feature"
```

**Note**: Pre-commit hooks will run automatically and may modify files. If changes are made, stage them and commit again.

### 7. Push and Create PR
```bash
git push origin feature/my-new-feature
```
Then open a Pull Request against `main`.

## Pre-commit Hooks

This repository uses [pre-commit](https://pre-commit.com/) to automatically format code and run checks before each commit.

### What the Hooks Do
The pre-commit hooks automatically:
- ✂️ Trim trailing whitespace
- 📄 Fix end of files (ensure newline at EOF)
- ✅ Validate YAML, JSON, and TOML syntax
- 🚫 Check for large files and private keys
- 🔍 Detect merge conflicts
- 🎨 Format Python code with **Black**
- 📦 Sort imports with **isort**

### First-Time Setup
```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Manual Runs
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

### If Hooks Make Changes
When hooks modify files, your commit will be aborted. Simply stage and commit again:
```bash
git add .
git commit -m "your message"
```

### Bypassing Hooks (Not Recommended)
```bash
git commit --no-verify -m "your message"
```
⚠️ **Warning**: Only use this in emergencies, as it may cause CI failures.

## Coding Guidelines

### Python Style
- Follow **PEP8** style guidelines
- Use **Black** for formatting (configured in pre-commit)
- Use **isort** for import sorting (configured in pre-commit)
- Run `flake8 .` to check for linting issues
- Maximum line length: 88 characters (Black default)

### Testing
- Write unit tests for all new features under `tests/`
- Maintain minimum 70% test coverage
- Use pytest fixtures and async tests where appropriate
- Test both success and error cases

### Commit Messages
Follow conventional commit format:
- `feat: add new feature`
- `fix: correct bug in X`
- `docs: update documentation`
- `style: format code`
- `refactor: restructure code`
- `test: add tests for X`
- `chore: update dependencies`

### LLM Prompts
- Use prompt templates from `prompts/v1/...` for all LLM calls
- Document prompt changes in commit messages
- Version prompts appropriately

## Issue & PR Templates
Please fill in all requested sections in our issue and PR templates to help us review faster.

## Questions?
Feel free to open an issue for any questions or concerns!
